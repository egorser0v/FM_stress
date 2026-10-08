"""Sparse representations inside conditional flow matching.

Independent small adaptations, not reproductions of released language-model
SAEs or CRsAE experiments. All heads map (interpolant, time, past) to velocity.
History autoencoders reconstruct ONLY the supplied past. ``auxiliary_loss()``
returns the weighted scalar for the most recent forward, to add to FM MSE.
``activity_statistics()`` returns detached diagnostics for that same forward.
See docs/sparse_model_design.md for definitions and scientific limitations.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .models import AdaLNMLP
from .extended_models import JointHead, TemporalUNet


def topk_mask(values: Tensor, k: int, *, magnitude: bool = False) -> Tensor:
    """Keep at most k coordinates along the last axis; exact zeros elsewhere."""
    if not 1 <= k <= values.shape[-1]:
        raise ValueError('topk must lie between 1 and latent_dim')
    ranking = values.abs() if magnitude else values
    indices = ranking.topk(k, dim=-1, sorted=False).indices
    return torch.zeros_like(values).scatter(-1, indices, values.gather(-1, indices))


class AuxiliaryHead(JointHead):
    """The cached loss is replaced on each forward, never accumulated."""
    def __init__(self, lookback, horizon, channels):
        super().__init__(lookback, horizon, channels)
        self._auxiliary: Tensor | None = None
        self._activity: dict[str, Tensor] = {}

    def auxiliary_loss(self, history: Tensor | None = None) -> Tensor:
        # The optional history argument is accepted for runner compatibility;
        # forward already encoded it, so no second reconstruction is performed.
        if self._auxiliary is None:
            raise RuntimeError('Call forward before auxiliary_loss')
        return self._auxiliary

    def activity_statistics(self) -> dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self._activity.items()}

    def _cache(self, auxiliary: Tensor, **statistics: Tensor) -> None:
        self._auxiliary = auxiliary
        self._activity = {name: value.detach() for name, value in statistics.items()}


def _make_core(lookback, horizon, channels, width, depth, time_dim, condition_dim):
    core = AdaLNMLP(horizon=horizon * channels, lookback=lookback * channels,
                   encoder_dim=condition_dim, width=width, depth=depth, time_dim=time_dim)
    core.encoder = nn.Identity()
    return core


class HistoryAutoencoderFM(AuxiliaryHead):
    """History -> ReLU (optionally TopK) code -> FM conditioning.

    The decoder has unit-L2 columns so representation scales cannot be changed
    arbitrarily by shrinking its coefficients and enlarging dictionary atoms.
    Dense and TopK controls have identical trainable parameters and initialization.
    """
    def __init__(self, lookback, horizon, channels, width=56, depth=3,
                 time_dim=32, latent_dim=64, topk=8, sparse=True,
                 reconstruction_weight=.05):
        super().__init__(lookback, horizon, channels)
        if not 1 <= topk <= latent_dim:
            raise ValueError('topk must lie between 1 and latent_dim')
        self.latent_dim, self.topk, self.sparse = latent_dim, topk, sparse
        self.reconstruction_weight = reconstruction_weight
        self.encoder = nn.Linear(lookback * channels, latent_dim)
        self.dictionary = nn.Parameter(torch.randn(lookback * channels, latent_dim))
        self.history_center = nn.Parameter(torch.zeros(lookback * channels))
        self.core = _make_core(lookback, horizon, channels, width, depth, time_dim, latent_dim)
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

    def encode_history(self, history: Tensor) -> Tensor:
        values = F.relu(self.encoder(history.flatten(1) - self.history_center))
        return topk_mask(values, self.topk) if self.sparse else values

    def reconstruct_history(self, code: Tensor) -> Tensor:
        return F.linear(code, F.normalize(self.dictionary, dim=0)) + self.history_center

    def forward(self, x, t, history):
        self.check(x, t, history)
        code = self.encode_history(history)
        reconstruction = self.reconstruct_history(code)
        recon_loss = F.mse_loss(reconstruction, history.flatten(1))
        self._cache(self.reconstruction_weight * recon_loss,
                    history_reconstruction_mse=recon_loss,
                    code_active_fraction=(code != 0).float().mean(),
                    code_abs_mean=code.abs().mean())
        return self.core(x.flatten(1), t, code).reshape_as(x)


class HistoryConvolutionalFM(AuxiliaryHead):
    """Three tied-dictionary proximal-gradient steps encode the history.

    This is CRsAE/ISTA-inspired, not a reproduction: same-padded multichannel
    convolutions, learned positive thresholds, and jointly trained FM conditioning.
    Dense control runs the identical steps with zero thresholds and no L1 loss.
    A rigorous Young-inequality operator-norm upper bound stabilizes the tied
    convolution/transpose-convolution updates; learned steps remain in (0, 2).
    """
    def __init__(self, lookback, horizon, channels, width=56, depth=3,
                 time_dim=32, latent_dim=64, conv_steps=3, conv_kernel=5,
                 sparse=True, reconstruction_weight=.05, sparsity_weight=.001):
        super().__init__(lookback, horizon, channels)
        if conv_kernel < 1 or conv_kernel % 2 != 1 or conv_steps < 1:
            raise ValueError('conv_kernel must be positive and odd; conv_steps >= 1')
        self.latent_dim, self.conv_steps = latent_dim, conv_steps
        self.padding, self.sparse = conv_kernel // 2, sparse
        self.reconstruction_weight, self.sparsity_weight = reconstruction_weight, sparsity_weight
        self.dictionary = nn.Parameter(torch.randn(latent_dim, channels, conv_kernel))
        self.history_center = nn.Parameter(torch.zeros(1, channels, 1))
        self.step_logits = nn.Parameter(torch.zeros(conv_steps))
        # A small threshold permits activation from the conservative normalized
        # dictionary, while retaining a differentiable sparsity-control parameter.
        if sparse:
            self.threshold_logits = nn.Parameter(torch.full((latent_dim,), math.log(math.expm1(.01))))
        else:
            self.register_parameter('threshold_logits', None)
        # Four ordered bins preserve coarse timing instead of global pooling.
        self.pool_bins = min(4, lookback)
        self.core = _make_core(lookback, horizon, channels, width, depth,
                               time_dim, latent_dim * self.pool_bins)

    def normalized_dictionary(self) -> Tensor:
        atoms = F.normalize(self.dictionary.flatten(1), dim=1).reshape_as(self.dictionary)
        # Each input/output-channel pair has convolution norm <= its kernel L1.
        # Frobenius norm of that nonnegative matrix bounds its spectral norm.
        upper = atoms.abs().sum(-1).square().sum().sqrt().clamp_min(1e-6)
        return atoms / upper

    def encode_history(self, history: Tensor) -> tuple[Tensor, Tensor]:
        centered = history.transpose(1, 2) - self.history_center
        weight = self.normalized_dictionary()
        code = centered.new_zeros((len(history), self.latent_dim, self.lookback))
        for index in range(self.conv_steps):
            reconstruction = F.conv_transpose1d(code, weight, padding=self.padding)
            step = 2 * self.step_logits[index].sigmoid()
            proposal = code + step * F.conv1d(centered - reconstruction, weight, padding=self.padding)
            if self.sparse:
                threshold = step * F.softplus(self.threshold_logits)[None, :, None]
                code = proposal.sign() * F.relu(proposal.abs() - threshold)
            else:
                code = proposal
        reconstruction = F.conv_transpose1d(code, weight, padding=self.padding) + self.history_center
        return code, reconstruction.transpose(1, 2)

    def forward(self, x, t, history):
        self.check(x, t, history)
        code, reconstruction = self.encode_history(history)
        recon_loss = F.mse_loss(reconstruction, history)
        l1 = code.abs().mean()
        auxiliary = self.reconstruction_weight * recon_loss
        if self.sparse:
            auxiliary = auxiliary + self.sparsity_weight * l1
        self._cache(auxiliary, history_reconstruction_mse=recon_loss,
                    code_active_fraction=(code != 0).float().mean(), code_abs_mean=l1)
        # Explicit ordered bins also support non-divisible lengths on MPS.
        condition = torch.stack([part.mean(-1) for part in torch.tensor_split(code, self.pool_bins, dim=-1)], dim=-1).flatten(1)
        return self.core(x.flatten(1), t, condition).reshape_as(x)


class VelocityDictionaryFM(AuxiliaryHead):
    """Signed velocity coefficients with optional magnitude TopK truncation.

    This is a velocity dictionary head, not an autoencoder. Dense/TopK use the
    identical architecture. Signed coefficients avoid imposing nonnegative
    velocities. A small random coefficient readout avoids permanently selecting the same
    initially tied atoms; dense and TopK controls share this initialization.
    """
    def __init__(self, lookback, horizon, channels, width=56, depth=3,
                 time_dim=32, latent_dim=64, topk=8, sparse=True):
        super().__init__(lookback, horizon, channels)
        if not 1 <= topk <= latent_dim:
            raise ValueError('topk must lie between 1 and latent_dim')
        self.latent_dim, self.topk, self.sparse = latent_dim, topk, sparse
        self.core = _make_core(lookback, horizon, channels, width, depth,
                               time_dim, lookback * channels)
        self.core.output_projection = nn.Linear(width, latent_dim)
        nn.init.normal_(self.core.output_projection.weight, std=.001)
        nn.init.zeros_(self.core.output_projection.bias)
        self.dictionary = nn.Parameter(torch.randn(horizon * channels, latent_dim))
        self._last_code: Tensor | None = None

    def coefficients(self, x, t, history) -> Tensor:
        values = self.core(x.flatten(1), t, history.flatten(1))
        return topk_mask(values, self.topk, magnitude=True) if self.sparse else values

    def forward(self, x, t, history):
        self.check(x, t, history)
        code = self.coefficients(x, t, history)
        self._last_code = code.detach()
        output = F.linear(code, F.normalize(self.dictionary, dim=0)).reshape_as(x)
        self._cache(output.new_zeros(()),
                    code_active_fraction=(code != 0).float().mean(), code_abs_mean=code.abs().mean())
        return output


class HardConcreteGate(nn.Module):
    """Channel gate from Louizos et al.; own checkpointed CPU random stream.

    Training gates are shared across batch and time, and independently sampled
    per forward. Evaluation uses the standard deterministic stretched sigmoid.
    No torch global RNG is touched by forward, preserving paired FM minibatches.
    """
    temperature = 2 / 3
    lower, upper = -.1, 1.1

    def __init__(self, channels: int, seed: int = 1729, initial_keep: float = .5):
        super().__init__()
        if channels < 1 or not 0 < initial_keep < 1:
            raise ValueError('channels >= 1 and initial_keep in (0, 1) required')
        self.log_alpha = nn.Parameter(torch.full((channels,), math.log(initial_keep / (1 - initial_keep))))
        self._generator = torch.Generator(device='cpu').manual_seed(seed)

    def get_extra_state(self):
        return {'rng_state': self._generator.get_state()}

    def set_extra_state(self, state):
        self._generator.set_state(state['rng_state'].cpu())

    def expected_nonzero(self) -> Tensor:
        return torch.sigmoid(self.log_alpha - self.temperature * math.log(-self.lower / self.upper))

    def deterministic_gate(self) -> Tensor:
        return (self.log_alpha.sigmoid() * (self.upper - self.lower) + self.lower).clamp(0, 1)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            noise = torch.rand(self.log_alpha.shape, generator=self._generator).to(x.device, x.dtype).clamp(1e-6, 1 - 1e-6)
            stretched = torch.sigmoid((noise.log() - torch.log1p(-noise) + self.log_alpha) / self.temperature)
            gate = (stretched * (self.upper - self.lower) + self.lower).clamp(0, 1)
        else:
            gate = self.deterministic_gate()
        return x * gate[None, :, None]


class L0TemporalUNet(TemporalUNet):
    """Original temporal U-Net plus gates after each encoder/decoder block.

    Gates select hidden channels, not individual weights. Implemented using
    dense operations, so no sparse-kernel speed or storage saving is claimed.
    """
    def __init__(self, lookback, horizon, channels, width=16, depth=3,
                 time_dim=32, l0_weight=.001, gate_seed=1729):
        super().__init__(lookback, horizon, channels, width, depth, time_dim)
        widths = [width * 2 ** i for i in range(depth)]
        self.encoder_gates = nn.ModuleList(HardConcreteGate(w, gate_seed + i) for i, w in enumerate(widths))
        self.decoder_gates = nn.ModuleList(HardConcreteGate(w, gate_seed + depth + i)
                                          for i, w in enumerate(reversed(widths[:-1])))
        self.l0_weight = l0_weight

    def gates(self):
        return list(self.encoder_gates) + list(self.decoder_gates)

    def auxiliary_loss(self, history: Tensor | None = None) -> Tensor:
        return self.l0_weight * torch.cat([g.expected_nonzero() for g in self.gates()]).mean()

    def activity_statistics(self) -> dict[str, float]:
        expected = torch.cat([g.expected_nonzero().detach() for g in self.gates()])
        deterministic = torch.cat([g.deterministic_gate().detach() for g in self.gates()])
        return {'gate_expected_active_fraction': float(expected.mean().cpu()),
                'gate_eval_active_fraction': float((deterministic != 0).float().mean().cpu()),
                'gate_eval_mean': float(deterministic.mean().cpu())}

    def forward(self, x, t, history):
        h = self.check(x, t, history)
        condition = self.time_embedding(t) + self.history_projection(h)
        z = self.input_projection(x.transpose(1, 2))
        skips = []
        for index, (encoder, gate) in enumerate(zip(self.encoders, self.encoder_gates)):
            z = gate(encoder(z, condition))
            skips.append(z)
            if index < len(self.down):
                z = self.down[index](z)
        for up, decoder, gate, skip in zip(self.up, self.decoders, self.decoder_gates, reversed(skips[:-1])):
            z = up(F.interpolate(z, size=skip.shape[-1], mode='nearest'))
            z = gate(decoder(torch.cat((z, skip), dim=1), condition))
        return self.output_projection(z).transpose(1, 2)


def create_sparse_model(head, lookback, horizon, channels, width=56, depth=3,
                        latent_dim=64, topk=8, conv_steps=3, conv_kernel=5,
                        reconstruction_weight=.05, sparsity_weight=.001,
                        l0_weight=.001, gate_seed=1729, time_dim=32):
    if width < 2 or depth < 1 or latent_dim < 1:
        raise ValueError('width >= 2, depth >= 1, latent_dim >= 1 required')
    if min(reconstruction_weight, sparsity_weight, l0_weight) < 0:
        raise ValueError('Auxiliary-loss weights must be nonnegative')
    common = dict(lookback=lookback, horizon=horizon, channels=channels,
                  width=width, depth=depth, time_dim=time_dim)
    if head in ('history_dense_ae', 'history_topk_ae'):
        return HistoryAutoencoderFM(**common, latent_dim=latent_dim, topk=topk,
                                    sparse=head == 'history_topk_ae', reconstruction_weight=reconstruction_weight)
    if head in ('history_conv_dense', 'history_conv_sparse'):
        return HistoryConvolutionalFM(**common, latent_dim=latent_dim, conv_steps=conv_steps,
                                      conv_kernel=conv_kernel, sparse=head == 'history_conv_sparse',
                                      reconstruction_weight=reconstruction_weight, sparsity_weight=sparsity_weight)
    if head in ('dictionary_dense', 'dictionary_topk'):
        return VelocityDictionaryFM(**common, latent_dim=latent_dim, topk=topk, sparse=head == 'dictionary_topk')
    if head == 'unet_l0':
        return L0TemporalUNet(**common, l0_weight=l0_weight, gate_seed=gate_seed)
    raise ValueError(f'Unknown sparse head: {head}')
