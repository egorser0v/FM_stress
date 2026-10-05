"""Small conditional velocity networks for the GP flow-matching stress test.

AdaLN architecture adapted from thuml/Sundial's official ``flow_loss.py``.
HiPPO/NPLR initialization adapted from state-spaces/s4 (Apache-2.0).
The real S4 kernel below is an exact finite-horizon realization of a rank-one
NPLR SSM with bilinear discretization, not a learned arbitrary convolution.
See docs/model_provenance.md and vendor/ for attribution and exact changes.
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LookbackEncoder(nn.Module):
    """Same small, trainable encoder architecture for every experiment cell."""

    def __init__(self, lookback: int, encoder_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(lookback, encoder_dim), nn.SiLU(),
                                 nn.Linear(encoder_dim, encoder_dim))

    def forward(self, history: Tensor) -> Tensor:
        return self.net(history)


class TimeEmbedding(nn.Module):
    def __init__(self, width: int, time_dim: int = 32):
        super().__init__()
        if time_dim < 2 or time_dim % 2:
            raise ValueError("time_dim must be positive and even")
        self.register_buffer("frequencies", torch.exp(
            -math.log(10000) * torch.arange(time_dim // 2) / (time_dim // 2)))
        self.net = nn.Sequential(nn.Linear(time_dim, width), nn.SiLU(),
                                 nn.Linear(width, width))

    def forward(self, t: Tensor) -> Tensor:
        phase = t.reshape(-1, 1) * 1000 * self.frequencies[None]
        return self.net(torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1))


class AdaLNBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 3 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale, gate = self.modulation(condition).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class AdaLNMLP(nn.Module):
    def __init__(self, horizon: int = 16, lookback: int = 32, width: int = 64,
                 encoder_dim: int = 32, depth: int = 3, time_dim: int = 32):
        super().__init__()
        self.horizon = horizon
        self.encoder = LookbackEncoder(lookback, encoder_dim)
        self.time_embedding = TimeEmbedding(width, time_dim)
        self.condition_projection = nn.Linear(encoder_dim, width)
        self.input_projection = nn.Linear(horizon, width)
        self.blocks = nn.ModuleList(AdaLNBlock(width) for _ in range(depth))
        self.final_norm = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.output_projection = nn.Linear(width, horizon)
        # Match the released TimeFlow AdaLN-Zero initialization.
        self.apply(_xavier_linear)
        for index in (0, 2):
            nn.init.normal_(self.time_embedding.net[index].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, x: Tensor, t: Tensor, history: Tensor) -> Tensor:
        condition = self.time_embedding(t) + self.condition_projection(self.encoder(history))
        z = self.input_projection(x)
        for block in self.blocks:
            z = block(z, condition)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        return self.output_projection(self.final_norm(z) * (1 + scale) + shift)


def _xavier_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@lru_cache(maxsize=16)
def _hippo_nplr(state_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Official S4 LegS rank correction + conjugate eigenbasis (CPU init only)."""
    if state_size < 4 or state_size % 2:
        raise ValueError("state_size must be even and at least four")
    q = torch.arange(state_size, dtype=torch.float64)
    r = torch.sqrt(2 * q + 1)
    a = -torch.tril(r[:, None] * r[None, :]) + torch.diag(q)
    p = torch.sqrt(q + 0.5)
    normal = a + p[:, None] * p[None, :]
    # Subtract scalar real diagonal to make the eigensolver input Hermitian.
    imag, vectors = torch.linalg.eigh((normal + 0.5 * torch.eye(state_size)) * -1j)
    vectors = vectors[:, :state_size // 2]
    eigenvalues = torch.complex(torch.full_like(imag[:state_size // 2], -0.5),
                                imag[:state_size // 2])
    p_complex = vectors.mH @ p.to(torch.complex128)
    b_complex = vectors.mH @ r.to(torch.complex128)
    return (eigenvalues.to(torch.complex64), p_complex.to(torch.complex64),
            b_complex.to(torch.complex64))


class S4NPLRKernel(nn.Module):
    """Rank-one S4-LegS kernel, evaluated using entirely real MPS operations.

    A = diag(a + i omega) - P P* in a conjugate-pair basis.
    In the stored real half-state [Re(z), Im(z)], the correction is -2 p p^T.
    Bilinear discretization is computed with a 2x2 block inverse plus the
    Sherman-Morrison identity, avoiding complex ops and matrix solves on MPS.
    """

    def __init__(self, width: int, state_size: int = 16, directions: int = 2,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        eigenvalues, p, b = _hippo_nplr(state_size)
        half = state_size // 2
        self.width, self.state_size = width, state_size
        self.log_dt = nn.Parameter(torch.empty(width).uniform_(math.log(dt_min), math.log(dt_max)))
        self.log_a_real = nn.Parameter(eigenvalues.real.neg().log().expand(width, half).clone())
        self.a_imag = nn.Parameter(eigenvalues.imag.expand(width, half).clone())
        self.p = nn.Parameter(torch.cat((p.real, p.imag)).expand(width, state_size).clone())
        self.b = nn.Parameter(torch.cat((b.real, b.imag)).expand(width, state_size).clone())
        # c stores the actual output row in real state coordinates.
        self.c = nn.Parameter(torch.randn(directions, width, state_size) / math.sqrt(state_size))
        for parameter in (self.log_dt, self.log_a_real, self.a_imag, self.p, self.b):
            parameter._optim = {"weight_decay": 0.0}

    def discretize(self) -> tuple[Tensor, Tensor]:
        half = self.state_size // 2
        h = self.log_dt.exp()[:, None] / 2
        ar = -self.log_a_real.exp()
        diagonal = 1 - h * ar
        imaginary = h * self.a_imag
        denominator = diagonal.square() + imaginary.square()
        d, w = diagonal / denominator, imaginary / denominator
        # Inverse of I-hD, D = [[a,-omega],[omega,a]].
        inv0 = torch.cat((torch.cat((torch.diag_embed(d), -torch.diag_embed(w)), dim=-1),
                          torch.cat((torch.diag_embed(w), torch.diag_embed(d)), dim=-1)), dim=-2)
        inv_p = (inv0 @ self.p.unsqueeze(-1)).squeeze(-1)
        p_inv = (self.p.unsqueeze(-2) @ inv0).squeeze(-2)
        correction_scale = 2 * h / (1 + 2 * h * (self.p * inv_p).sum(-1, keepdim=True))
        inverse = inv0 - correction_scale[..., None] * inv_p[..., :, None] * p_inv[..., None, :]
        eye = torch.eye(2 * half, dtype=inverse.dtype, device=inverse.device)
        a_bar = 2 * inverse - eye
        b_bar = (inverse @ (2 * h * self.b).unsqueeze(-1)).squeeze(-1)
        return a_bar, b_bar

    def forward(self, length: int) -> Tensor:
        a_bar, b_bar = self.discretize()
        # Krylov sequence [B, AB, ..., A^(L-1)B], doubling in O(log L) launches.
        krylov = b_bar.unsqueeze(-1)
        power = a_bar
        while krylov.shape[-1] < length:
            krylov = torch.cat((krylov, power @ krylov), dim=-1)
            if krylov.shape[-1] < length:
                power = power @ power
        return torch.einsum("dhn,hnl->dhl", self.c, krylov[..., :length])


class S4TemporalLayer(nn.Module):
    def __init__(self, width: int, state_size: int = 16):
        super().__init__()
        self.kernel = S4NPLRKernel(width, state_size, directions=2)
        self.skip = nn.Parameter(torch.randn(width))
        self.mix = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch, horizon, width]. Exactly matches the finite bidirectional
        # FFT convolution convention used by TSFlow's official S4 layer.
        length = x.shape[1]
        kernel = self.kernel(length)
        index = torch.arange(length, device=x.device)
        offset = index[:, None] - index[None, :]
        forward = kernel[0][:, offset.clamp(min=0)] * (offset >= 0)
        backward = kernel[1][:, (-offset - 1).clamp(min=0)] * (offset < 0)
        y = torch.einsum("bih,hoi->boh", x, forward + backward)
        return self.mix(F.gelu(y + x * self.skip))


class S4ResidualBlock(nn.Module):
    """Three of these give the TSFlow-style univariate residual/skip backbone."""
    def __init__(self, width: int, encoder_dim: int, state_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.temporal = S4TemporalLayer(width, state_size)
        self.time_projection = nn.Linear(width, width)
        self.history_projection = nn.Linear(encoder_dim, width)
        self.residual_projection = nn.Linear(width, width)
        self.skip_projection = nn.Linear(width, width)

    def forward(self, x: Tensor, time: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        z = x + self.time_projection(time)[:, None, :]
        z = z + self.temporal(self.norm(z))
        z = z + self.history_projection(history)[:, None, :]
        z = torch.tanh(z) * torch.sigmoid(z)
        return x + self.residual_projection(z), self.skip_projection(z)


class S4Velocity(nn.Module):
    def __init__(self, horizon: int = 16, lookback: int = 32, width: int = 64,
                 encoder_dim: int = 32, depth: int = 3, state_size: int = 16,
                 time_dim: int = 32):
        super().__init__()
        if depth != 3:
            raise ValueError("The assignment specifies exactly three S4 residual blocks")
        self.horizon = horizon
        self.encoder = LookbackEncoder(lookback, encoder_dim)
        self.time_embedding = TimeEmbedding(width, time_dim)
        self.input_projection = nn.Linear(1, width)
        self.blocks = nn.ModuleList(S4ResidualBlock(width, encoder_dim, state_size) for _ in range(depth))
        # The last residual output is never consumed: only its skip is used.
        # Do not count/train a dead final residual projection.
        self.blocks[-1].residual_projection = nn.Identity()
        self.output_projection = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        self.apply(_xavier_linear)
        nn.init.zeros_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)

    def forward(self, x: Tensor, t: Tensor, history: Tensor) -> Tensor:
        h, time = self.encoder(history), self.time_embedding(t)
        z = F.relu(self.input_projection(x.unsqueeze(-1)))
        skips = []
        for block in self.blocks:
            z, skip = block(z, time, h)
            skips.append(skip)
        return self.output_projection(torch.stack(skips).sum(0)).squeeze(-1)


def create_model(kind: str, horizon: int = 16, lookback: int = 32, width: int = 64,
                 encoder_dim: int = 32, depth: int = 3, state_size: int = 16,
                 time_dim: int = 32) -> nn.Module:
    common = dict(horizon=horizon, lookback=lookback, width=width,
                  encoder_dim=encoder_dim, depth=depth, time_dim=time_dim)
    if kind in {"mlp", "whitened_mlp", "mlp_white"}:
        return AdaLNMLP(**common)
    if kind == "s4":
        return S4Velocity(**common, state_size=state_size)
    raise ValueError(f"Unknown model kind: {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def optimizer_groups(model: nn.Module, lr: float, weight_decay: float = 0.0) -> list[dict]:
    """Honor official S4's no-decay rule on A, B, P, and step sizes."""
    groups: dict[float, list] = {}
    for parameter in model.parameters():
        if parameter.requires_grad:
            decay = getattr(parameter, "_optim", {}).get("weight_decay", weight_decay)
            groups.setdefault(decay, []).append(parameter)
    return [{"params": parameters, "lr": lr, "weight_decay": decay}
            for decay, parameters in groups.items()]
