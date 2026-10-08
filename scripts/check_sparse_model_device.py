"""Independent CPU/MPS parity diagnostics for every new sparse FM head.

Uses tiny nontrivial models and actual MPS without operator fallback. Includes
history reconstruction/gate/routing auxiliary losses in backward, private-gate
checkpoint continuation, global-RNG invariance, and one AdamW step. This is a
numerical implementation check, not an experiment or a quality comparison.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.sparse_models import create_sparse_model, HardConcreteGate
from fm_stress.sparse_moe import TinyMoE
from fm_stress.extended_models import parameter_counts
from fm_stress.models import optimizer_groups
from fm_stress.experiment import save_json

HEADS = ('history_dense_ae', 'history_topk_ae', 'history_conv_dense',
         'history_conv_sparse', 'dictionary_dense', 'dictionary_topk',
         'unet_l0', 'moe_dense', 'moe_top2')


def make(head, channels):
    if head.startswith('moe_'):
        return TinyMoE(9, 15, channels, width=12, experts=4,
                       topk=2 if head == 'moe_top2' else None)
    return create_sparse_model(head, lookback=9, horizon=15, channels=channels,
                               width=12, depth=3, time_dim=8, latent_dim=16,
                               topk=4, conv_steps=3, conv_kernel=5,
                               gate_seed=95000 + channels)


def discrepancy(a, b, atol, rtol):
    a, b = a.detach().cpu(), b.detach().cpu()
    difference = float((a - b).abs().max())
    reference = float(a.abs().max())
    tolerance = atol + rtol * reference
    return {'max_abs_difference': difference, 'max_abs_reference': reference,
            'tolerance': tolerance, 'passed': bool(torch.isfinite(b).all() and difference <= tolerance)}


def gate_states(model):
    return [m.get_extra_state()['rng_state'].clone()
            for m in model.modules() if isinstance(m, HardConcreteGate)]


def checked_forward(model, x, t, history):
    cpu_rng = torch.random.get_rng_state().clone()
    mps_rng = torch.mps.get_rng_state().clone()
    output = model(x, t, history)
    return output, {'cpu_global_rng_unchanged': torch.equal(cpu_rng, torch.random.get_rng_state()),
                    'mps_global_rng_unchanged': torch.equal(mps_rng, torch.mps.get_rng_state())}


@torch.no_grad()
def support(model, head, x, t, history):
    if head in ('history_topk_ae', 'history_dense_ae'):
        return model.encode_history(history) != 0
    if head in ('history_conv_sparse', 'history_conv_dense'):
        return model.encode_history(history)[0] != 0
    if head in ('dictionary_topk', 'dictionary_dense'):
        return model.coefficients(x, t, history) != 0
    if head.startswith('moe_'):
        z = torch.nn.functional.silu(model.state(x.flatten(1)) + model.history(history.flatten(1)) + model.time(t))
        probs = model.router(z).softmax(-1)
        if model.topk is None:
            return probs > 0
        indices = probs.topk(model.topk, dim=-1).indices
        return torch.zeros_like(probs).scatter(-1, indices, 1).bool()
    return None


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError('Actual MPS device is required')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Silent MPS fallback must be disabled')
    torch.set_num_threads(4)
    torch.empty(1, device='mps')  # Initialize device before RNG-invariance checks.
    rows = []
    for channels in (1, 3, 7, 8):
        for head in HEADS:
            torch.manual_seed(3100 + channels)
            cpu = make(head, channels)
            with torch.no_grad():
                for module in cpu.modules():
                    weight = getattr(module, 'weight', None)
                    if isinstance(weight, torch.nn.Parameter) and weight.ndim >= 2 and not weight.count_nonzero():
                        torch.nn.init.normal_(weight, std=.05)
            # Deep copy preserves nested private-generator extra state exactly.
            state = copy.deepcopy(cpu.state_dict())
            mps = make(head, channels).to('mps')
            mps.load_state_dict(state)
            x, history = torch.randn(4, 15, channels), torch.randn(4, 9, channels)
            t = torch.tensor([0., .25, .7, 1.])
            xm, hm, tm = x.to('mps'), history.to('mps'), t.to('mps')
            target = .3 * x.roll(1, dims=-1) + history.mean(1)[:, None]
            a, rng_cpu = checked_forward(cpu, x, t, history)
            b, rng_mps = checked_forward(mps, xm, tm, hm)
            assert a.abs().max() > 1e-7, 'Parity cannot pass through trivial zero output'
            forward = discrepancy(a, b, 2e-4, 5e-4)
            auxiliary_a, auxiliary_b = cpu.auxiliary_loss(), mps.auxiliary_loss()
            auxiliary = discrepancy(auxiliary_a, auxiliary_b, 2e-5, 5e-4)
            loss_a = (a - target).square().mean() + auxiliary_a
            loss_b = (b - target.to('mps')).square().mean() + auxiliary_b
            loss_a.backward()
            loss_b.backward()
            gradients = []
            for (name, p), (other, q) in zip(cpu.named_parameters(), mps.named_parameters()):
                assert name == other and p.grad is not None and q.grad is not None, name
                assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(), name
                gradients.append({'parameter': name, **discrepancy(p.grad, q.grad, 3e-4, 2e-3)})
            support_cpu = support(cpu, head, x, t, history)
            support_mps = support(mps, head, xm, tm, hm)
            disagreements = None if support_cpu is None else int((support_cpu.cpu() != support_mps.cpu()).sum())
            opt_a = torch.optim.AdamW(optimizer_groups(cpu, .001, .0001))
            opt_b = torch.optim.AdamW(optimizer_groups(mps, .001, .0001))
            opt_a.step()
            opt_b.step()
            with torch.no_grad():
                post_a, post_rng_cpu = checked_forward(cpu, x, t, history)
                post_b, post_rng_mps = checked_forward(mps, xm, tm, hm)
                post = discrepancy(post_a, post_b, 5e-4, 2e-3)
                cpu.eval()
                mps.eval()
                gates_before = gate_states(mps)
                eval_a = cpu(x, t, history)
                eval_b, eval_rng_mps = checked_forward(mps, xm, tm, hm)
                repeat = mps(xm, tm, hm)
                evaluation = discrepancy(eval_a, eval_b, 5e-4, 2e-3)
                deterministic = bool(torch.equal(eval_b, repeat))
                gate_rng_eval_unchanged = all(torch.equal(before, after)
                                               for before, after in zip(gates_before, gate_states(mps)))
            rng_checks = {'train_cpu': rng_cpu, 'train_mps': rng_mps,
                          'post_cpu': post_rng_cpu, 'post_mps': post_rng_mps,
                          'eval_mps': eval_rng_mps}
            passed = (forward['passed'] and auxiliary['passed'] and post['passed']
                      and evaluation['passed'] and deterministic and gate_rng_eval_unchanged
                      and all(g['passed'] for g in gradients)
                      and all(all(v.values()) for v in rng_checks.values())
                      and disagreements in (None, 0))
            row = {'head': head, 'channels': channels, 'shape': [4, 15, channels],
                   'parameters': parameter_counts(cpu), 'nonzero_forward': forward,
                   'auxiliary_loss': auxiliary, 'cpu_loss': float(loss_a.detach()),
                   'mps_loss': float(loss_b.detach().cpu()), 'gradient_comparisons': gradients,
                   'train_support_disagreements': disagreements,
                   'post_adamw_step': post, 'evaluation': evaluation,
                   'evaluation_exactly_repeatable': deterministic,
                   'private_gate_rng_unchanged_in_evaluation': gate_rng_eval_unchanged,
                   'rng_checks': rng_checks, 'passed': passed}
            rows.append(row)
            print(head, 'C', channels, 'forward', forward['max_abs_difference'],
                  'aux', auxiliary['max_abs_difference'],
                  'maxgrad', max(g['max_abs_difference'] for g in gradients),
                  'post', post['max_abs_difference'], 'support', disagreements,
                  'PASS' if passed else 'FAIL', flush=True)
            del cpu, mps, opt_a, opt_b
            torch.mps.empty_cache()
    files = ('fm_stress/sparse_models.py', 'fm_stress/sparse_moe.py', 'scripts/check_sparse_model_device.py')
    result = {'status': 'passed' if all(r['passed'] for r in rows) else 'failed',
              'device': 'mps', 'torch': torch.__version__, 'fallback_enabled': False,
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
              'rows': rows,
              'scope': '36 tiny nontrivial CPU/MPS model comparisons; all new heads; channels1/3/7/8, odd history9/horizon15; auxiliary gradients, private RNG checkpoint continuation, deterministic eval, one AdamW update. No scientific experiment results.'}
    destination = ROOT / 'results/diagnostics/sparse_model_parity.json'
    save_json(destination, result)
    print('Wrote', destination, 'status', result['status'], flush=True)
    if result['status'] != 'passed':
        raise AssertionError('Sparse-model parity failed; inspect JSON')


if __name__ == '__main__':
    main()
