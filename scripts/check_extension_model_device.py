"""Nontrivial full-model CPU/MPS forward, gradients and optimizer-step parity.

Runs tiny diagnostic tensors, not training experiments. Opens zero-initialized
maps so equality cannot pass just because both freshly created models output0.
"""
from pathlib import Path
import json
import os
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fm_stress.extended_models import create_extended_model,parameter_counts
from fm_stress.models import optimizer_groups
from fm_stress.experiment import save_json


def discrepancy(a,b,atol,rtol):
    a,b=a.detach().cpu(),b.detach().cpu()
    absolute=float((a-b).abs().max())
    reference=float(a.abs().max())
    return {'max_abs_difference':absolute,'max_abs_reference':reference,
            'tolerance':atol+rtol*reference,'passed':bool(absolute<=atol+rtol*reference)}


def main():
    if not torch.backends.mps.is_available():raise RuntimeError('Actual MPS required')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise RuntimeError('Disable silent MPS fallback for this check')
    torch.set_num_threads(4)
    rows=[]
    for channels in (1,3,7,8):
        for head in ('mlp','sparse_mlp','unet','s4'):
            torch.manual_seed(200+channels)
            options=dict(head=head,lookback=9,horizon=15,channels=channels,width=12,
                         density=.5,depth=3,state_size=8,time_dim=8)
            cpu=create_extended_model(**options)
            with torch.no_grad():
                for module in cpu.modules():
                    weight=getattr(module,'weight',None)
                    if isinstance(weight,torch.nn.Parameter) and weight.ndim>=2 and not weight.count_nonzero():
                        torch.nn.init.normal_(weight,std=.05)
            mps=create_extended_model(**options).to('mps')
            mps.load_state_dict(cpu.state_dict())
            x,h,t=torch.randn(3,15,channels),torch.randn(3,9,channels),torch.tensor([0.,.4,1.])
            target=.3*x.roll(1,dims=-1)+h.mean(1)[:,None]
            a,b=cpu(x,t,h),mps(x.to('mps'),t.to('mps'),h.to('mps'))
            assert a.abs().max()>1e-7,'Nontrivial output check failed'
            forward=discrepancy(a,b,2e-4,5e-4)
            loss_a=(a-target).square().mean();loss_b=(b-target.to('mps')).square().mean()
            loss_a.backward();loss_b.backward()
            grads=[]
            for (name,p),(other,q) in zip(cpu.named_parameters(),mps.named_parameters()):
                assert name==other and p.grad is not None and q.grad is not None
                assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all()
                grads.append({'parameter':name,**discrepancy(p.grad,q.grad,3e-4,2e-3)})
            opt_a=torch.optim.AdamW(optimizer_groups(cpu,.001,.0001))
            opt_b=torch.optim.AdamW(optimizer_groups(mps,.001,.0001))
            opt_a.step();opt_b.step()
            with torch.no_grad():post=discrepancy(cpu(x,t,h),mps(x.to('mps'),t.to('mps'),h.to('mps')),5e-4,2e-3)
            passed=forward['passed'] and post['passed'] and all(g['passed'] for g in grads)
            row={'head':head,'channels':channels,'shape':[3,15,channels],'options':options,
                 'parameters':parameter_counts(cpu),'nonzero_forward':forward,
                 'cpu_loss':float(loss_a.detach()),'mps_loss':float(loss_b.detach().cpu()),
                 'gradient_comparisons':grads,'post_adamw_step':post,'passed':passed}
            rows.append(row)
            print(head,'channels',channels,'forward',forward['max_abs_difference'],
                  'maxgrad',max(g['max_abs_difference'] for g in grads),'post',post['max_abs_difference'],'PASS' if passed else 'FAIL',flush=True)
            del cpu,mps,opt_a,opt_b
            torch.mps.empty_cache()
    result={'status':'passed' if all(r['passed'] for r in rows) else 'failed',
            'device':'mps','torch':torch.__version__,'mps_available':True,'fallback_enabled':False,
            'rows':rows,'scope':'Tiny diagnostic models at odd horizon15; nonzero maps, allparameter gradients and oneAdamWupdate. No experiment results produced.'}
    save_json(ROOT/'results/diagnostics/extension_model_parity.json',result)
    if result['status']!='passed':raise AssertionError('Model parity threshold failed; inspect JSON')

if __name__=='__main__':main()
