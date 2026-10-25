#!/usr/bin/env python3
"""Check the F64/width56 MLP's immutable decoder-complement directions.

For v(x,t,h)=W a(x,t,h)+b, N^T W=0 implies N^T x_1=N^T x_0+N^T b
on any Euler grid whose steps sum to one. This identity does not assume linear
hidden layers, conditioning quality, or optimization convergence.
"""
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import check_kernel_scan as ind

def main():
    out=ROOT/'results/followup-reproduction'
    cfg=ind.read(ROOT/'results/kernel-scan/config.json')
    rows=[]
    for p in sorted((out/'generation_runs').glob('*.json')):
        audit=ind.read(p);r=audit['identity']
        if r['head']!='mlp':continue
        group=ROOT/'results/kernel-scan'/r['kernel']/r['geometry_id']/f'draw{r["data_seed"]}'
        record=ind.read(group/'runs'/f'{r["source"]}_mlp.json')
        checkpoint=group/'checkpoints'/f'{r["source"]}_mlp.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
        W=state['base.output_projection.weight'].numpy().astype(float)
        bias=state['base.output_projection.bias'].numpy().astype(float)
        U,s,_=np.linalg.svd(W,full_matrices=True);rank=int((s>s[0]*1e-7).sum());N=U[:,rank:]
        spec={**record['geometry'],'threshold':record['geometry']['normalization_threshold']}
        n,m=cfg['n_ensemble_contexts'],cfg['ensemble_size']
        eps=ind.source_draw(spec,n*m,r['source'],r['data_seed']+1000).reshape(n,m,-1)
        archive=ROOT/audit['generation']['samples'];a=np.load(archive);gen=a['ensemble'].astype(float)
        error=gen@N-eps@N-bias@N
        test=ind.windows(spec,cfg['n_test'],r['data_seed']+2)
        l,f=spec['lookback'],spec['horizon']
        k=ind.covariance(spec['kernel'],l+f,spec['lengthscale'],spec['variance'],spec['nugget'])
        conditional=k[l:,l:]-k[l:,:l]@np.linalg.solve(k[:l,:l],k[:l,l:]);conditional=(conditional+conditional.T)/2
        target_conditional_trace=float(np.trace(N.T@conditional@N)*np.mean(1/test.scale[:n]**2))
        source_trace=float(np.trace(N.T@ind.source_cov(spec,r['source'])@N))
        marginal_trace=float(np.var(test.target@N,axis=0,ddof=1).sum())
        rows.append({'identity':r,'checkpoint_sha256':ind.sha(checkpoint),'sample_sha256':ind.sha(archive),
            'decoder_rank':rank,'horizon':f,'immutable_dimensions':f-rank,
            'decoder_null_residual_max_abs':float(np.abs(N.T@W).max()),
            'euler_identity_max_abs_error':float(np.abs(error).max()),
            'euler_identity_rmse':float(np.sqrt(np.mean(error**2))),
            'immutable_source_variance_trace':source_trace,
            'privileged_target_conditional_variance_trace':target_conditional_trace,
            'test_target_marginal_variance_trace':marginal_trace,
            'variance_ratio_source_over_target_marginal':source_trace/marginal_trace,
            'variance_ratio_source_over_privileged_target':source_trace/target_conditional_trace,
            'observed_ensemble_variance_trace':float(np.var(gen@N,axis=1,ddof=1).sum(axis=-1).mean()),
            'observed_source_variance_trace':float(np.var(eps@N,axis=1,ddof=1).sum(axis=-1).mean())})
    torch.manual_seed(0)
    z=torch.randn(56,dtype=torch.float64,requires_grad=True)
    ln=torch.nn.LayerNorm(56,eps=1e-6,elementwise_affine=False).double()
    jac=torch.autograd.functional.jacobian(ln,z).numpy();sv=np.linalg.svd(jac,compute_uv=False)
    layernorm={'width':56,'epsilon':1e-6,'smallest_singular_values':sv[-3:].tolist(),
        'rank_threshold_1e-12':int((sv>sv[0]*1e-12).sum()),
        'rank_threshold_1e-5':int((sv>sv[0]*1e-5).sum()),
        'interpretation':'Exact generic rank is width-1 for positive epsilon; width-2 is the zero-epsilon limit or an effective numerical rank.'}
    result={'complete':len(rows)==9,'expected_runs':9,'runs':rows,'layernorm_rank_check':layernorm,
        'interpretation':'The decoder-complement invariant is exact for this architecture. The target conditional covariance uses privileged raw history and is a descriptive reference. It is not the covariance available to the normalized-history network.',
        'formula':'N.T W=0 => N.T x_1=N.T x_0+N.T b; width56,horizon64 implies at least8 immutable directions.',
        'audit_script_sha256':ind.sha(__file__)}
    (out/'decoder_span.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
