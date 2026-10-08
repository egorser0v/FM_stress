"""Leakage-free multivariate synthetic and chronological real-series windows.

This is a separate extension; original experiment modules remain unchanged.

API
---
``bundle = build_data(config)`` returns ``ExtendedData`` with ``splits`` mapping
train/val/test to ``WindowDataset(context[N,L,C], target[N,F,C], starts[N])``,
``source_covariances`` mapping white/gp to [F*C,F*C], and JSON-safe ``metadata``.
``sample_source(bundle,n,source,seed)`` returns [N,F,C]. Flattening is always
C-order, time first and channels last: flattened index = time*C + channel.

Config is a flat dictionary with kind='synthetic'|'real', lookback and horizon.
Synthetic: channels, channel_correlation, lengthscale, nugget, seed,
n_train/n_val/n_test. Real: path, format='csv' (header and columns) or 'text'
(no header, numeric, delimiter=',' and optional channel_indices); split_ends
(three exclusive row ends) or split_fractions=[.7,.1,.2]; strides and max_windows
are optional dictionaries keyed by train/val/test. No window crosses a split.

Sources share a regularized train-target channel correlation matrix. 'white'
means temporally white, not independent across channels. 'gp' adds an RBF temporal
covariance (source_lengthscale=8 by default). Sources have unit marginal channel
variance rather than matching the empirical target variances. Both covariances
use a normalized temporal nugget so their marginal scales match exactly.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ('train', 'val', 'test')


@dataclass
class WindowDataset:
    context: np.ndarray
    target: np.ndarray
    starts: np.ndarray

    def __len__(self):
        return len(self.target)


@dataclass
class ExtendedData:
    splits: dict[str, WindowDataset]
    source_covariances: dict[str, np.ndarray]
    metadata: dict[str, Any]


def equicorrelation(channels: int, rho: float):
    if channels < 1:
        raise ValueError('channels must be positive')
    if channels > 1 and not -1/(channels-1) < rho < 1:
        raise ValueError('channel_correlation must define a positive definite matrix')
    return np.eye(channels) if channels == 1 else (1-rho)*np.eye(channels)+rho*np.ones((channels,channels))


def rbf_temporal(size: int, lengthscale: float, nugget: float = 1e-6, unit_diagonal: bool = False):
    if size < 1 or lengthscale <= 0 or nugget <= 0:
        raise ValueError('size, lengthscale and nugget must be positive')
    t=np.arange(size,dtype=np.float64)
    cov=np.exp(-.5*((t[:,None]-t[None,:])/lengthscale)**2)+nugget*np.eye(size)
    return cov/(1+nugget) if unit_diagonal else cov


def fit_channel_scaler(training_points):
    x=np.asarray(training_points,dtype=np.float64)
    if x.ndim < 2 or not np.isfinite(x).all():
        raise ValueError('Finite arrays with a final channel dimension are required')
    points=x.reshape(-1,x.shape[-1])
    if len(points)<2:
        raise ValueError('At least two training points are required')
    mean=points.mean(axis=0)
    std=points.std(axis=0,ddof=0)
    # Deterministic constant-channel fallback; no validation/test information.
    scale=np.where(std>1e-8,std,1.)
    return mean,scale,std


def _synthetic(config):
    l,f,c=int(config['lookback']),int(config['horizon']),int(config.get('channels',1))
    rho=float(config.get('channel_correlation',.6))
    temporal=rbf_temporal(l+f,float(config.get('lengthscale',8.)),float(config.get('nugget',1e-6)))
    channel=equicorrelation(c,rho)
    tc,cc=np.linalg.cholesky(temporal),np.linalg.cholesky(channel)
    raw={}
    seed=int(config.get('seed',97041))
    for offset,split in enumerate(SPLITS):
        n=int(config['n_'+split])
        if n<2:
            raise ValueError('Each synthetic split requires at least two independent windows')
        z=np.random.default_rng(seed+offset).normal(size=(n,l+f,c))
        raw[split]=np.einsum('ij,njk,lk->nil',tc,z,cc,optimize=True)
    mean,scale,std=fit_channel_scaler(raw['train'])
    splits={}
    for split,x in raw.items():
        normalized=(x-mean)/scale
        splits[split]=WindowDataset(normalized[:,:l],normalized[:,l:],np.arange(len(x),dtype=np.int64))
    metadata={'kind':'synthetic','kernel':'separable RBF(time) x equicorrelation(channels)',
        'lengthscale':float(config.get('lengthscale',8.)),'nugget':float(config.get('nugget',1e-6)),
        'population_channel_covariance':channel.tolist(),
        'split_seeds':{split:seed+i for i,split in enumerate(SPLITS)},
        'start_index_meaning':'Independent-window ordinal within each split, not a continuous-series timestamp',
        'scaler_fit':'All points of independent TRAIN joint history/future windows only',
        'scaler_mean':mean.tolist(),'scaler_scale':scale.tolist(),'scaler_raw_std':std.tolist(),
        'scaler_ddof':0,'channel_names':[f'channel_{i}' for i in range(c)]}
    return splits,metadata


def _read_real(config):
    path=Path(config['path'])
    fmt=config.get('format','csv')
    delimiter=config.get('delimiter',',')
    if fmt=='csv':
        with path.open(newline='') as f:
            reader=csv.DictReader(f,delimiter=delimiter)
            names=config.get('columns')
            if not names:
                names=[x for x in reader.fieldnames if x!='date']
            if len(set(names))!=len(names) or any(name not in reader.fieldnames for name in names):
                raise ValueError('CSV columns must be present and unique')
            values=[]; dates=[]
            for row in reader:
                values.append([float(row[name]) for name in names])
                if 'date' in row:
                    dates.append(datetime.fromisoformat(row['date']))
        x=np.asarray(values,dtype=np.float64)
        if dates and any(b<=a for a,b in zip(dates,dates[1:])):
            raise ValueError('Real-series timestamps must be strictly increasing')
        timestamp_range=[dates[0].isoformat(),dates[-1].isoformat()] if dates else None
    elif fmt=='text':
        x=np.loadtxt(path,delimiter=delimiter,ndmin=2)
        indices=config.get('channel_indices',list(range(x.shape[1])))
        if len(set(indices))!=len(indices) or not indices or min(indices)<0 or max(indices)>=x.shape[1]:
            raise ValueError('Text channel_indices must be unique valid columns')
        x=x[:,indices]
        names=[f'channel_{i}' for i in indices]
        timestamp_range=None
    else:
        raise ValueError('Real format must be csv or text')
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Real series must contain finite numeric rows; no implicit imputation')
    return x,names,{'path':str(path),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'format':fmt,'timestamp_range':timestamp_range,'total_rows':len(x)}


def chronological_bounds(n, config):
    if 'split_ends' in config:
        ends=[int(i) for i in config['split_ends']]
    else:
        fractions=np.asarray(config.get('split_fractions',[.7,.1,.2]),dtype=float)
        if fractions.shape!=(3,) or (fractions<=0).any() or not np.isclose(fractions.sum(),1):
            raise ValueError('Three positive split fractions summing to one are required')
        ends=[int(n*fractions[0]),int(n*(fractions[0]+fractions[1])),n]
    if len(ends)!=3 or not 0<ends[0]<ends[1]<ends[2]<=n:
        raise ValueError('split_ends must be three increasing exclusive row boundaries')
    return {split:(start,end) for split,start,end in zip(SPLITS,[0]+ends[:-1],ends)}


def make_windows(series,start,end,lookback,horizon,stride=1,cap=None):
    if stride<1 or lookback<1 or horizon<1 or start<0 or end>len(series) or start>=end:
        raise ValueError('Invalid chronological window parameters')
    starts=np.arange(start,end-lookback-horizon+1,stride,dtype=np.int64)
    if not len(starts):
        raise ValueError('Split is too short for a complete history/future window')
    if cap is not None:
        if int(cap)<1:
            raise ValueError('Window cap must be positive')
        if len(starts)>int(cap):
            starts=starts[np.linspace(0,len(starts)-1,int(cap),dtype=np.int64)]
    contexts=np.stack([series[i:i+lookback] for i in starts])
    targets=np.stack([series[i+lookback:i+lookback+horizon] for i in starts])
    return WindowDataset(contexts,targets,starts)


def _real(config):
    x,names,file_metadata=_read_real(config)
    bounds=chronological_bounds(len(x),config)
    mean,scale,std=fit_channel_scaler(x[:bounds['train'][1]])
    normalized=(x-mean)/scale
    l,f=int(config['lookback']),int(config['horizon'])
    strides=config.get('strides',{})
    caps=config.get('max_windows',{})
    splits={split:make_windows(normalized,start,end,l,f,int(strides.get(split,1)),caps.get(split))
            for split,(start,end) in bounds.items()}
    metadata={'kind':'real','file':file_metadata,'channel_names':names,
        'split_row_bounds':{split:list(b) for split,b in bounds.items()},
        'omitted_tail_rows':len(x)-bounds['test'][1],
        'start_index_meaning':'Global zero-based row of the history start in the input file',
        'window_starts':{split:ds.starts.tolist() for split,ds in splits.items()},
        'scaler_fit':'Unique rows in chronological TRAIN segment only, before windowing',
        'scaler_mean':mean.tolist(),'scaler_scale':scale.tolist(),'scaler_raw_std':std.tolist(),
        'scaler_ddof':0,
        'chronology':'Every history and target is fully inside its assigned split; no borrowing past training rows for validation histories'}
    return splits,metadata


def estimate_channel_correlation(training_targets,shrinkage=.05):
    if not 0<shrinkage<=1:
        raise ValueError('Positive channel shrinkage <=1 guarantees a nonsingular source')
    x=np.asarray(training_targets,dtype=np.float64)
    if x.ndim!=3 or len(x)<1 or not np.isfinite(x).all():
        raise ValueError('Training targets must be finite [N,F,C]')
    points=x.reshape(-1,x.shape[-1])
    centered=points-points.mean(axis=0)
    cov=centered.T@centered/max(len(points)-1,1)
    scale=np.sqrt(np.maximum(np.diag(cov),0))
    # A constant channel has no estimated relationship; give it independent
    # unit-variance source coordinates rather than dividing by zero.
    safe=np.where(scale>1e-12,scale,1.)
    corr=cov/np.outer(safe,safe)
    np.fill_diagonal(corr,1.)
    corr=(1-shrinkage)*corr+shrinkage*np.eye(x.shape[-1])
    return (corr+corr.T)/2


def build_data(config):
    """Construct immutable split arrays and declared source covariance metadata."""
    config=dict(config)
    l,f=int(config['lookback']),int(config['horizon'])
    if l<1 or f<2:
        raise ValueError('lookback>=1 and horizon>=2 are required')
    kind=config.get('kind','synthetic')
    if kind=='synthetic':
        splits,metadata=_synthetic(config)
    elif kind=='real':
        splits,metadata=_real(config)
    else:
        raise ValueError('kind must be synthetic or real')
    c=splits['train'].target.shape[-1]
    if 'channels' in config and int(config['channels'])!=c:
        raise ValueError('Requested channels do not match selected real columns')
    shrinkage=float(config.get('source_channel_shrinkage',.05))
    correlation=estimate_channel_correlation(splits['train'].target,shrinkage)
    ell=float(config.get('source_lengthscale',8.))
    nugget=float(config.get('source_nugget',1e-6))
    temporal=rbf_temporal(f,ell,nugget,unit_diagonal=True)
    sources={'white':np.kron(np.eye(f),correlation),'gp':np.kron(temporal,correlation)}
    if config.get('include_independent_white',False):
        sources['independent_white']=np.eye(f*c)
    metadata.update({'config':config,'lookback':l,'horizon':f,'channels':c,
        'split_sizes':{split:len(ds) for split,ds in splits.items()},
        'flattening':'C-order [time,channel]; flat index = time*channels + channel',
        'source_channel_correlation':correlation.tolist(),'source_channel_shrinkage':shrinkage,
        'source_channel_fit':'TRAIN target windows only, points pooled across windows and horizon; repeated overlapping observations retain their window frequency',
        'source_scale':'Unit marginal channel variance. Empirical standardized target variances need not be exactly one.',
        'source_lengthscale':ell,'source_nugget':nugget,
        'source_covariance_definition':{'white':'I_F x train channel correlation (temporally white, correlated channels)',
            'gp':'[(RBF_F + nugget I)/(1+nugget)] x same train channel correlation'},
        'source_covariances':{key:value.tolist() for key,value in sources.items()}})
    return ExtendedData(splits,sources,metadata)


def sample_source(bundle: ExtendedData,n: int,source: str,seed=0):
    """Independent source draws, [N,F,C], using a separate local RNG stream."""
    if n<1 or source not in bundle.source_covariances:
        raise ValueError('Positive n and a declared source family are required')
    cov=bundle.source_covariances[source]
    rng=seed if isinstance(seed,np.random.Generator) else np.random.default_rng(seed)
    z=rng.standard_normal((n,len(cov)))
    return (z@np.linalg.cholesky(cov).T).reshape(n,bundle.metadata['horizon'],bundle.metadata['channels'])
