"""Joint multivariate heads for a separate exploratory extension.

No previous model or checkpoint is changed. Dense/sparse MLPs adapt the pinned
TimeFlow AdaLN implementation in models.py. S4 reuses the numerically audited
rank-one HiPPO/NPLR temporal layer. U-Net is a small conventional temporal
encoder/decoder, not a claimed reproduction of a particular paper/checkpoint.
Every head conditions on the identical flattened full history (identity
representation); each family learns its own subsequent conditioning maps.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .models import AdaLNMLP, TimeEmbedding, S4ResidualBlock, optimizer_groups


class MaskedLinear(nn.Module):
    """Fixed structured-by-row random connectivity; dense storage/computation.

    Each output retains k=max(1, round(density*input_size)) randomly selected
    input connections. Buffers persist in checkpoints and never regrow. Biases
    are dense. This is static weight sparsity, NOT mixture-of-experts routing
    and NOT a sparse-kernel speed/memory optimization.
    """
    def __init__(self, original: nn.Linear, density: float):
        super().__init__()
        if not 0 < density <= 1:
            raise ValueError('density must be in (0, 1]')
        self.in_features, self.out_features = original.in_features, original.out_features
        active = max(1, round(density*self.in_features))
        # Exact per-row degree prevents entirely disconnected output units.
        selected = torch.rand(self.out_features,self.in_features,device=original.weight.device).argsort(dim=1)[:,:active]
        mask = torch.zeros_like(original.weight).scatter_(1,selected,1)
        self.register_buffer('mask',mask)
        self.weight = nn.Parameter(original.weight.detach().clone()/math.sqrt(active/self.in_features))
        self.bias = None if original.bias is None else nn.Parameter(original.bias.detach().clone())
        self.requested_density = density

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x,self.weight*self.mask,self.bias)


def _mask_linear_maps(module: nn.Module, density: float) -> None:
    for name,child in list(module.named_children()):
        if isinstance(child,nn.Linear):
            setattr(module,name,MaskedLinear(child,density))
        else:
            _mask_linear_maps(child,density)


def parameter_counts(model: nn.Module) -> dict:
    """Allocated trainable scalars and functionally active scalars separately.

    Masked-out entries still occupy Parameter/optimizer storage, but cannot
    affect the function. Do not interpret effective count as runtime savings.
    """
    allocated=sum(p.numel() for p in model.parameters() if p.requires_grad)
    masked_allocated=masked_active=0
    for module in model.modules():
        if isinstance(module,MaskedLinear) and module.weight.requires_grad:
            masked_allocated+=module.weight.numel()
            masked_active+=int(module.mask.count_nonzero().item())
    return {'allocated':allocated,'active':allocated-masked_allocated+masked_active,
            'allocated_trainable':allocated,'effective_trainable':allocated-masked_allocated+masked_active,
            'masked_weight_allocated':masked_allocated,'masked_weight_active':masked_active,
            'realized_weight_density':None if not masked_allocated else masked_active/masked_allocated}


class JointHead(nn.Module):
    def __init__(self,lookback: int,horizon: int,channels: int):
        super().__init__()
        if min(lookback,horizon,channels)<1:
            raise ValueError('lookback, horizon and channels must be positive')
        self.lookback,self.horizon,self.channels=lookback,horizon,channels
        self.history_encoder=nn.Identity()

    def check(self,x,t,history):
        if x.ndim!=3 or tuple(x.shape[1:])!=(self.horizon,self.channels):
            raise ValueError('x must have shape [batch,horizon,channels]')
        if history.shape!=(len(x),self.lookback,self.channels):
            raise ValueError('history must have shape [batch,lookback,channels]')
        if t.shape not in ((len(x),),(len(x),1)):
            raise ValueError('t must have shape [batch] or [batch,1]')
        return self.history_encoder(history.reshape(len(x),-1))


class JointMLP(JointHead):
    def __init__(self,lookback,horizon,channels,width=56,depth=3,density=1.,time_dim=32):
        super().__init__(lookback,horizon,channels)
        self.core=AdaLNMLP(horizon=horizon*channels,lookback=lookback*channels,
                          encoder_dim=lookback*channels,width=width,depth=depth,time_dim=time_dim)
        self.core.encoder=nn.Identity()
        if density<1:
            _mask_linear_maps(self.core,density)
        self.density=density

    def forward(self,x,t,history):
        h=self.check(x,t,history)
        return self.core(x.reshape(len(x),-1),t,h).reshape_as(x)


def _init_maps(module):
    if isinstance(module,(nn.Linear,nn.Conv1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:nn.init.zeros_(module.bias)


def _init_time(embedding):
    for layer in (embedding.net[0],embedding.net[2]):
        nn.init.normal_(layer.weight,std=.02)


class JointS4(JointHead):
    """Real S4 across TIME, with joint channel projections; not channels-as-batch."""
    def __init__(self,lookback,horizon,channels,width=56,depth=3,state_size=16,time_dim=32):
        super().__init__(lookback,horizon,channels)
        if depth!=3:raise ValueError('The S4 comparison retains exactly three residual blocks')
        self.time_embedding=TimeEmbedding(width,time_dim)
        self.input_projection=nn.Linear(channels,width)
        self.blocks=nn.ModuleList(S4ResidualBlock(width,lookback*channels,state_size) for _ in range(depth))
        self.blocks[-1].residual_projection=nn.Identity()
        self.output_projection=nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,channels))
        self.apply(_init_maps);_init_time(self.time_embedding)
        nn.init.zeros_(self.output_projection[-1].weight);nn.init.zeros_(self.output_projection[-1].bias)

    def forward(self,x,t,history):
        h=self.check(x,t,history);time=self.time_embedding(t)
        z=F.relu(self.input_projection(x));skips=[]
        for block in self.blocks:
            z,skip=block(z,time,h);skips.append(skip)
        return self.output_projection(torch.stack(skips).sum(0))


class ConditionedConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,condition_dim):
        super().__init__()
        self.conv1=nn.Conv1d(in_channels,out_channels,3,padding=1)
        self.norm1=nn.GroupNorm(1,out_channels)
        self.condition=nn.Linear(condition_dim,2*out_channels)
        self.conv2=nn.Conv1d(out_channels,out_channels,3,padding=1)
        self.norm2=nn.GroupNorm(1,out_channels)
        self.residual=nn.Identity() if in_channels==out_channels else nn.Conv1d(in_channels,out_channels,1)

    def forward(self,x,condition):
        z=self.norm1(self.conv1(x))
        shift,scale=self.condition(condition).chunk(2,dim=-1)
        z=F.silu(z*(1+scale[:,:,None])+shift[:,:,None])
        z=F.silu(self.norm2(self.conv2(z)))
        return (z+self.residual(x))/math.sqrt(2)


class TemporalUNet(JointHead):
    """Joint-channel 1-D U-Net, temporal stride-2 downsampling and skip fusion.

    depth counts resolution levels (default 3 means two down/up stages).
    Nearest interpolation targets the exact skip length, including odd lengths.
    """
    def __init__(self,lookback,horizon,channels,width=16,depth=3,time_dim=32):
        super().__init__(lookback,horizon,channels)
        if depth<2:raise ValueError('U-Net requires at least two resolution levels')
        self.time_embedding=TimeEmbedding(width,time_dim)
        self.history_projection=nn.Linear(lookback*channels,width)
        widths=[width*2**i for i in range(depth)]
        self.input_projection=nn.Conv1d(channels,width,1)
        self.encoders=nn.ModuleList(ConditionedConvBlock(w,w,width) for w in widths)
        self.down=nn.ModuleList(nn.Conv1d(widths[i],widths[i+1],3,stride=2,padding=1) for i in range(depth-1))
        self.up=nn.ModuleList(nn.Conv1d(widths[i+1],widths[i],3,padding=1) for i in reversed(range(depth-1)))
        self.decoders=nn.ModuleList(ConditionedConvBlock(2*widths[i],widths[i],width) for i in reversed(range(depth-1)))
        self.output_projection=nn.Conv1d(width,channels,1)
        self.apply(_init_maps);_init_time(self.time_embedding)
        nn.init.zeros_(self.output_projection.weight);nn.init.zeros_(self.output_projection.bias)

    def forward(self,x,t,history):
        h=self.check(x,t,history)
        condition=self.time_embedding(t)+self.history_projection(h)
        z=self.input_projection(x.transpose(1,2));skips=[]
        for index,encoder in enumerate(self.encoders):
            z=encoder(z,condition);skips.append(z)
            if index<len(self.down):z=self.down[index](z)
        for up,decoder,skip in zip(self.up,self.decoders,reversed(skips[:-1])):
            z=up(F.interpolate(z,size=skip.shape[-1],mode='nearest'))
            z=decoder(torch.cat((z,skip),dim=1),condition)
        return self.output_projection(z).transpose(1,2)


def create_extended_model(head,lookback,horizon,channels,width=56,depth=3,density=.25,state_size=16,time_dim=32):
    """Return a joint [B,F,C] velocity model; seed torch before construction."""
    if width<2 or depth<1:raise ValueError('width >=2 and depth >=1 are required')
    if not 0<density<=1:raise ValueError('density must be in (0,1]')
    common=dict(lookback=lookback,horizon=horizon,channels=channels,width=width,depth=depth,time_dim=time_dim)
    if head in ('mlp','dense_mlp'):return JointMLP(**common,density=1.)
    if head in ('sparse','sparse_mlp'):return JointMLP(**common,density=density)
    if head in ('unet','u_net'):return TemporalUNet(**common)
    if head=='s4':return JointS4(**common,state_size=state_size)
    raise ValueError(f'Unknown extended head: {head}')
