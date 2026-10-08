"""Small deterministic routed FM velocity head; not a Time-MoE reproduction.

Both controls allocate all experts. Dense uses every expert; sparse retains top2
router probabilities and renormalizes. Dense tensor computation deliberately
kept for MPS compatibility; active routing does not imply wall-clock savings.
"""
import torch
from torch import nn
from .extended_models import JointHead
from .models import TimeEmbedding


class TinyMoE(JointHead):
    def __init__(self, lookback, horizon, channels, width=48, experts=4, topk=None, balance_weight=.01):
        super().__init__(lookback,horizon,channels)
        self.experts_count, self.topk, self.balance_weight = experts, topk, balance_weight
        self.time = TimeEmbedding(width,32)
        self.history = nn.Linear(lookback*channels,width)
        self.state = nn.Linear(horizon*channels,width)
        self.router = nn.Linear(width,experts)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,horizon*channels)) for _ in range(experts))
        for net in self.experts:
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
        self._aux = None; self._stats = {}

    def forward(self,x,t,history):
        h=self.check(x,t,history)
        z=torch.nn.functional.silu(self.state(x.flatten(1))+self.history(h)+self.time(t))
        probs=self.router(z).softmax(-1)
        weights=probs
        if self.topk is not None:
            values,indices=probs.topk(self.topk,dim=-1)
            weights=torch.zeros_like(probs).scatter(-1,indices,values)
            weights=weights/weights.sum(-1,keepdim=True)
        values=torch.stack([expert(z) for expert in self.experts],dim=1)
        # Same differentiable importance regularizer for both routing controls.
        importance=probs.mean(0)
        self._aux=self.balance_weight*(self.experts_count*importance.square().sum()-1.)
        self._stats={'routing_fraction':(weights>0).float().mean().detach(),
                     'router_importance_min':importance.min().detach(),
                     'router_importance_max':importance.max().detach()}
        return (values*weights[:,:,None]).sum(1).reshape_as(x)

    def auxiliary_loss(self):
        return self._aux

    def activity_statistics(self):
        return self._stats
