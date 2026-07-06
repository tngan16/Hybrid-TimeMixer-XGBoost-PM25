"""Compact TimeMixer implementation for multiscale PM2.5 forecasting.

The module follows TimeMixer's central design ideas—multiscale downsampling,
seasonal/trend decomposition, cross-scale mixing and scale-specific future
predictors—while remaining small enough for a university workstation.
"""
import torch
from torch import nn
import torch.nn.functional as F

class MovingAverageDecomposition(nn.Module):
    def __init__(self,kernel_size=25): super().__init__(); self.kernel_size=kernel_size
    def forward(self,x):
        pad=self.kernel_size//2; z=x.transpose(1,2); trend=F.avg_pool1d(F.pad(z,(pad,pad),mode="replicate"),self.kernel_size,stride=1).transpose(1,2); return x-trend,trend

class ComponentEncoder(nn.Module):
    def __init__(self,n_features,hidden,dropout): super().__init__(); self.net=nn.Sequential(nn.Linear(n_features,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden)); self.norm=nn.LayerNorm(hidden)
    def forward(self,x): return self.norm(self.net(x)).mean(dim=1)

class ScaleMixer(nn.Module):
    def __init__(self,n_scales,hidden,dropout):
        super().__init__(); self.scale_mlp=nn.Sequential(nn.Linear(n_scales,n_scales*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(n_scales*2,n_scales)); self.channel_mlp=nn.Sequential(nn.Linear(hidden,hidden*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden*2,hidden)); self.norm1=nn.LayerNorm(hidden); self.norm2=nn.LayerNorm(hidden)
    def forward(self,x):
        x=self.norm1(x+self.scale_mlp(x.transpose(1,2)).transpose(1,2)); return self.norm2(x+self.channel_mlp(x))

class TimeMixer(nn.Module):
    def __init__(self,n_features,n_horizons,input_length=168,hidden=64,scales=(1,2,4,8),dropout=.1,blocks=2,decomposition_kernel=25):
        super().__init__(); self.scales=tuple(scales); self.decomposition=MovingAverageDecomposition(decomposition_kernel); self.seasonal=nn.ModuleList([ComponentEncoder(n_features,hidden,dropout) for _ in scales]); self.trend=nn.ModuleList([ComponentEncoder(n_features,hidden,dropout) for _ in scales]); self.fuse=nn.ModuleList([nn.Sequential(nn.Linear(hidden*2,hidden),nn.GELU(),nn.Dropout(dropout)) for _ in scales]); self.mixers=nn.ModuleList([ScaleMixer(len(scales),hidden,dropout) for _ in range(blocks)]); self.predictors=nn.ModuleList([nn.Linear(hidden,n_horizons) for _ in scales]); self.scale_logits=nn.Parameter(torch.zeros(len(scales))); self.embedding_dim=hidden*len(scales)
    def _downsample(self,x,scale):
        if scale==1: return x
        return F.avg_pool1d(x.transpose(1,2),kernel_size=scale,stride=scale,ceil_mode=False).transpose(1,2)
    def encode(self,x):
        encoded=[]
        for i,scale in enumerate(self.scales):
            sequence=self._downsample(x,scale); seasonal,trend=self.decomposition(sequence); encoded.append(self.fuse[i](torch.cat([self.seasonal[i](seasonal),self.trend[i](trend)],dim=-1)))
        mixed=torch.stack(encoded,dim=1)
        for block in self.mixers: mixed=block(mixed)
        return mixed
    def forward(self,x,return_embedding=False):
        mixed=self.encode(x); weights=torch.softmax(self.scale_logits,dim=0); scale_predictions=torch.stack([head(mixed[:,i]) for i,head in enumerate(self.predictors)],dim=1); forecast=(scale_predictions*weights.view(1,-1,1)).sum(dim=1); embedding=mixed.flatten(1); return (forecast,embedding) if return_embedding else forecast
