import torch
from torch import nn

class LSTMForecaster(nn.Module):
    def __init__(self,n_features,n_horizons,hidden=64,layers=2,dropout=.1):
        super().__init__(); self.lstm=nn.LSTM(n_features,hidden,num_layers=layers,batch_first=True,dropout=dropout if layers>1 else 0.0); self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,n_horizons))
    def forward(self,x,return_embedding=False):
        _,(h,_)=self.lstm(x); emb=h[-1]; pred=self.head(emb); return (pred,emb) if return_embedding else pred
