"""DLinear-style modern simple comparator."""
import torch
from torch import nn


class DLinearForecaster(nn.Module):
    def __init__(self,input_length,n_horizons,kernel_size=25):
        super().__init__()
        self.kernel_size=kernel_size
        self.trend=nn.Linear(input_length,n_horizons)
        self.seasonal=nn.Linear(input_length,n_horizons)

    def forward(self,x,return_embedding=False):
        series=x[:,:,0]
        pad=(self.kernel_size-1)//2
        padded=torch.nn.functional.pad(
            series[:,None,:],(pad,pad),mode="replicate"
        )
        trend=torch.nn.functional.avg_pool1d(
            padded,self.kernel_size,stride=1
        )[:,0,:]
        seasonal=series-trend
        prediction=self.trend(trend)+self.seasonal(seasonal)
        embedding=torch.stack([trend.mean(1),seasonal.mean(1)],dim=1)
        return (prediction,embedding) if return_embedding else prediction
