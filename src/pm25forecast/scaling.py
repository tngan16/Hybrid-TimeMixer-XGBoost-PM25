"""Train-only standardization for neural inputs and targets."""
from dataclasses import dataclass
import numpy as np
@dataclass
class ArrayStandardizer:
    mean: np.ndarray|None=None
    std: np.ndarray|None=None
    def fit(self,array,axes): self.mean=np.nanmean(array,axis=axes,keepdims=True); self.std=np.nanstd(array,axis=axes,keepdims=True)+1e-6; return self
    def transform(self,array):
        if self.mean is None: raise RuntimeError("Standardizer has not been fitted")
        return (array-self.mean)/self.std
    def inverse(self,array): return array*self.std+self.mean
