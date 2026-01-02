import torch
import torch.nn as nn

class RealToAlgebra(nn.Module):
    def __init__(self, algebra_dim, mode='pad'):
        super().__init__()
        self.dim = algebra_dim
        self.mode = mode

    def forward(self, x):
        ch_dim = 1 if x.dim() > 2 else -1

        if self.mode == 'pad':
            pad_shape = list(x.shape)
            pad_shape[ch_dim] = pad_shape[ch_dim] * (self.dim - 1)
            
            zeros = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
            
            return torch.cat([x, zeros], dim=ch_dim)
        
        elif self.mode == 'repeat':
            repeats = [1] * x.dim()
            repeats[ch_dim] = self.dim
            return x.repeat(*repeats)
        
        else:
             raise NotImplementedError(f"Mode {self.mode} not implemented.")