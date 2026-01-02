import torch
import torch.nn as nn

class AlgebraToReal(nn.Module):
    def __init__(self, algebra_dim, mode='magnitude'):
        super().__init__()
        self.dim = algebra_dim
        self.mode = mode

    def forward(self, x):
        ch_dim = 1 if x.dim() > 2 else -1

        if self.mode == 'magnitude':
            x_poly = x.unflatten(ch_dim, (-1, self.dim))
            
            target_dim = ch_dim + 1 if ch_dim >= 0 else -1
            
            return torch.norm(x_poly, p=2, dim=target_dim)
            
        elif self.mode == 'first':
            x_poly = x.unflatten(ch_dim, (-1, self.dim))
            
            target_dim = ch_dim + 1 if ch_dim >= 0 else -1
            return x_poly.select(dim=target_dim, index=0)

        elif self.mode == 'flatten':
            return x
            
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")