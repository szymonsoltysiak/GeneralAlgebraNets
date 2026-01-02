import torch
import torch.nn as nn

class RealToAlgebra(nn.Module):
    def __init__(self, algebra_dim, mode='pad'):
        super().__init__()
        self.dim = algebra_dim
        self.mode = mode # 'pad' or 'learnable'

    def forward(self, x):
        # x shape: [Batch, Features]
        if self.mode == 'pad':
            # Place real data in the scalar part, pad the rest with zeros
            # Example: [x] -> [x, 0, 0, 0] for Quaternion
            batch, width = x.shape
            zeros = torch.zeros(batch, width * (self.dim - 1), device=x.device)
            # Interleaving is tricky, appending is easier but check your algebra structure
            # Ideally, we want [x1, 0, 0, 0, x2, 0, 0, 0...]
            # A simple reshaping trick:
            x_expanded = x.unsqueeze(-1) # [B, W, 1]
            pad = torch.zeros(batch, width, self.dim - 1, device=x.device)
            out = torch.cat([x_expanded, pad], dim=-1)
            return out.flatten(1) # [B, W*4]