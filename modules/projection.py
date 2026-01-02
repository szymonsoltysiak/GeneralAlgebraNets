import torch
import torch.nn as nn

class AlgebraToReal(nn.Module):
    def __init__(self, algebra_dim, mode='magnitude'):
        super().__init__()
        self.dim = algebra_dim
        self.mode = mode

    def forward(self, x):
        # x shape: [Batch, Features * Dim]
        if self.mode == 'magnitude':
            # Reshape to separate algebraic components
            # [B, Features, Dim]
            x_reshaped = x.view(x.shape[0], -1, self.dim)
            # Compute Euclidean norm across the algebraic dimension
            # Result: [B, Features]
            return torch.norm(x_reshaped, p=2, dim=2)
            
        elif self.mode == 'flatten':
             # Just treat all components as independent real features
             return x