import torch
import torch.nn as nn
import torch.nn.functional as F

class AlgebraLinear(nn.Module):
    def __init__(self, in_features, out_features, algebra: 'Algebra', bias=True):
        super().__init__()
        self.algebra = algebra
        
        # Logical features (e.g., 16 inputs / 4 dims = 4 quaternions)
        assert in_features % algebra.dim == 0
        assert out_features % algebra.dim == 0
        
        self.feat_in = in_features // algebra.dim
        self.feat_out = out_features // algebra.dim
        
        # Create independent parameters for each component
        # We store them in a ParameterList so PyTorch finds them
        self.components = nn.ParameterList([
            nn.Parameter(torch.empty(self.feat_out, self.feat_in))
            for _ in range(algebra.dim)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize each component
        for p in self.components:
            nn.init.kaiming_uniform_(p, a=2.236)

    def forward(self, x):
        # 1. Ask the algebra to build the big matrix
        W_constrained = self.algebra.expand_matrix(self.components)
        
        # 2. Standard linear pass
        return F.linear(x, W_constrained, self.bias)