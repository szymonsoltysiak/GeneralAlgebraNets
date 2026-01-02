import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import Algebra

class AlgebraLinear(nn.Module):
    def __init__(self, in_features, out_features, algebra: Algebra, bias=True):
        super().__init__()
        self.algebra = algebra
        
        assert in_features % algebra.mat_dim == 0, f"Input {in_features} must be div by {algebra.mat_dim}"
        assert out_features % algebra.mat_dim == 0
        
        self.feat_in = in_features // algebra.mat_dim
        self.feat_out = out_features // algebra.mat_dim
        
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
        for p in self.components:
            nn.init.kaiming_uniform_(p, a=2.236)

    def forward(self, x):
        W_constrained = self.algebra.expand_matrix(self.components)
        return F.linear(x, W_constrained, self.bias)