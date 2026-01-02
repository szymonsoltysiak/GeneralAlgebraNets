import torch
import torch.nn as nn
import torch.nn.functional as F
from core.algebra import Algebra

class AlgebraConv1d(nn.Module):
    """
    1D Convolution for Algebraic Data (e.g. Time Series, Audio).
    Weights shape: [Out, In, Kernel_Length]
    """
    def __init__(self, in_channels, out_channels, kernel_size, algebra: Algebra, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.algebra = algebra
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if in_channels % algebra.mat_dim != 0:
            raise ValueError(f"In_channels {in_channels} must be divisible by {algebra.mat_dim}")
        if out_channels % algebra.mat_dim != 0:
            raise ValueError(f"Out_channels {out_channels} must be divisible by {algebra.mat_dim}")
            
        self.feat_in = in_channels // algebra.mat_dim
        self.feat_out = out_channels // algebra.mat_dim
        
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        
        self.components = nn.ParameterList([
            nn.Parameter(torch.empty(self.feat_out, self.feat_in, k))
            for _ in range(algebra.dim)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.components:
            nn.init.kaiming_uniform_(p, a=2.236)

    def forward(self, x):
        W_constrained = self.algebra.expand_matrix(self.components)
        
        return F.conv1d(
            input=x, 
            weight=W_constrained, 
            bias=self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            groups=self.groups
        )

class AlgebraConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, algebra: Algebra, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.algebra = algebra
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        if in_channels % algebra.mat_dim != 0:
            raise ValueError(f"In_channels {in_channels} must be divisible by {algebra.mat_dim}")
        if out_channels % algebra.mat_dim != 0:
            raise ValueError(f"Out_channels {out_channels} must be divisible by {algebra.mat_dim}")
            
        self.feat_in = in_channels // algebra.mat_dim
        self.feat_out = out_channels // algebra.mat_dim
        
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        
        self.components = nn.ParameterList([
            nn.Parameter(torch.empty(self.feat_out, self.feat_in, k, k))
            for _ in range(algebra.dim)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.components:
            nn.init.kaiming_uniform_(p, a=2.236)

    def forward(self, x):
        W_constrained = self.algebra.expand_matrix(self.components)
        
        return F.conv2d(
            input=x, 
            weight=W_constrained, 
            bias=self.bias, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            groups=self.groups
        )