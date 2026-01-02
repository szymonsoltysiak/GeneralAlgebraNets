import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitActivation(nn.Module):
    """
    Applies a standard real-valued activation to each component independently.
    
    Example: For a Complex number (a + bi), SplitReLU outputs:
    ReLU(a) + i*ReLU(b)
    
    Works automatically for any tensor shape because standard activations 
    are element-wise.
    """
    def __init__(self, activation_fn=nn.ReLU()):
        super().__init__()
        self.act = activation_fn

    def forward(self, x):
        return self.act(x)


class MagnitudeActivation(nn.Module):
    """
    Applies an activation based on the norm (magnitude) of the algebraic number.
    Formula: out = z * ReLU(|z| + bias) / |z|
    
    PRESERVES: Orientation/Phase
    CHANGES: Magnitude
    """
    def __init__(self, features, algebra_dim, bias=True):
        super().__init__()
        self.dim = algebra_dim
        
        assert features % algebra_dim == 0, "Features must be divisible by algebra_dim"
        self.logical_features = features // algebra_dim
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.logical_features))
        else:
            self.register_parameter('bias', None)
            
        self.epsilon = 1e-5

    def forward(self, x):
        ch_dim = 1 if x.dim() > 2 else -1
        
        x_poly = x.unflatten(ch_dim, (self.logical_features, self.dim))
        
        alg_dim = ch_dim + 1 if ch_dim != -1 else -1
        
        magnitude = torch.norm(x_poly, p=2, dim=alg_dim, keepdim=True)
        
        if self.bias is not None:
            shape = [1] * x_poly.dim()
            shape[ch_dim] = self.logical_features
            
            b = self.bias.view(*shape)
            active_magnitude = F.relu(magnitude + b)
        else:
            active_magnitude = F.relu(magnitude)
            
        scale = active_magnitude / (magnitude + self.epsilon)
        
        out_poly = x_poly * scale
        
        return out_poly.flatten(ch_dim, alg_dim)


class GatedAlgebraActivation(nn.Module):
    """
    z_out = z_in * Sigmoid(w * |z| + b)
    """
    def __init__(self, in_features, algebra_dim):
        super().__init__()
        self.dim = algebra_dim
        assert in_features % algebra_dim == 0
        self.logical_features = in_features // algebra_dim
        
        self.gate_w = nn.Parameter(torch.ones(self.logical_features))
        self.gate_b = nn.Parameter(torch.zeros(self.logical_features))

    def forward(self, x):
        ch_dim = 1 if x.dim() > 2 else -1
        
        x_poly = x.unflatten(ch_dim, (self.logical_features, self.dim))
        alg_dim = ch_dim + 1 if ch_dim != -1 else -1
        
        norms = torch.norm(x_poly, p=2, dim=alg_dim, keepdim=True)
        
        shape = [1] * x_poly.dim()
        shape[ch_dim] = self.logical_features
        
        w = self.gate_w.view(*shape)
        b = self.gate_b.view(*shape)
        
        gate = torch.sigmoid(w * norms + b)
        
        out = x_poly * gate
        
        return out.flatten(ch_dim, alg_dim)