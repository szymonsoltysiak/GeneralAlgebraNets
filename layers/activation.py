import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitActivation(nn.Module):
    def __init__(self, activation_fn=nn.ReLU()):
        super().__init__()
        self.act = activation_fn

    def forward(self, x):
        return self.act(x)

class MagnitudeActivation(nn.Module):
    def __init__(self, features, algebra_dim, bias=True):
        super().__init__()
        self.dim = algebra_dim
        assert features % algebra_dim == 0
        self.logical_features = features // algebra_dim
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.logical_features))
        else:
            self.register_parameter('bias', None)
        self.epsilon = 1e-5

    def _get_ch_dim(self, x):
        expected_size = self.logical_features * self.dim
        if x.shape[-1] == expected_size:
            return -1
        elif x.dim() > 1 and x.shape[1] == expected_size:
            return 1
        return 1 if x.dim() > 2 else -1

    def forward(self, x):
        ch_dim = self._get_ch_dim(x)
        
        # 1. Unflatten to expose Algebra Dimension
        x_poly = x.unflatten(ch_dim, (self.logical_features, self.dim))
        
        # Calculate indices for the split dimensions
        # If ch_dim was -1, the new dims are at -2 and -1
        if ch_dim == -1:
            feat_dim = -2
            alg_dim = -1
        else:
            feat_dim = ch_dim
            alg_dim = ch_dim + 1

        # 2. Compute Magnitude
        magnitude = torch.norm(x_poly, p=2, dim=alg_dim, keepdim=True)
        
        # 3. Apply Bias
        if self.bias is not None:
            # Create broadcasting shape
            shape = [1] * x_poly.dim()
            # Handle negative indexing for the feature dimension
            target_idx = feat_dim if feat_dim >= 0 else x_poly.dim() + feat_dim
            shape[target_idx] = self.logical_features
            
            b = self.bias.view(*shape)
            active_magnitude = F.relu(magnitude + b)
        else:
            active_magnitude = F.relu(magnitude)
            
        # 4. Scale
        scale = active_magnitude / (magnitude + self.epsilon)
        out_poly = x_poly * scale
        
        # 5. Flatten back (Fixing the bug here)
        return out_poly.flatten(feat_dim, alg_dim)


class GatedAlgebraActivation(nn.Module):
    def __init__(self, in_features, algebra_dim):
        super().__init__()
        self.dim = algebra_dim
        assert in_features % algebra_dim == 0
        self.logical_features = in_features // algebra_dim
        
        self.gate_w = nn.Parameter(torch.ones(self.logical_features))
        self.gate_b = nn.Parameter(torch.zeros(self.logical_features))

    def _get_ch_dim(self, x):
        expected_size = self.logical_features * self.dim
        if x.shape[-1] == expected_size:
            return -1
        elif x.dim() > 1 and x.shape[1] == expected_size:
            return 1
        return 1 if x.dim() > 2 else -1

    def forward(self, x):
        ch_dim = self._get_ch_dim(x)
        
        x_poly = x.unflatten(ch_dim, (self.logical_features, self.dim))
        
        if ch_dim == -1:
            feat_dim = -2
            alg_dim = -1
        else:
            feat_dim = ch_dim
            alg_dim = ch_dim + 1
            
        norms = torch.norm(x_poly, p=2, dim=alg_dim, keepdim=True)
        
        shape = [1] * x_poly.dim()
        target_idx = feat_dim if feat_dim >= 0 else x_poly.dim() + feat_dim
        shape[target_idx] = self.logical_features
        
        w = self.gate_w.view(*shape)
        b = self.gate_b.view(*shape)
        
        gate = torch.sigmoid(w * norms + b)
        out = x_poly * gate
        
        return out.flatten(feat_dim, alg_dim)


class SpectralActivation(nn.Module):
    def __init__(self, in_features, algebra, activation_fn=F.relu):
        super().__init__()
        self.algebra = algebra
        self.act = activation_fn
        if in_features % algebra.dim != 0:
            raise ValueError(f"Features {in_features} not divisible by {algebra.dim}")
        self.features = in_features // algebra.dim
        
    def _get_ch_dim(self, x):
        expected_size = self.features * self.algebra.dim
        if x.shape[-1] == expected_size:
            return -1
        elif x.dim() > 1 and x.shape[1] == expected_size:
            return 1
        return 1 if x.dim() > 2 else -1
        
    def forward(self, x):
        ch_dim = self._get_ch_dim(x)

        x_poly = x.unflatten(ch_dim, (self.features, self.algebra.dim))
        
        if ch_dim == -1:
            feat_dim = -2
            alg_dim = -1
        else:
            feat_dim = ch_dim
            alg_dim = ch_dim + 1

        mat = self.algebra.vector_to_matrix(x_poly)

        try:
            U, S, Vh = torch.linalg.svd(mat)
        except RuntimeError:
            eps = 1e-6 * torch.eye(mat.shape[-1], device=mat.device)
            U, S, Vh = torch.linalg.svd(mat + eps)
        
        S_activated = self.act(S)
        new_mat = U @ torch.diag_embed(S_activated) @ Vh
        out_poly = self.algebra.matrix_to_vector(new_mat)
        
        return out_poly.flatten(feat_dim, alg_dim)