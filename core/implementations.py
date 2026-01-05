import torch
from .algebra import Algebra

# ==========================================
# 1. STANDARD HYPERCOMPLEX ALGEBRAS
# ==========================================

class QuaternionAlgebra(Algebra):
    """
    Standard Hamilton Quaternions (H). 
    Constraint: 4 shared parameters arranged in the specific non-commutative structure.
    """
    @property
    def dim(self): return 4
    @property
    def mat_dim(self): return 4

    def expand_matrix(self, weights):
        r, i, j, k = weights
        row1 = torch.cat([r, -i, -j, -k], dim=1)
        row2 = torch.cat([i,  r, -k,  j], dim=1)
        row3 = torch.cat([j,  k,  r, -i], dim=1)
        row4 = torch.cat([k, -j,  i,  r], dim=1)
        return torch.cat([row1, row2, row3, row4], dim=0)

    def vector_to_matrix(self, x):
        """ Maps vector [..., 4] -> Matrix [..., 4, 4] """
        r, i, j, k = x.unbind(dim=-1)
        row1 = torch.stack([r, -i, -j, -k], dim=-1)
        row2 = torch.stack([i,  r, -k,  j], dim=-1)
        row3 = torch.stack([j,  k,  r, -i], dim=-1)
        row4 = torch.stack([k, -j,  i,  r], dim=-1)
        return torch.stack([row1, row2, row3, row4], dim=-2)

    def matrix_to_vector(self, mat):
        """ Reconstructs vector from first column of matrix """
        return mat[..., :, 0]


class ComplexAlgebra(Algebra):
    """
    Standard Complex Numbers (C).
    Constraint: 2 parameters (real, imag).
    """
    @property
    def dim(self): return 2
    @property
    def mat_dim(self): return 2

    def expand_matrix(self, weights):
        r, i = weights
        row1 = torch.cat([r, -i], dim=1)
        row2 = torch.cat([i,  r], dim=1)
        return torch.cat([row1, row2], dim=0)

    def vector_to_matrix(self, x):
        r, i = x.unbind(dim=-1)
        row1 = torch.stack([r, -i], dim=-1)
        row2 = torch.stack([i,  r], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def matrix_to_vector(self, mat):
        return mat[..., :, 0]


class RealAlgebra(Algebra):
    """
    Standard Real Numbers (R).
    """
    @property
    def dim(self): return 1
    @property
    def mat_dim(self): return 1

    def expand_matrix(self, weights):
        return weights[0]

    def vector_to_matrix(self, x):
        # 1D vector to 1x1 matrix is just adding a dimension
        return x.unsqueeze(-1)

    def matrix_to_vector(self, mat):
        return mat.squeeze(-1)


# ==========================================
# 2. EXOTIC / GEOMETRIC ALGEBRAS
# ==========================================

class DualNumberAlgebra(Algebra):
    """
    Dual Numbers: z = a + b*epsilon, where epsilon^2 = 0.
    """
    @property
    def dim(self): return 2
    @property
    def mat_dim(self): return 2

    def expand_matrix(self, weights):
        a, b = weights
        zeros = torch.zeros_like(a)
        row1 = torch.cat([a, zeros], dim=1)
        row2 = torch.cat([b, a],     dim=1)
        return torch.cat([row1, row2], dim=0)

    def vector_to_matrix(self, x):
        a, b = x.unbind(dim=-1)
        zeros = torch.zeros_like(a)
        row1 = torch.stack([a, zeros], dim=-1)
        row2 = torch.stack([b, a],     dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def matrix_to_vector(self, mat):
        return mat[..., :, 0]


class SplitComplexAlgebra(Algebra):
    """
    Split-Complex: z = x + y*j, where j^2 = +1.
    """
    @property
    def dim(self): return 2
    @property
    def mat_dim(self): return 2

    def expand_matrix(self, weights):
        x, y = weights
        row1 = torch.cat([x, y], dim=1)
        row2 = torch.cat([y, x], dim=1)
        return torch.cat([row1, row2], dim=0)

    def vector_to_matrix(self, x):
        x_val, y_val = x.unbind(dim=-1)
        row1 = torch.stack([x_val, y_val], dim=-1)
        row2 = torch.stack([y_val, x_val], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def matrix_to_vector(self, mat):
        return mat[..., :, 0]


# ==========================================
# 3. MATRIX & LIE ALGEBRAS
# ==========================================

class MatrixAlgebra(Algebra):
    """
    General Matrix Algebra M_n(R).
    """
    def __init__(self, n):
        self.n = n
        
    @property
    def dim(self): return self.n * self.n 
    @property
    def mat_dim(self): return self.n

    def expand_matrix(self, weights):        
        n = self.n
        rows = []
        for i in range(n):
            row_chunks = []
            for j in range(n):
                idx = i * n + j
                row_chunks.append(weights[idx])
            rows.append(torch.cat(row_chunks, dim=1))
        return torch.cat(rows, dim=0)

    def vector_to_matrix(self, x):
        # x is already flattened n*n parameters. Just reshape.
        # [..., n*n] -> [..., n, n]
        base_shape = x.shape[:-1]
        return x.view(*base_shape, self.n, self.n)

    def matrix_to_vector(self, mat):
        # Flatten back
        return mat.flatten(-2, -1)


class SOnAlgebra(Algebra):
    """
    Lie Algebra so(n): Skew-symmetric n x n matrices.
    """
    def __init__(self, n):
        self.n = n
        self.num_params = (n * (n - 1)) // 2

    @property
    def dim(self): return self.num_params
    @property
    def mat_dim(self): return self.n 

    def expand_matrix(self, weights):
        n = self.n
        ref = weights[0]
        zeros = torch.zeros_like(ref)
        grid = [[zeros for _ in range(n)] for _ in range(n)]
        
        weight_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                w = weights[weight_idx]
                grid[i][j] = w
                grid[j][i] = -w
                weight_idx += 1
        
        rows = [torch.cat(row_list, dim=1) for row_list in grid]
        return torch.cat(rows, dim=0)

    def vector_to_matrix(self, x):
        """ Maps parameters [n(n-1)/2] -> Skew-Symmetric Matrix [n, n] """
        n = self.n
        batch_dims = x.shape[:-1]
        mat = torch.zeros(*batch_dims, n, n, device=x.device, dtype=x.dtype)
        
        weight_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                w = x[..., weight_idx]
                mat[..., i, j] = w
                mat[..., j, i] = -w
                weight_idx += 1
        return mat

    def matrix_to_vector(self, mat):
        """ Extracts parameters from upper triangle """
        n = self.n
        params = []
        for i in range(n):
            for j in range(i + 1, n):
                params.append(mat[..., i, j])
        return torch.stack(params, dim=-1)