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
    def dim(self):
        return 4

    @property
    def mat_dim(self):
        return 4

    def expand_matrix(self, weights):
        r, i, j, k = weights
        # Row-major Hamilton Product
        row1 = torch.cat([r, -i, -j, -k], dim=1)
        row2 = torch.cat([i,  r, -k,  j], dim=1)
        row3 = torch.cat([j,  k,  r, -i], dim=1)
        row4 = torch.cat([k, -j,  i,  r], dim=1)
        return torch.cat([row1, row2, row3, row4], dim=0)

class ComplexAlgebra(Algebra):
    """
    Standard Complex Numbers (C).
    Constraint: 2 parameters (real, imag) arranged to emulate complex multiplication.
    """
    @property
    def dim(self):
        return 2

    @property
    def mat_dim(self):
        return 2

    def expand_matrix(self, weights):
        r, i = weights
        # [ r  -i ]
        # [ i   r ]
        row1 = torch.cat([r, -i], dim=1)
        row2 = torch.cat([i,  r], dim=1)
        return torch.cat([row1, row2], dim=0)

class RealAlgebra(Algebra):
    """
    Standard Real Numbers (R).
    This reduces the AlgebraNet to a standard Neural Network.
    Useful for benchmarking and baselines within the same framework.
    """
    @property
    def dim(self):
        return 1

    @property
    def mat_dim(self):
        return 1

    def expand_matrix(self, weights):
        return weights[0]

# ==========================================
# 2. EXOTIC / GEOMETRIC ALGEBRAS
# ==========================================

class DualNumberAlgebra(Algebra):
    """
    Dual Numbers: z = a + b*epsilon, where epsilon^2 = 0.
    Used often in kinematics and forward-mode automatic differentiation.
    Structure:
    [ a   0 ]
    [ b   a ]
    """
    @property
    def dim(self):
        return 2

    @property
    def mat_dim(self):
        return 2

    def expand_matrix(self, weights):
        a, b = weights
        zeros = torch.zeros_like(a)
        
        row1 = torch.cat([a, zeros], dim=1)
        row2 = torch.cat([b, a],     dim=1)
        return torch.cat([row1, row2], dim=0)

class SplitComplexAlgebra(Algebra):
    """
    Split-Complex (Hyperbolic) Numbers: z = x + y*j, where j^2 = +1.
    Associated with Lorentzian geometry and Minkowski spacetime (1+1 dimensions).
    Structure:
    [ x   y ]
    [ y   x ]
    """
    @property
    def dim(self):
        return 2

    @property
    def mat_dim(self):
        return 2

    def expand_matrix(self, weights):
        x, y = weights
        row1 = torch.cat([x, y], dim=1)
        row2 = torch.cat([y, x], dim=1)
        return torch.cat([row1, row2], dim=0)

# ==========================================
# 3. MATRIX & LIE ALGEBRAS
# ==========================================

class MatrixAlgebra(Algebra):
    """
    General Matrix Algebra M_n(R).
    This corresponds to GL(n, R) style weights.
    
    Instead of a scalar weight, every connection is an n x n block of FREE parameters.
    """
    def __init__(self, n):
        self.n = n
        
    @property
    def dim(self):
        return self.n * self.n 

    @property
    def mat_dim(self):
        return self.n

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

class SOnAlgebra(Algebra):
    """
    Lie Algebra so(n): Skew-symmetric n x n matrices.
    Used for learning n-dimensional rotations.
    """
    def __init__(self, n):
        self.n = n
        self.num_params = (n * (n - 1)) // 2

    @property
    def dim(self):
        return self.num_params

    @property
    def mat_dim(self):
        return self.n 

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