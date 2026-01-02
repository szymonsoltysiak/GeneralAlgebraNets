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

    def expand_matrix(self, weights):
        r, i = weights
        # [ r  -i ]
        # [ i   r ]
        row1 = torch.cat([r, -i], dim=1)
        row2 = torch.cat([i,  r], dim=1)
        return torch.cat([row1, row2], dim=0)

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

    def expand_matrix(self, weights):
        a, b = weights
        # Row 1: a, 0
        # Row 2: b, a
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

    def expand_matrix(self, weights):
        x, y = weights
        # Note the symmetry compared to Complex (no negative sign)
        row1 = torch.cat([x, y], dim=1)
        row2 = torch.cat([y, x], dim=1)
        return torch.cat([row1, row2], dim=0)

# ==========================================
# 3. MATRIX & LIE ALGEBRAS
# ==========================================

class MatrixAlgebra(Algebra):
    """
    General Matrix Algebra M_n(R).
    This corresponds to the user's request for GL(n, R) style weights.
    
    Instead of a scalar weight, every connection is an n x n block of FREE parameters.
    Constraint: None (Dense blocks), but enforces block-locality in the network structure.
    """
    def __init__(self, n):
        self.n = n
        
    @property
    def dim(self):
        return self.n * self.n # A 2x2 matrix has 4 parameters

    def expand_matrix(self, weights):
        # 'weights' is a list of n*n tensors.
        # We need to arrange them into an n x n block structure.
        # Let's assume the list is row-major: w_00, w_01, w_10, w_11...
        
        n = self.n
        rows = []
        
        for i in range(n):
            # Collect the n chunks for this row
            row_chunks = []
            for j in range(n):
                idx = i * n + j
                row_chunks.append(weights[idx])
            
            # Concatenate them horizontally
            rows.append(torch.cat(row_chunks, dim=1))
            
        # Concatenate rows vertically
        return torch.cat(rows, dim=0)

class SO3Algebra(Algebra):
    """
    Lie Algebra so(3): Skew-symmetric 3x3 matrices.
    Used for learning infinitesimal 3D rotations (angular velocities).
    
    Parameters: 3 (x, y, z)
    Structure:
    [  0  -z   y ]
    [  z   0  -x ]
    [ -y   x   0 ]
    """
    @property
    def dim(self):
        return 3

    def expand_matrix(self, weights):
        x, y, z = weights
        zeros = torch.zeros_like(x)
        
        # Build the skew-symmetric block
        row1 = torch.cat([zeros, -z,      y], dim=1)
        row2 = torch.cat([z,     zeros,  -x], dim=1)
        row3 = torch.cat([-y,    x,   zeros], dim=1)
        
        return torch.cat([row1, row2, row3], dim=0)