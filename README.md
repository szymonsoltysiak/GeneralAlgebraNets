# AlgebraNets: A Modular Framework for Hypercomplex & Geometric Deep Learning

**AlgebraNets** is a PyTorch-based framework for building neural networks that operate over arbitrary algebraic structures. Instead of standard real numbers ($\mathbb{R}$), neurons and weights can exist in **Complex Space ($\mathbb{C}$)**, **Quaternion Space ($\mathbb{H}$)**, **Dual Numbers ($\mathbb{D}$)**, or even **Lie Algebras like $\mathfrak{so}(n)$**.

This allows the network to learn geometric primitives (rotations, scales, screw motions) directly, providing better generalization, parameter efficiency, and mathematical interpretability for tasks involving geometry, physics, or signal processing.

> ** Everything is a Matrix**
> The mathematical foundation of this framework is that **all finite-dimensional associative algebras can be represented as real matrices**.
> * A Complex number $a+bi$ acts exactly like the matrix $\begin{pmatrix} a & -b \\ b & a \end{pmatrix}$.
> * A Quaternion acts like a $4 \times 4$ skew-symmetric block.

---

## Architecture Overview

The framework is built on a modular "Algebra-First" design. You define the algebra once, and all layers (Linear, Conv, Activation) automatically adapt their weight matrices and logic to respect that algebra's multiplication rules.

### 1. Core Abstractions (`core/`)
* **`Algebra` (Base Class):** The brain of the operation. It defines:
    * `dim`: Number of parameters (e.g., 4 for Quaternions).
    * `mat_dim`: Physical vector size (e.g., 4 for Quat, $N$ for SO($N$)).
    * `expand_matrix()`: Tiling logic to enforce algebraic multiplication (e.g., Hamilton product).

### 2. Implemented Algebras (`core/implementations.py`)

| Algebra | Dim | Description | Best Use Case |
| :--- | :--- | :--- | :--- |
| **`RealAlgebra`** | 1 | Standard $\mathbb{R}$ (Identity). | Baseline comparison. |
| **`ComplexAlgebra`** | 2 | $z = x + iy$ | Signal processing (Audio/Radio), 2D rotation. |
| **`QuaternionAlgebra`** | 4 | $q = r + xi + yj + zk$ | 3D rotations, Robotics, Satellites, RGB+D images. |
| **`DualNumberAlgebra`** | 2 | $z = a + b\epsilon$ ($\epsilon^2=0$) | Kinematics, Screw theory, Auto-diff. |
| **`SplitComplexAlgebra`** | 2 | $z = x + jy$ ($j^2=1$) | Lorentzian geometry, Minkowski spacetime. |
| **`SOnAlgebra(n)`** | $n(n-1)/2$ | Skew-symmetric matrices $\mathfrak{so}(n)$. | $N$-dimensional rotations, Manifold learning. |
| **`MatrixAlgebra(n)`** | $n^2$ | Full $n \times n$ blocks. | Block-dense networks. |

### 3. Layers (`layers/`)
* **`AlgebraLinear`:** A fully connected layer where weights are algebraic numbers.
    * *Logic:* Replaces $y=Wx$ with algebraic multiplication using block matrices.
* **`AlgebraConv1d` / `AlgebraConv2d`:** Convolutional layers with algebraic kernels.
    * *Logic:* Exploits PyTorch's `groups` or weight expansion to perform sliding window algebraic products.

### 4. Activations (`layers/activation.py`)

Standard activations (ReLU) applied element-wise can destroy the geometric properties of algebraic numbers (e.g., changing the direction of a vector). We provide geometrically valid alternatives:

* **`MagnitudeActivation` (ModReLU):**
* *Formula:* 
* *Logic:* Scales the vector radially based on its magnitude. **Preserves orientation/phase** perfectly. Ideal for Quaternions and Complex numbers.


* **`GatedAlgebraActivation`:**
* *Formula:* 
* *Logic:* Uses a learnable "gate" (sigmoid) to scale the amplitude. More expressive than MagnitudeActivation.


* **`SpectralActivation`:**
* *Logic:* Computes the **Eigenvalues** (Spectrum) of the algebraic element (via matrix representation), applies a non-linearity to them, and reconstructs the element.
* *Best For:* Matrix Algebras, Split-Complex numbers, and structures with distinct real eigenvalues.


* **`SplitActivation`:**
* *Logic:* Applies standard ReLU to each component independently. Fast, but not rotationally invariant.

### 5. Modules (`modules/`)
* **`RealToAlgebra` (Lifting):** Maps standard input data (Images, Audio) into the algebraic domain.
    * *Modes:* `pad` (zero-padding), `repeat`, `linear` (learned projection).
* **`AlgebraToReal` (Projection):** Maps algebraic features back to real-world predictions (logits).
    * *Modes:* `magnitude` (Norm), `first` (Real component), `flatten`.

---

## Quick Start

### Example: Quaternion CNN for MNIST
```python
from core.implementations import QuaternionAlgebra
from layers.conv import AlgebraConv2d
from layers.activation import GatedAlgebraActivation
from modules.lifting import RealToAlgebra
from modules.projection import AlgebraToReal

# 1. Define Algebra
ALGEBRA = QuaternionAlgebra() # 4 dimensions

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Lift 1-channel image to 4-channel Quaternion (Pads zeros)
        self.lift = RealToAlgebra(algebra_dim=4, mode='pad')
        
        # Conv Layers: 1 Logical In -> 8 Logical Out (Physical: 4 -> 32 channels)
        self.conv1 = AlgebraConv2d(1 * ALGEBRA.mat_dim, 8 * ALGEBRA.mat_dim, 3, ALGEBRA, padding=1)
        self.act1 = GatedAlgebraActivation(8 * ALGEBRA.mat_dim, ALGEBRA.mat_dim)
        
        self.conv2 = AlgebraConv2d(8 * ALGEBRA.mat_dim, 16 * ALGEBRA.mat_dim, 3, ALGEBRA, padding=1)
        self.act2 = GatedAlgebraActivation(16 * ALGEBRA.mat_dim, ALGEBRA.mat_dim)

        self.pool = nn.MaxPool2d(2)
        
        # Linear Layer
        input_dim = (16 * 7 * 7) * ALGEBRA.mat_dim
        self.fc = AlgebraLinear(input_dim, 10 * ALGEBRA.mat_dim, ALGEBRA)

        # Project back to Real
        self.project = AlgebraToReal(algebra_dim=ALGEBRA.mat_dim, mode='magnitude')

    def forward(self, x):
        x = self.lift(x)      
        x = self.conv1(x)     
        x = self.act1(x)
        x = self.pool(x)      
        x = self.conv2(x)     
        x = self.act2(x)
        x = self.pool(x)
        
        x = x.flatten(1)
        
        x = self.fc(x)
        
        return self.project(x)
