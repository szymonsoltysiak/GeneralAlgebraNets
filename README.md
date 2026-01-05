# AlgebraNets: Hypercomplex & Geometric Deep Learning Framework

**AlgebraNets** is a PyTorch framework for building neural networks over arbitrary associative algebras. Instead of real numbers ($\mathbb{R}$), neurons operate in **Complex Space ($\mathbb{C}$)**, **Quaternion Space ($\mathbb{H}$)**, **Dual Numbers ($\mathbb{D}$)**, or **Lie Algebras like $\mathfrak{so}(n)$**.

> **💡 The Core Idea: Everything is a Matrix**
> The foundation of this framework is that **all finite-dimensional associative algebras can be represented as real matrices**.
> * A Complex number $a+bi$ acts like $\begin{pmatrix} a & -b \\ b & a \end{pmatrix}$.
> * A Quaternion acts like a $4 \times 4$ block matrix.
>
> **AlgebraNets** automates this isomorphism. You define the algebra, and the framework constructs the structured weight matrices ($\psi(W)$) to enforce the algebra's geometric rules.

---

## 📐 Mathematical Framework

We formalize the network not in vector space $\mathbb{R}^n$, but in the category of **Modules over an Algebra $\mathcal{A}$**.

### 1. The Spaces
* **Input Space ($\mathcal{X}$):** $\mathbb{R}^{N_{in}}$ (Standard Real Data).
* **Feature Space ($\mathcal{F}$):** $\mathcal{A}^{C}$ (A module of rank $C$ over $\mathcal{A}$).
    * *Note:* Physically stored as a tensor of shape `[..., C, Algebra_Dim]`.
* **Output Space ($\mathcal{Y}$):** $\mathbb{R}^{N_{out}}$ (Logits/Regression targets).

### 2. The Operators
The network is a composition of functions mapping between these spaces:

$$F(x) = \rho \circ L_N \circ \sigma \circ \dots \circ L_1 \circ \phi(x)$$

| Operator | Symbol | Map | Description |
| :--- | :--- | :--- | :--- |
| **Lifting** | $\phi$ | $\mathbb{R}^{N} \to \mathcal{A}^{C}$ | Embeds real data into the algebra (e.g., zero-padding imaginary parts). |
| **Linear** | $L$ | $\mathcal{A}^{C_{in}} \to \mathcal{A}^{C_{out}}$ | Affine transformation $L(x) = W \cdot x + b$ using algebraic multiplication. |
| **Activation** | $\sigma$ | $\mathcal{A} \to \mathcal{A}$ | Element-wise non-linearity preserving geometric structure (e.g., ModReLU). |
| **Projection** | $\rho$ | $\mathcal{A}^{C} \to \mathbb{R}^{N}$ | Collapses the algebra back to real numbers (e.g., via Norm) for the loss function. |

---

## 🏗️ Architecture & Components

### 1. Implemented Algebras (`core/`)

| Algebra | Dim | Best Use Case |
| :--- | :--- | :--- |
| **`ComplexAlgebra`** | 2 | Signal processing, Phase-shift invariance. |
| **`QuaternionAlgebra`** | 4 | 3D Rotations, Robotics, RGB+D analysis. |
| **`DualNumberAlgebra`** | 2 | Kinematics, Screw theory. |
| **`SOnAlgebra(n)`** | $n(n-1)/2$ | Manifold learning on $N$-dimensional rotations. |
| **`MatrixAlgebra(n)`** | $n^2$ | Block-dense representations. |

### 2. Layers (`layers/`)

* **`AlgebraLinear`:** Replaces standard weights with algebraic block matrices.
* **`AlgebraConv1d` / `AlgebraConv2d`:** Convolutional filters where every pixel is an algebraic number.
    * *Effect:* Naturally invariant to phase (Complex) or 3D orientation (Quaternion).
* **`AlgebraTransformerLayer`:** 
    * **Geometric Attention:** Projects $Q, K, V$ algebraically. Attention scores are computed via geometric alignment (real-valued dot product), while Values are aggregated algebraically.

### 3. Activations (`layers/activation.py`)

| Name | Formula / Logic | Best For |
| :--- | :--- | :--- |
| **`Magnitude`** | $ z \cdot \text{ReLU}(\|z\|+b)/\|z\| $ | **Quaternions, Complex**. Preserves orientation/phase. |
| **`Gated`** | $ z \cdot \sigma(w\|z\|+b) $ | **Lie Groups ($\mathfrak{so}(n)$)**. Learnable amplitude gating. |
| **`Spectral`** | $U \cdot \sigma(\Sigma) \cdot V^H$ | **Matrix/Split Algebras**. Filters energy on the spectrum (SVD). |

---

## 🚀 Quick Start: Quaternion MNIST

```python
import torch.nn as nn
from core.implementations import QuaternionAlgebra
from layers.conv import AlgebraConv2d
from layers.activation import GatedAlgebraActivation
from modules.lifting import RealToAlgebra
from modules.projection import AlgebraToReal

ALGEBRA = QuaternionAlgebra() # 4 Dim

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Lift: 1-channel img -> 4-channel Quaternion (0-padded)
        self.lift = RealToAlgebra(algebra_dim=4, mode='pad')
        
        # 2. Conv: 1 Logical In -> 8 Logical Out (Physical: 4->32 ch)
        self.conv1 = AlgebraConv2d(4, 32, kernel_size=3, algebra=ALGEBRA, padding=1)
        self.act1  = GatedAlgebraActivation(32, algebra_dim=4)
        
        # 3. Project: Collapse 4D algebra to 1D Magnitude
        self.proj = AlgebraToReal(algebra_dim=4, mode='magnitude')
        self.fc   = nn.Linear(8 * 14 * 14, 10) # Standard classifier

    def forward(self, x):
        x = self.lift(x)      # [B, 1, 28, 28] -> [B, 4, 28, 28]
        x = self.act1(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = self.proj(x)      # [B, 32, 14, 14] -> [B, 8, 14, 14]
        return self.fc(x.flatten(1))