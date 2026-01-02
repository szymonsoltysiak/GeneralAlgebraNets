import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Imports from your project structure
from core.implementations import QuaternionAlgebra
from layers.linear import AlgebraLinear
from layers.activation import GatedAlgebraActivation
from modules.lifting import RealToAlgebra
from modules.projection import AlgebraToReal

# ==========================================
# 1. CONFIGURATION
# ==========================================
ALGEBRA = QuaternionAlgebra()
INPUT_DIM = 3      # Swiss Roll has 3 dimensions (x, y, z)
HIDDEN_DIM = 16    # 16 Logical Quaternions (16 * 4 = 64 real params)
CLASSES = 4        # We will split the roll into 4 colored segments

# ==========================================
# 2. THE MODEL
# ==========================================
class SwissRollQNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Lift: 3 Real Coordinates -> 3 Quaternions (Scalar parts)
        # Logical Input: 3. Real Input to layer: 3 * 4 = 12
        self.lift = RealToAlgebra(algebra_dim=ALGEBRA.dim, mode='pad')
        
        # 2. Layer 1: Mixes the X, Y, Z coordinates in Quaternion space
        self.layer1 = AlgebraLinear(INPUT_DIM * ALGEBRA.dim, HIDDEN_DIM * ALGEBRA.dim, ALGEBRA)
        self.act1   = GatedAlgebraActivation(HIDDEN_DIM * ALGEBRA.dim, ALGEBRA.dim)
        
        # 3. Layer 2: Deeper geometric mixing
        self.layer2 = AlgebraLinear(HIDDEN_DIM * ALGEBRA.dim, HIDDEN_DIM * ALGEBRA.dim, ALGEBRA)
        self.act2   = GatedAlgebraActivation(HIDDEN_DIM * ALGEBRA.dim, ALGEBRA.dim)
        
        # 4. Project: Quaternion -> Real Magnitude
        self.project = AlgebraToReal(algebra_dim=ALGEBRA.dim, mode='magnitude')
        
        # 5. Classifier: Standard linear layer to predict segment class
        self.classifier = nn.Linear(HIDDEN_DIM, CLASSES)

    def forward(self, x):
        x = self.lift(x)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.project(x)
        return self.classifier(x)

# ==========================================
# 3. DATA PREPARATION
# ==========================================
def get_data(n_samples=1500):
    # Generate Swiss Roll
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.1)
    
    # Quantize the continuous 't' (manifold position) into discrete classes
    # This creates the "colored bands" along the roll
    t_min, t_max = t.min(), t.max()
    labels = ((t - t_min) / (t_max - t_min) * CLASSES).astype(int)
    labels = np.clip(labels, 0, CLASSES - 1)
    
    # Standardize inputs (Critical for Neural Networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Convert to Tensor
    X = torch.FloatTensor(X)
    y = torch.LongTensor(labels)
    
    return X, y, t  # Return t for plotting coloring later

# ==========================================
# 4. TRAINING & PLOTTING
# ==========================================
def train_and_plot():
    # Setup
    X, y, t_continuous = get_data(2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = SwissRollQNet()
    optimizer = optim.Adam(model.parameters(), lr=0.015)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training Quaternion Network on {len(X_train)} samples...")
    
    loss_history = []
    
    # Training Loop
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if epoch % 20 == 0:
            acc = (out.argmax(dim=1) == y_train).float().mean()
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train Acc: {acc:.2f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        test_preds = test_out.argmax(dim=1)
        acc = (test_preds == y_test).float().mean()
        print(f"\nFinal Test Accuracy: {acc:.2%}")

    # ==========================================
    # VISUALIZATION
    # ==========================================
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Ground Truth (The Manifold)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    p1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', s=20, alpha=0.8)
    ax1.set_title("Ground Truth (Manifold Segments)")
    ax1.view_init(7, -80) # Angle that shows the roll structure best
    fig.colorbar(p1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot 2: Model Predictions (Generalization)
    # We predict on the WHOLE dataset to see decision boundaries
    with torch.no_grad():
        all_preds = model(X).argmax(dim=1).numpy()
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    p2 = ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=all_preds, cmap='viridis', s=20, alpha=0.8)
    ax2.set_title(f"Quaternion Net Predictions (Acc: {acc:.2%})")
    ax2.view_init(7, -80)
    fig.colorbar(p2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    print("Displaying plots...")
    plt.show()

if __name__ == "__main__":
    train_and_plot()