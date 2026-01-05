import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.implementations import ComplexAlgebra
from layers.linear import AlgebraLinear
from layers.activation import GatedAlgebraActivation
from modules.lifting import RealToAlgebra
from modules.projection import AlgebraToReal

# ==========================================
# 1. CONFIGURATION
# ==========================================
ALGEBRA = ComplexAlgebra()
INPUT_DIM = 3      # Swiss Roll has 3 dimensions (x, y, z)
HIDDEN_DIM = 16    
CLASSES = 4        # We will split the roll into 4 colored segments

# ==========================================
# 2. THE MODEL
# ==========================================
class SwissRollAlgebraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = RealToAlgebra(algebra_dim=ALGEBRA.mat_dim, mode='pad')
        
        input_size = INPUT_DIM * ALGEBRA.mat_dim
        hidden_size = HIDDEN_DIM * ALGEBRA.mat_dim
        
        self.layer1 = AlgebraLinear(input_size, hidden_size, ALGEBRA)
        
        self.act1   = GatedAlgebraActivation(hidden_size, ALGEBRA.mat_dim)
        
        self.layer2 = AlgebraLinear(hidden_size, hidden_size, ALGEBRA)
        self.act2   = GatedAlgebraActivation(hidden_size, ALGEBRA.mat_dim)
        
        self.project = AlgebraToReal(algebra_dim=ALGEBRA.mat_dim, mode='first')
        
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
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.1)
    
    t_min, t_max = t.min(), t.max()
    labels = ((t - t_min) / (t_max - t_min) * CLASSES).astype(int)
    labels = np.clip(labels, 0, CLASSES - 1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X = torch.FloatTensor(X)
    y = torch.LongTensor(labels)
    
    return X, y, t

# ==========================================
# 4. TRAINING & PLOTTING
# ==========================================
def train_and_plot():
    X, y, t_continuous = get_data(2000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = SwissRollAlgebraNet()
    optimizer = optim.Adam(model.parameters(), lr=0.015)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training Algebra Network on {len(X_train)} samples...")
    
    loss_history = []
    
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

    model.eval()
    with torch.no_grad():
        test_out = model(X_test)
        test_preds = test_out.argmax(dim=1)
        acc = (test_preds == y_test).float().mean()
        print(f"\nFinal Test Accuracy: {acc:.2%}")

    fig = plt.figure(figsize=(16, 6))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    p1 = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', s=20, alpha=0.8)
    ax1.set_title("Ground Truth (Manifold Segments)")
    ax1.view_init(7, -80)
    fig.colorbar(p1, ax=ax1, fraction=0.046, pad=0.04)

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