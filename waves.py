import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from core.implementations import ComplexAlgebra
from layers.conv import AlgebraConv1d
from layers.linear import AlgebraLinear
from layers.activation import MagnitudeActivation
from modules.lifting import RealToAlgebra
from modules.projection import AlgebraToReal

# ==========================================
# 1. CONFIGURATION
# ==========================================
ALGEBRA = ComplexAlgebra()
BATCH_SIZE = 32
LR = 0.01


# ==========================================
# 1. DATA GENERATION (Synthetic Signals)
# ==========================================
def generate_signals(n_samples=1000, seq_len=100):
    """
    Generates noisy sine waves.
    Class 0: Low Freq (1Hz)
    Class 1: High Freq (3Hz)
    Class 2: Mixed (1Hz + 3Hz)
    """
    t = np.linspace(0, 4*np.pi, seq_len)
    X = []
    y = []
    
    for _ in range(n_samples):
        label = np.random.randint(0, 3)
        noise = np.random.normal(0, 0.3, seq_len)
        
        phase = np.random.uniform(0, 2*np.pi)
        
        if label == 0:
            signal = np.sin(1 * t + phase)
        elif label == 1:
            signal = np.sin(3 * t + phase)
        else:
            signal = np.sin(1 * t + phase) + np.sin(3 * t + phase)
            
        X.append(signal + noise)
        y.append(label)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    return torch.tensor(X).unsqueeze(1), torch.tensor(y)

# ==========================================
# 2. THE MODEL
# ==========================================
class SignalAlgebraNet(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        self.lift = RealToAlgebra(algebra_dim=algebra.mat_dim, mode='pad')
        
        self.conv1 = AlgebraConv1d(1*algebra.mat_dim, 8*algebra.mat_dim, kernel_size=5, algebra=algebra, padding=2)
        self.act1  = MagnitudeActivation(8*algebra.mat_dim, algebra.mat_dim)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = AlgebraConv1d(8*algebra.mat_dim, 16*algebra.mat_dim, kernel_size=5, algebra=algebra, padding=2)
        self.act2  = MagnitudeActivation(16*algebra.mat_dim, algebra.mat_dim)
        self.pool2 = nn.AdaptiveAvgPool1d(1) # Global Pool -> [B, C, 1]
        
        self.flatten = nn.Flatten()
        input_dim = 16 * algebra.mat_dim
        self.fc = AlgebraLinear(input_dim, 3*algebra.mat_dim, algebra)
        
        self.project = AlgebraToReal(algebra_dim=algebra.mat_dim, mode='magnitude')

    def forward(self, x):
        x = self.lift(x)      # [B, 2, 100] (Complex)
        
        x = self.conv1(x)     # [B, 16, 100]
        x = self.act1(x)
        x = self.pool1(x)     # [B, 16, 50]
        
        x = self.conv2(x)     # [B, 32, 50]
        x = self.act2(x)
        x = self.pool2(x)     # [B, 32, 1] (Global Average)
        
        x = self.flatten(x)   # [B, 32]
        x = self.fc(x)        # [B, 6] (3 Complex classes)
        
        return self.project(x) # [B, 3] (Magnitudes)

# ==========================================
# 3. TRAINING
# ==========================================
def run_experiment():    
    print("Generating data...")
    X_train, y_train = generate_signals(1000)
    X_test, y_test = generate_signals(200)
    
    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SignalAlgebraNet(ALGEBRA)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training {ALGEBRA.__class__.__name__} Net on 1D Signals...")
    
    losses = []
    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
        acc = 100 * correct / total
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Train Acc: {acc:.1f}%")

    model.eval()
    with torch.no_grad():
        out_test = model(X_test)
        preds_test = out_test.argmax(dim=1)
        test_acc = 100 * (preds_test == y_test).float().mean()
        print(f"\nFinal Test Accuracy: {test_acc:.1f}%")
        
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    
    plt.subplot(1, 2, 2)
    indices = np.random.choice(len(X_test), 5, replace=False)
    for i in indices:
        sig = X_test[i, 0].numpy()
        true_lbl = y_test[i].item()
        pred_lbl = preds_test[i].item()
        color = 'g' if true_lbl == pred_lbl else 'r'
        plt.plot(sig, color=color, alpha=0.6, label=f"T:{true_lbl} P:{pred_lbl}")
    
    plt.title("Sample Test Signals (Green=Correct)")
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()