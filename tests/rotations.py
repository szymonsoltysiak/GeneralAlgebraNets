import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Imports
from core.implementations import QuaternionAlgebra
from layers.linear import AlgebraLinear
from layers.transformer import AlgebraTransformerLayer
from modules.lifting import RealToAlgebra
from modules.projection import AlgebraToReal

# ==========================================
# 1. CONFIGURATION
# ==========================================
ALGEBRA = QuaternionAlgebra()
BATCH_SIZE = 32
LR = 0.002
SEQ_LEN = 50
D_MODEL = 64      # Embedding Size (Must be divisible by 4)
NUM_HEADS = 4     # 4 Heads working on Quaternions

# ==========================================
# 2. DATA GENERATION (3D Rotations)
# ==========================================
def euler_to_quaternion(roll, pitch, yaw):
    """Converts Euler angles to Quaternions."""
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.stack([qw, qx, qy, qz], axis=1)

def generate_rotations(n_samples=1000, seq_len=50):
    """
    Generates sequences of Quaternions.
    0: Roll (X-axis)
    1: Pitch (Y-axis)
    2: Yaw (Z-axis)
    3: Chaos (Random)
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        label = np.random.randint(0, 4)
        
        # Base noise for all angles
        roll  = np.random.normal(0, 0.1, seq_len)
        pitch = np.random.normal(0, 0.1, seq_len)
        yaw   = np.random.normal(0, 0.1, seq_len)
        
        # Add systematic motion based on class
        velocity = np.linspace(0, 4*np.pi, seq_len) # 2 full rotations
        
        if label == 0:   # Roll
            roll += velocity
        elif label == 1: # Pitch
            pitch += velocity
        elif label == 2: # Yaw
            yaw += velocity
        else:            # Chaos
            roll  += np.random.normal(0, 2.0, seq_len)
            pitch += np.random.normal(0, 2.0, seq_len)
            yaw   += np.random.normal(0, 2.0, seq_len)
            
        # Convert to Quaternions [Seq, 4]
        quat_seq = euler_to_quaternion(roll, pitch, yaw)
        
        X.append(quat_seq)
        y.append(label)
        
    X = np.array(X, dtype=np.float32) # [Samples, Seq, 4]
    y = np.array(y, dtype=np.int64)
    
    return torch.tensor(X), torch.tensor(y)

# ==========================================
# 3. THE MODEL
# ==========================================
class MotionAlgebraTransformer(nn.Module):
    def __init__(self, algebra):
        super().__init__()
        self.algebra = algebra
        
        # 1. Lift: Input is ALREADY 4D Quaternions. 
        # We assume input is [B, Seq, 4]. We map this to D_MODEL.
        # We use AlgebraLinear for this projection.
        self.input_proj = AlgebraLinear(4, D_MODEL, algebra)
        
        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        
        # 3. Encoder
        self.encoder = AlgebraTransformerLayer(
            d_model=D_MODEL, 
            num_heads=NUM_HEADS, 
            algebra=algebra, 
            dim_feedforward=128, 
            dropout=0.1
        )
        
        # 4. Classifier
        self.fc = AlgebraLinear(D_MODEL, 4*algebra.mat_dim, algebra)
        self.project = AlgebraToReal(algebra_dim=algebra.mat_dim, mode='magnitude')

    def forward(self, x):
        # x: [B, Seq, 4] (Quaternions)
        
        # Project: [B, Seq, 4] -> [B, Seq, 64]
        # This mixes the quaternion components algebraically
        x = self.input_proj(x)
        
        x = x + self.pos_embed
        
        # Transformer
        x = self.encoder(x)   # -> [B, Seq, 64]
        
        # Global Avg Pool
        x = x.mean(dim=1)     # -> [B, 64]
        
        # Classify
        x = self.fc(x)        # -> [B, 16] (4 Quaternions)
        return self.project(x)# -> [B, 4] (Magnitudes)

# ==========================================
# 4. TRAINING
# ==========================================
def run_experiment():    
    print("Generating geometric data (Quaternions)...")
    X_train, y_train = generate_rotations(1000, SEQ_LEN)
    X_test, y_test = generate_rotations(200, SEQ_LEN)
    
    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MotionAlgebraTransformer(ALGEBRA)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training {ALGEBRA.__class__.__name__} Transformer on 3D Motion...")
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    
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

    # Plot
    plt.plot(losses, marker='o')
    plt.title("Quaternion Transformer Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    run_experiment()