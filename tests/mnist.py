import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from core.implementations import SOnAlgebra
from layers.linear import AlgebraLinear
from layers.conv import AlgebraConv2d
from layers.activation import GatedAlgebraActivation
from modules.lifting import RealToAlgebra
from modules.projection import AlgebraToReal

# ==========================================
# 1. CONFIGURATION
# ==========================================
ALGEBRA = SOnAlgebra(n=4)  # Using so(4) Algebra
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001

# ==========================================
# 2. THE MODEL
# ==========================================
class MNISTAlgebraNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lift = RealToAlgebra(algebra_dim=ALGEBRA.mat_dim, mode='pad')
        
        self.conv1 = AlgebraConv2d(1 * ALGEBRA.mat_dim, 8 * ALGEBRA.mat_dim, 3, ALGEBRA, padding=1)
        self.act1 = GatedAlgebraActivation(8 * ALGEBRA.mat_dim, ALGEBRA.mat_dim)
        
        self.conv2 = AlgebraConv2d(8 * ALGEBRA.mat_dim, 16 * ALGEBRA.mat_dim, 3, ALGEBRA, padding=1)
        self.act2 = GatedAlgebraActivation(16 * ALGEBRA.mat_dim, ALGEBRA.mat_dim)
        
        self.pool = nn.MaxPool2d(2)
        
        input_dim = (16 * 7 * 7) * ALGEBRA.mat_dim
        
        self.fc = AlgebraLinear(input_dim, 10 * ALGEBRA.mat_dim, ALGEBRA)
        
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

# ==========================================
# 2. EVALUATION UTILS
# ==========================================
def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    correct = 0
    total = 0
    
    print("\nEvaluating...")
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Quaternion AlgebraNet')
    plt.show()

# ==========================================
# 3. MAIN RUNNER
# ==========================================
def run():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_set = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Setup
    model = MNISTAlgebraNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) # Decays LR
    criterion = nn.CrossEntropyLoss()
    
    print(f"Initialized {ALGEBRA.__class__.__name__} Network with {sum(p.numel() for p in model.parameters())} params.")

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 200 == 199:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 200:.4f}")
                running_loss = 0.0
        
        # Step LR Scheduler
        scheduler.step()
        
        # Validation after every epoch
        acc, _, _ = evaluate_model(model, test_loader, device)
        print(f"--> Epoch {epoch+1} Test Accuracy: {acc:.2f}%")

    # --- Final Evaluation ---
    print("\nTraining Complete. Running Final Evaluation...")
    final_acc, preds, true_labels = evaluate_model(model, test_loader, device)
    print(f"Final Accuracy: {final_acc:.2f}%")
    
    # Plot Confusion Matrix
    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(true_labels, preds, class_names)

if __name__ == "__main__":
    run()