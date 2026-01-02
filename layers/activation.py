import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitActivation(nn.Module):
    """
    Applies a standard real-valued activation to each component independently.
    
    Example: For a Complex number (a + bi), SplitReLU outputs:
    ReLU(a) + i*ReLU(b)
    
    Pros: Very fast, standard gradients.
    Cons: Not rotationally equivariant. Breaks the geometry.
    """
    def __init__(self, activation_fn=nn.ReLU()):
        super().__init__()
        self.act = activation_fn

    def forward(self, x):
        # Since our "Algebra Tensor" is just a flattened real tensor,
        # standard element-wise application IS split activation.
        return self.act(x)


class MagnitudeActivation(nn.Module):
    """
    Applies an activation based on the norm (magnitude) of the algebraic number.
    Often called 'ModReLU' in Complex/Quaternion literature.
    
    Formula: 
    scale = ReLU(|z| + bias) / |z|
    out = z * scale
    
    Result: The vector 'z' is scaled radially. Its direction (phase/orientation) 
    is perfectly preserved.
    """
    def __init__(self, features, algebra_dim, bias=True):
        """
        Args:
            features (int): Number of independent algebraic features (channels).
                            (e.g., if input is 128 floats and dim is 4, features=32)
            algebra_dim (int): Dimensionality of the algebra (2=Complex, 4=Quat).
            bias (bool): Whether to learn a threshold parameter (the 'dead zone' radius).
        """
        super().__init__()
        self.dim = algebra_dim
        self.features = features
        
        # Learnable bias parameter.
        # It determines the "radius" below which the neuron is inactive.
        # Initialized to 0 or slightly negative to ensure initial flow.
        if bias:
            self.bias = nn.Parameter(torch.zeros(features))
        else:
            self.register_parameter('bias', None)
            
        self.epsilon = 1e-5 # For numerical stability

    def forward(self, x):
        # x shape: [Batch, Total_Real_Dimensions]
        batch_size = x.shape[0]
        
        # 1. Reshape to separate algebraic components
        # View as: [Batch, Features, Algebra_Dim]
        x_reshaped = x.view(batch_size, self.features, self.dim)
        
        # 2. Compute Magnitude (L2 Norm) along the last dimension
        # Shape: [Batch, Features, 1]
        magnitude = torch.norm(x_reshaped, p=2, dim=2, keepdim=True)
        
        # 3. Calculate Scaling Factor
        if self.bias is not None:
            # Broadcast bias: [Features] -> [1, Features, 1]
            b = self.bias.view(1, -1, 1)
            # ReLU(|z| + b). If |z| + b < 0, the neuron dies.
            active_magnitude = F.relu(magnitude + b)
        else:
            active_magnitude = F.relu(magnitude)
            
        # 4. Apply Scaling
        # We divide by magnitude to get the unit vector, then multiply by active_magnitude
        # optimization: scale = active_magnitude / (magnitude + eps)
        scale = active_magnitude / (magnitude + self.epsilon)
        
        # Broadcast scale to all components
        # [B, F, 1] * [B, F, D] = [B, F, D]
        out_reshaped = x_reshaped * scale
        
        # 5. Flatten back to original shape
        return out_reshaped.view(batch_size, -1)

class GatedAlgebraActivation(nn.Module):
    """
    A more advanced activation where a separate "Gate" determines the amplitude.
    
    z_out = z_in * Sigmoid(Linear(z_in))
    """
    def __init__(self, in_features, algebra_dim):
        super().__init__()
        self.dim = algebra_dim
        self.features = in_features // algebra_dim
        
        # A lightweight real-valued projection to learn the gate
        # Maps magnitude -> scalar gate value
        self.gate_b = nn.Parameter(torch.zeros(self.features))
        self.gate_w = nn.Parameter(torch.ones(self.features))

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.features, self.dim)
        
        # Compute norms: [B, F]
        norms = torch.norm(x_reshaped, p=2, dim=2)
        
        # Compute Gate: Sigmoid(w * norm + b)
        # This allows the network to learn smooth "on/off" regions based on amplitude
        gate = torch.sigmoid(self.gate_w * norms + self.gate_b)
        
        # Apply gate (broadcasting)
        # [B, F, D] * [B, F, 1]
        out = x_reshaped * gate.unsqueeze(-1)
        
        return out.view(batch_size, -1)