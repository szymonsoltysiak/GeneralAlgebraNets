import torch
import torch.nn as nn
import math
from layers.linear import AlgebraLinear

class AlgebraMultiheadAttention(nn.Module):
    """
    Multi-Head Attention where Q, K, V projections are Algebraic.
    
    The 'Attention Score' is computed in the underlying real embedding space 
    (Geometric alignment), but the 'Value' mixing preserves the algebraic structure.
    """
    def __init__(self, embed_dim, num_heads, algebra, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.algebra = algebra
        self.head_dim = embed_dim // num_heads
        
        # Ensure dimensions align with algebra
        assert self.head_dim % algebra.mat_dim == 0, "Head dimension must be divisible by algebra vector size"
        
        # Algebraic Projections
        self.q_proj = AlgebraLinear(embed_dim, embed_dim, algebra)
        self.k_proj = AlgebraLinear(embed_dim, embed_dim, algebra)
        self.v_proj = AlgebraLinear(embed_dim, embed_dim, algebra)
        
        self.out_proj = AlgebraLinear(embed_dim, embed_dim, algebra)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        """
        x: [Batch, Seq_Len, Embed_Dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Project Algebraically
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape for Multi-Head
        # [B, Seq, Heads, Head_Dim] -> [B, Heads, Seq, Head_Dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Geometric Attention Scores
        # We flatten the algebraic structure to Real for the dot product similarity.
        # This implies: "How geometrically similar are these two multi-vectors?"
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 4. Context Aggregation
        # Weights (Real) * Values (Algebraic)
        # Since Algebra acts as a Vector Space over R, this scalar mult is valid.
        context = torch.matmul(attn_weights, v)
        
        # 5. Restore Shape
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.out_proj(context)

class AlgebraTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, algebra, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Attention Block
        self.self_attn = AlgebraMultiheadAttention(d_model, num_heads, algebra, dropout)
        self.norm1 = nn.LayerNorm(d_model) # LayerNorm operates on the magnitude basically
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed Forward Block (Algebraic MLP)
        self.linear1 = AlgebraLinear(d_model, dim_feedforward, algebra)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = AlgebraLinear(dim_feedforward, d_model, algebra)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # For non-linearity, we use a geometric activation
        # MagnitudeActivation is usually best for Transformers
        from layers.activation import MagnitudeActivation
        self.activation = MagnitudeActivation(dim_feedforward, algebra.mat_dim)

    def forward(self, src, mask=None):
        # 1. Self-Attention
        src2 = self.self_attn(src, mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 2. Feed Forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src