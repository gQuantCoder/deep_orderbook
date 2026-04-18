import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformers.
    
    This implementation follows the original Transformer paper's positional encoding,
    which uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize the positional encoding.
        
        Args:
            d_model (int): The embedding dimension
            max_len (int): Maximum sequence length
            dropout (float): Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and store
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x) 