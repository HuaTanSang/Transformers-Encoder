import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout ,max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Compute the positional encodings once in log space
     
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
       
        pe = self.pe[:, :x.size(1)] 
        pe = pe.expand(x.size(0), -1, -1)   
        x = x + pe
        return self.dropout(x)

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super(LearnedPositionalEmbedding, self).__init__()

        # Compute the positional embedding once in log space.
        pe = nn.Parameter(torch.randn(1, max_len, d_model))
        self.register_parameter('pe', pe)

        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] 
        return self.dropout(x)
