import torch
from positional_encoding import PositionalEncoding, LearnedPositionalEmbedding

from torch import nn
from torch.nn import functional as F
from vocab import Vocab
from attention import ScaledDotProductAttention

# from modules.attention import ScaledDotProductAttention
# from modules.positional_encoding import PositionalEncoding

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, head: int, d_ff: int, dropout: float):
        super().__init__()

        self.self_attn = ScaledDotProductAttention(head, d_model)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, attention_mask: torch.Tensor):
        features = self.self_attn(src, src, src, attention_mask=attention_mask)

        features = features + self.dropout1(src)
        features = self.norm1(features)

        features_ffw = self.linear2(self.dropout(F.relu(self.linear1(features))))
        features_ffw = features_ffw + self.dropout2(features)
        features_ffw = self.norm2(features_ffw)

        return features_ffw
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, head: int, n_layer: int, d_ff: int, dropout: float):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, head, d_ff, dropout) for _ in range(n_layer)
        ])

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, attention_mask)

        return outputs
    
class TransformerEncoderModel_positional(nn.Module):
    def __init__(self, d_model: int, head: int, n_layer: int, d_ff: int, dropout: float, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        
        self.embedding = nn.Embedding(vocab.total_tokens, d_model)
        self.PE = LearnedPositionalEmbedding(d_model, dropout)

        self.encoder = TransformerEncoder(d_model, head, n_layer, d_ff, dropout)
        
        self.lm_head = nn.Linear(d_model, vocab.total_labels)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss(ignore_index=2)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        attention_mask = generate_padding_mask(input_ids, self.vocab.pad_idx)

        input_embs = self.embedding(input_ids)
        features = self.PE(input_embs)
        features = self.encoder(features, attention_mask)
        # only use the <cls> token to classify the input sentence
        features = features[:, 0]
        logits = self.dropout(self.lm_head(features))

        labels = torch.where(labels == 0, 2, labels)

        return logits, self.loss(logits, labels)


class TransformerEncoderModel_learnedpositional(nn.Module):
    def __init__(self, d_model: int, head: int, n_layer: int, d_ff: int, dropout: float, vocab: Vocab):
        super().__init__()

        self.vocab = vocab
        
        self.embedding = nn.Embedding(vocab.total_tokens, d_model)
        self.PE = PositionalEncoding(d_model, dropout)

        self.encoder = TransformerEncoder(d_model, head, n_layer, d_ff, dropout)
        
        self.lm_head = nn.Linear(d_model, vocab.total_labels)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss(ignore_index=2)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor):
        attention_mask = generate_padding_mask(input_ids, self.vocab.pad_idx)

        input_embs = self.embedding(input_ids)
        features = self.PE(input_embs)
        features = self.encoder(features, attention_mask)
        # only use the <cls> token to classify the input sentence
        features = features[:, 0]
        logits = self.dropout(self.lm_head(features))

        labels = torch.where(labels == 0, 2, labels)

        return logits, self.loss(logits, labels)

def generate_padding_mask(sequences: torch.Tensor, padding_value: int = 0) -> torch.Tensor:
    '''
        sequences: (bs, seq_len, dim)
    '''
    
    if len(sequences.shape) == 2: # (bs, seq_len)
        __seq = sequences.unsqueeze(dim=-1) # (bs, seq_len, 1)
    else:
        __seq = sequences
    
    mask = (torch.sum(__seq, dim=-1) == (padding_value * __seq.shape[-1])) # (b_s, seq_len)
    
    return mask # (bs, seq_len)
