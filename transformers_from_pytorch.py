import torch
from torch import nn
from vocab import Vocab
from positional_encoding import PositionalEncoding
from transformer_encoder_model import generate_padding_mask

class TransformersEncoder_pytorch(nn.Module): 
    def __init__(self, d_model: int, head: int, n_layer: int, d_ff: int, dropout: float, vocab: Vocab): 
        super().__init__() 

        self.vocab = vocab 

        self.embedding = nn.Embedding(vocab.total_tokens, d_model) 
        self.PE = PositionalEncoding(d_model, dropout) 
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, head, d_ff, dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_layer)

        self.lm_head = nn.Linear(d_model, vocab.total_labels)
        self.dropout = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss(ignore_index=2) 
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor): 
        # attention_mask = generate_padding_mask(input_ids, self.vocab.pad_idx)

        input_embs = self.embedding(input_ids) 
        features = self.PE(input_embs)
        features = self.encoder(features) 

        features = features[:, 0] 
        logits = self.dropout(self.lm_head(features))
        labels = torch.where(labels == 0, 2, labels)
        
        return logits, self.loss(logits, labels)
