import math
import torch
from typing import Tuple
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, nhead, d_hid,
                 nlayers, max_len,weights_matrix, infer, dropout=0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.embed_dim = embed_dim
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, max_len)
        encoder_layers = TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=d_hid, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        if not infer:
            self.encoder.weight.data.copy_(torch.tensor(weights_matrix))
            self.encoder.weight.requires_grad = True
        
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, vocab_size]
        """
        src = self.encoder(src) * math.sqrt(self.embed_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = torch.mean(output,dim=0)
        return output

class TransformerIntentModel(nn.Module):
    def __init__(self,vocab,max_len,num_classes,weights_matrix=None,infer=False):
        super(TransformerIntentModel, self).__init__()
        self.vocab_size = len(vocab)
        self.num_classes = num_classes
        self.max_len = max_len
        self.embed_dim = 100
        self.num_head = 5
        self.ff_dim = 256
        self.fc_dim = 64
        self.num_layers = 2
        self.dropout = 0.2

        self.transformer = TransformerModel(self.vocab_size,self.embed_dim,self.num_head,self.ff_dim,self.num_layers,self.max_len,weights_matrix,infer,self.dropout)
        self.net = nn.Sequential(nn.Linear(self.embed_dim, self.fc_dim),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.20),
                                 nn.Linear(self.fc_dim, self.num_classes))
        
    def forward(self, inputs):
        output = self.transformer(inputs)
        output = self.net(output)
        return output