import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTMIntentModel(nn.Module):
    def __init__(self,vocab,num_classes,weights_matrix=None,infer=False):
        super(LSTMIntentModel, self).__init__()
        self.vocab_size = len(vocab)
        self.embed_dim = 100
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        if not infer:
            self.embedding.weight.data.copy_(torch.tensor(weights_matrix))
            self.embedding.weight.requires_grad = True
        self.fc_dim = 64
        self.num_classes = num_classes
        self.final_dim = 100
        self.lstm_dim = 50
        self.encoder = nn.LSTM(self.embed_dim, self.lstm_dim,batch_first=True,bidirectional=True)
        self.net = nn.Sequential(nn.Linear(self.final_dim, self.fc_dim),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.30),
                                 nn.Linear(self.fc_dim, self.num_classes))
        
    def forward(self, s1, s1_len):
        s1 = self.embedding(s1)
        s1 = pack_padded_sequence(s1, s1_len.cpu(), batch_first=True, enforce_sorted=False)
        
        output_packed, (hidden,cell) = self.encoder(s1)
        # bidrection 2 layer into 1 dim
        hidden = hidden.transpose(1,0).contiguous().view(hidden.shape[1], -1)
        out = self.net(hidden)
        return out