import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class LSTMDataset(Dataset):
    def __init__(self,df,vocab):
        self.df = df
        self.vocab = vocab
        self.utters = df['utterance']
        self.intents = df['intent']

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        utter = self.utters[index]
        intent = self.intents[index]

        encoded_utter = [] 
        encoded_utter += self.vocab.encode(utter)

        return torch.tensor(encoded_utter), torch.tensor(intent)

class MyCollate:
    def __init__(self,padIdx):
        self.padIdx = padIdx

    def __call__(self,batch):
        (xx,zz) = zip(*batch)
        utter = pad_sequence(xx, batch_first=True, padding_value= self.padIdx)
        utter_len = [len(x) for x in xx]

        return utter,torch.tensor(utter_len),torch.tensor(zz) 