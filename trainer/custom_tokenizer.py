import pickle 

class Vocabulary:
    def __init__(self, freqThresold):
        self.itos = {0: '<PAD>',1: '<UNK>'}
        self.stoi = {'<PAD>': 0,'<UNK>': 1}
        self.freqThresold = freqThresold

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenizer(text):
        return [tok.lower() for tok in str(text).split(" ")]

    def build_vocabulary(self,sentenceList):
        freq = {}

        idx = 2

        for sent in sentenceList:
            for word in self.tokenizer(sent):
                #print(word)
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1

                if freq[word] == self.freqThresold:
                    self.itos[idx] = word
                    self.stoi[word] = idx 
                    idx += 1

    def encode(self,text):
        tokenizedText = self.tokenizer(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenizedText
        ]

    def storeVocab(self,name):
        print("Saving Vocab Dict...")
        with open('itos.pkl', 'wb') as f:
            pickle.dump(self.itos, f, pickle.HIGHEST_PROTOCOL)

        with open('stoi.pkl', 'wb') as f:
            pickle.dump(self.stoi, f, pickle.HIGHEST_PROTOCOL)