import os
import torch
import pandas as pd
import fasttext
import numpy as np
import torch.nn as nn
from .utils import explode,print_metrics,model_saver
from .bert_dataset import BertDataset
from .bert_model import BertIntentModel
from .lstm_dataset import LSTMDataset,MyCollate
from .lstm_model import LSTMIntentModel
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from .custom_tokenizer import Vocabulary

class Trainer:
    def __init__(self,intent_collection,config):
        self.intent_collection = intent_collection
        self.config = config
        self.data_path = config.DATA_PATH
        self.name = datetime.now().strftime("%Y%m%d%H%M%S")

    def data_creator(self):
        file_name = '/training_{}.xlsx'.format(self.name)
        intents = []
        utterances = []
        for data in self.intent_collection.find({}):
            intents.append(data['intent'])
            utterances.append(data['utterances'])

        df = pd.DataFrame(list(zip(intents, utterances)),columns=['intent','utterance'])
        df = explode(df, lst_cols=['utterance'])
        df = df[['utterance','intent']]
        df.to_excel(self.data_path+file_name,index=False)
        return None 

    def get_latest_file(self):
        self.data_creator()
        files = os.listdir(self.data_path)
        latest_file_name = '/training_{}.xlsx'.format(max([file.split('_')[1].split('.')[0] for file in files]))
        return latest_file_name

    def data_loader(self):
        training_file = self.data_path + self.get_latest_file()
        df = pd.read_excel(training_file)
        df.dropna(inplace=True)
        df = df.sample(frac=1).reset_index(drop=True)

        le = LabelEncoder()
        df['intent'] = le.fit_transform(df['intent'])
        num_labels = len(le.classes_)

        df_train, df_val = train_test_split(df, test_size=.10,stratify = df.intent)
        df_train.reset_index(drop=True, inplace=True)
        df_val.reset_index(drop=True, inplace=True)

        return df,df_train,df_val,num_labels,le

    def bert_loader(self):
        print("Loading Data for Training...")
        df,df_train,df_val,num_labels,le = self.data_loader()
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        token_lens = []
        for txt in df.utterance:
            tokens = tokenizer.encode(str(txt), max_length=512,truncation=True)
            token_lens.append(len(tokens))
        max_len = int(np.percentile(token_lens,99))
        print("Total Data:",df.shape[0])
        trainDataset = BertDataset(df_train,self.config.MODEL_CONFIG['bert']['tokenizer'], max_len)
        valDataset = BertDataset(df_val, self.config.MODEL_CONFIG['bert']['tokenizer'], max_len)

        trainLoader = DataLoader(
            dataset=trainDataset,
            batch_size=self.config.MODEL_CONFIG['bert']['train_batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        valLoader = DataLoader(
            dataset=valDataset,
            batch_size=self.config.MODEL_CONFIG['bert']['val_batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        return trainLoader, valLoader, num_labels, le, max_len

    def bert_trainer(self):
        epochs = self.config.MODEL_CONFIG['bert']['epochs']
        device = self.config.MODEL_CONFIG['bert']['device']
        
        def validate(model, valLoader):
            model.eval()
            val_targets = []
            val_outputs = []
            with torch.no_grad():
                for _, data in enumerate(valLoader):
                    ids = data['ids'].to(device, dtype=torch.long)
                    mask = data['mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                    targets = data['targets'].to(device, dtype=torch.long)
                    outputs = model(ids, mask, token_type_ids)
                    _, preds = torch.max(outputs, dim=1)
                    loss = loss_fun(outputs, targets)
                    epoch_loss = loss.item()
                    val_targets.extend(targets.cpu().detach().numpy().tolist())
                    val_outputs.extend(preds.cpu().detach().numpy().tolist())

            return print_metrics(val_targets,val_outputs, epoch_loss,'Validation')
        
        trainLoader, valLoader, num_labels, le, max_len = self.bert_loader()
        model = BertIntentModel(num_labels,self.config.MODEL_CONFIG['bert']['model_config']).to(device) 
        # model = BertIntentModel(num_labels,BertConfig()).to(device) 
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.MODEL_CONFIG['bert']['lr'])
        loss_fun = torch.nn.CrossEntropyLoss().to(device)
        prev_loss = float('inf')
        print("Training Started...")
        print("Device:",device)
        for epoch in range(1,epochs+1):
            print(f'Epoch: {epoch}')
            #eval_metrics["epochs"].append(epoch)
            model.train()
            epoch_loss = 0
            # training actual and prediction for each epoch for printing metrics
            train_targets = []
            train_outputs = []
            for _, data in enumerate(trainLoader):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.long)

                outputs = model(ids, mask, token_type_ids)

                #softmax = torch.nn.Softmax(dim=1)
                #_, preds = torch.max(softmax(outputs), dim=1)
                
                _, preds = torch.max(outputs, dim=1)

                loss = loss_fun(outputs, targets)
                epoch_loss = loss.item()
                train_targets.extend(targets.cpu().detach().numpy().tolist())
                train_outputs.extend(preds.cpu().detach().numpy().tolist())

                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # calculating the evaluation scores for both training and validation data
            train_accuracy,train_loss = print_metrics(train_targets,train_outputs,epoch_loss, 'Training')
            val_accuracy, val_loss = validate(model, valLoader)

            if val_loss < prev_loss:
                print("Val loss decrease from {} to {}:".format(prev_loss,val_loss))
                prev_loss = val_loss
                checkpoint = {"state_dict": model.state_dict()}
                model_saver(le,max_len,checkpoint,filename = self.name)

        return None

    def lstm_loader(self):
        def build_vocab_and_embedding_fasttext(df):
            vocab = Vocabulary(1)
            vocab.build_vocabulary(df['utterance'].tolist())

            matrix_len = len(vocab.itos)
            weights_matrix = np.zeros((matrix_len, 100))
            words_found = 0
            emb_dim = 100

            model = fasttext.load_model("cc.en.100.bin")
            
            for ind,(i, word) in enumerate(vocab.itos.items()):
                if ind > 1:
                    try: 
                        weights_matrix[i] = model[word]
                        words_found += 1
                    except KeyError:
                        weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))

            return vocab,weights_matrix
        
        df,df_train,df_val,num_labels,le = self.data_loader()
        vocab,weights_matrix = build_vocab_and_embedding_fasttext(df_train)

        trainDataset = LSTMDataset(df_train, vocab)
        valDataset = LSTMDataset(df_val, vocab)

        padIdx = trainDataset.vocab.stoi['<PAD>']
    
        trainLoader = DataLoader(
            dataset =trainDataset,
            batch_size= self.config.MODEL_CONFIG['lstm']['train_batch_size'],
            shuffle = True, 
            drop_last=True,
            collate_fn = MyCollate(padIdx = padIdx ),
            
        )

        valLoader = DataLoader(
            dataset = valDataset,
            batch_size= self.config.MODEL_CONFIG['lstm']['val_batch_size'],
            shuffle = True, 
            drop_last=True,
            collate_fn = MyCollate(padIdx = padIdx )
        )

        return trainLoader, valLoader, num_labels, le, weights_matrix, vocab
        
    def lstm_trainer(self):
        epochs = self.config.MODEL_CONFIG['lstm']['epochs']
        device = self.config.MODEL_CONFIG['lstm']['device']

        def validate(model,valLoader):
            eval_acc = 0
            batch_loader = valLoader 
            model.eval()
            with torch.no_grad():
                val_targets = []
                val_outputs = []
                val_losses = []
                for iters, (sent1,sent1_len, label) in enumerate(batch_loader):
                    sent1 = sent1.to(device, dtype=torch.long)
                    sent1_len = sent1_len.to(device, dtype=torch.long)
                    label = label.to(device, dtype=torch.long)
                    out = model(sent1,sent1_len)
                    loss = criterion(out, label)
                    preds = torch.argmax(out, dim=1)
                    val_targets.extend(label.cpu().detach().numpy().tolist())
                    val_outputs.extend(preds.cpu().detach().numpy().tolist())
                    val_losses.append(loss.item())
                eval_loss = sum(val_losses)/len(val_losses)
            return print_metrics(val_targets,val_outputs, eval_loss,'Validation')

        trainLoader, valLoader, num_labels, le, weights_matrix, vocab = self.lstm_loader()
        model = LSTMIntentModel(vocab,weights_matrix,len(le.classes_)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        criterion = nn.CrossEntropyLoss()
        prev_loss = float('inf')
        for epoch in range(1,epochs+1):
            print(f'Epoch: {epoch}')

            # training actual and prediction for each epoch for printing metrics
            train_targets = []
            train_outputs = []
            losses = []
            model.train()
            for iters, (sent1,sent1_len, label) in enumerate(trainLoader):
                sent1 = sent1.to(device, dtype=torch.long)
                sent1_len = sent1_len.to(device, dtype=torch.long)
                label = label.to(device, dtype=torch.long)
                out = model(sent1,sent1_len)
                
                loss = criterion(out, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = torch.argmax(out, dim=1)
                losses.append(loss.item())
                train_targets.extend(label.cpu().detach().numpy().tolist())
                train_outputs.extend(preds.cpu().detach().numpy().tolist())


                
            train_loss = sum(losses)/len(losses)
            train_accuracy,train_loss = print_metrics(train_targets,train_outputs,train_loss, 'Training')
            val_acc,val_loss = validate(model, valLoader)
            if val_loss < prev_loss:
                print("Val loss decrease from {} to {}:".format(prev_loss,val_loss))
                prev_loss = val_loss
                checkpoint = {"state_dict": model.state_dict()}
                model_saver(le,32,checkpoint,filename = self.name)

        return None