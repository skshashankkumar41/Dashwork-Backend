import os
import torch
import pandas as pd
import numpy as np
from .utils import explode
from .bert_dataset import BertDataset
from .bert_model import BertModel
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self,intent_collection,config):
        self.intent_collection = intent_collection
        self.config = config
        self.data_path = config.DATA_PATH

    def data_creator(self):
        file_name = '/training_{}.xlsx'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
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
        files = os.listdir(self.data_path)
        latest_file_name = '/training_{}.xlsx'.format(max([file.split('_')[1].split('.')[0] for file in files]))
        return latest_file_name

    def data_loader(self):
        training_file = self.data_path + self.get_latest_file()
        df = pd.read_excel(training_file)
        df = df.sample(frac=1).reset_index(drop=True)

        le = LabelEncoder()
        df['intent'] = le.fit_transform(df['intent'])
        num_labels = len(le.classes_)

        df_train, df_val = train_test_split(df, test_size=.10,stratify = df.intent)
        df_train.reset_index(drop=True, inplace=True)
        df_val.reset_index(drop=True, inplace=True)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        token_lens = []
        for txt in df.utterance:
            tokens = tokenizer.encode(str(txt), max_length=512,truncation=True)
            token_lens.append(len(tokens))
        max_len = np.percentile(token_lens,99)

        return df,df_train,df_val,num_labels,max_len,le

    def bert_loader(self):
        df,df_train,df_val,num_labels,max_len,le = self.data_loader()
        trainDataset = BertDataset(df_train,self.config['tokenizer'], max_len)
        valDataset = BertDataset(df_val, self.config['tokenizer'], max_len)

        trainLoader = DataLoader(
            dataset=trainDataset,
            batch_size=self.config['train_batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        valLoader = DataLoader(
            dataset=valDataset,
            batch_size=self.config['val_batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        return trainLoader, valLoader, num_labels, le

    def bert_trainer(self):
        epochs = self.config['epochs']
        device = self.config['device']
        trainLoader, valLoader, num_labels, le = self.bert_loader
        model = BertModel(num_labels,self.config['model_config']).to(device) 
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['lr'])
        prev_loss = float('inf')
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
                # save_checkpoint(checkpoint,filename = '/content/drive/My Drive/Ori/sentiment/sent_model_25k_v1.pth')

    


        
