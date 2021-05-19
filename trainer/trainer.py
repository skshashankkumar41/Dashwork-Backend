import os
import pandas as pd
import numpy as np
from .utils import explode
from datetime import datetime
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self,intent_collection,data_path):
        self.intent_collection = intent_collection
        self.data_path = data_path

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


        
