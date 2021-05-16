import pandas as pd
from .utils import explode
from datetime import datetime

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