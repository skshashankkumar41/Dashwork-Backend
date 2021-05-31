from flask import request
from flask_restful import Resource
import pandas as pd
from datetime import datetime

class DataUpload(Resource):
    def __init__(self,collection):
        self.collection = collection

    def post(self):
        file = request.files['file']
        intents = self.collection.distinct('intent')
        try:
            df = pd.read_csv(file)
        except:
            df = pd.read_excel(file)
        
        df = df.groupby(['intent'])['utterance'].apply(list).reset_index()

        for i in range(len(df)):
            intent = df.iloc[i,0]
            utterances = df.iloc[i,1]
            if intent in intents:
                self.collection.update_one({'intent':intent},{"$addToSet": {"utterances": {"$each":utterances}}})
            else:
                date_created =datetime.utcnow()
                data = {
                    'intent':intent,
                    'utterances':list(set(utterances)),
                    'date_created':date_created
                }

                self.collection.insert_one(data)
                # self.collection.update_one({'intent':intent},{"$addToSet": {"utterances": {"$each":utterances}}})

        return {'response':"Data Updated"}
    