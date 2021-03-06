from flask import request
from flask_restful import Resource
from datetime import datetime

class AddIntent(Resource):
    def __init__(self,collection):
        self.collection = collection

    def get(self):
        return "YOYOYOYOYO"

    def post(self):
        intent = request.json['intent_name']
        # date_created = datetime.timestamp(datetime.utcnow())
        if self.checkExist(intent): return {'response':'intent already exist!!','status':'warning'}
        
        date_created =datetime.utcnow()
        data = {
            'intent':intent,
            'utterances':[],
            'date_created':date_created
        }

        self.collection.insert_one(data)
        
        return {'response':'intent created','status':'success'}
    
    def checkExist(self,intent_name):
        intent_data = list(self.collection.find({'intent':intent_name}))
        
        return True if len(intent_data) == 1 else False