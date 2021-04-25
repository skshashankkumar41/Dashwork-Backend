import json
import pymongo 
from datetime import datetime
from flask import Flask,request, jsonify
from flask_restful import Api, Resource, reqparse


class AddIntent(Resource):
    def __init__(self,collection):
        self.collection = collection

    def get(self):
        return "YOYOYOYOYO"

    def post(self):
        intent = request.json['intent_name']
        # date_created = datetime.timestamp(datetime.utcnow())
        if self.checkExist(intent): return {'response':'intent already exist!!'}
        
        date_created =datetime.utcnow()
        data = {
            'intent':intent,
            'utterances':[],
            'date_created':date_created
        }

        self.collection.insert_one(data)
        
        return {'response':'intent created'}
    
    def checkExist(self,intent_name):
        intent_data = list(self.collection.find({'intent':intent_name}))
        
        return True if len(intent_data) == 1 else False