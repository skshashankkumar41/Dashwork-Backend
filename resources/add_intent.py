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
        date_created =datetime.utcnow()
        data = {
            'intent':intent,
            'utterances':[],
            'date_created':date_created
        }

        self.collection.insert_one(data)
        return 'Intent Created'