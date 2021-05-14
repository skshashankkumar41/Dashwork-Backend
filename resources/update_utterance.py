import json
import pymongo 
from datetime import datetime
from flask import Flask,request, jsonify
from flask_restful import Api, Resource, reqparse


class UpdateUtterance(Resource):
    def __init__(self,collection):
        self.collection = collection

    def post(self):
        intent = request.json['intent_name']
        utterance = request.json['utterance']
        updated_utterance = request.json['updated_utterance']

        if updated_utterance not in list(self.collection.find({'intent':intent}))[0]['utterances']:
            self.collection.update_one( 
                { "intent": intent,"utterances":utterance},
                { "$set": { "utterances.$" : updated_utterance } })
        
            return {'response':'utterance updated'}

        else:
            return {'response':"utterance updation failed:already exist!!"}
