import json
import pymongo 
from datetime import datetime
from flask import Flask,request, jsonify
from flask_restful import Api, Resource, reqparse

class GetUtterances(Resource):
    def __init__(self,collection):
        self.collection = collection

    def post(self):
        intent = request.json['intent_name']
        utterances = list(self.collection.find({'intent':intent}))[0]['utterances']
        return {'utterances':utterances}
    