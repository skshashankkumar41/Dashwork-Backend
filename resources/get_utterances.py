from flask import request
from flask_restful import Resource

class GetUtterances(Resource):
    def __init__(self,collection):
        self.collection = collection

    def post(self):
        intent = request.json['intent_name']
        utterances = list(self.collection.find({'intent':intent}))[0]['utterances']
        return {'utterances':utterances}
    