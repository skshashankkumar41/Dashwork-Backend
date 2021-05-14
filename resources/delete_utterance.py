from flask import request
from flask_restful import Resource


class DeleteUtterance(Resource):
    def __init__(self,collection):
        self.collection = collection

    def post(self):
        intent = request.json['intent_name']
        utterance = request.json['utterance']
        
        self.collection.update_one(
                            { "intent": intent},
                            { "$pull": { "utterances": utterance }})
        
        return {'response':'utterance deleted'}
