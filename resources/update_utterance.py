from flask import request
from flask_restful import Resource


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
        
            return {'response':'utterance updated','status':'success'}

        else:
            return {'response':"utterance updation failed: already exist!!","status":"error"}
