from flask import request
from flask_restful import Resource

class DeleteIntent(Resource):
    def __init__(self,collection):
        self.collection = collection

    def post(self):
        intent = request.json['intent_name']
        
        self.collection.delete_one({'intent':intent})
        
        return {'response':'intent deleted'}
