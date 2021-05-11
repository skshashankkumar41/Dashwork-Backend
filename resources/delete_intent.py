import json
import pymongo 
from datetime import datetime
from flask import Flask,request, jsonify
from flask_restful import Api, Resource, reqparse


class DeleteIntent(Resource):
    def __init__(self,collection):
        self.collection = collection

    def post(self):
        intent = request.json['intent_name']
        
        self.collection.delete_one({'intent':intent})
        
        return {'response':'intent deleted'}
