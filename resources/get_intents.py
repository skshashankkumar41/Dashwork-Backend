import json
import pymongo 
from datetime import datetime
from flask import Flask,request, jsonify
from flask_restful import Api, Resource, reqparse

class GetIntent(Resource):
    def __init__(self,collection):
        self.collection = collection

    def post(self):
        
        intents = self.collection.distinct('intent')
        return {'intents':intents}
    