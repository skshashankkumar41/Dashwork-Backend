from flask import request
from flask_restful import Resource
from trainer.trainer import Trainer


class TrainModel(Resource):
    def __init__(self,collection,config):
        self.collection = collection
        self.trainer =Trainer(collection,config)

    def get(self):
        self.trainer.lstm_trainer()
        
        return {'response':'model trained'}
