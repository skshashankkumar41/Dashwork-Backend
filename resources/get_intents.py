from flask_restful import  Resource

class GetIntent(Resource):
    def __init__(self,collection):
        self.collection = collection

    def get(self):
        
        intents = self.collection.distinct('intent')
        return {'intents':intents}
    