import json 
import pymongo
from bson import json_util
from flask import Flask,jsonify
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource
from resources.add_intent import AddIntent
from resources.get_intents import GetIntent

app = Flask(__name__)
api = Api(app)

cors = CORS(app, resources={
    r"/*": {
       "origins": "*"
    }
})


client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
db = client['dashwork']
intent_collection = db.intent

api.add_resource(AddIntent, "/add_intent/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(GetIntent, "/get_intents/" ,resource_class_kwargs={'collection': intent_collection})


# @app.route('/')
# def index():
#     return "YOYO"
#     # return json.dumps(collection.find_one(), sort_keys=True, indent=4, default=json_util.default)

if __name__ == "__main__":
    app.run(debug=True)
