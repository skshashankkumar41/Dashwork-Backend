import json
from bson import json_util
from flask import Flask,jsonify
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource
from resources.add_intent import AddIntent
from resources.get_intents import GetIntent
from resources.delete_intent import DeleteIntent
from resources.add_utterance import AddUtterance
from resources.get_utterances import GetUtterances
from mongo_connector import db_connector

app = Flask(__name__)
api = Api(app)

cors = CORS(app, resources={
    r"/*": {
       "origins": "*"
    }
})

intent_collection = db_connector()

api.add_resource(AddIntent, "/add_intent/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(GetIntent, "/get_intents/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(DeleteIntent, "/delete_intent/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(AddUtterance, "/add_utterance/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(GetUtterances, "/get_utterances/" ,resource_class_kwargs={'collection': intent_collection})


# @app.route('/')
# def index():
#     return "YOYO"
#     # return json.dumps(collection.find_one(), sort_keys=True, indent=4, default=json_util.default)

if __name__ == "__main__":
    # app.run(host='192.168.0.105',debug=True)
    app.run(debug=True)
