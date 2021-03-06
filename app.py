import config
from flask import Flask,jsonify
from trainer.trainer import Trainer
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource
from resources.add_intent import AddIntent
from resources.get_intents import GetIntent
from resources.delete_intent import DeleteIntent
from resources.add_utterance import AddUtterance
from resources.get_utterances import GetUtterances
from resources.delete_utterance import DeleteUtterance
from resources.update_utterance import UpdateUtterance
from resources.train_model import TrainModel
from resources.data_upload import DataUpload
from resources.predict import Predict
from mongo_connector import db_connector
from trainer.utils import model_loader,lstm_model_loader,transformer_model_loader

app = Flask(__name__)
api = Api(app)

cors = CORS(app, resources={
    r"/*": {
       "origins": "*"
    }
})

intent_collection = db_connector()
model,encoder,tokenizer,max_len,model_name = transformer_model_loader()
# trainer = Trainer(intent_collection,config.DATA_PATH)
# resp = trainer.data_creator()

api.add_resource(AddIntent, "/add_intent/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(GetIntent, "/get_intents/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(DeleteIntent, "/delete_intent/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(AddUtterance, "/add_utterance/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(GetUtterances, "/get_utterances/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(DeleteUtterance, "/delete_utterance/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(UpdateUtterance, "/update_utterance/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(TrainModel, "/train_model/" ,resource_class_kwargs={'collection': intent_collection,'config':config})
api.add_resource(DataUpload, "/upload_file/" ,resource_class_kwargs={'collection': intent_collection})
api.add_resource(Predict, "/predict/" ,resource_class_kwargs={'collection': intent_collection,'model':model,'encoder':encoder,'tokenizer':tokenizer,'model_name':model_name,'max_len':max_len,'config':config})


# @app.route('/')
# def index():
#     return "YOYO"
#     # return json.dumps(collection.find_one(), sort_keys=True, indent=4, default=json_util.default)

if __name__ == "__main__":
    # app.run(host='192.168.0.105',debug=True)
    app.run(debug=True)
