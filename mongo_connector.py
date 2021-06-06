import pymongo
import conf

def db_connector():
    client = pymongo.MongoClient('mongodb+srv://{}:{}@cluster0.4h9nz.mongodb.net/dashwork?retryWrites=true&w=majority'.format(conf.mongo_username,conf.mongo_password))
    db = client['dashwork']
    intent_collection = db.intent
    return intent_collection