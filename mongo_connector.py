import pymongo

def db_connector():
    client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')
    db = client['dashwork']
    intent_collection = db.intent
    return intent_collection