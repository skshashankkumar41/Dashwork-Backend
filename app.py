from flask import Flask,jsonify
import pymongo
from bson import json_util
import json 
client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')

mydb = client['dw']
collection = mydb.collection

app = Flask(__name__)

@app.route('/')
def index():
    return json.dumps(collection.find_one(), sort_keys=True, indent=4, default=json_util.default)

if __name__ == "__main__":
    app.run(debug=True)
