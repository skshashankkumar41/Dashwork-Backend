import pymongo
from flask import Flask,jsonify
client = pymongo.MongoClient('mongodb://127.0.0.1:27017/')

mydb = client['dw']
collection = mydb.collection

print(type(jsonify(collection.find_one())))

