import os
from pymongo import MongoClient
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/ecosort')

client = None
db = None

def init_db():
    global client, db
    client = MongoClient(MONGO_URI)
    db = client.ecosort

def get_db():
    if db is None:
        init_db()
    return db

def create_admin():
    users = get_db().users
    if users.count_documents({'username': 'admin'}) == 0:
        users.insert_one({
            'username': 'admin',
            'password_hash': generate_password_hash('password'),
            'created_at': datetime.datetime.utcnow()
        })

def find_user(username):
    users = get_db().users
    user = users.find_one({'username': username})
    return user

def verify_password(username, password):
    user = find_user(username)
    if user and check_password_hash(user['password_hash'], password):
        return True
    return False

def log_prediction(username, waste_type, confidence):
    predictions = get_db().predictions
    predictions.insert_one({
        'username': username,
        'waste_type': waste_type,
        'confidence': confidence,
        'timestamp': datetime.datetime.utcnow()
    })

def get_user_stats(username):
    predictions = get_db().predictions
    count = predictions.count_documents({'username': username})
    return {'predictions': count}

def get_global_stats():
    predictions = get_db().predictions
    users_coll = get_db().users
    return {
        'total_users': users_coll.count_documents({}),
        'total_predictions': predictions.count_documents({})
    }
