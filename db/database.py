from pymongo import MongoClient
from config import Config

class MongoDB:
    _client = None
    _db = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            cls._client = MongoClient(Config.MONGO_URI)
        return cls._client

    @classmethod
    def get_db(cls):
        if cls._db is None:
            cls._db = cls.get_client()[Config.DATABASE_NAME]
        return cls._db

    @classmethod
    def get_embeddings_collection(cls):
        return cls.get_db()[Config.COLLECTION_NAME]