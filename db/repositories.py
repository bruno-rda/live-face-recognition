import numpy as np
from db.database import MongoDB
from config import Config

class FaceRepository:
    def __init__(self):
        self.collection = MongoDB.get_embeddings_collection()

    def insert_embedding(self, name: str, emb: np.ndarray) -> None:
        self.collection.insert_one({
            'name': name,
            Config.VECTOR_SEARCH_FIELD_PATH: emb.flatten().tolist()
        })

    def is_name_taken(self, name: str) -> bool:
        return self.collection.find_one({'name': name}) is not None

    def update_name(self, old_name: str, new_name: str) -> bool:
        return bool(
            self.collection.update_one(
                {'name': old_name},
                {'$set': {'name': new_name}}
            ).modified_count
        )

    def delete_name(self, name: str) -> bool:
        return bool(
            self.collection.delete_one(
                {'name': name}
            ).deleted_count
        )

    def get_all_names(self) -> list[str]:
        res = self.collection.find({}, {'name': 1, '_id': 0})
        return sorted([x['name'] for x in res])

    def get_count(self) -> int:
        return self.collection.count_documents({})

    def search_face(self, embedding_to_check: np.ndarray) -> tuple[str, float, bool]:
        res = self.collection.aggregate([
            {
                "$vectorSearch": {
                    "index": Config.VECTOR_SEARCH_INDEX_NAME,
                    "path": Config.VECTOR_SEARCH_FIELD_PATH,
                    "queryVector": embedding_to_check.flatten().tolist(),
                    "numCandidates": 10,
                    "limit": 1
                }
            },
            {
                "$project": {
                    "name": 1,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]).to_list()

        if res:
            match_res = res[0]
            return (
                match_res['name'], 
                match_res['score'], 
                match_res['score'] >= Config.FACE_SEARCH_THRESHOLD
            )
        
        return '', 0, False