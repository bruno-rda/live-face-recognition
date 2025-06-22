import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # MongoDB configuration
    MONGO_URI = os.getenv('MONGO_URI')
    DATABASE_NAME = os.getenv('DATABASE_NAME')
    COLLECTION_NAME = os.getenv('COLLECTION_NAME')
    VECTOR_SEARCH_INDEX_NAME = os.getenv('VECTOR_SEARCH_INDEX_NAME')
    VECTOR_SEARCH_FIELD_PATH = os.getenv('VECTOR_SEARCH_FIELD_PATH')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')

    # InsightFace configuration
    INSIGHTFACE_MODEL_NAME = 'buffalo_l' # Or 'buffalo_s'
    INSIGHTFACE_PROVIDERS = ['CPUExecutionProvider'] # Or ['CUDAExecutionProvider']
    INSIGHTFACE_CTX_ID = -1 # 0 for GPU, -1 for CPU
    FACE_SEARCH_THRESHOLD = 0.75
    
    # Gradio streaming interval
    STREAM_INTERVAL = 0.1

# Validate required environment variables
def validate_required_env_vars():
    required_vars = [
        'MONGO_URI',
        'DATABASE_NAME',
        'COLLECTION_NAME',
        'VECTOR_SEARCH_INDEX_NAME',
        'VECTOR_SEARCH_FIELD_PATH',
        'ADMIN_PASSWORD',
    ]
    
    for var in required_vars:
        if getattr(Config, var) is None:
            raise ValueError(f'Missing required environment variable: {var}')

validate_required_env_vars()