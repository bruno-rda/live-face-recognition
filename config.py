import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_URI = os.getenv('MONGO_URI')
    DATABASE_NAME = os.getenv('DATABASE_NAME')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')
    INSIGHTFACE_MODEL_NAME = 'buffalo_l' # Or 'buffalo_s'
    INSIGHTFACE_PROVIDERS = ['CPUExecutionProvider'] # Or ['CUDAExecutionProvider']
    INSIGHTFACE_CTX_ID = -1 # 0 for GPU, -1 for CPU
    FACE_SEARCH_THRESHOLD = 0.75
    
    # Gradio streaming interval
    STREAM_INTERVAL = 0.1