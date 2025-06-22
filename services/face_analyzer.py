from insightface.app import FaceAnalysis
import numpy as np
from config import Config

class FaceAnalyzer:
    def __init__(self):
        print('Initializing FaceAnalysis model...')
        self.app = FaceAnalysis(
            name=Config.INSIGHTFACE_MODEL_NAME,
            providers=Config.INSIGHTFACE_PROVIDERS,
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(ctx_id=Config.INSIGHTFACE_CTX_ID)
        print('FaceAnalysis model initialization complete.')

    def compute_embeddings(self, image: np.ndarray):
        if image is None:
            return []
        try:
            faces = self.app.get(image)
            return faces
        except Exception as e:
            print(f'Error computing embeddings: {e}')
            return []