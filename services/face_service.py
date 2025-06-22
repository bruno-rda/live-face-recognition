import numpy as np
import cv2
from db.repositories import FaceRepository
from services.face_analyzer import FaceAnalyzer
from config import Config

class FaceService:
    def __init__(self):
        self.face_analyzer = FaceAnalyzer()
        self.face_repository = FaceRepository()

    def process_frame_for_prediction(self, frame: np.ndarray):
        image_out = frame.copy()
        faces = self.face_analyzer.compute_embeddings(frame)

        if not faces:
            return image_out

        for face in faces:
            name, similarity, match = self.face_repository.search_face(face.embedding)
            bbox = face.bbox.astype(int)
            coord = (bbox[0], bbox[1])

            if match:
                label = f'{name} ({similarity:.2f})'
                color = (92, 184, 92)
            else:
                if name and similarity > Config.FACE_SEARCH_THRESHOLD: # Show if somewhat similar
                     label = f'Unknown (~{name} {similarity:.2f})'
                     color = (200, 150, 0) # Yellow for uncertain
                else:
                     label = 'Unknown'
                     color = (250, 17, 61) # Red for unknown

            cv2.rectangle(image_out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            font_scale = 1.5
            font_thickness = 2
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(image_out, (coord[0], coord[1] - h - 5), (coord[0] + w, coord[1]), color, -1)
            cv2.putText(
                image_out, label, (coord[0], coord[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
            )
        return image_out

    def process_frame_for_registration_preview(self, frame: np.ndarray):
        if frame is None:
            return frame

        image_out = frame.copy()
        faces = self.face_analyzer.compute_embeddings(frame)

        if faces:
            face = faces[0] # use the first detected face
            bbox = face.bbox.astype(int)
            coord = (bbox[0], bbox[1])
            
            label = 'Face to Register'
            color = (72, 114, 211)

            cv2.rectangle(image_out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)

            font_scale = 1.5
            font_thickness = 2
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(image_out, (coord[0], coord[1] - h - 5), (coord[0] + w, coord[1]), color, -1)
            cv2.putText(
                image_out, label, (coord[0], coord[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness
            )

            if len(faces) > 1:
                for face in faces[1:]:
                    bbox = face.bbox.astype(int)
                    coord = (bbox[0], bbox[1])
                    label = 'Ignored'
                    color = (250, 17, 61)
                    cv2.rectangle(image_out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
        return image_out

    def register_new_face(self, name: str, frame_snapshot: np.ndarray) -> tuple[str, bool]:
        if frame_snapshot is None:
            return 'Error: No image captured. Please ensure camera is active.', False

        if not name or not name.strip():
            return 'Error: Name cannot be empty.', False

        name = name.strip()

        if self.face_repository.is_name_taken(name):
            return f'Error: Name "{name}" is already registered.', False

        faces = self.face_analyzer.compute_embeddings(frame_snapshot)

        if not faces:
            return 'Error: No face detected in the snapshot.', False

        embedding_to_register = faces[0].embedding
        existing_name, similarity, match = self.face_repository.search_face(embedding_to_register)

        if existing_name is not None and match:
            return (
                f'Error: This face seems already registered as "{existing_name}" '
                f'(Similarity: {similarity:.2f}). Cannot re-register.',
                False
            )

        self.face_repository.insert_embedding(name, embedding_to_register)
        print(f'Registered "{name}". Total embeddings: {self.face_repository.get_count()}')
        return f'Success: User "{name}" registered!', True

    def rename_existing_face(self, old_name: str, new_name: str, password: str) -> tuple[str, bool]:
        if not old_name:
            return 'Error: Please select a name to rename.', False
        if not new_name or not new_name.strip():
            return 'Error: New name cannot be empty.', False
        if not self.face_repository.is_name_taken(old_name):
            return f'Error: "{old_name}" not found (might have been deleted).', False
        if self.face_repository.is_name_taken(new_name.strip()):
            return f'Error: Name "{new_name.strip()}" already exists.', False
        if old_name == new_name.strip():
            return 'Info: New name is the same as the old name. No changes made.', False
        if password != Config.ADMIN_PASSWORD:
            return 'Error: Incorrect password.', False

        new_name = new_name.strip()
        success = self.face_repository.update_name(old_name, new_name)

        if success:
            return f'Success: Renamed "{old_name}" to "{new_name}".', True
        else:
            return f'Error: Failed to rename "{old_name}" to "{new_name}".', False

    def delete_existing_face(self, name_to_delete: str, password: str) -> tuple[str, bool]:
        if not name_to_delete:
            return 'Error: Please select a name to delete.', False
        if not self.face_repository.is_name_taken(name_to_delete):
            return f'Error: "{name_to_delete}" not found (might have been deleted).', False
        if password != Config.ADMIN_PASSWORD:
            return 'Error: Incorrect password.', False

        success = self.face_repository.delete_name(name_to_delete)

        if success:
            return f'Success: Deleted "{name_to_delete}".', True
        else:
            return f'Error: Failed to delete "{name_to_delete}".', False

    def get_all_registered_names(self) -> list[str]:
        return self.face_repository.get_all_names()

    def get_registered_count(self) -> int:
        return self.face_repository.get_count()

    def __deepcopy__(self, memo):
        # Since object will be a value in the gr.State dict
        # it must have an implemented deepcopy method
        return self