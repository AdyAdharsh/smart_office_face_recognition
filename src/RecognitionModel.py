# src/RecognitionModel.py

import os
import pickle
import cv2
import numpy as np

# Import your existing core logic functions
# NOTE: Ensure these paths are correct relative to where you run the code
from src.detect import detect_face
from src.embed import get_embedding as extract_embedding
from src.utils import load_embeddings as load_known_embeddings, save_embeddings as save_known_embeddings
from deepface import DeepFace

class RecognitionModel:
    """
    Wraps the core detection and recognition logic for use by the GUI.
    Handles persistent data and model loading.
    """
    
    def __init__(self, embedding_path='data/embeddings.pkl'):
        self.embedding_path = embedding_path
        self.known_embeddings, self.known_names = self._load_data()
        self.recognize_threshold = 0.5  # Cosine similarity threshold

    def _load_data(self):
        """Loads known embeddings from the pickle file."""
        if not os.path.exists(self.embedding_path):
            print("INFO: Embeddings file not found. Starting with empty database.")
            return [], []
        
        known_embeddings, known_names = load_known_embeddings(self.embedding_path)
        return known_embeddings, known_names

    # NOTE: The get_detector_function method is REMOVED as detect_face handles the mode internally.

    def process_frame(self, frame: np.ndarray, detector_mode='cnn'):
        """
        Detects faces in a frame, extracts embeddings, and performs recognition.

        Returns: (processed_frame, log_message, recognized_user)
        """
        detected_results = []
        
        try:
            # --- FIX: Map the internal detector mode name to the DeepFace backend name ---
            if detector_mode == 'cnn':
                backend_name = 'mtcnn'  # DeepFace name for CNN-based MTCNN detection
            elif detector_mode == 'classical':
                backend_name = 'opencv' # DeepFace name for Haar Cascade detection
            else:
                backend_name = 'mtcnn'
            # --------------------------------------------------------------------------
            
            # We must use the RGB version here for DeepFace's raw API calls
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Detect faces and get bounding boxes using the mapped backend_name
            detected_results = DeepFace.extract_faces(
                frame_rgb, 
                detector_backend=backend_name, # <-- USING THE MAPPED NAME
                enforce_detection=False
            )
            
        except Exception as e:
            # Handle case where DeepFace/CV fails entirely (e.g., model loading error)
            print(f"DeepFace/CV detection failed in RecognitionModel: {e}")
            return frame, "ERROR: Detection failed", None
        
        recognized_user = None
        log_message = "No face detected"
        
        # Now iterate over the structured results
        for item in detected_results:
            region = item["facial_area"]
            
            # Extract coordinates (x, y, w, h)
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Crop the face image (use the original BGR frame for coordinates, though colors are handled below)
            face_img = frame[y:y+h, x:x+w] 
            face_img = cv2.resize(face_img, (160, 160)) # Resize for FaceNet (160x160)
            
            # Convert to RGB for the embedding step, as FaceNet models usually expect RGB
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # 2. Extract embedding
            embedding = extract_embedding(face_img_rgb) # Use the RGB image for embedding

            if embedding is not None and self.known_embeddings:
                # 3. Recognition (using Cosine Similarity)
                distances = np.dot(self.known_embeddings, embedding) / (
                            np.linalg.norm(self.known_embeddings, axis=1) * np.linalg.norm(embedding)
                        )
                best_match_index = np.argmax(distances)
                best_distance = distances[best_match_index]

                if best_distance > self.recognize_threshold:
                    recognized_user = self.known_names[best_match_index]
                    log_message = f"Access Granted: {recognized_user} ({best_distance:.2f})"
                    color = (0, 255, 0) # Green (BGR)
                else:
                    recognized_user = "Unknown"
                    log_message = f"Access Denied (Confidence: {best_distance:.2f})"
                    color = (0, 0, 255) # Red (BGR)
            else:
                color = (255, 255, 0) # Yellow (BGR) (Face detected, but no DB or embedding failed)
                log_message = "Face detected, no DB or bad embedding"

            # Draw bounding box and text (using BGR frame)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if recognized_user:
                cv2.putText(frame, recognized_user, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # If no faces were detected, log_message remains "No face detected"
        # If detection was successful, log_message holds the last recognition result.
        
        return frame, log_message, recognized_user