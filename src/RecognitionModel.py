# src/RecognitionModel.py

import os
import pickle
import cv2
import numpy as np
from deepface import DeepFace 

# Import your existing core logic functions
from src.detect import detect_face
from src.embed import get_embedding as extract_embedding
from src.utils import load_embeddings as load_known_embeddings, save_embeddings as save_known_embeddings

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
        """Loads known embeddings and names from the pickle file."""
        if not os.path.exists(self.embedding_path):
            print("INFO: Embeddings file not found. Starting with empty database.")
            return [], []
        
        known_embeddings, known_names = load_known_embeddings(self.embedding_path)
        return known_embeddings, known_names


    def process_frame(self, frame: np.ndarray, detector_mode='cnn'):
        """
        Detects faces in a frame, extracts embeddings, and performs recognition.

        Returns: (processed_frame, log_msg, recognized_user, log_data)
        """
        detected_results = []
        
        # Initialize log variables
        recognized_user = None
        log_message = "No face detected"
        log_data = None # Will store {'user_id': ..., 'status': ..., 'confidence': ...}
        
        try:
            # --- Map the internal detector mode name to the DeepFace backend name ---
            if detector_mode == 'cnn':
                backend_name = 'mtcnn'
            elif detector_mode == 'classical':
                backend_name = 'opencv'
            else:
                backend_name = 'mtcnn'
            # --------------------------------------------------------------------------
            
            # Convert to RGB for DeepFace's raw API calls
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. Detect faces and get bounding boxes
            detected_results = DeepFace.extract_faces(
                frame_rgb, 
                detector_backend=backend_name, 
                enforce_detection=False
            )
            
        except Exception as e:
            # Handle case where DeepFace/CV fails entirely
            # log_data is None in this error case
            print(f"DeepFace/CV detection failed in RecognitionModel: {e}")
            return frame, "ERROR: Detection failed", None, None
        
        
        # Now iterate over the structured results
        for item in detected_results:
            region = item["facial_area"]
            
            # Extract coordinates (x, y, w, h)
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Crop the face image (use the original BGR frame)
            face_img = frame[y:y+h, x:x+w] 
            face_img = cv2.resize(face_img, (160, 160)) 
            
            # Convert to RGB for the embedding step
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # 2. Extract embedding
            embedding = extract_embedding(face_img_rgb) 

            best_distance = 0.0 # Default confidence
            current_user_id = "Unknown"
            current_status = "Denied"
            color = (0, 0, 255) # Default Red (BGR)

            if embedding is not None and self.known_embeddings:
                # 3. Recognition (using Cosine Similarity)
                distances = np.dot(self.known_embeddings, embedding) / (
                            np.linalg.norm(self.known_embeddings, axis=1) * np.linalg.norm(embedding)
                        )
                best_match_index = np.argmax(distances)
                best_distance = float(distances[best_match_index])

                if best_distance > self.recognize_threshold:
                    current_user_id = self.known_names[best_match_index]
                    current_status = "Granted"
                    log_message = f"Access Granted: {current_user_id} ({best_distance:.2f})"
                    color = (0, 255, 0) # Green
                else:
                    current_user_id = "Unknown"
                    current_status = "Denied"
                    log_message = f"Access Denied (Confidence: {best_distance:.2f})"
                    # Color is already Red
            else:
                log_message = "Face detected, no DB or embedding failed"
                color = (255, 255, 0) # Yellow
            
            # Assign final recognized user name (the last one detected/recognized in the frame)
            recognized_user = current_user_id
            
            # --- NEW: Package the structured data for logging ---
            # This is created for EVERY face detected in the frame.
            log_data = {
                'user_id': current_user_id,
                'status': current_status,
                'confidence': best_distance
            }
            # ----------------------------------------------------


            # Draw bounding box and text (using BGR frame)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if recognized_user:
                cv2.putText(frame, recognized_user, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # If no faces were detected, log_message and log_data remain their initialized values
        
        # --- MODIFICATION: RETURN 4 VALUES ---
        return frame, log_message, recognized_user, log_data
        # -------------------------------------