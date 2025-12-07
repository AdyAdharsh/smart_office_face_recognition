import cv2
import numpy as np
from deepface import DeepFace # <-- REQUIRED IMPORT for save_user_from_frame
from src.detect import detect_face
from src.embed import get_embedding
from src.utils import load_embeddings, save_embeddings

# --- 1. ORIGINAL CLI REGISTRATION FUNCTION (RETAINED) ---

def register_user(db_path='data/embeddings.pkl', detector='cnn'):
    """
    CLI function for registering a user via webcam interaction.
    (Kept for compatibility with the old CLI entry point)
    """
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture your face.")
    print(f"Using '{detector}' face detector.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
            
        cv2.imshow("Register - Press 'c' or 'q'", frame)
        key = cv2.waitKey(1)

        if key == ord('c'):
            # NOTE: Your detect_face function in src/detect.py handles BGR to RGB conversion internally.
            faces = detect_face(frame, detector=detector) 
            
            if not faces:
                print("No face detected.")
                continue
                
            # faces is a list of cropped, resized face images (NumPy arrays)
            face = faces[0]
            
            # --- CLI INPUTS ---
            name = input("Enter your name: ").strip()
            if not name:
                print("Name cannot be empty. Skipping registration.")
                continue
            
            # For CLI, we can simplify storage by using the name as the ID
            user_id = name 
            # ------------------
            
            # Ensure the face image is RGB before embedding (if not guaranteed by detect_face)
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            embedding = get_embedding(face_rgb) # Use the RGB version for FaceNet

            db = load_embeddings(db_path)
            
            # Structure data for GUI compatibility: {ID: {'name': NAME, 'embedding': EMBEDDING}}
            db[user_id] = {'name': name, 'embedding': embedding} 
            
            save_embeddings(db_path, db)
            print(f"User '{name}' (ID: {user_id}) registered successfully.")
            break
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- 2. GUI HELPER FUNCTION (NEW) ---

def save_user_from_frame(frame, user_id, user_name):
    """
    Saves a new user from a given frame, user ID, and user name.
    This function is called by the GUI RegistrationDialog.
    """
    # 1. Detect face in the frame using DeepFace's raw extractor for bounding box/structure
    try:
        # NOTE: DeepFace.extract_faces expects the frame to be BGR or RGB. 
        # Since the frame comes directly from the webcam via the thread, it is BGR.
        detected_faces = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=True)
    except Exception as e:
        print(f"Registration detection failed: {e}")
        return False # No face found or detection failed
        
    # Ensure exactly one face is detected for registration quality
    if len(detected_faces) != 1:
        return False 

    # 2. Extract the cropped face image from the structured result
    # DeepFace's 'face' output is already cropped, resized (160x160),