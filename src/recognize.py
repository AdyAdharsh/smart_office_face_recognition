import cv2
from src.detect import detect_face
from src.embed import get_embedding
from src.utils import load_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recognize_user(db_path='data/embeddings.pkl', detector='cnn'):
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture and recognize.")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Recognize - Press 'c'", frame)
        key = cv2.waitKey(1)

        if key == ord('c'):
            faces = detect_face(frame, detector=detector)
            if not faces:
                print("No face detected.")
                continue

            face = faces[0]
            embedding = get_embedding(face)
            db = load_embeddings(db_path)

            if not db:
                print("No registered users.")
                continue

            names = list(db.keys())
            embeddings = np.array(list(db.values()))

            similarities = cosine_similarity([embedding], embeddings)[0]
            best_match_idx = np.argmax(similarities)
            best_score = similarities[best_match_idx]

            if best_score > 0.7:  # Threshold for recognition
                print(f"Hello, {names[best_match_idx]}!")
            else:
                print("Face not recognized.")

            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()