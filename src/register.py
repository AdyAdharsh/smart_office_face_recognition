import cv2
from src.detect import detect_face
from src.embed import get_embedding
from src.utils import load_embeddings, save_embeddings

def register_user(db_path='data/embeddings.pkl', detector='cnn'):
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture your face.")
    print(f"Using '{detector}' face detector.")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Register - Press 'c'", frame)
        key = cv2.waitKey(1)

        if key == ord('c'):
            faces = detect_face(frame, detector=detector)  # 👈 Pass the detector
            if not faces:
                print("No face detected.")
                continue
            face = faces[0]
            name = input("Enter your name: ")
            embedding = get_embedding(face)

            db = load_embeddings(db_path)
            db[name] = embedding
            save_embeddings(db_path, db)
            print(f"User '{name}' registered.")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()