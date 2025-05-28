import cv2
from deepface import DeepFace

# Load OpenCV Haar Cascade
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(frame, detector='cnn'):
    faces = []

    if detector == 'classical':
        print("Using classical OpenCV Haar Cascade for face detection")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in results:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            faces.append(face)

    else:
        print("Using CNN-based detection (DeepFace - MTCNN backend)")
        try:
            detected_faces = DeepFace.extract_faces(frame, detector_backend='mtcnn', enforce_detection=False)
            for item in detected_faces:
                face = item["face"]
                face = cv2.resize(face, (160, 160))
                faces.append(face)
        except Exception as e:
            print("DeepFace detection failed:", e)

    return faces