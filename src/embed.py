from deepface import DeepFace
import cv2

def get_embedding(face_img):
    embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)[0]["embedding"]
    return embedding