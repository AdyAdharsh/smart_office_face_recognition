# Smart Office Face Recognition System

This is a real-time face recognition system for smart office access control. It supports both modern CNN-based face detection and classical OpenCV-based detection.

##  Features

- CNN-based face detection (MTCNN via DeepFace)
- Classical face detection (Haar Cascades – OpenCV)
- Embedding extraction using FaceNet
- Face registration and persistent storage (Pickle DB)
- Real-time face recognition with cosine similarity
- Modular and extendable design

---

# Project Structure


smart_office_face_recognition/
│
├── run.py                   # Entry point
├── data/                    # Stores embeddings.pkl
├── src/
│   ├── detect.py            # Face detection (CNN and classical)
│   ├── embed.py             # Embedding logic
│   ├── register.py          # Register user
│   ├── recognize.py         # Recognize user
│   └── utils.py             # Load/save helper



# Activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt


# How to Use

# Register a user using CNN-based detection (default)
python run.py --mode register --detector cnn

# Recognize a user using CNN-based detection (default)
python run.py --mode recognize --detector cnn

# Register a user using Haar Cascade (bonus)
python run.py --mode register --detector classical

# Recognize a user using Haar Cascade (bonus)
python run.py --mode recognize --detector classical


Most Importantly: 

Press c to capture during registration. 

Press q to quit the webcam view.



