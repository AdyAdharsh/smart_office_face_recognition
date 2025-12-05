# Smart Office Face Recognition System

A comprehensive real-time face recognition solution for smart office access control. This system leverages both modern (CNN-based) and classical (OpenCV Haar Cascade) methods to ensure flexibility, accuracy, and speed. 

---

## 📚 Table of Contents

| Section                      | Description                                  |
|------------------------------|----------------------------------------------|
| [Features](#features)      | System highlights and key functionalities    |
| [Project Structure](#project-structure) | Overview of project files and organization   |
| [Prerequisites](#prerequisites) | Requirements before installing               |
| [Installation](#installation) | Step-by-step guide to set up the environment |
| [Configuration](#configuration) | Storage and system customization info         |
| [Usage](#usage)           | How to register and recognize users          |
| [Controls](#controls)      | Key controls for camera operation            |
| [Troubleshooting](#troubleshooting) | Tips for resolving common issues             |
| [Contribution](#contribution) | Guidelines for contributing to the project    |
| [License](#license)       | Licensing details                            |
| [Acknowledgments](#acknowledgments) | Credits to supporting projects               |
| [Contact](#contact)       | Getting help or reporting bugs               |


---

## Features

- **CNN-Based Detection**  
  Uses MTCNN (via DeepFace) for high-precision face detection.

- **Classical Detection**  
  Employs OpenCV’s Haar Cascade classifier for lightweight, traditional face detection.

- **Face Embedding Extraction**  
  Applies FaceNet to convert faces into numerical embeddings, which are stored and compared during recognition.

- **User Registration & Persistent Storage**  
  Allows new users to register their faces, which are stored in a Pickle database for future recognition.

- **Real-Time Recognition**  
  Authenticates users by comparing real-time webcam input against stored embeddings using cosine similarity.

- **Modular, Extendable Design**  
  Clean, maintainable code layout allows easy customization and addition of new features.

---

## Project Structure

```
smart_office_face_recognition/
│
├── run.py              # Main entry point for CLI operations
├── data/               # Persistent data (embeddings.pkl etc.)
├── requirements.txt    # Python package dependencies
├── src/
│   ├── detect.py       # Face detection logic (CNN & classical)
│   ├── embed.py        # Embedding extraction logic
│   ├── register.py     # User registration functionality
│   ├── recognize.py    # User recognition functionality
│   └── utils.py        # Helper functions for data handling
```

---

## Prerequisites

Before installation, ensure you have the following:

- **Python 3.7+**  
  Check your version:  
  ```bash
  python3 --version
  ```

- **pip** (Python package installer)  
  Upgrade pip if necessary:  
  ```bash
  pip install --upgrade pip
  ```

- **Operating System Compatibility**  
  Works on Linux, macOS, and Windows. On Windows, verify webcam drivers and Python installation.

- **Webcam**  
  A functional webcam for capturing live images during registration and recognition.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AdyAdharsh/smart_office_face_recognition.git
cd smart_office_face_recognition
```

### 2. Set Up the Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows (Command Prompt)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you encounter issues with package installation (especially for OpenCV), refer to [Troubleshooting](#troubleshooting) below.

---

## Configuration

All registered face embeddings and user data are stored in `data/embeddings.pkl` by default. You may change the storage directory or filename in the corresponding utility functions within `src/utils.py` or as arguments in `run.py`.

---

## Usage

### Register a User

1. **Using CNN-Based Detector (Recommended & Default):**
   ```bash
   python run.py --mode register --detector cnn
   ```

2. **Using Haar Cascade Detector (Classical):**
   ```bash
   python run.py --mode register --detector classical
   ```

Follow the on-screen instructions; the webcam feed will open. Position your face within the frame and:

- Press **`c`** to capture your registration photo.
- Enter your **username** or identifier when prompted.

### Recognize a User

1. **Using CNN-Based Detector (Recommended & Default):**
   ```bash
   python run.py --mode recognize --detector cnn
   ```

2. **Using Haar Cascade Detector (Classical):**
   ```bash
   python run.py --mode recognize --detector classical
   ```

The webcam will identify registered users in real time. Recognition feedback is displayed directly on the feed and/or console output.

---

## Controls

- **`c`**: Capture image for registration.
- **`q`**: Quit or close the webcam window.

---

## Troubleshooting

- **OpenCV Installation Issues:**  
  Some environments (especially Windows) may require installing extra packages:
  ```bash
  pip install opencv-python opencv-contrib-python
  ```

- **Webcam Not Detected:**  
  Ensure the webcam is connected and not in use by another application. Try a different port or restart your computer.

- **Dependencies Not Installing:**  
  Upgrade pip; check your Python version or consider creating a fresh virtual environment.

- **Permission Errors:**  
  Run your terminal or IDE as administrator (Windows) or use `sudo` where appropriate (Linux/macOS).

---

## Contribution

We welcome community contributions! Please:

- Fork the repository and create a feature branch.
- Submit a pull request with a clear description of changes.
- Review [issues](https://github.com/AdyAdharsh/smart_office_face_recognition/issues) for existing bugs, feature requests, or enhancements.
- Follow conventional commit messages and document your code.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

This work is built upon:
- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV](https://opencv.org/)
- [FaceNet](https://github.com/davidsandberg/facenet)

Special thanks to all contributors and open-source community members.

---

## Contact

For questions, bug reports, or feature requests:

- **GitHub Issues:** [Open an Issue](https://github.com/AdyAdharsh/smart_office_face_recognition/issues)
- **Email:** adharshnandy@gmail.com (replace with your actual contact email)
