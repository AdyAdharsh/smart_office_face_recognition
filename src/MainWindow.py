# src/MainWindow.py

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QTextEdit, QSizePolicy
)
from src.CameraThread import CameraThread
from src.RecognitionModel import RecognitionModel

# Import local modules
from .CameraThread import CameraThread
from .RecognitionModel import RecognitionModel

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Office Face Recognition System")
        self.setGeometry(100, 100, 1000, 700)
        
        # 1. Initialize the Recognition Model (Backend)
        self.model = RecognitionModel()

        # 2. Initialize the Camera Thread (Engine)
        self.camera_thread = CameraThread(model=self.model)
        self.camera_thread.frame_ready.connect(self.update_video_feed)
        self.camera_thread.log_message.connect(self.update_log)

        self._setup_ui()
        
    def _setup_ui(self):
        # --- Central Widget & Main Layout ---
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # --- Left Panel: Video Feed ---
        video_panel = QVBoxLayout()
        
        self.video_label = QLabel("Webcam Feed will appear here...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setFixedSize(640, 480) # Fixed size for the video display
        video_panel.addWidget(self.video_label)
        
        # --- Controls ---
        controls_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Recognition")
        self.start_button.clicked.connect(self.start_recognition)
        
        self.stop_button = QPushButton("Stop Recognition")
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False) # Initially disabled

        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["cnn", "classical"])
        self.detector_combo.currentTextChanged.connect(self.change_detector)
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(QLabel("Detector:"))
        controls_layout.addWidget(self.detector_combo)
        
        video_panel.addLayout(controls_layout)
        main_layout.addLayout(video_panel)

        # --- Right Panel: Log/Registration ---
        right_panel = QVBoxLayout()
        
        log_label = QLabel("--- Access Log ---")
        right_panel.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_panel.addWidget(self.log_text)
        
        # Placeholder for registration functionality (to be fully built later)
        register_label = QLabel("--- Registration (Placeholder) ---")
        self.register_button = QPushButton("Go to Registration Screen")
        
        right_panel.addWidget(register_label)
        right_panel.addWidget(self.register_button)

        main_layout.addLayout(right_panel)

    @Slot(QImage)
    def update_video_feed(self, image):
        """Receives QImage from the thread and displays it in the QLabel."""
        # Convert QImage to QPixmap and scale to fit the label
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    @Slot(str)
    def update_log(self, message):
        """Receives log message from the thread and updates the text area."""
        self.log_text.append(message)
        
    def start_recognition(self):
        """Starts the camera thread."""
        if not self.camera_thread.isRunning():
            self.camera_thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.update_log("INFO: Recognition started.")

    def stop_recognition(self):
        """Stops the camera thread."""
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.update_log("INFO: Recognition stopped.")

    @Slot(str)
    def change_detector(self, text):
        """Passes the new detector mode to the camera thread."""
        self.camera_thread.set_detector_mode(text)
        self.update_log(f"INFO: Detector mode switched to '{text}'.")

# --- Main Entry Point ---

def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

# To run the GUI from your main project file, you would update run.py to call this function.