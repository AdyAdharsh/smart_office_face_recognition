# src/MainWindow.py

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QComboBox, QTextEdit, QSizePolicy
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap, QImage

# Import local modules
from src.CameraThread import CameraThread
from src.RecognitionModel import RecognitionModel
from src.LogManager import LogManager 
from src.RegistrationDialog import RegistrationDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Office Face Recognition System")
        self.setGeometry(100, 100, 1000, 700)
        
        # 1. Initialize the Log Manager (DB connection)
        self.log_manager = LogManager()

        # 2. Initialize the Recognition Model (Backend)
        self.model = RecognitionModel()

        # 3. Initialize the Camera Thread (Engine)
        self.camera_thread = CameraThread(model=self.model)
        
        # Connect signals
        # NOTE: This line requires update_video_feed to be defined!
        self.camera_thread.frame_ready.connect(self.update_video_feed) 
        self.camera_thread.log_message.connect(self.update_live_console_log) # For internal messages/errors
        
        # --- NEW CONNECTION: Connect structured log event to the database handler ---
        self.camera_thread.log_event.connect(self.handle_log_event)
        # -------------------------------------------------------------------------

        self._setup_ui()
        
        # Start the thread and populate the log display immediately
        self.start_recognition() # Automatically start the camera feed
        self.refresh_log_display() # Load any existing logs on startup
        
    def _setup_ui(self):
        # --- Central Widget & Main Layout ---
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # --- Left Panel: Video Feed and Controls ---
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
        self.start_button.setEnabled(False) # Already started in __init__
        
        self.stop_button = QPushButton("Stop Recognition")
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(True) 

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
        
        log_label = QLabel("--- Access Log (DB) ---")
        right_panel.addWidget(log_label)
        
        # Log Text Area (Now displays DB contents)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontPointSize(9) # Smaller font for more entries
        right_panel.addWidget(self.log_text)
        
        # Placeholder for registration functionality 
        register_label = QLabel("--- Registration ---")
        self.register_button = QPushButton("Go to Registration Screen")
        self.register_button.clicked.connect(self._open_registration_dialog) # Connects to the new method
        right_panel.addWidget(register_label)
        right_panel.addWidget(self.register_button)

        main_layout.addLayout(right_panel)

    @Slot(QImage)
    def update_video_feed(self, image): # <--- DEFINITION RESTORED (FIX #1)
        """Receives QImage from the thread and displays it in the QLabel."""
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    @Slot(str)
    def update_live_console_log(self, message):
        """Receives simple messages/errors from the thread and adds them to the GUI log."""
        # Use appendHtml to make system messages distinct
        self.log_text.append(f"<span style='color: blue;'>{message}</span>")

    @Slot(str, str, float)
    def handle_log_event(self, user_id, status, confidence):
        """
        Receives structured event from the thread and logs it to the DB.
        This runs every time a face is detected/recognized.
        """
        # 1. Write the event to SQLite
        self.log_manager.log_access_event(user_id, status, confidence)
        
        # 2. Refresh the visible log display (calls DB again)
        self.refresh_log_display()
        
    def refresh_log_display(self):
        """Fetches the 20 most recent logs from DB and updates the QTextEdit."""
        recent_logs = self.log_manager.get_recent_logs(limit=20)
        self.log_text.clear()
        
        # Display logs in reverse order (newest at the bottom of the log)
        for ts, status, user_id, confidence in recent_logs:
            # Format display string
            display_user = user_id if user_id and user_id != 'Unknown' else "N/A"
            display_conf = f"{confidence:.2f}" if confidence is not None else "0.00"
            color = "green" if status == "Granted" else "red"
            
            line = f"<span style='color:{color};'>{ts} - **{status.upper()}**: {display_user} (Conf: {display_conf})</span>"
            self.log_text.append(line)
            
    def start_recognition(self):
        """Starts the camera thread."""
        if not self.camera_thread.isRunning():
            self.camera_thread.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.update_live_console_log("INFO: Recognition started.")

    def stop_recognition(self):
        """Stops the camera thread."""
        if self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.update_live_console_log("INFO: Recognition stopped.")

    @Slot(str)
    def change_detector(self, text):
        """Passes the new detector mode to the camera thread."""
        self.camera_thread.set_detector_mode(text)
        self.update_live_console_log(f"INFO: Detector mode switched to '{text}'.")

    def _open_registration_dialog(self): # <--- DEFINITION ADDED (FIX #2)
        """Opens the registration dialog, pausing recognition if necessary."""
        was_running = self.camera_thread.isRunning()
        
        # Stop recognition while the user registers to ensure stability
        if was_running:
            self.stop_recognition()
            self.stop_button.setEnabled(False) 
            
        dialog = RegistrationDialog(self.camera_thread, self.model, parent=self)
        
        # Reload embeddings after new registration so the recognition model instantly knows the new user
        dialog.registration_complete.connect(self.model._load_data) 

        dialog.exec() # Run the dialog modal
        
        # Resume recognition after the dialog closes (if it was running before)
        if was_running:
            self.start_recognition()
            self.start_button.setEnabled(False) 
            self.stop_button.setEnabled(True)

# --- Main Entry Point ---

def run_gui():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    run_gui()