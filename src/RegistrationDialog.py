# src/RegistrationDialog.py

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np

# Import core registration logic
# NOTE: You'll need to adapt src/register.py to accept the frame and name directly,
# rather than initiating its own webcam capture.
from src import register 

class RegistrationDialog(QDialog):
    # Signal emitted when a new user is successfully saved
    registration_complete = Signal()
    
    def __init__(self, camera_thread, recognition_model, parent=None):
        super().__init__(parent)
        self.setWindowTitle("User Registration")
        self.setFixedSize(650, 600)
        
        self.camera_thread = camera_thread
        self.recognition_model = recognition_model
        self.captured_frame = None # Holds the NumPy array of the captured image
        
        self._setup_ui()
        
        # Connect to the main camera thread to receive live frames for preview
        self.camera_thread.frame_ready.connect(self._update_preview)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Top Section: Inputs and Preview ---
        input_preview_layout = QHBoxLayout()
        
        # A. Preview Window (Displays the live feed)
        self.preview_label = QLabel("Live Camera Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(300, 300)
        input_preview_layout.addWidget(self.preview_label)
        
        # B. Registration Inputs
        input_group = QGroupBox("User Details")
        input_layout = QVBoxLayout(input_group)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter User Name (e.g., John Doe)")
        
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("Enter User ID (e.g., JD101)")
        
        self.capture_button = QPushButton("1. Capture Photo (Press 'c')")
        self.capture_button.clicked.connect(self._capture_photo)
        
        self.save_button = QPushButton("2. Save User & Embedding")
        self.save_button.setEnabled(False) # Disabled until capture is complete
        self.save_button.clicked.connect(self._save_user)
        
        input_layout.addWidget(self.name_input)
        input_layout.addWidget(self.id_input)
        input_layout.addWidget(self.capture_button)
        input_layout.addWidget(self.save_button)
        
        input_preview_layout.addWidget(input_group)
        main_layout.addLayout(input_preview_layout)
        
        # --- Bottom Section: Status ---
        self.status_label = QLabel("Status: Ready to capture.")
        main_layout.addWidget(self.status_label)


    @Slot(QImage)
    def _update_preview(self, image):
        """Receives live QImage from the CameraThread and displays it."""
        pixmap = QPixmap.fromImage(image)
        self.preview_label.setPixmap(
            pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _capture_photo(self):
        """Grabs the current frame from the camera thread for registration."""
        
        # To get the raw numpy array, you must momentarily disconnect the signal, 
        # let the thread emit, and process the raw image data.
        
        # A simpler approach: use the latest QImage passed to the preview, 
        # convert it back to NumPy array (BGRA) and then to BGR for DeepFace/OpenCV.
        
        current_pixmap = self.preview_label.pixmap()
        if current_pixmap is None:
            QMessageBox.critical(self, "Error", "No active camera feed detected.")
            return

        # Convert QPixmap back to QImage
        qimage = current_pixmap.toImage()
        
        # Convert QImage to a NumPy array (BGR format for OpenCV/DeepFace)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        # Ensure correct buffer size and format conversion
        ptr.setsize(height * width * 4) 
        arr = np.array(ptr).reshape((height, width, 4))  # Assuming ARGB32 or similar (4 channels)
        
        # Convert ARGB/RGBA to BGR
        self.captured_frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        
        self.save_button.setEnabled(True)
        self.status_label.setText("Status: Photo captured. Ready to save user.")
        QMessageBox.information(self, "Success", "Photo captured. Enter details and click 'Save User'.")


    def _save_user(self):
        """Saves the captured face embedding and user details."""
        user_name = self.name_input.text().strip()
        user_id = self.id_input.text().strip()
        
        if not user_name or not user_id:
            QMessageBox.warning(self, "Input Error", "Please enter both User Name and User ID.")
            return

        if self.captured_frame is None:
            QMessageBox.critical(self, "Error", "No image captured. Click 'Capture Photo' first.")
            return
            
        try:
            # 1. Adapt your register.py function to be callable with the frame and name.
            # Assuming you modify src/register.py to have a function:
            # save_new_user(frame, user_id, user_name, model_instance)
            
            # Since register.py probably expects the global DeepFace model setup, 
            # we'll pass the frame and rely on the existing logic in a new way.
            
            # For simplicity, let's assume register.py has been modified to:
            # register.save_user_from_frame(frame, user_id, user_name)
            
            # Call the modified registration logic
            success = register.save_user_from_frame(self.captured_frame, user_id, user_name)
            
            if success:
                QMessageBox.information(self, "Success", f"User {user_name} ({user_id}) registered successfully!")
                self.registration_complete.emit()
                self.accept() # Close the dialog on success
            else:
                QMessageBox.critical(self, "Failed", "Registration failed. No single face detected or embedding failed.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred during saving: {e}")
            
    def reject(self):
        # Disconnect the preview update signal when closing the dialog
        self.camera_thread.frame_ready.disconnect(self._update_preview)
        super().reject()

    def _open_registration_dialog(self):
        """Opens the registration dialog."""
        was_running = self.camera_thread.isRunning()
        
        # Stop recognition while the user registers to ensure stability
        if was_running:
            self.stop_recognition()
            self.stop_button.setEnabled(False) # Prevent user from stopping a stopped thread
            
        dialog = RegistrationDialog(self.camera_thread, self.model, parent=self)
        dialog.registration_complete.connect(self.model._load_data) # Reload embeddings after new registration

        dialog.exec() # Run the dialog
        
        # Resume recognition after the dialog closes (if it was running before)
        if was_running:
            self.start_recognition()
            self.start_button.setEnabled(False) # Already starting it
            self.stop_button.setEnabled(True)    