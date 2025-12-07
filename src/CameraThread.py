# src/CameraThread.py

import cv2
import time
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QImage
from src.RecognitionModel import RecognitionModel

class CameraThread(QThread):
    # Signals must be defined on the class level
    frame_ready = Signal(QImage)
    log_message = Signal(str)
    
    # --- NEW SIGNAL FOR STRUCTURED DB LOGGING ---
    # Emits: user_id (str), status (str), confidence (float)
    log_event = Signal(str, str, float) 
    # -------------------------------------------
    
    def __init__(self, model: RecognitionModel, camera_index=0, parent=None):
        super().__init__(parent)
        self.model = model
        self.camera_index = camera_index
        self._is_running = True
        self.detector_mode = 'cnn' # Default detection mode

    def run(self):
        """The main loop that runs in the separate thread."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.log_message.emit("ERROR: Cannot open webcam!")
            self._is_running = False
            return

        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                self.log_message.emit("ERROR: Failed to read frame from camera.")
                break

            # --- MODIFICATION: RecognitionModel now returns structured log data ---
            # It now returns 4 values: processed_frame, log_msg, recognized_user, log_data
            processed_frame, log_msg, recognized_user, log_data = self.model.process_frame(
                frame, 
                detector_mode=self.detector_mode
            )
            # --------------------------------------------------------------------

            # Convert OpenCV BGR image (NumPy array) to QImage (RGB format)
            h, w, ch = processed_frame.shape
            bytes_per_line = ch * w
            
            # The QImage MUST be created from an RGB image
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Emit the signals back to the main thread
            self.frame_ready.emit(qt_image)
            
            # If the RecognitionModel returned structured log data, emit both logs
            if log_data:
                # 1. Emit the verbose log message for the GUI console
                self.log_message.emit(log_msg)
                
                # 2. Emit the structured data for the Database
                user_id = log_data.get('user_id', 'Unknown')
                status = log_data.get('status', 'Denied')
                confidence = log_data.get('confidence', 0.0)
                
                self.log_event.emit(user_id, status, confidence)
            
            # Control the framerate (adjust as needed)
            time.sleep(0.01) 
            
        cap.release()
        
    def stop(self):
        """Gracefully stops the thread."""
        self._is_running = False
        self.wait() # Wait for the thread to finish execution
        
    @Slot(str)
    def set_detector_mode(self, mode):
        """Slot to change the detector mode from the main thread."""
        self.detector_mode = mode