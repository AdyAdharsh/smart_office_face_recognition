# app.py (WebRTC Implementation for Camera Access)

import streamlit as st
import cv2
import numpy as np
import time

# --- NEW IMPORTS for WebRTC ---
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# ------------------------------

# Import your core logic
from src.RecognitionModel import RecognitionModel
from src.LogManager import LogManager

# --- Streamlit Session State for Log Management ---
# Global function to display logs since the transformer cannot access the main page directly
def update_log_display(log_manager, log_placeholder):
    recent_logs = log_manager.get_recent_logs(limit=15)
    log_content = ""
    for ts, status, user_id, confidence in recent_logs:
        display_user = user_id if user_id and user_id != 'Unknown' else "N/A"
        display_conf = f" ({confidence:.2f})" if confidence is not None else "0.00"
        color = "green" if status == "Granted" else "red"
        
        log_content += f"<p style='color:{color}; font-size:14px; margin:0;'>{ts} - **{status.upper()}**: {display_user}{display_conf}</p>"
    
    log_placeholder.markdown(log_content, unsafe_allow_html=True)


# --- Video Processing Class (Replaces the while loop) ---
class FaceRecognitionTransformer(VideoTransformerBase):
    """
    Handles frame processing for the WebRTC stream.
    This class runs on a separate server thread and only processes the image data.
    """
    def __init__(self, model, log_manager, detector_mode):
        self.model = model
        self.log_manager = log_manager
        self.detector_mode = detector_mode
        self.last_log_time = time.time()
        
    def transform(self, frame):
        # 1. Convert video stream frame (AVFrame) to NumPy array (BGR format for OpenCV)
        img = frame.to_ndarray(format="bgr24")

        # 2. Process frame using your existing logic (Returns: frame, log_msg, recognized_user, log_data)
        processed_frame, log_msg, recognized_user, log_data = self.model.process_frame(
            img, 
            detector_mode=self.detector_mode
        )
        
        # 3. Log to DB (runs every time a face is processed)
        if log_data and log_data.get('user_id') != 'Unknown':
            self.log_manager.log_access_event(
                log_data.get('user_id'), 
                log_data.get('status'), 
                log_data.get('confidence')
            )
        
        # 4. Return the processed frame (BGR format) to the WebRTC streamer
        return processed_frame


# --- Initialization (runs once) ---
@st.cache_resource
def load_resources():
    st.info("Loading ML models (FaceNet, etc.). This happens only once...")
    return RecognitionModel(), LogManager()

model, log_manager = load_resources()

# --- Page Setup ---
st.set_page_config(page_title="Smart Office Access System", layout="wide")
st.title("Smart Office Face Recognition Demo")

# --- UI Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Recognition Feed")

with col2:
    st.header("Access Log (DB)")
    LOG_TEXT = st.empty() # Placeholder for the dynamically updated log

# --- Controls ---
st.markdown("---")
st.markdown("### Controls")
mode_select = st.selectbox(
    "Select Detector Mode:",
    options=['cnn', 'classical'],
    index=0
)
st.markdown("---")


# --- WebRTC Streamer (Replaces the OpenCV Loop) ---

ctx = webrtc_streamer(
    key="face-recognition-stream",
    # Pass a factory function that creates the transformer instance, passing necessary arguments
    video_processor_factory=lambda: FaceRecognitionTransformer(
        model=model, 
        log_manager=log_manager, 
        detector_mode=mode_select
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)

# --- Log Display Loop (Updates the GUI Log Separately) ---

if ctx.state.playing:
    st.sidebar.success("Webcam Stream Active. Processing...")
    
    # Loop to refresh the log display while the stream is active
    while True:
        update_log_display(log_manager, LOG_TEXT)
        time.sleep(1) # Refresh log every second
        
else:
    # Display message if stream is not active
    if not ctx.state.last_video_frame:
        st.info("Click 'START' above and allow camera access to begin recognition.")