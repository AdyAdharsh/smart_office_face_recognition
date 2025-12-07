# app.py (Streamlit WebRTC Cloud Deployment)

import streamlit as st
import cv2
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# NOTE: Assuming src.RecognitionModel and src.LogManager exist and are updated
from src.RecognitionModel import RecognitionModel
from src.LogManager import LogManager 

# --- INITIALIZATION ---

# Initialize your Recognition Model and Log Manager.
# Streamlit's @st.cache_resource decorator handles the slow, synchronous loading of Keras/TF 
# models by running it only once and caching the result, preventing the timeout issue 
# you experienced with FastAPI.
@st.cache_resource
def load_resources():
    """Loads the heavy Recognition Model and LogManager only once."""
    st.write("Initializing ML Models (This takes a moment)...")
    model = RecognitionModel()
    log_manager = LogManager()
    st.write("Initialization complete.")
    return model, log_manager

# Load the resources globally
model, log_manager = load_resources()

# --- VIDEO TRANSFORMER CLASS ---

class FaceRecognitionTransformer(VideoTransformerBase):
    """
    A class that takes a video frame, processes it for face recognition, 
    and returns the annotated frame.
    """
    def __init__(self, model, log_manager):
        self.model = model
        self.log_manager = log_manager

    def transform(self, frame):
        # Convert the incoming video frame (from WebRTC) to an OpenCV array
        img = frame.to_ndarray(format="bgr")

        # 1. Process Frame (Your Recognition Logic)
        # This calls your core function to perform detection and recognition
        processed_img, log_msg, recognized_user, log_data = self.model.process_frame(img, detector_mode='cnn')

        # 2. Log Event
        if log_data and log_data.get('user_id') != 'Unknown':
            self.log_manager.log_access_event(
                log_data.get('user_id'), 
                log_data.get('status'), 
                log_data.get('confidence')
            )

        # Return the annotated frame back to the Streamlit video component
        return processed_img

# --- STREAMLIT UI LAYOUT ---

st.set_page_config(page_title="Smart Office Face Recognition", layout="wide")
st.title("👨‍💼 Smart Office Face Recognition System")
st.markdown("---")

# Instructions and Live Feed Section
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live Attendance & Access Control")
    st.info("Please allow camera access. The system is running recognition models directly on the video stream.")

    # 3. WebRTC Streamer Setup
    # This widget handles camera access, video streaming, and uses the transformer class
    webrtc_streamer(
        key="smart-office-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: FaceRecognitionTransformer(model, log_manager),
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.header("Access Log")
    
    # Refresh log every 5 seconds (to simulate real-time updates)
    time.sleep(1) # Simple sleep to allow model time to process the first frame
    
    # Display the latest access log data
    latest_logs = log_manager.get_recent_logs(limit=10)
    
    if latest_logs:
        st.subheader("Latest Recognized Entries")
        # Display logs in a table/dataframe format
        st.dataframe(latest_logs, use_container_width=True)
    else:
        st.warning("No successful entries recorded yet.")

st.markdown("---")
st.caption("Powered by Streamlit, WebRTC, and Deepface models.")