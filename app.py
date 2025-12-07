# app.py (New File for Streamlit Web Demo)

import streamlit as st
import cv2
import numpy as np
import time

# Import your core logic
# Ensure these imports use the correct absolute paths if needed, 
# though running from the root usually handles 'src' package correctly.
from src.RecognitionModel import RecognitionModel
from src.LogManager import LogManager

# --- Initialization (runs once) ---
# Use Streamlit's cache to ensure heavy models only load once
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
    # Streamlit component to hold the video feed
    FRAME_HOLDER = st.image([], caption="Webcam Stream", width=640)

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


# --- Real-Time Processing Loop ---

# Use Streamlit's session state to manage the camera state (optional, but good practice)
if 'start_camera' not in st.session_state:
    st.session_state.start_camera = True

if st.session_state.start_camera:
    
    # Use OpenCV to capture the video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open webcam. Please ensure your camera is not in use.")
    else:
        # Streamlit requires a container to show progress
        status_text = st.empty()
        status_text.info("Recognition running... Look into the camera.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                status_text.warning("Failed to read frame.")
                break
                
            # Call your existing recognition logic (Returns: frame, log_msg, recognized_user, log_data)
            processed_frame, log_msg, recognized_user, log_data = model.process_frame(
                frame, 
                detector_mode=mode_select
            )
            
            # 1. Update Video Feed (Streamlit uses RGB)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            FRAME_HOLDER.image(processed_frame_rgb, channels="RGB", width=640)

            # 2. Log to DB and Update Display 
            if log_data and log_data.get('user_id') != 'Unknown': # Only log successful or definite attempts
                # Log the event to SQLite
                log_manager.log_access_event(
                    log_data.get('user_id'), 
                    log_data.get('status'), 
                    log_data.get('confidence')
                )
                
            # Display logs from DB
            recent_logs = log_manager.get_recent_logs(limit=15)
            log_content = ""
            for ts, status, user_id, confidence in recent_logs:
                display_user = user_id if user_id and user_id != 'Unknown' else "N/A"
                display_conf = f" ({confidence:.2f})" if confidence is not None else "0.00"
                color = "green" if status == "Granted" else "red"
                
                log_content += f"<p style='color:{color}; font-size:14px; margin:0;'>{ts} - **{status.upper()}**: {display_user}{display_conf}</p>"
            
            LOG_TEXT.markdown(log_content, unsafe_allow_html=True)
            
            # Streamlit loop needs to yield control quickly
            time.sleep(0.01)

    # Cleanup
    cap.release()
    status_text.warning("Application stopped.")