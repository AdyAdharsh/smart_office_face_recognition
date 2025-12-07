# app.py (Streamlit WebRTC Cloud Deployment)

import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# NOTE: Assuming src.RecognitionModel and src.LogManager exist and are updated
from src.RecognitionModel import RecognitionModel
from src.LogManager import LogManager

# --------------------------------------------------------
# PAGE CONFIG (must be at the VERY top for Streamlit)
# --------------------------------------------------------
st.set_page_config(
    page_title="Smart Office Face Recognition",
    layout="wide",
)

# --------------------------------------------------------
# INITIALIZATION (cached models - heavy loading)
# --------------------------------------------------------
@st.cache_resource
def load_resources():
    """Loads the heavy Recognition Model and LogManager only once."""
    st.write("Initializing ML Models (This takes a moment)...")
    model = RecognitionModel()
    log_manager = LogManager()
    st.write("Initialization complete.")
    return model, log_manager

model, log_manager = load_resources()


# --------------------------------------------------------
# VIDEO TRANSFORMER (WebRTC)
# --------------------------------------------------------
class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self, model, log_manager):
        self.model = model
        self.log_manager = log_manager
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr")
        self.frame_count += 1

        # ----------------------------------------
        # Process only every 5 frames (boost FPS)
        # ----------------------------------------
        if self.frame_count % 5 != 0:
            return img

        # ----------------------------------------
        # HEAVY FACE RECOGNITION
        # ----------------------------------------
        processed_img, log_msg, recognized_user, log_data = self.model.process_frame(
            img, detector_mode='cnn'
        )

        # If model returns None, avoid crash
        if processed_img is None:
            return img

        # ----------------------------------------
        # LOG EVENTS
        # ----------------------------------------
        if log_data and log_data.get("user_id") != "Unknown":
            self.log_manager.log_access_event(
                log_data.get("user_id"),
                log_data.get("status"),
                log_data.get("confidence")
            )

        return processed_img


# --------------------------------------------------------
# USER INTERFACE
# --------------------------------------------------------
st.title("👨‍💼 Smart Office Face Recognition System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

# ------------------------------
# LEFT COLUMN - LIVE WEBCAM
# ------------------------------
with col1:
    st.header("Live Attendance & Access Control")
    st.info("Please allow camera access. The system is running recognition models directly on the video stream.")

    webrtc_streamer(
        key="smart-office-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: FaceRecognitionTransformer(model, log_manager),
        async_processing=True,
        media_stream_constraints={"video": True, "audio": False},
    )

# ------------------------------
# RIGHT COLUMN - LOG DISPLAY
# ------------------------------
with col2:
    st.header("Access Log")

    # Auto-refresh every 5 seconds
    st.autorefresh(interval=5000, key="refresh_logs")

    latest_logs = log_manager.get_recent_logs(limit=10)

    if latest_logs:
        st.subheader("Latest Recognized Entries")
        st.dataframe(latest_logs, use_container_width=True)
    else:
        st.warning("No successful entries recorded yet.")

st.markdown("---")
st.caption("Powered by Streamlit, WebRTC, and Deepface models.")