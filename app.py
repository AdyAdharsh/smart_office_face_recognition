# app.py

import cv2
import time
from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from src.RecognitionModel import RecognitionModel
from src.LogManager import LogManager # Your in-memory LogManager
import uvicorn

# --- INITIALIZATION ---
model = RecognitionModel()
log_manager = LogManager()
templates = Jinja2Templates(directory="templates")

app = FastAPI()

# --- Generator function that produces JPEG frames ---
# NOTE: This must be synchronous so OpenCV's blocking read() works in a threadpool
def generate_frames(model: RecognitionModel, log_manager: LogManager, camera_index=0):
    # Use WebRTC/similar logic if deploying to cloud, but this local code is for the structure
    cap = cv2.VideoCapture(camera_index) 

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return 

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 1. Process Frame (Your Recognition Logic)
        # Your model.process_frame returns: processed_frame, log_msg, recognized_user, log_data
        processed_frame, _, _, log_data = model.process_frame(frame, detector_mode='cnn')
        
        # 2. Encode Frame to JPEG
        # Use JPEG encoding for better browser compatibility and streaming performance
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        
        if not ret:
            continue

        # 3. Log Event (Access Log)
        if log_data and log_data.get('user_id') != 'Unknown':
            log_manager.log_access_event(
                log_data.get('user_id'), 
                log_data.get('status'), 
                log_data.get('confidence')
            )

        # 4. Yield the Frame for the StreamingResponse
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

# --- Endpoint Definitions ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the HTML template."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/video_feed")
async def video_feed():
    """Endpoint that returns the video stream."""
    # The StreamingResponse uses the generator function and sets the necessary media type
    return StreamingResponse(
        generate_frames(model, log_manager),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )

# --- Log Endpoint (Recruiter can hit this to see the DB) ---
@app.get("/logs")
async def get_logs():
    """Endpoint to return the current log data."""
    return {"logs": log_manager.get_recent_logs(limit=50)}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)