import os
import sys
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO


app = FastAPI()

def get_video_source():
    if sys.platform.startswith('win'):
        return cv2.VideoCapture(0, cv2.CAP_DSHOW)
    elif sys.platform.startswith('linux'):
        return cv2.VideoCapture('/dev/video0')
    else:
        raise OSError('Unsupported OS')

models_path = "./data/models"
weights = 'best.pt'
# model = YOLO(os.path.join(models_path, weights))
model = YOLO('yolov8n.pt')
def generate_frames():
    cap = get_video_source()
    if not cap.isOpened():
        raise OSError('Cannot open camera')
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(source=frame, show=False, tracker='bytetrack.yaml')
            for result in results:
                frame = result.plot()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()
            

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get('check_folder/{folder_name}')
def check_folder(folder_name: str):
    return os.listdir(folder_name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5500, reload=True)