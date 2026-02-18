from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
from modules.face_recognition_module.recognizer import FaceRecognizer

app = FastAPI()

recognizer = FaceRecognizer()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize for performance
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        results = recognizer.recognize(small_frame)

        for result in results:
            top, right, bottom, left = result["location"]
            name = result["name"]
            reg_no = result["reg_no"]

            # Scale back
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            label = f"{name} | {reg_no}" if name != "Unknown" else "Unknown"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Intent Aware Exam Surveillance</title>
            <style>
                body {
                    margin: 0;
                    background-color: black;
                }
                img {
                    width: 100vw;
                    height: 100vh;
                    object-fit: cover;
                }
            </style>
        </head>
        <body>
            <img src="/video" />
        </body>
    </html>
    """




@app.get("/video")
def video():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

from fastapi.responses import FileResponse

@app.get("/dashboard")
def dashboard():
    return FileResponse("exam_dashboard.html")
