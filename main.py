from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import cv2

from modules.behavior_analysis.pose_estimator import HeadPoseEstimator

app = FastAPI()

pose_estimator = HeadPoseEstimator()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

latest_pose_data = {}


@app.get("/")
def root():
    return {"message": "AI Examination Surveillance System Running"}


def generate_frames():
    global latest_pose_data

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yaw, pitch, roll, blink, mouth = pose_estimator.estimate(frame)

        latest_pose_data = {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "blink_count": blink,
            "mouth_open": mouth
        }

        cv2.putText(frame, f"Yaw: {round(yaw,2)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Pitch: {round(pitch,2)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Roll: {round(roll,2)}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Blink: {blink}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Mouth: {mouth}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/pose")
def get_pose():
    return JSONResponse(latest_pose_data)