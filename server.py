from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime
from ultralytics import YOLO

from modules.face_detection.mediapipe_detector import MediaPipeFaceDetector
from modules.face_recognition_module.arcface_recognizer import ArcFaceRecognizer
from modules.tracking.centroid_tracker import CentroidTracker
from modules.behavior_analysis.pose_estimator import HeadPoseEstimator

app = FastAPI()

# ================= INIT =================

detector = MediaPipeFaceDetector()
recognizer = ArcFaceRecognizer()
tracker = CentroidTracker()
pose_estimator = HeadPoseEstimator()

yolo_model = YOLO("yolov8n.pt")

# ================= CAMERA =================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

# ================= CSV SETUP =================

CSV_FILE = "behavioral_baseline_log_dataset.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "timestamp",
            "student_id",
            "yaw",
            "pitch",
            "roll",
            "blink_rate",
            "mouth_ratio",
            "mobile_ratio"
        ])

# ================= WINDOW SETTINGS =================

WINDOW_DURATION = 30
window_start = time.time()

window_data = {}
latest_pose = {}
identity_lock = {}

# ================= LOG FUNCTION =================

def log_behavior(student_id, yaw, pitch, roll,
                 blink_rate, mouth_ratio, mobile_ratio):

    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            student_id,
            round(yaw, 2),
            round(pitch, 2),
            round(roll, 2),
            round(blink_rate, 2),
            round(mouth_ratio, 2),
            round(mobile_ratio, 2)
        ])

# ================= FRAME LOOP =================

def generate_frames():
    global window_start

    while True:
        success, frame = cap.read()
        if not success:
            break

        # ================= MOBILE DETECTION =================

        mobile_detected = 0
        results = yolo_model(frame, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = yolo_model.names[cls_id]

            if label == "cell phone":
                mobile_detected = 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 0, 255), 2)
                cv2.putText(frame, "Mobile",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255), 2)

        # ================= FACE DETECTION =================

        faces = detector.detect(frame)
        rects = []

        for (top, right, bottom, left) in faces:
            if right - left > 80 and bottom - top > 80:
                rects.append((left, top, right, bottom))

        objects = tracker.update(rects)
        latest_pose.clear()

        for object_id, centroid in objects.items():

            if len(rects) == 0:
                continue

            (left, top, right, bottom) = rects[0]
            face_crop = frame[top:bottom, left:right]

            if face_crop.size == 0:
                continue

            # Recognition Lock
            if object_id not in identity_lock:
                name, reg_no = recognizer.recognize(face_crop)
                if reg_no:
                    identity_lock[object_id] = (name, reg_no)
            else:
                name, reg_no = identity_lock[object_id]

            if object_id not in identity_lock:
                continue

            student_id = reg_no

            yaw, pitch, roll, blink, mouth = pose_estimator.estimate(face_crop)

            # ================= WINDOW STORE =================

            if student_id not in window_data:
                window_data[student_id] = {
                    "yaw": [],
                    "pitch": [],
                    "roll": [],
                    "blink": 0,
                    "mouth": 0,
                    "mobile": 0,
                    "frames": 0
                }

            window_data[student_id]["yaw"].append(yaw)
            window_data[student_id]["pitch"].append(pitch)
            window_data[student_id]["roll"].append(roll)
            window_data[student_id]["blink"] += blink
            window_data[student_id]["mouth"] += mouth
            window_data[student_id]["mobile"] += mobile_detected
            window_data[student_id]["frames"] += 1

            # ================= DASHBOARD DATA =================

            latest_pose[object_id] = {
                "name": name,
                "reg_no": reg_no,
                "yaw": round(float(yaw), 2),
                "pitch": round(float(pitch), 2),
                "roll": round(float(roll), 2),
                "blink": blink,
                "mouth": mouth,
                "mobile": mobile_detected
            }

            # Draw label
            label = f"ID {object_id} | {name} | {reg_no}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            cv2.rectangle(frame,
                          (left, top - text_height - 10),
                          (left + text_width + 6, top),
                          (0, 0, 0), -1)

            cv2.putText(frame,
                        label,
                        (left + 3, top - 5),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA)

            cv2.rectangle(frame,
                          (left, top),
                          (right, bottom),
                          (0, 255, 0), 2)

        # ================= WINDOW FLUSH =================

        if time.time() - window_start >= WINDOW_DURATION:

            for student_id, data in window_data.items():

                if data["frames"] == 0:
                    continue

                mean_yaw = np.mean(data["yaw"])
                mean_pitch = np.mean(data["pitch"])
                mean_roll = np.mean(data["roll"])

                blink_rate = data["blink"] / WINDOW_DURATION
                mouth_ratio = data["mouth"] / data["frames"]
                mobile_ratio = data["mobile"] / data["frames"]

                log_behavior(
                    student_id,
                    mean_yaw,
                    mean_pitch,
                    mean_roll,
                    blink_rate,
                    mouth_ratio,
                    mobile_ratio
                )

            window_data.clear()
            window_start = time.time()

        ret, buffer = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        )

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() +
               b"\r\n")


# ================= ROUTES =================

@app.get("/", response_class=HTMLResponse)
def home():
    with open("exam_dashboard.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/pose")
def get_pose():
    return latest_pose