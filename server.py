from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime
from ultralytics import YOLO
import pandas as pd

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

# ================= GLOBALS =================

WINDOW_DURATION = 30
window_start = time.time()

window_data = {}
latest_pose = {}
identity_lock = {}

# ===== NEW: baseline correction variables =====
baseline_yaw = None
baseline_pitch = None
DEAD_ZONE = 8

# ================= PROBABILITY FUNCTION =================

def compute_probability(student_id, yaw, pitch, blink_rate, mobile, mouth):

    if mobile == 1:
        return 1.0, "High"

    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_yaw < 15 and abs_pitch < 15:
        if mouth == 1:
            return 0.4, "Moderate"
        return 0.05, "Normal"

    if abs_yaw < 35 and abs_pitch < 35:
        if mouth == 1:
            return 0.9, "High"
        return 0.5, "Moderate"

    return 0.9, "High"

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
    global window_start, baseline_yaw, baseline_pitch

    while True:
        success, frame = cap.read()
        if not success:
            break

        mobile_detected = 0

        # ================= MOBILE DETECTION =================

        results = yolo_model(frame, conf=0.3, verbose=False)[0]

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

            if object_id not in identity_lock:
                name, reg_no = recognizer.recognize(face_crop)
                if reg_no and reg_no != "Unknown":
                    identity_lock[object_id] = (name, reg_no.strip())
                else:
                    continue

            name, reg_no = identity_lock[object_id]
            student_id = reg_no.strip()

            yaw, pitch, roll, blink, mouth = pose_estimator.estimate(face_crop)

            # ===== NEW: Baseline auto calibration =====
            if baseline_yaw is None:
                baseline_yaw = yaw
                baseline_pitch = pitch

            yaw -= baseline_yaw
            pitch -= baseline_pitch

            # ===== NEW: Dead-zone filter =====
            if abs(yaw) < DEAD_ZONE:
                yaw = 0
            if abs(pitch) < DEAD_ZONE:
                pitch = 0
            if abs(roll) < DEAD_ZONE:
                roll = 0

            blink_rate_live = blink / 30.0

            probability, risk = compute_probability(
                student_id,
                yaw,
                pitch,
                blink_rate_live,
                mobile_detected,
                mouth
            )

            latest_pose[object_id] = {
                "name": name,
                "reg_no": student_id,
                "yaw": round(float(yaw), 2),
                "pitch": round(float(pitch), 2),
                "roll": round(float(roll), 2),
                "blink": blink,
                "mouth": mouth,
                "mobile": mobile_detected,
                "probability": probability,
                "risk": risk
            }

            cv2.rectangle(frame,
                          (left, top),
                          (right, bottom),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        f"ID {object_id} | {name} | {student_id}",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)

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