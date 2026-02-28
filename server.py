from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import csv
import os
import time
from datetime import datetime
from ultralytics import YOLO

from modules.face_detection.mediapipe_detector import MediaPipeFaceDetector
from modules.face_recognition_module.arcface_recognizer import ArcFaceRecognizer
from modules.tracking.centroid_tracker import CentroidTracker
from modules.behavior_analysis.pose_estimator import HeadPoseEstimator
from modules.behavior_analysis.probabilistic_fusion_engine import ProbabilisticFusionEngine

app = FastAPI()

# ================= INIT =================

detector = MediaPipeFaceDetector()
recognizer = ArcFaceRecognizer()
tracker = CentroidTracker()
pose_estimator = HeadPoseEstimator()
fusion_engine = ProbabilisticFusionEngine(window_size=5)

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
            "max_yaw_dev",
            "max_pitch_dev",
            "max_roll_dev",
            "blink_rate",
            "mouth_seen",
            "mobile_seen"
        ])

# ================= GLOBALS =================

latest_pose = {}
identity_lock = {}
last_log_time = {}
behavior_window_flags = {}
behavior_window_metrics = {}

LOG_INTERVAL = 30
baseline_yaw = None
baseline_pitch = None
DEAD_ZONE = 5

# ================= LOG FUNCTION =================

def log_behavior(student_id, yaw, pitch, roll,
                 blink_rate, mouth_seen, mobile_seen):

    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            student_id,
            round(yaw, 2),
            round(pitch, 2),
            round(roll, 2),
            round(blink_rate, 3),
            mouth_seen,
            mobile_seen
        ])

# ================= FRAME LOOP =================

def generate_frames():
    global baseline_yaw, baseline_pitch

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
                cv2.putText(frame, "Mobile Phone",
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255), 2)

        # ================= FACE DETECTION =================

        faces = detector.detect(frame)
        rects = []

        for (top, right, bottom, left) in faces:
            width = right - left
            height = bottom - top
            if width > 40 and height > 40:
                rects.append((left, top, right, bottom))

        objects = tracker.update(rects)
        latest_pose.clear()

        for object_id, centroid in objects.items():

            if object_id >= len(rects):
                continue

            (left, top, right, bottom) = rects[object_id]

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

            if baseline_yaw is None:
                baseline_yaw = yaw
                baseline_pitch = pitch

            yaw -= baseline_yaw
            pitch -= baseline_pitch

            if abs(yaw) < DEAD_ZONE:
                yaw = 0
            if abs(pitch) < DEAD_ZONE:
                pitch = 0
            if abs(roll) < DEAD_ZONE:
                roll = 0

            blink_rate_live = blink / 30.0

            probability, risk,reason = fusion_engine.update(
                student_id=student_id,
                yaw=yaw,
                pitch=pitch,
                blink=blink_rate_live,
                mouth=mouth,
                mobile=mobile_detected
            )

            # ===== Window Tracking =====

            if student_id not in behavior_window_flags:
                behavior_window_flags[student_id] = {
                    "mobile_seen": 0,
                    "mouth_seen": 0
                }

            if student_id not in behavior_window_metrics:
                behavior_window_metrics[student_id] = {
                    "max_yaw": 0,
                    "max_pitch": 0,
                    "max_roll": 0
                }

            # Track deviations
            behavior_window_metrics[student_id]["max_yaw"] = max(
                behavior_window_metrics[student_id]["max_yaw"], abs(yaw)
            )
            behavior_window_metrics[student_id]["max_pitch"] = max(
                behavior_window_metrics[student_id]["max_pitch"], abs(pitch)
            )
            behavior_window_metrics[student_id]["max_roll"] = max(
                behavior_window_metrics[student_id]["max_roll"], abs(roll)
            )

            if mobile_detected == 1:
                behavior_window_flags[student_id]["mobile_seen"] = 1

            if mouth == 1:
                behavior_window_flags[student_id]["mouth_seen"] = 1

            current_time = time.time()

            if student_id not in last_log_time:
                last_log_time[student_id] = current_time

            # ===== STRICT STORE CONDITION =====
            if current_time - last_log_time[student_id] >= LOG_INTERVAL:

                flags = behavior_window_flags[student_id]
                metrics = behavior_window_metrics[student_id]

                if flags["mobile_seen"] == 1 and flags["mouth_seen"] == 1:

                    log_behavior(
                        student_id,
                        metrics["max_yaw"],
                        metrics["max_pitch"],
                        metrics["max_roll"],
                        blink_rate_live,
                        flags["mouth_seen"],
                        flags["mobile_seen"]
                    )

                # Reset window
                behavior_window_flags[student_id] = {
                    "mobile_seen": 0,
                    "mouth_seen": 0
                }

                behavior_window_metrics[student_id] = {
                    "max_yaw": 0,
                    "max_pitch": 0,
                    "max_roll": 0
                }

                last_log_time[student_id] = current_time

            # Draw face
            cv2.rectangle(frame, (left, top),
                          (right, bottom), (0, 255, 0), 2)

            cv2.putText(frame,
                        f"ID {object_id} | {name} | {student_id}",
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255), 2)

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
                "risk": risk,
                "reason": reason,
                "sri": fusion_engine.get_student_risk_index(student_id)
            }

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

@app.get("/report/{student_id}")
def get_exam_report(student_id: str):
    return fusion_engine.generate_exam_report(student_id)