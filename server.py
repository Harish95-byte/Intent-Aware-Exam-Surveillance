from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2

from modules.face_detection.mediapipe_detector import MediaPipeFaceDetector
from modules.face_recognition_module.lbph_recognizer import LBPHRecognizer
from modules.tracking.centroid_tracker import CentroidTracker
from modules.behavior_analysis.pose_estimator import HeadPoseEstimator

app = FastAPI()

detector = MediaPipeFaceDetector()
recognizer = LBPHRecognizer()
tracker = CentroidTracker()
pose_estimator = HeadPoseEstimator()

# Camera setup
cap = None
for i in range(5):
    temp = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp.isOpened():
        print(f"Using camera index {i}")
        cap = temp
        break

if cap is None:
    raise Exception("No camera found")

print("✅ Camera started successfully")

latest_pose = {}
previous_pose = {}   # 🔥 NEW: store previous frame values


def generate_frames():
    global latest_pose, previous_pose

    while True:
        success, frame = cap.read()
        if not success:
            break

        latest_pose.clear()   # 🔥 DO NOT reset dictionary reference

        faces = detector.detect(frame)

        rects = []
        for (top, right, bottom, left) in faces:
            rects.append((left, top, right, bottom))

        objects = tracker.update(rects)

        for object_id, centroid in objects.items():

            min_distance = float("inf")
            matched_box = None

            for (left, top, right, bottom) in rects:
                cX = int((left + right) / 2)
                cY = int((top + bottom) / 2)

                distance = ((centroid[0] - cX) ** 2 +
                            (centroid[1] - cY) ** 2) ** 0.5

                if distance < min_distance:
                    min_distance = distance
                    matched_box = (left, top, right, bottom)

            if matched_box is None:
                continue

            (left, top, right, bottom) = matched_box

            face_crop = frame[top:bottom, left:right]
            if face_crop.size == 0:
                continue

            name, reg_no = recognizer.recognize(face_crop)

            # Raw pose
            raw_yaw, raw_pitch, raw_roll = pose_estimator.estimate(face_crop)

            # 🔥 STABILITY FILTER
            alpha = 0.8       # smoothing strength
            dead_zone = 1.2   # ignore tiny movement

            prev = previous_pose.get(object_id)

            if prev:
                # Exponential smoothing
                yaw = alpha * prev["yaw"] + (1 - alpha) * raw_yaw
                pitch = alpha * prev["pitch"] + (1 - alpha) * raw_pitch
                roll = alpha * prev["roll"] + (1 - alpha) * raw_roll

                # Dead-zone filtering
                if abs(yaw - prev["yaw"]) < dead_zone:
                    yaw = prev["yaw"]

                if abs(pitch - prev["pitch"]) < dead_zone:
                    pitch = prev["pitch"]

                if abs(roll - prev["roll"]) < dead_zone:
                    roll = prev["roll"]

            else:
                yaw, pitch, roll = raw_yaw, raw_pitch, raw_roll

            # Save for next frame memory
            previous_pose[object_id] = {
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll
            }

            # Store output for frontend
            latest_pose[object_id] = {
                "name": name,
                "reg_no": reg_no,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll)
            }

            label = f"ID {object_id} | {name} | {reg_no}"

            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )


@app.get("/", response_class=HTMLResponse)
def home():
    with open("exam_dashboard.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/video")
def video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/pose")
def get_pose():
    return latest_pose