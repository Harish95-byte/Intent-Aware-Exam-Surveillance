from modules.face_detection.mediapipe_detector import FaceDetector
from modules.face_recognition_module.lbph_recognizer import FaceRecognizer
import cv2


def run_system():
    cap = cv2.VideoCapture(0)

    detector = FaceDetector()
    recognizer = FaceRecognizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = recognizer.recognize(frame)

        for result in results:
            top, right, bottom, left = result["location"]
            name = result["name"]
            reg_no = result["reg_no"]

            label = f"{name} | {reg_no}" if name != "Unknown" else "Unknown"

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Intent Aware Exam Surveillance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system()

import uvicorn

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)