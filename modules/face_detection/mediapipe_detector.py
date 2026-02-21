import cv2
import mediapipe as mp


class MediaPipeFaceDetector:
    def __init__(self, confidence=0.6):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=confidence
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        faces = []

        if results.detections:
            h, w, _ = frame.shape

            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                xmin = int(bbox.xmin * w)
                ymin = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                top = max(0, ymin)
                left = max(0, xmin)
                bottom = min(h, ymin + height)
                right = min(w, xmin + width)

                faces.append((top, right, bottom, left))

        return faces