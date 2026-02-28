import cv2
import numpy as np
import mediapipe as mp


class HeadPoseEstimator:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.blink_counter = 0
        self.eye_closed_frames = 0

        self.prev_yaw = 0
        self.prev_pitch = 0
        self.prev_roll = 0

    # ================= EAR =================
    def calculate_EAR(self, landmarks, eye_points, w, h):

        pts = []
        for p in eye_points:
            pts.append(np.array([
                landmarks[p].x * w,
                landmarks[p].y * h
            ]))

        vertical1 = np.linalg.norm(pts[1] - pts[5])
        vertical2 = np.linalg.norm(pts[2] - pts[4])
        horizontal = np.linalg.norm(pts[0] - pts[3])

        if horizontal == 0:
            return 0

        return (vertical1 + vertical2) / (2.0 * horizontal)

    # ================= MAR =================
    def calculate_MAR(self, landmarks, w, h):

        top = np.array([
            landmarks[13].x * w,
            landmarks[13].y * h
        ])

        bottom = np.array([
            landmarks[14].x * w,
            landmarks[14].y * h
        ])

        return np.linalg.norm(top - bottom)

    # ================= MAIN =================
    def estimate(self, frame):

        h, w, _ = frame.shape

        # Slight brightness stabilization
        frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=-5)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return 0, 0, 0, self.blink_counter, 0

        landmarks = results.multi_face_landmarks[0].landmark

        # ================= HEAD POSE =================

        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[61].x * w, landmarks[61].y * h),
            (landmarks[291].x * w, landmarks[291].y * h)
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])

        focal_length = w
        center = (w / 2, h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0]
        yaw = angles[1]
        roll = angles[2]

        # Clamp angles
        yaw = max(min(yaw, 40), -40)
        pitch = max(min(pitch, 40), -40)
        roll = max(min(roll, 40), -40)

        # Smooth angles
        yaw = 0.7 * self.prev_yaw + 0.3 * yaw
        pitch = 0.7 * self.prev_pitch + 0.3 * pitch
        roll = 0.7 * self.prev_roll + 0.3 * roll

        self.prev_yaw = yaw
        self.prev_pitch = pitch
        self.prev_roll = roll

        # ================= BLINK (Sensitive) =================

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        left_EAR = self.calculate_EAR(landmarks, left_eye, w, h)
        right_EAR = self.calculate_EAR(landmarks, right_eye, w, h)

        ear = (left_EAR + right_EAR) / 2.0

        EAR_THRESHOLD = 0.28
        CONSEC_FRAMES = 1

        if ear < EAR_THRESHOLD:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames >= CONSEC_FRAMES:
                self.blink_counter += 1
            self.eye_closed_frames = 0

        blink_count = self.blink_counter

        # ================= MOUTH (Talking Sensitive) =================

        mar = self.calculate_MAR(landmarks, w, h)

        face_height = abs((landmarks[152].y - landmarks[10].y) * h)

        if face_height == 0:
            mouth_open = 0
        else:
            normalized_mar = mar / face_height

            # More sensitive threshold for normal talking
            if normalized_mar > 0.045:
                mouth_open = 1
            else:
                mouth_open = 0

        return yaw, pitch, roll, blink_count, mouth_open