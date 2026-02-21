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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def estimate(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return 0.0, 0.0, 0.0

        face_landmarks = results.multi_face_landmarks[0]

        img_h, img_w, _ = frame.shape

        # --- SolvePnP for Yaw & Pitch ---
        landmark_ids = [33, 263, 1, 61, 291, 199]

        face_2d = []
        face_3d = []

        for idx in landmark_ids:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array([
            [focal_length, 0, img_w / 2],
            [0, focal_length, img_h / 2],
            [0, 0, 1]
        ])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d,
            face_2d,
            cam_matrix,
            dist_matrix
        )

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = angles[0] * 360
        yaw = angles[1] * 360

        # --- 🔥 Accurate Roll Using Eye Alignment ---
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        x1, y1 = left_eye.x * img_w, left_eye.y * img_h
        x2, y2 = right_eye.x * img_w, right_eye.y * img_h

        roll = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        return yaw, pitch, roll