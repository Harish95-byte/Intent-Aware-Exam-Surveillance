import os
import cv2
import numpy as np
import pandas as pd
import face_recognition


class FaceRecognizer:

    def __init__(self, db_path="student_db.csv", image_folder="students"):
        self.db_path = db_path
        self.image_folder = image_folder

        self.known_encodings = []
        self.known_names = []
        self.known_regnos = []

        self.load_database()

    def load_database(self):
        print("Loading student database...")

        df = pd.read_csv(self.db_path)

        for _, row in df.iterrows():
            image_path = os.path.join(self.image_folder, row["filename"])

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"No face found in {row['filename']}")
                continue

            self.known_encodings.append(encodings[0])
            self.known_names.append(row["name"])
            self.known_regnos.append(row["reg_no"])

        print("Database loaded successfully.")

    def recognize(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        results = []

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):

            name = "Unknown"
            reg_no = ""

            if len(self.known_encodings) > 0:
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                # 🔥 Strict tolerance (adjust if needed)
                if face_distances[best_match_index] < 0.45:
                    name = self.known_names[best_match_index]
                    reg_no = self.known_regnos[best_match_index]

            results.append({
                "location": (top, right, bottom, left),
                "name": name,
                "reg_no": reg_no
            })

        return results