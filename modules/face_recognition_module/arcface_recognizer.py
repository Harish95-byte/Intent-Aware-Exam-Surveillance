import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from insightface.app import FaceAnalysis


class ArcFaceRecognizer:
    def __init__(self,
                 image_folder="students",
                 csv_path="student_db.csv",
                 threshold=0.7):

        self.threshold = threshold

        # Initialize InsightFace
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(320, 320))

        self.known_faces = {}
        self.load_database(image_folder, csv_path)

    def load_database(self, image_folder, csv_path):
        print("📂 Loading student database (ArcFace)...")

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            filename = row["filename"]
            name = row["name"]
            reg_no = row["reg_no"]

            image_path = os.path.join(image_folder, filename)

            if not os.path.exists(image_path):
                print(f"⚠ Image not found: {filename}")
                continue

            image = cv2.imread(image_path)
            faces = self.app.get(image)

            if len(faces) == 0:
                print(f"⚠ No face detected in {filename}")
                continue

            embedding = faces[0].embedding

            self.known_faces[name] = {
                "embedding": embedding,
                "reg_no": reg_no
            }

        print(f"✅ Loaded {len(self.known_faces)} students")

    def recognize(self, face_img):

        faces = self.app.get(face_img)

        if len(faces) == 0:
            return "Unknown", "Unknown"

        embedding = faces[0].embedding

        min_dist = float("inf")
        identity = "Unknown"
        reg_no = "Unknown"

        for name, data in self.known_faces.items():
            dist = cosine(embedding, data["embedding"])

            if dist < min_dist:
                min_dist = dist
                identity = name
                reg_no = data["reg_no"]

        if min_dist > self.threshold:
            return "Unknown", "Unknown"

        return identity, reg_no