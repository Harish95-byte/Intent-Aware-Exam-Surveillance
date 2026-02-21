import os
import cv2
import numpy as np
import pandas as pd


class LBPHRecognizer:

    def __init__(self, dataset_path="students", db_path="student_db.csv"):
        self.dataset_path = dataset_path
        self.db_path = db_path
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self.student_info = {}
        self.load_student_database()
        self.train()

    def load_student_database(self):
        if not os.path.exists(self.db_path):
            print("student_db.csv not found!")
            return

        df = pd.read_csv(self.db_path)

        for _, row in df.iterrows():
            filename = row["filename"]
            name = row["name"]
            reg_no = row["reg_no"]

            # Remove extension for matching
            key = os.path.splitext(filename)[0]
            self.student_info[key] = (name, reg_no)

    def train(self):
        faces = []
        labels = []
        label_id = 0

        if not os.path.exists(self.dataset_path):
            print("Dataset folder not found!")
            return

        for image_name in os.listdir(self.dataset_path):

            if not image_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(self.dataset_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (200, 200))

            faces.append(img)
            labels.append(label_id)

            # Store filename without extension
            person_key = os.path.splitext(image_name)[0]
            self.label_map[label_id] = person_key

            label_id += 1

        if len(faces) > 0:
            self.recognizer.train(faces, np.array(labels))
            print("LBPH Model Trained Successfully")
        else:
            print("No training images found!")

    def recognize(self, face_img):

        if len(self.label_map) == 0:
            return "Unknown", ""

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (200, 200))

        try:
            label, confidence = self.recognizer.predict(gray)
        except:
            return "Unknown", ""

        if confidence < 80:
            person_key = self.label_map.get(label, None)

            if person_key in self.student_info:
                name, reg_no = self.student_info[person_key]
                return name, reg_no

        return "Unknown", ""