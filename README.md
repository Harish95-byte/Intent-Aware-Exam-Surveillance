# Intent-Aware-Exam-Surveillance

Project Overview

This project proposes an AI-based examination surveillance system that predicts cheating intention by analyzing behavioral deviations of students during exams.
Unlike traditional proctoring systems that detect cheating actions directly, this system first learns a student’s normal behavior (baseline) and then identifies unusual changes to estimate cheating probability.

The system integrates Computer Vision, Machine Learning, and CI/CD automation using Azure DevOps.

Problem Statement

Online and offline examinations require continuous monitoring to ensure academic integrity.
Existing AI proctoring systems generate high false positives because they rely only on action-based detection.
This project addresses the issue by introducing an intent-aware approach, reducing false alerts and improving fairness.

Proposed Solution

Learn baseline behavior of each student

Monitor real-time facial, eye, and head movements

Detect behavior deviations

Predict cheating intention score

Generate alerts and reports based on probability

ML Model used:

| Module               | Model               |
| -------------------- | ------------------- |
| Video Capture        | OpenCV              |
| Face Detection       | YOLOv8-Face         |
| Face Tracking        | Deep SORT           |
| Eye Detection        | MediaPipe Face Mesh |
| Head Pose Estimation | OpenCV PnP          |
| Feature Extraction   | ResNet-18           |
| Baseline Learning    | Autoencoder         |
| Deviation Detection  | Isolation Forest    |
| Temporal Modeling    | LSTM                |
| Intention Scoring    | Bayesian Network    |

Technologies Used

Python

OpenCV

MediaPipe

Scikit-learn

PyTorch / TensorFlow

GitHub

Azure DevOps Pipelines (CI)

Advantages

Reduces false cheating alerts

Personalized behavior analysis

Early intention prediction

Scalable and automated

Industry-level DevOps workflow
