📌 Intent-Aware AI-Based Examination Surveillance System
Using Behavioral Pattern Deviation & Machine Learning

[AI Exam project.pdf](https://github.com/user-attachments/files/25357141/AI.Exam.project.pdf)

1️⃣ Introduction

The Intent-Aware AI-Based Examination Surveillance System is an AI-driven online proctoring solution designed to detect suspicious student behavior during examinations.

The system analyzes behavioral patterns and predicts the probability of cheating using machine learning techniques instead of simple rule-based detection.

2️⃣ Project Objective

Monitor students in real-time using webcam input

Analyze facial presence, eye movement, and head orientation

Learn individual baseline behavior

Detect deviations from normal behavior

Compute a probability-based cheating intention score

Generate alerts for examiners when necessary

3️⃣ Project Scope
What We Are Going To Do

Monitor students during online examinations using webcam

Track face detection, eye movement, and head movement

Learn normal behavior during the initial phase

Detect deviations from baseline behavior

Calculate cheating probability using Machine Learning

Generate warnings or alerts for examiners

4️⃣ Implementation Approach
How We Are Going To Do It

Capture live video using webcam

Detect and track facial features

Extract behavioral features (gaze direction, movement frequency, stability)

Create a personalized baseline behavior profile

Compare real-time data with baseline behavior

Feed deviation metrics into the Machine Learning model

Generate risk level based on predicted probability

5️⃣ Machine Learning Model
Model Used: Logistic Regression

The system uses Logistic Regression to compute the cheating probability score.

P(cheating) = 1 / (1 + e^-(wX + b))


Where:

X = Feature vector

w = Model weights

b = Bias

Model Justification

Produces probability output

Computationally efficient

Suitable for real-time applications

Easy to interpret and explain academically

6️⃣ Technologies Used

Python

OpenCV

Scikit-learn

FastAPI / Flask

HTML, CSS, JavaScript

MySQL / PostgreSQL

7️⃣ System Workflow

Student Authentication

Baseline Behavior Learning

Continuous Video Monitoring

Face & Eye Tracking

Feature Extraction

Behavior Deviation Analysis

Logistic Regression Prediction

Risk Level Classification

Alert & Logging

*Risk Level Classification

| Score Range | Risk Level | Action  |
|------------|------------|---------|
| 0.0 – 0.3  | Low        | Normal  |
| 0.3 – 0.7  | Medium     | Warning |
| 0.7 – 1.0  | High       | Alert   |

8️⃣ Key Features

Real-time behavioral monitoring

Personalized baseline learning

Probability-based cheating detection

Multi-level risk classification

Secure alert and logging mechanism

Examiner dashboard integration

Modules used :

Face Detection (YOLOv8-Face or HOG)
Face Tracking (Deep SORT or simple centroid tracking)
Eye + Head Pose (MediaPipe + PnP)
Temporal Behavior Modeling (LSTM)
Baseline Normal Behavior Learning (Autoencoder)
Deviation Detection (Isolation Forest)
Bayesian Intent Scoring Engine ⭐ (Core Patent Claim)
Adaptive Decision Threshold + Logging
