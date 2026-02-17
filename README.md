📌 Intent-Aware AI-Based Examination Surveillance System
Using Behavioral Pattern Deviation & Machine Learning

[AI Exam project.pdf](https://github.com/user-attachments/files/25357141/AI.Exam.project.pdf)

📖 Project Overview

The Intent-Aware AI-Based Examination Surveillance System is a real-time intelligent proctoring solution designed to detect suspicious behavior during online examinations.

Unlike traditional monitoring systems, this project focuses on:

Behavioral pattern learning

Deviation detection

Probability-based cheating intention scoring

Real-time alert generation

The system uses Logistic Regression to compute a Cheating Intention Score between 0 and 1.

🏗️ System Architecture

The system follows a modular, AI-driven layered architecture:

🔹 1. Input Layer

Webcam / Live Camera Feed

Student Authentication Data

🔹 2. Processing Layer

Video Capture Module

Face Detection & Tracking Module

Eye & Head Movement Tracking Module

🔹 3. Intelligence Layer

Baseline Behavior Learning Module

Behavior Deviation Analysis Module

Intention Scoring Engine (ML Model)

🔹 4. Application Layer

Alert & Logging Module

Examiner Dashboard

🔹 5. Storage Layer

Behavior Data Repository

Event Logs & Reports

🤖 Machine Learning Model
🔹 Model Used: Logistic Regression

Logistic Regression is selected because:

✔ Outputs probabilities (ideal for intention scoring)

✔ Computationally efficient for real-time systems

✔ Easy to interpret and justify academically

✔ Works well with structured numerical features

✔ Suitable for small to medium datasets

📊 Cheating Intention Formula

P(cheating) = 1 / (1 + e^-(wX + b))

Where:
- X = Feature vector  
- w = Model weights  
- b = Bias  

🔄 System Workflow
Step 1: Student Authentication

Student logs into the system

Identity verification performed

Step 2: Baseline Behavior Learning

System observes normal behavior

No alerts generated

Personalized baseline profile created

Step 3: Continuous Monitoring

Live webcam feed captured

Face & eye tracking performed

Step 4: Feature Extraction

Behavior converted into numerical values:

Eye gaze variance

Head movement frequency

Face stability score

Deviation duration

Deviation count

Step 5: Deviation Analysis

Live behavior compared with baseline

Anomalies detected

Step 6: Intention Scoring

Logistic Regression computes cheating probability
​
Step 7: Decision Engine

| Score Range | Risk Level | Action  |
|------------|------------|---------|
| 0.0 – 0.3  | Low        | Normal  |
| 0.3 – 0.7  | Medium     | Warning |
| 0.7 – 1.0  | High       | Alert   |

Step 8: Alert & Logging

Suspicious events recorded securely

Evidence preserved

Alerts shown to examiner

Step 9: Report Generation

Post-exam analysis report created

Student → System : Login  
System → Camera : Start Capture  
Camera → Tracking Module : Send Frames  
Tracking Module → ML Model : Send Features  
ML Model → Decision Engine : Intention Score  
Decision Engine → System : Risk Level  
System → Examiner : Alert (if high)  
System → Database : Store Logs  

🚀 Key Features

Real-time behavioral monitoring

Personalized baseline learning

Probability-based cheating detection

Multi-level risk classification

Alert & evidence logging

Examiner dashboard interface

Post-exam reporting system

🛠️ Technologies (Proposed Implementation)

Python

OpenCV

Machine Learning (Scikit-learn)

FastAPI / Flask

HTML, CSS, JavaScript

Database (MySQL / PostgreSQL)

🎯 Research Contribution

This system introduces an intent-based probabilistic model instead of binary cheating detection, enabling:

Reduced false positives

Personalized monitoring

Scalable real-time surveillance

Explainable AI decision-making

📌 Future Enhancements

Deep Learning-based intention prediction

Multi-camera support

Emotion detection integration

Adaptive model retraining

Cloud-based deployment
