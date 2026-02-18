📌 Intent-Aware AI-Based Examination Surveillance System
Using Behavioral Pattern Deviation & Machine Learning

A Multi-Modal Behavioral Deviation Fusion Framework for Probabilistic Cheating Intention Inference
1️⃣ Technical Field of Invention

The present invention relates to:

Artificial Intelligence-based surveillance systems

Real-time behavioral analytics

Probabilistic intent inference

Examination integrity monitoring

Specifically, this system introduces a multi-layer behavioral deviation fusion engine to estimate cheating intention probability in examination environments.

2️⃣ Problem Statement

Existing online proctoring systems:

Use simple rule-based triggers

Depend on fixed thresholds

Lack personalized behavioral baselines

Do not model temporal behavioral deviation

Cannot probabilistically infer intention

There is a need for a system that:

Learns normal behavior per candidate

Detects deviations across multiple modalities

Models behavior over time

Produces a probabilistic cheating intent score

3️⃣ Summary of the Invention

The proposed system introduces:

A Multi-Modal Behavioral Deviation Fusion Framework
with Temporal Modeling and Probabilistic Intent Scoring.

The invention integrates:

Facial presence stability

Eye gaze direction patterns

Head orientation dynamics

Behavioral baseline learning

Temporal deviation modeling

Bayesian probabilistic inference

to compute : p(cheating | Behavioral Features)
System Architecture
Core Modules

1️⃣ Face Detection Module

Detects candidate presence in real-time

2️⃣ Face Tracking Module

Maintains persistent identity across frames

3️⃣ Eye Landmark & Head Pose Module

Extracts gaze direction and orientation angles

4️⃣ Behavioral Feature Extraction Module

Computes movement frequency

Gaze deviation duration

Pose variance

Face stability metrics

5️⃣ Baseline Behavior Learning Module

Learns normal behavioral profile

Personalized per candidate

6️⃣ Deviation Detection Module

Measures statistical distance from baseline

7️⃣ Temporal Behavior Modeling Module

Models sequential behavior patterns over time

8️⃣ Bayesian Intent Scoring Engine ⭐ (Core Innovation)

Fuses multi-modal deviations

Computes probabilistic cheating intention

9️⃣ Adaptive Decision Threshold Module

Converts probability into actionable risk levels

🔟 Alert & Logging Module

Stores behavioral evidence

Generates examiner reports

System Architecture
Core Modules

1️⃣ Face Detection Module

Detects candidate presence in real-time

2️⃣ Face Tracking Module

Maintains persistent identity across frames

3️⃣ Eye Landmark & Head Pose Module

Extracts gaze direction and orientation angles

4️⃣ Behavioral Feature Extraction Module

Computes movement frequency

Gaze deviation duration

Pose variance

Face stability metrics

5️⃣ Baseline Behavior Learning Module

Learns normal behavioral profile

Personalized per candidate

6️⃣ Deviation Detection Module

Measures statistical distance from baseline

7️⃣ Temporal Behavior Modeling Module

Models sequential behavior patterns over time

8️⃣ Bayesian Intent Scoring Engine ⭐ (Core Innovation)

Fuses multi-modal deviations

Computes probabilistic cheating intention

9️⃣ Adaptive Decision Threshold Module

Converts probability into actionable risk levels

🔟 Alert & Logging Module

Stores behavioral evidence

Generates examiner reports

Mathematical Framework:
Let:
Gi = Gaze deviation metric
Hi = Head orientation variance
Fi = Face Stability score
Di = Behavioral deviation from baseline
Ti = Temporal anomaly factor
The cheating probability is computed as : P(cheating) = f(Gi,Hi,Fi,Di,Ti)
using :
Logistic modeling
Bayesian Inference
Temporal Weighting 

Risk Classification:

| Probability Range | Risk Level | Action            |
| ----------------- | ---------- | ----------------- |
| 0.0 – 0.3         | Low        | Normal Monitoring |
| 0.3 – 0.7         | Medium     | Warning Flag      |
| 0.7 – 1.0         | High       | Alert & Logging   |

Technical Advantages Over Existing Systems

✔ Personalized behavioral baseline learning
✔ Multi-modal behavioral fusion
✔ Temporal modeling of deviations
✔ Probability-based intention inference
✔ Adaptive thresholding mechanism
✔ Explainable AI framework


Implementation Technologies

Python

OpenCV

Scikit-learn

Deep Learning frameworks

FastAPI (real-time streaming)

Relational database for behavioral logs


Novelty & Patent Claim Focus

The novelty does NOT lie in:

Face detection algorithms

Eye tracking models

Existing deep learning architectures

The novelty lies in:

The probabilistic fusion of multi-modal behavioral deviations with temporal modeling to compute cheating intention likelihood.

This integrated behavioral deviation fusion architecture constitutes the core inventive step.

Future Extensions

Multi-room monitoring

Distributed exam hall surveillance

Edge-device deployment

Federated behavioral learning

Modules used :

Face Detection (YOLOv8-Face or HOG)
Face Tracking (Deep SORT or simple centroid tracking)
Eye + Head Pose (MediaPipe + PnP)
Temporal Behavior Modeling (LSTM)
Baseline Normal Behavior Learning (Autoencoder)
Deviation Detection (Isolation Forest)
Bayesian Intent Scoring Engine ⭐ (Core Patent Claim)
Adaptive Decision Threshold + Logging
