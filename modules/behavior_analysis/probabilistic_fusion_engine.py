import time
from collections import deque, defaultdict
import numpy as np


class ProbabilisticFusionEngine:

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = defaultdict(deque)
        self.baseline = {}

        # Behavioral Intelligence
        self.persistence_memory = defaultdict(int)
        self.student_risk_index = defaultdict(float)
        self.reason_memory = defaultdict(str)

        # Event Counters
        self.event_counter = defaultdict(lambda: {
            "High": 0,
            "Moderate": 0,
            "Normal": 0
        })

    def update(self, student_id, yaw, pitch, blink, mouth, mobile):

        current_time = time.time()

        # Adaptive baseline
        if student_id not in self.baseline:
            self.baseline[student_id] = {
                "yaw": yaw,
                "pitch": pitch,
                "blink": blink
            }
        else:
            self.baseline[student_id]["yaw"] = 0.95 * self.baseline[student_id]["yaw"] + 0.05 * yaw
            self.baseline[student_id]["pitch"] = 0.95 * self.baseline[student_id]["pitch"] + 0.05 * pitch
            self.baseline[student_id]["blink"] = 0.95 * self.baseline[student_id]["blink"] + 0.05 * blink

        yaw_dev = abs(yaw - self.baseline[student_id]["yaw"])
        pitch_dev = abs(pitch - self.baseline[student_id]["pitch"])
        blink_dev = abs(blink - self.baseline[student_id]["blink"])

        self.buffer[student_id].append({
            "time": current_time,
            "yaw_dev": yaw_dev,
            "pitch_dev": pitch_dev,
            "blink_dev": blink_dev,
            "mouth": mouth,
            "mobile": mobile
        })

        while self.buffer[student_id] and \
                current_time - self.buffer[student_id][0]["time"] > self.window_size:
            self.buffer[student_id].popleft()

        return self.compute_intelligent_risk(student_id)

    # ================= RISK ENGINE =================

    def compute_intelligent_risk(self, student_id):

        data = list(self.buffer[student_id])
        if len(data) == 0:
            return 0.0, "Normal", "No activity"

        # Mobile escalation
        if any(d["mobile"] == 1 for d in data):
            self.persistence_memory[student_id] += 2
            self.reason_memory[student_id] = "Elevated external activity"
            return self.escalate(student_id, 1.0)

        yaw_mean = np.mean([d["yaw_dev"] for d in data])
        pitch_mean = np.mean([d["pitch_dev"] for d in data])
        mouth_rate = np.mean([d["mouth"] for d in data])

        head_movement = max(yaw_mean, pitch_mean)

        if head_movement > 30 and mouth_rate > 0.3:
            self.persistence_memory[student_id] += 2
            self.reason_memory[student_id] = "Strong head turn + speaking"
            return self.escalate(student_id, 0.95)

        if 10 < head_movement <= 30 and mouth_rate > 0.3:
            self.persistence_memory[student_id] += 1
            self.reason_memory[student_id] = "Head turn + speaking"
            return self.escalate(student_id, 0.9)

        if head_movement > 15:
            self.persistence_memory[student_id] += 1
            self.reason_memory[student_id] = "Head deviation"
            return self.escalate(student_id, 0.6)

        if mouth_rate > 0.4:
            self.persistence_memory[student_id] += 1
            self.reason_memory[student_id] = "Speaking detected"
            return self.escalate(student_id, 0.7)

        # Normal behavior decay
        decay_factor = 0.98
        self.student_risk_index[student_id] *= decay_factor

        self.persistence_memory[student_id] = max(
            self.persistence_memory[student_id] - 1, 0
        )

        self.reason_memory[student_id] = "Stable behavior"
        self.event_counter[student_id]["Normal"] += 1

        return 0.05, "Normal", "Stable behavior"

    # ================= ESCALATION =================

    def escalate(self, student_id, base_probability):

        persistence_factor = min(self.persistence_memory[student_id] * 0.1, 0.3)
        probability = min(base_probability + persistence_factor, 1.0)

        # Risk decay + accumulation
        decay_factor = 0.98
        alpha = 0.05

        previous_sri = self.student_risk_index[student_id]
        new_sri = decay_factor * previous_sri + alpha * probability
        self.student_risk_index[student_id] = new_sri

        if probability > 0.8:
            risk = "High"
        elif probability > 0.5:
            risk = "Moderate"
        else:
            risk = "Normal"

        self.event_counter[student_id][risk] += 1

        reason = self.reason_memory[student_id]

        return probability, risk, reason

    # ================= ACCESS METHODS =================

    def get_student_risk_index(self, student_id):
        return round(self.student_risk_index[student_id], 3)

    def generate_exam_report(self, student_id):
        return {
            "student_id": student_id,
            "final_sri": round(self.student_risk_index[student_id], 3),
            "high_events": self.event_counter[student_id]["High"],
            "moderate_events": self.event_counter[student_id]["Moderate"],
            "normal_events": self.event_counter[student_id]["Normal"]
        }