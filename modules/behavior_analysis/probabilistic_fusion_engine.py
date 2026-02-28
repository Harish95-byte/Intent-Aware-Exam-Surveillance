import time
from collections import deque, defaultdict
import numpy as np


class ProbabilisticFusionEngine:

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = defaultdict(deque)
        self.baseline = {}

        # Behavioral intelligence
        self.persistence_memory = defaultdict(int)
        self.student_risk_index = defaultdict(float)
        self.reason_memory = defaultdict(str)

        self.escalation_level = defaultdict(int)
        self.critical_counter = defaultdict(int)
        self.last_critical_time = defaultdict(float)

        # Event counters
        self.event_counter = defaultdict(lambda: {
            "High": 0,
            "Moderate": 0,
            "Normal": 0
        })

    # ================= UPDATE =================

    def update(self, student_id, yaw, pitch, blink, mouth, mobile):

        current_time = time.time()

        # -------- Adaptive Baseline --------
        if student_id not in self.baseline:
            self.baseline[student_id] = {
                "yaw": yaw,
                "pitch": pitch,
                "blink": blink
            }
        else:
            self.baseline[student_id]["yaw"] = \
                0.95 * self.baseline[student_id]["yaw"] + 0.05 * yaw
            self.baseline[student_id]["pitch"] = \
                0.95 * self.baseline[student_id]["pitch"] + 0.05 * pitch
            self.baseline[student_id]["blink"] = \
                0.95 * self.baseline[student_id]["blink"] + 0.05 * blink

        yaw_dev = abs(yaw - self.baseline[student_id]["yaw"])
        pitch_dev = abs(pitch - self.baseline[student_id]["pitch"])

        self.buffer[student_id].append({
            "time": current_time,
            "yaw_dev": yaw_dev,
            "pitch_dev": pitch_dev,
            "mouth": mouth,
            "mobile": mobile
        })

        # Remove old entries
        while self.buffer[student_id] and \
                current_time - self.buffer[student_id][0]["time"] > self.window_size:
            self.buffer[student_id].popleft()

        return self.compute_intelligent_risk(student_id)

    # ================= RISK ENGINE =================

    def compute_intelligent_risk(self, student_id):

        data = list(self.buffer[student_id])
        if len(data) == 0:
            return 0.0, "Normal", "No activity"

        # -------- Mobile Immediate High --------
        if any(d["mobile"] == 1 for d in data):
            self.reason_memory[student_id] = "External device detected"
            return self.escalate(student_id, 1.0)

        yaw_mean = np.mean([d["yaw_dev"] for d in data])
        pitch_mean = np.mean([d["pitch_dev"] for d in data])
        mouth_rate = np.mean([d["mouth"] for d in data])

        head_movement = max(yaw_mean, pitch_mean)

        # -------- Contextual Rules --------

        if head_movement > 30 and mouth_rate > 0.3:
            self.reason_memory[student_id] = "Strong head turn with speech"
            return self.escalate(student_id, 0.95)

        if 10 < head_movement <= 30 and mouth_rate > 0.3:
            self.reason_memory[student_id] = "Head turn with speech"
            return self.escalate(student_id, 0.85)

        if head_movement > 15:
            self.reason_memory[student_id] = "Head deviation"
            return self.escalate(student_id, 0.6)

        if mouth_rate > 0.4:
            self.reason_memory[student_id] = "Speech-like activity"
            return self.escalate(student_id, 0.7)

        # -------- Normal + Decay --------
        self.student_risk_index[student_id] *= 0.98
        self.event_counter[student_id]["Normal"] += 1
        self.reason_memory[student_id] = "Stable"

        return 0.05, "Normal", "Stable"

    # ================= ESCALATION =================

    def escalate(self, student_id, base_probability):

        probability = min(base_probability, 1.0)

        # Increase cumulative SRI
        decay_factor = 0.98
        self.student_risk_index[student_id] *= decay_factor

        escalation_weight = 1 + (self.escalation_level[student_id] * 0.2)

        self.student_risk_index[student_id] += probability * 0.05 * escalation_weight

        if probability > 0.8:
            risk = "High"
            self.event_counter[student_id]["High"] += 1
        elif probability > 0.5:
            risk = "Moderate"
            self.event_counter[student_id]["Moderate"] += 1
        else:
            risk = "Normal"
            self.event_counter[student_id]["Normal"] += 1

        return probability, risk, self.reason_memory[student_id]

    # ================= SRI =================

    def get_student_risk_index(self, student_id):
        return round(self.student_risk_index[student_id], 3)

    # ================= FINAL REPORT =================

    def generate_exam_report(self, student_id):

        sri = self.student_risk_index[student_id]
        high = self.event_counter[student_id]["High"]
        moderate = self.event_counter[student_id]["Moderate"]
        normal = self.event_counter[student_id]["Normal"]

        total = high + moderate + normal
        stability_index = round((normal / total) * 100, 2) if total > 0 else 100

        if sri > 1.5:
            risk_level = "Critical"
            recommendation = "Manual review required"
        elif sri > 0.8:
            risk_level = "High"
            recommendation = "Flag for observation"
        elif sri > 0.4:
            risk_level = "Moderate"
            recommendation = "Monitor closely"
        else:
            risk_level = "Low"
            recommendation = "No action required"

        return {
            "student_id": student_id,
            "exam_behavior_summary": {
                "final_sri": round(sri, 3),
                "risk_level": risk_level,
                "behavioral_pattern": self.reason_memory[student_id],
                "high_risk_episodes": high,
                "moderate_risk_episodes": moderate,
                "stability_index": stability_index,
                "system_recommendation": recommendation
            }
        }