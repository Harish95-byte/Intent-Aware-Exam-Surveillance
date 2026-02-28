import json
from datetime import datetime
from collections import defaultdict

class EventLogger:

    def __init__(self):
        self.event_store = defaultdict(list)

    def log_event(self, student_id, risk, reason, sri):

        event = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk": risk,
            "reason": reason,
            "sri_at_event": round(sri, 3)
        }

        self.event_store[student_id].append(event)

    def get_student_events(self, student_id):
        return self.event_store.get(student_id, [])