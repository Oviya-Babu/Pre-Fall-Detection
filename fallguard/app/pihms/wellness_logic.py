"""
Wellness Logic — Medication tracking, meal/exercise aggregation, wellness risk.
"""

import time
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class WellnessTracker:
    """
    Aggregates medication adherence, meal patterns, and exercise tracking
    into a unified wellness risk score (0-50).
    """

    def __init__(self):
        # Medication schedules: {med_id: {...schedule...}}
        self._medications: Dict[str, Dict] = {}
        self._medication_log: List[Dict] = []

        # Meal tracking
        self._meal_log: List[Dict] = []
        self._baseline_meal_count = 3  # Expected meals/day

        # Exercise tracking
        self._exercise_sessions: List[Dict] = []
        self._min_sessions_per_week = 2

        # Last reminder check
        self._last_check = 0

    # ── Medication Management ───────────────────────────────────────

    def add_medication(self, med_id: str, name: str,
                       schedule_hours_utc: List[int],
                       dosage: str = '',
                       reminder_minutes_before: int = 5):
        """Register a medication schedule."""
        self._medications[med_id] = {
            'med_id': med_id,
            'name': name,
            'dosage': dosage,
            'schedule_hours_utc': schedule_hours_utc,
            'reminder_minutes_before': reminder_minutes_before,
        }
        logger.info(f"Medication added: {name} ({med_id})")

    def check_medication_reminders(self) -> List[Dict]:
        """Check if any medication reminders are due. Call every ~60 seconds."""
        now = time.time()
        if now - self._last_check < 55:  # Avoid checking too frequently
            return []
        self._last_check = now

        reminders = []
        current_hour = datetime.now(timezone.utc).hour
        current_minute = datetime.now(timezone.utc).minute

        for med_id, med in self._medications.items():
            for sched_hour in med['schedule_hours_utc']:
                reminder_min = med['reminder_minutes_before']
                # Check if we're within the reminder window
                total_min_now = current_hour * 60 + current_minute
                total_min_sched = sched_hour * 60
                diff = total_min_sched - total_min_now

                if 0 <= diff <= reminder_min:
                    reminders.append({
                        'alert_type': 'MEDICATION_REMINDER',
                        'med_id': med_id,
                        'medication_name': med['name'],
                        'dosage': med['dosage'],
                        'scheduled_hour': sched_hour,
                        'action': 'Take now',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                    })
        return reminders

    def log_medication_taken(self, med_id: str,
                             status: str = 'ON_TIME'):
        """Record that a medication was taken."""
        self._medication_log.append({
            'med_id': med_id,
            'timestamp': time.time(),
            'status': status,
        })
        logger.info(f"Medication taken: {med_id} ({status})")

    def get_adherence_7day(self) -> float:
        """Calculate 7-day medication adherence rate."""
        cutoff = time.time() - (7 * 86400)
        recent = [m for m in self._medication_log if m['timestamp'] >= cutoff]
        if not recent:
            return 1.0  # No data = assume compliant

        taken = sum(1 for m in recent if m['status'] in ('ON_TIME', 'LATE'))
        return taken / len(recent) if recent else 1.0

    # ── Meal Tracking ───────────────────────────────────────────────

    def add_meal_log(self, meal_type: str, duration_seconds: float = 0):
        """Record a meal event."""
        self._meal_log.append({
            'meal_type': meal_type,
            'timestamp': time.time(),
            'duration_seconds': duration_seconds,
        })

    def get_meals_today(self) -> int:
        """Count meals logged today."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0).timestamp()
        return sum(1 for m in self._meal_log if m['timestamp'] >= today_start)

    # ── Exercise Tracking ───────────────────────────────────────────

    def add_exercise_session(self, session_data: Dict):
        """Record an exercise session."""
        self._exercise_sessions.append({
            'timestamp': time.time(),
            **session_data,
        })

    def get_exercise_count_7day(self) -> int:
        """Count exercise sessions in last 7 days."""
        cutoff = time.time() - (7 * 86400)
        return sum(1 for s in self._exercise_sessions if s['timestamp'] >= cutoff)

    # ── Wellness Risk Score ─────────────────────────────────────────

    def get_wellness_risk(self) -> float:
        """
        Compute wellness risk score (0-50).
        Higher = worse adherence.
        """
        risk = 0.0

        # Medication adherence
        adherence = self.get_adherence_7day()
        if adherence < 0.80:
            risk += 15  # Low adherence = +15

        # Meal tracking
        meals_today = self.get_meals_today()
        if meals_today < self._baseline_meal_count - 1:
            risk += 10  # Missed meals

        # Exercise
        sessions = self.get_exercise_count_7day()
        if sessions < self._min_sessions_per_week:
            risk += 10  # Insufficient exercise

        # 3+ days without exercise
        if self._exercise_sessions:
            last_exercise = max(s['timestamp'] for s in self._exercise_sessions)
            days_since = (time.time() - last_exercise) / 86400
            if days_since >= 3:
                risk += 15  # 3+ days without exercise

        return min(50.0, risk)

    def get_wellness_summary(self) -> Dict:
        """Get comprehensive wellness summary."""
        return {
            'medication_adherence_7day': round(self.get_adherence_7day(), 2),
            'meals_today': self.get_meals_today(),
            'exercise_sessions_7day': self.get_exercise_count_7day(),
            'wellness_risk_score': self.get_wellness_risk(),
            'medications_registered': len(self._medications),
        }
