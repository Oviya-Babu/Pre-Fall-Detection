"""
Activity Monitoring — State machine for daily activity tracking + anomaly detection.
CPU-only, rule-based. Target: 85%+ activity classification accuracy.
"""

import numpy as np
import time
import json
import logging
from collections import deque, defaultdict
from typing import Dict, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Activity state codes
LYING = 0
SITTING = 1
STANDING_IDLE = 2
WALKING = 3
REACHING = 4
TRANSFERRING = 5

STATE_NAMES = {
    LYING: 'LYING', SITTING: 'SITTING', STANDING_IDLE: 'STANDING_IDLE',
    WALKING: 'WALKING', REACHING: 'REACHING', TRANSFERRING: 'TRANSFERRING'
}


class ActivityMonitor:
    """
    Activity state machine with anomaly detection.
    Tracks: time in each state, transitions, hourly distribution.
    """

    def __init__(self, fps: int = 30, hysteresis_frames: int = 3):
        self.fps = fps
        self.hysteresis = hysteresis_frames

        # Current state tracking
        self._current_state = STANDING_IDLE
        self._candidate_state = STANDING_IDLE
        self._candidate_count = 0
        self._state_start_time = time.time()

        # Daily accumulation
        self._state_durations: Dict[int, float] = defaultdict(float)  # seconds
        self._transitions_count = 0
        self._hourly_activity: Dict[int, Dict[int, float]] = defaultdict(
            lambda: defaultdict(float))  # hour -> state -> seconds

        # Motion intensity
        self._motion_intensities: deque = deque(maxlen=fps * 60)  # 1 min buffer

        # Anomaly detection
        self._baseline_loaded = False
        self._hourly_baseline: Dict[int, Dict[str, float]] = {}
        self._learning_mode = True
        self._learning_data: Dict[int, List[Dict[str, float]]] = defaultdict(list)
        self._day_count = 0

        # Frame counter
        self._frame_count = 0

    def update(self, features: Dict[str, float]) -> Dict:
        """
        Process one frame and update activity state.

        Args:
            features: Dict from FeatureExtractor.extract()

        Returns:
            Dict with current state, duration, anomaly_score
        """
        if features is None:
            return self._make_result()

        self._frame_count += 1

        # Classify activity from features
        raw_state = int(features.get('activity_state', STANDING_IDLE))

        # Apply hysteresis: require N consecutive frames in new state
        if raw_state != self._current_state:
            if raw_state == self._candidate_state:
                self._candidate_count += 1
            else:
                self._candidate_state = raw_state
                self._candidate_count = 1

            if self._candidate_count >= self.hysteresis:
                self._transition_to(raw_state)
        else:
            self._candidate_state = raw_state
            self._candidate_count = 0

        # Accumulate duration
        now = time.time()
        elapsed = min(1.0 / self.fps, now - self._state_start_time)
        self._state_durations[self._current_state] += elapsed
        self._state_start_time = now

        # Track hourly distribution
        hour = datetime.now(timezone.utc).hour
        self._hourly_activity[hour][self._current_state] += elapsed

        # Motion intensity
        hip_speed = features.get('hip_speed', 0.0)
        self._motion_intensities.append(hip_speed)

        return self._make_result()

    def _transition_to(self, new_state: int):
        """Handle state transition."""
        old_name = STATE_NAMES.get(self._current_state, '?')
        new_name = STATE_NAMES.get(new_state, '?')
        logger.debug(f"Activity transition: {old_name} → {new_name}")
        self._current_state = new_state
        self._transitions_count += 1
        self._state_start_time = time.time()

    def _make_result(self) -> Dict:
        """Build result dict for current frame."""
        intensity = float(np.mean(self._motion_intensities)) if self._motion_intensities else 0.0
        return {
            'current_state': self._current_state,
            'state_name': STATE_NAMES.get(self._current_state, 'UNKNOWN'),
            'duration_in_state_sec': time.time() - self._state_start_time,
            'transitions_today': self._transitions_count,
            'motion_intensity': round(intensity, 4),
            'anomaly_score': self.compute_anomaly_score(),
        }

    def compute_anomaly_score(self) -> float:
        """
        Compute anomaly score via KL divergence from baseline.
        Returns 0.0 if in learning mode or no baseline.
        """
        if self._learning_mode or not self._hourly_baseline:
            return 0.0

        hour = datetime.now(timezone.utc).hour
        baseline = self._hourly_baseline.get(hour)
        if not baseline:
            return 0.0

        # Current distribution for this hour
        current_dist = self._hourly_activity.get(hour, {})
        total_current = sum(current_dist.values()) + 1e-6

        # KL divergence: D_KL(current || baseline)
        kl = 0.0
        for state in range(6):
            p = current_dist.get(state, 0.0) / total_current
            q = baseline.get(str(state), 1e-6)
            if p > 1e-6:
                kl += p * np.log(max(p, 1e-10) / max(q, 1e-10))

        return float(np.clip(kl, 0.0, 2.0))

    def set_baseline(self, hourly_baseline: Dict[int, Dict[str, float]]):
        """Load a learned baseline for anomaly detection."""
        self._hourly_baseline = hourly_baseline
        self._baseline_loaded = True
        self._learning_mode = False
        logger.info("Activity baseline loaded for anomaly detection")

    def get_daily_summary(self) -> Dict:
        """Generate daily activity summary."""
        total_sec = sum(self._state_durations.values()) or 1.0
        intensity = float(np.mean(self._motion_intensities)) if self._motion_intensities else 0.0

        # Determine mobility trend
        lying_pct = self._state_durations.get(LYING, 0) / total_sec
        walking_pct = self._state_durations.get(WALKING, 0) / total_sec
        if lying_pct > 0.75:
            trend = 'declining'
        elif walking_pct > 0.3:
            trend = 'active'
        else:
            trend = 'stable'

        summary = {
            'walking_minutes': round(self._state_durations.get(WALKING, 0) / 60, 1),
            'sitting_minutes': round(self._state_durations.get(SITTING, 0) / 60, 1),
            'lying_minutes': round(self._state_durations.get(LYING, 0) / 60, 1),
            'standing_minutes': round(self._state_durations.get(STANDING_IDLE, 0) / 60, 1),
            'transfers_count': self._transitions_count,
            'active_intensity_avg': round(intensity, 4),
            'anomaly_flags': 1 if self.compute_anomaly_score() > 0.3 else 0,
            'mobility_trend': trend,
            'activity_timeline': self._build_timeline(),
        }
        return summary

    def _build_timeline(self) -> List[Dict]:
        """Build hourly activity timeline."""
        timeline = []
        for hour in sorted(self._hourly_activity.keys()):
            hour_data = self._hourly_activity[hour]
            if not hour_data:
                continue
            dominant_state = max(hour_data, key=hour_data.get)
            timeline.append({
                'hour': hour,
                'activity': STATE_NAMES.get(dominant_state, 'UNKNOWN'),
                'duration_min': round(sum(hour_data.values()) / 60, 1),
            })
        return timeline

    def reset_daily(self):
        """Reset daily counters (call at midnight)."""
        self._state_durations = defaultdict(float)
        self._transitions_count = 0
        self._hourly_activity = defaultdict(lambda: defaultdict(float))
        logger.info("Activity monitor daily reset")
