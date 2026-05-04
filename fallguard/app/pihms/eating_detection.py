"""
Eating Detection — Pose-based eating inference + manual meal logging support.
"""

import time
import logging
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EatingDetector:
    """
    Detects eating activity via pose heuristics:
    - Hand near mouth + sitting posture + repetitive arm motion
    """

    def __init__(self, fps: int = 30,
                 hand_mouth_threshold: float = 0.08,
                 min_eating_frames: int = 60):
        self.fps = fps
        self.hand_mouth_threshold = hand_mouth_threshold
        self.min_eating_frames = min_eating_frames

        self._eating_frame_count = 0
        self._current_meal_start: Optional[float] = None
        self._meals_today: List[Dict] = []
        self._baseline_meal_times: List[float] = []  # learned expected times
        self._learning_mode = True
        self._learning_days = 0

    def update(self, features: Dict[str, float]) -> Dict:
        """Process one frame for eating detection."""
        if features is None:
            return {'eating_detected': False, 'meal_in_progress': False}

        is_eating_frame = self._check_eating_posture(features)

        if is_eating_frame:
            self._eating_frame_count += 1
            if self._current_meal_start is None:
                self._current_meal_start = time.time()
        else:
            if self._eating_frame_count >= self.min_eating_frames:
                # Meal detected — log it
                duration = time.time() - self._current_meal_start if self._current_meal_start else 0
                self._meals_today.append({
                    'timestamp': self._current_meal_start,
                    'duration_seconds': duration,
                    'source': 'pose_detected',
                })
                logger.info(f"Eating detected: {duration:.0f}s duration")
            self._eating_frame_count = 0
            self._current_meal_start = None

        meal_in_progress = self._eating_frame_count >= self.min_eating_frames

        return {
            'eating_detected': meal_in_progress,
            'meal_in_progress': meal_in_progress,
            'meals_today_count': len(self._meals_today),
            'eating_frame_count': self._eating_frame_count,
        }

    def _check_eating_posture(self, f: Dict) -> bool:
        """Check if current posture looks like eating."""
        l_dist = f.get('left_wrist_mouth_dist', 1.0)
        r_dist = f.get('right_wrist_mouth_dist', 1.0)
        activity = f.get('activity_state', 2)

        hand_near_mouth = min(l_dist, r_dist) < self.hand_mouth_threshold
        is_sitting = activity == 1  # SITTING

        return hand_near_mouth and is_sitting

    def log_manual_meal(self, meal_type: str, notes: str = '') -> Dict:
        """Log a manually reported meal."""
        meal = {
            'timestamp': time.time(),
            'meal_type': meal_type,
            'source': 'manual',
            'notes': notes,
            'duration_seconds': 0,
        }
        self._meals_today.append(meal)
        logger.info(f"Manual meal logged: {meal_type}")
        return meal

    def get_meal_summary(self) -> List[Dict]:
        """Return today's meal log."""
        return list(self._meals_today)

    def get_meal_anomaly_score(self) -> float:
        """Check if meal pattern deviates from baseline."""
        if self._learning_mode or not self._baseline_meal_times:
            return 0.0
        # Count missed meals (not detected within ±1 hour of baseline)
        missed = 0
        for expected_time in self._baseline_meal_times:
            found = any(
                abs(m['timestamp'] - expected_time) < 3600
                for m in self._meals_today
                if m.get('timestamp')
            )
            if not found:
                missed += 1
        return min(1.0, missed / max(len(self._baseline_meal_times), 1))

    def reset_daily(self):
        """Reset daily meal tracking."""
        self._meals_today = []
        self._eating_frame_count = 0
        self._current_meal_start = None
