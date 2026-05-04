"""
Pre-Fall Detection — Predictive fall risk scoring via rule-based biomechanics.
Zero new models. CPU-only. Target: 80%+ sensitivity, <8% false positives.

Tuned for both:
  - Real MoveNet keypoints (high precision, low noise)
  - Anatomical skeleton estimator (proportional estimates, moderate noise)
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PreFallDetector:
    """
    Predictive pre-fall detection combining:
    1. Gait instability tracking
    2. Center-of-Mass oscillation analysis
    3. Repeated imbalance detection
    4. Postural instability (trunk lean + CoM over BoS)
    5. Temporal trend analysis
    """

    def __init__(self, fps: int = 30):
        self.fps = fps

        # ── Gait Instability (Component 3.1) ────────────────────────
        self._gait_baseline_values: deque = deque(maxlen=900)  # 30s baseline
        self._gait_baseline_set = False
        self._gait_baseline_mean = 0.0
        self._gait_baseline_std = 0.0
        self._gait_deviation_window: deque = deque(maxlen=90)  # 3s window

        # ── CoM Oscillation (Component 3.2) ─────────────────────────
        self._com_trajectory: deque = deque(maxlen=int(12 * fps))  # 12s window
        self._com_baseline_amp = None
        self._com_baseline_freq = None
        self._com_baseline_samples: deque = deque(maxlen=int(30 * fps))

        # ── Repeated Imbalance (Component 3.3) ──────────────────────
        self._near_fall_events: deque = deque(maxlen=int(5 * 60 * fps))
        self._near_fall_timestamps: deque = deque(maxlen=100)

        # ── Postural Instability (Component 3.4 — NEW) ──────────────
        self._trunk_lean_history: deque = deque(maxlen=int(5 * fps))  # 5s
        self._com_bos_history: deque = deque(maxlen=int(5 * fps))

        # ── Temporal Trend (Component 3.5) ──────────────────────────
        self._risk_history: deque = deque(maxlen=int(60 * fps))  # 60s trend
        self._instability_flag = 0.0
        self._sway_ratio = 0.0
        self._imbalance_count = 0
        self._postural_score = 0.0
        self._trend_slope = 0.0

        # ── Alert State ─────────────────────────────────────────────
        self._temporal_validation_frames = int(0.5 * fps)
        self._above_threshold_count = 0

        # ── Adaptive sensitivity (for anatomical skeleton) ──────────
        # After baseline period, if all signals remain near zero,
        # sensitivity is gradually increased.
        self._low_signal_counter = 0
        self._sensitivity_boost = 1.0  # multiplier, grows if signals are flat

    def update(self, features: Dict[str, float]) -> Dict:
        """
        Process one frame of features and return pre-fall risk assessment.

        Args:
            features: Dict from FeatureExtractor.extract()

        Returns:
            Dict with risk_score (0-100), biomarkers, latency estimate
        """
        if features is None:
            return {'risk_score': 0, 'biomarkers': [], 'latency_sec': 0,
                    'risk_validated': False, 'components': {}}

        biomarkers = []

        # ── 1. Gait Instability ─────────────────────────────────────
        gait_score = self._assess_gait_instability(features)
        if gait_score > 0.25:
            biomarkers.append('GAIT_INSTABILITY')
        self._instability_flag = gait_score

        # ── 2. CoM Sway ────────────────────────────────────────────
        sway_score = self._assess_com_sway(features)
        if sway_score > 0.25:
            biomarkers.append('EXCESSIVE_SWAY')
        self._sway_ratio = sway_score

        # ── 3. Repeated Imbalance ──────────────────────────────────
        imb_score = self._assess_repeated_imbalance(features)
        if imb_score > 0.35:
            biomarkers.append('REPEATED_IMBALANCE')
        self._imbalance_count = imb_score

        # ── 4. Postural Instability (trunk lean + CoM over BoS) ────
        postural_score = self._assess_postural_instability(features)
        if postural_score > 0.25:
            biomarkers.append('POSTURAL_INSTABILITY')
        self._postural_score = postural_score

        # ── 5. Temporal Trend ──────────────────────────────────────
        self._trend_slope = self._assess_trend()

        # ── Adaptive Sensitivity ───────────────────────────────────
        self._update_sensitivity(gait_score, sway_score, imb_score, postural_score)

        # ── Composite Risk Score ───────────────────────────────────
        raw_risk = (
            0.30 * self._instability_flag +
            0.25 * self._sway_ratio +
            0.10 * min(1.0, self._imbalance_count) +
            0.25 * self._postural_score +
            0.10 * max(0, self._trend_slope)
        ) * 100 * self._sensitivity_boost

        risk_score = int(np.clip(raw_risk, 0, 100))
        self._risk_history.append(risk_score)

        # ── Temporal Validation (require persistent signal) ────────
        if risk_score >= 60:
            self._above_threshold_count += 1
        else:
            self._above_threshold_count = max(0, self._above_threshold_count - 1)

        validated = self._above_threshold_count >= self._temporal_validation_frames

        # Estimate latency to potential fall
        latency_sec = 0.0
        if risk_score >= 60 and validated:
            latency_sec = max(1.0, 10.0 - (risk_score - 60) * 0.25)

        return {
            'risk_score': risk_score,
            'risk_validated': validated,
            'biomarkers': biomarkers,
            'latency_sec': round(latency_sec, 1),
            'components': {
                'gait_instability': round(self._instability_flag, 3),
                'sway_ratio': round(self._sway_ratio, 3),
                'imbalance_score': round(self._imbalance_count, 3),
                'postural_instability': round(self._postural_score, 3),
                'trend_slope': round(self._trend_slope, 3),
                'sensitivity_boost': round(self._sensitivity_boost, 2),
            }
        }

    def _assess_gait_instability(self, f: Dict) -> float:
        """Assess gait instability from step width, symmetry, and hip speed variance."""
        step_std = f.get('step_width_std', 0.0)
        symmetry = f.get('gait_symmetry', 0.0)
        hip_speed = f.get('hip_speed', 0.0)

        # Composite metric: step variability + asymmetry + speed irregularity
        metric = step_std + max(0, 0.5 + symmetry)

        # Build baseline
        if not self._gait_baseline_set:
            self._gait_baseline_values.append(metric)
            if len(self._gait_baseline_values) >= 150:  # 5 seconds (faster baseline)
                vals = np.array(self._gait_baseline_values)
                self._gait_baseline_mean = float(np.mean(vals))
                self._gait_baseline_std = float(np.std(vals)) + 1e-6
                self._gait_baseline_set = True
            return 0.0

        # Deviation from baseline
        deviation = (metric - self._gait_baseline_mean) / self._gait_baseline_std
        self._gait_deviation_window.append(deviation)

        if len(self._gait_deviation_window) >= 15:  # 0.5 seconds (faster response)
            recent_dev = np.array(self._gait_deviation_window)
            if len(recent_dev) >= 60:
                trend = float(np.mean(recent_dev[-30:]) - np.mean(recent_dev[:30]))
            else:
                trend = float(np.mean(recent_dev))
            # Normalize to [0, 1] — lower divisor = more sensitive
            score = np.clip(trend / 2.0, 0.0, 1.0)
            return float(score)
        return 0.0

    def _assess_com_sway(self, f: Dict) -> float:
        """Assess Center-of-Mass sway amplitude."""
        com_x = f.get('com_x', 0.0)
        self._com_trajectory.append(com_x)

        if len(self._com_trajectory) < 30:  # Need 1 second minimum (was 2s)
            self._com_baseline_samples.append(com_x)
            return 0.0

        current_amp = float(np.std(list(self._com_trajectory)[-30:]))

        # Establish baseline
        if self._com_baseline_amp is None:
            if len(self._com_baseline_samples) >= 150:  # 5 seconds (was 10s)
                self._com_baseline_amp = float(np.std(
                    list(self._com_baseline_samples))) + 1e-6
            else:
                self._com_baseline_samples.append(com_x)
                return 0.0

        # Sway ratio — more sensitive scaling
        ratio = current_amp / self._com_baseline_amp
        score = np.clip((ratio - 1.0) / 1.5, 0.0, 1.0)  # >1.5× = 0.33, >2.5× = 1.0
        return float(score)

    def _assess_repeated_imbalance(self, f: Dict) -> float:
        """Detect repeated near-fall recovery events."""
        ankle_speed_l = f.get('left_ankle_speed', 0.0)
        ankle_speed_r = f.get('right_ankle_speed', 0.0)
        knee_vel_l = abs(f.get('left_knee_angular_vel', 0.0))
        knee_vel_r = abs(f.get('right_knee_angular_vel', 0.0))
        hip_acc = f.get('hip_acceleration_mag', 0.0)

        # Detect near-fall event: rapid corrective movement
        max_ankle = max(ankle_speed_l, ankle_speed_r)
        max_knee_vel = max(knee_vel_l, knee_vel_r)

        # Lowered thresholds for anatomical skeleton (which has smaller values)
        is_near_fall = (
            (max_ankle > 3.0 and max_knee_vel > 60 and hip_acc > 1.2) or
            (max_ankle > 4.0 and hip_acc > 2.0) or
            (max_knee_vel > 100 and hip_acc > 2.0)
        )

        if is_near_fall:
            self._near_fall_timestamps.append(
                f.get('frame_number', 0) / self.fps)

        # Count near-falls in 5-minute window
        if len(self._near_fall_timestamps) >= 2:
            recent = [t for t in self._near_fall_timestamps
                      if t > self._near_fall_timestamps[-1] - 300]  # 5 min
            count = len(recent)
            # 3+ near-falls in 5 min = max risk
            return float(np.clip(count / 3.0, 0.0, 1.0))
        return 0.0

    def _assess_postural_instability(self, f: Dict) -> float:
        """
        Assess postural instability from trunk lean angle and
        Center-of-Mass deviation from Base of Support.
        This component responds to the absolute posture state,
        not just relative changes.
        """
        trunk_lean = f.get('trunk_lean_angle', 0.0)
        com_over_bos = f.get('com_over_bos_x', 0.0)
        body_height = f.get('body_height', 0.5)

        self._trunk_lean_history.append(trunk_lean)
        self._com_bos_history.append(com_over_bos)

        # Trunk lean scoring (degrees)
        # 0-10° = normal, 10-20° = mild concern, 20-35° = warning, >35° = critical
        if trunk_lean > 35:
            trunk_score = 1.0
        elif trunk_lean > 20:
            trunk_score = (trunk_lean - 20) / 15.0
        elif trunk_lean > 10:
            trunk_score = (trunk_lean - 10) / 20.0  # Mild
        else:
            trunk_score = 0.0

        # CoM deviation from BoS center
        # Normalized by body height: >15% deviation is concerning
        if body_height > 0.01:
            com_deviation_pct = com_over_bos / body_height
        else:
            com_deviation_pct = 0.0

        com_score = np.clip(com_deviation_pct / 0.15, 0.0, 1.0)

        # Trunk lean variability (rapid oscillation = instability)
        trunk_var_score = 0.0
        if len(self._trunk_lean_history) >= 30:
            trunk_std = float(np.std(list(self._trunk_lean_history)[-30:]))
            trunk_var_score = np.clip(trunk_std / 5.0, 0.0, 1.0)

        # Composite: 40% trunk lean + 30% CoM deviation + 30% variability
        score = 0.40 * trunk_score + 0.30 * com_score + 0.30 * trunk_var_score
        return float(np.clip(score, 0.0, 1.0))

    def _assess_trend(self) -> float:
        """Assess risk score trend (increasing = deteriorating)."""
        if len(self._risk_history) < 30:  # Was 60, now faster
            return 0.0
        recent = np.array(list(self._risk_history))
        # Linear regression slope over recent frames
        x = np.arange(len(recent))
        if len(x) < 2:
            return 0.0
        slope = float(np.polyfit(x, recent, 1)[0])
        # Normalize: positive slope = worsening
        return float(np.clip(slope * 10, -1.0, 1.0))

    def _update_sensitivity(self, *scores):
        """
        Adaptive sensitivity: if all signals are persistently near-zero
        for a long time, gradually boost sensitivity so the system can
        detect subtler instability patterns (e.g., anatomical skeleton).
        Caps at 1.5× to prevent false positives.
        """
        all_low = all(s < 0.05 for s in scores)
        if all_low and self._gait_baseline_set:
            self._low_signal_counter += 1
            # After 10 seconds of zero signals, start boosting
            if self._low_signal_counter > self.fps * 10:
                self._sensitivity_boost = min(1.5,
                    1.0 + (self._low_signal_counter - self.fps * 10) / (self.fps * 60))
        else:
            self._low_signal_counter = max(0, self._low_signal_counter - 5)
            if self._sensitivity_boost > 1.0:
                self._sensitivity_boost = max(1.0, self._sensitivity_boost - 0.01)

    def get_alert_level(self, risk_score: int) -> str:
        """Map risk score to alert level string."""
        if risk_score >= 85:
            return 'RED'
        elif risk_score >= 70:
            return 'ESCALATED_YELLOW'
        elif risk_score >= 55:
            return 'YELLOW'
        return 'GREEN'
