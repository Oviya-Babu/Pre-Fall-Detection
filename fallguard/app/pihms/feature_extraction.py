"""
Feature Extraction — Biomechanical feature computation from skeleton keypoints.
CPU-only, zero models. Target latency: <5ms per frame.
Computes ~50 features: joint angles, velocities, gait params, postural metrics.
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# MoveNet keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at joint b formed by segments a-b and b-c (degrees)."""
    return _angle_between(a[:2] - b[:2], c[:2] - b[:2])


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint of two keypoints (using y,x only)."""
    return (a[:2] + b[:2]) / 2.0


class FeatureExtractor:
    """
    Extracts ~50 biomechanical features per frame from 17 MoveNet keypoints.
    Maintains a rolling window for velocity/acceleration computation.
    """

    def __init__(self, window_size: int = 360, fps: int = 30,
                 min_confidence: float = 0.3):
        self.window_size = window_size
        self.fps = fps
        self.min_conf = min_confidence
        self.dt = 1.0 / fps

        # Rolling buffers (pre-allocated deques)
        self._kp_history: deque = deque(maxlen=window_size)
        self._com_history: deque = deque(maxlen=window_size)
        self._ankle_l_history: deque = deque(maxlen=window_size)
        self._ankle_r_history: deque = deque(maxlen=window_size)
        self._frame_count = 0

    def extract(self, keypoints: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Extract all features from a single frame of keypoints.

        Args:
            keypoints: (17, 3) array [y, x, confidence] in normalized coords

        Returns:
            Dict of ~50 named features, or None if keypoints invalid
        """
        if keypoints is None or keypoints.shape != (17, 3):
            return None

        kp = keypoints
        self._kp_history.append(kp.copy())
        self._frame_count += 1

        features: Dict[str, float] = {}

        # ── Joint Angles ────────────────────────────────────────────────
        features['angle_left_elbow'] = _joint_angle(kp[L_SHOULDER], kp[L_ELBOW], kp[L_WRIST])
        features['angle_right_elbow'] = _joint_angle(kp[R_SHOULDER], kp[R_ELBOW], kp[R_WRIST])
        features['angle_left_shoulder'] = _joint_angle(kp[L_HIP], kp[L_SHOULDER], kp[L_ELBOW])
        features['angle_right_shoulder'] = _joint_angle(kp[R_HIP], kp[R_SHOULDER], kp[R_ELBOW])
        features['angle_left_hip'] = _joint_angle(kp[L_SHOULDER], kp[L_HIP], kp[L_KNEE])
        features['angle_right_hip'] = _joint_angle(kp[R_SHOULDER], kp[R_HIP], kp[R_KNEE])
        features['angle_left_knee'] = _joint_angle(kp[L_HIP], kp[L_KNEE], kp[L_ANKLE])
        features['angle_right_knee'] = _joint_angle(kp[R_HIP], kp[R_KNEE], kp[R_ANKLE])

        # ── Spine / Trunk ───────────────────────────────────────────────
        shoulder_mid = _midpoint(kp[L_SHOULDER], kp[R_SHOULDER])
        hip_mid = _midpoint(kp[L_HIP], kp[R_HIP])
        spine_vec = shoulder_mid - hip_mid
        vertical = np.array([-1.0, 0.0])  # upward in image coords
        features['trunk_lean_angle'] = _angle_between(spine_vec, vertical)
        features['spine_length'] = float(np.linalg.norm(spine_vec))

        # ── Center of Mass (CoM) ────────────────────────────────────────
        # Approximate CoM as weighted average of torso keypoints
        com = (shoulder_mid * 0.3 + hip_mid * 0.7)  # hip-weighted
        self._com_history.append(com.copy())
        features['com_x'] = float(com[1])
        features['com_y'] = float(com[0])

        # ── Base of Support ─────────────────────────────────────────────
        ankle_mid = _midpoint(kp[L_ANKLE], kp[R_ANKLE])
        features['base_of_support_width'] = float(abs(kp[L_ANKLE][1] - kp[R_ANKLE][1]))
        features['com_over_bos_x'] = float(abs(com[1] - ankle_mid[1]))

        # ── Body Height ─────────────────────────────────────────────────
        head_y = min(kp[NOSE][0], kp[L_EAR][0], kp[R_EAR][0])
        ankle_y = max(kp[L_ANKLE][0], kp[R_ANKLE][0])
        features['body_height'] = float(ankle_y - head_y)

        # ── Ankle Positions ─────────────────────────────────────────────
        self._ankle_l_history.append(kp[L_ANKLE][:2].copy())
        self._ankle_r_history.append(kp[R_ANKLE][:2].copy())

        # ── Velocities (frame-to-frame deltas) ─────────────────────────
        if len(self._kp_history) >= 2:
            prev = self._kp_history[-2]
            # Hip velocity
            hip_mid_prev = _midpoint(prev[L_HIP], prev[R_HIP])
            hip_vel = (hip_mid - hip_mid_prev) / self.dt
            features['hip_velocity_x'] = float(hip_vel[1])
            features['hip_velocity_y'] = float(hip_vel[0])
            features['hip_speed'] = float(np.linalg.norm(hip_vel))

            # CoM velocity
            if len(self._com_history) >= 2:
                com_prev = self._com_history[-2]
                com_vel = (com - com_prev) / self.dt
                features['com_velocity_x'] = float(com_vel[1])
                features['com_velocity_y'] = float(com_vel[0])
                features['com_speed'] = float(np.linalg.norm(com_vel))

            # Ankle velocities
            if len(self._ankle_l_history) >= 2:
                lv = (self._ankle_l_history[-1] - self._ankle_l_history[-2]) / self.dt
                rv = (self._ankle_r_history[-1] - self._ankle_r_history[-2]) / self.dt
                features['left_ankle_speed'] = float(np.linalg.norm(lv))
                features['right_ankle_speed'] = float(np.linalg.norm(rv))

            # Joint angular velocities (degrees/sec)
            features['left_knee_angular_vel'] = (
                features['angle_left_knee'] -
                _joint_angle(prev[L_HIP], prev[L_KNEE], prev[L_ANKLE])
            ) / self.dt
            features['right_knee_angular_vel'] = (
                features['angle_right_knee'] -
                _joint_angle(prev[R_HIP], prev[R_KNEE], prev[R_ANKLE])
            ) / self.dt
            features['left_hip_angular_vel'] = (
                features['angle_left_hip'] -
                _joint_angle(prev[L_SHOULDER], prev[L_HIP], prev[L_KNEE])
            ) / self.dt
            features['right_hip_angular_vel'] = (
                features['angle_right_hip'] -
                _joint_angle(prev[R_SHOULDER], prev[R_HIP], prev[R_KNEE])
            ) / self.dt
        else:
            for key in ['hip_velocity_x', 'hip_velocity_y', 'hip_speed',
                         'com_velocity_x', 'com_velocity_y', 'com_speed',
                         'left_ankle_speed', 'right_ankle_speed',
                         'left_knee_angular_vel', 'right_knee_angular_vel',
                         'left_hip_angular_vel', 'right_hip_angular_vel']:
                features[key] = 0.0

        # ── Accelerations ──────────────────────────────────────────────
        if len(self._kp_history) >= 3:
            prev2 = self._kp_history[-3]
            prev1 = self._kp_history[-2]
            hip_mid_p2 = _midpoint(prev2[L_HIP], prev2[R_HIP])
            hip_mid_p1 = _midpoint(prev1[L_HIP], prev1[R_HIP])
            v1 = (hip_mid_p1 - hip_mid_p2) / self.dt
            v2 = (hip_mid - hip_mid_prev) / self.dt
            acc = (v2 - v1) / self.dt
            features['hip_acceleration_x'] = float(acc[1])
            features['hip_acceleration_y'] = float(acc[0])
            features['hip_acceleration_mag'] = float(np.linalg.norm(acc))
        else:
            features['hip_acceleration_x'] = 0.0
            features['hip_acceleration_y'] = 0.0
            features['hip_acceleration_mag'] = 0.0

        # ── Gait Parameters ────────────────────────────────────────────
        if len(self._ankle_l_history) >= 10:
            recent_l = np.array(list(self._ankle_l_history)[-10:])
            recent_r = np.array(list(self._ankle_r_history)[-10:])

            # Step width: lateral distance between ankles (x-axis)
            step_widths = np.abs(recent_l[:, 1] - recent_r[:, 1])
            features['step_width_mean'] = float(np.mean(step_widths))
            features['step_width_std'] = float(np.std(step_widths))

            # Ankle vertical oscillation (proxy for step detection)
            l_y_std = float(np.std(recent_l[:, 0]))
            r_y_std = float(np.std(recent_r[:, 0]))
            features['left_ankle_y_oscillation'] = l_y_std
            features['right_ankle_y_oscillation'] = r_y_std

            # Gait symmetry: correlation between left and right ankle vertical motion
            if l_y_std > 1e-6 and r_y_std > 1e-6:
                l_norm = (recent_l[:, 0] - np.mean(recent_l[:, 0])) / l_y_std
                r_norm = (recent_r[:, 0] - np.mean(recent_r[:, 0])) / r_y_std
                features['gait_symmetry'] = float(np.mean(l_norm * r_norm))
            else:
                features['gait_symmetry'] = 0.0
        else:
            features['step_width_mean'] = 0.0
            features['step_width_std'] = 0.0
            features['left_ankle_y_oscillation'] = 0.0
            features['right_ankle_y_oscillation'] = 0.0
            features['gait_symmetry'] = 0.0

        # ── CoM Sway ───────────────────────────────────────────────────
        if len(self._com_history) >= 10:
            recent_com = np.array(list(self._com_history)[-10:])
            features['sway_amplitude_x'] = float(np.std(recent_com[:, 1]))
            features['sway_amplitude_y'] = float(np.std(recent_com[:, 0]))
            features['sway_path_length'] = float(
                np.sum(np.linalg.norm(np.diff(recent_com, axis=0), axis=1))
            )
        else:
            features['sway_amplitude_x'] = 0.0
            features['sway_amplitude_y'] = 0.0
            features['sway_path_length'] = 0.0

        # ── Arm Positions (for eating/reaching detection) ──────────────
        features['left_wrist_mouth_dist'] = float(
            np.linalg.norm(kp[L_WRIST][:2] - kp[NOSE][:2]))
        features['right_wrist_mouth_dist'] = float(
            np.linalg.norm(kp[R_WRIST][:2] - kp[NOSE][:2]))
        features['left_arm_elevation'] = float(
            kp[L_SHOULDER][0] - kp[L_WRIST][0])  # positive = hand above shoulder
        features['right_arm_elevation'] = float(
            kp[R_SHOULDER][0] - kp[R_WRIST][0])

        # ── Activity State (heuristic classification) ──────────────────
        features['activity_state'] = self._classify_activity(features, kp)

        # ── Confidence Metrics ──────────────────────────────────────────
        features['avg_confidence'] = float(np.mean(kp[:, 2]))
        features['min_confidence'] = float(np.min(kp[:, 2]))
        features['frame_number'] = float(self._frame_count)

        return features

    def _classify_activity(self, features: Dict, kp: np.ndarray) -> float:
        """
        Heuristic activity state classification.
        Returns: 0=LYING, 1=SITTING, 2=STANDING_IDLE, 3=WALKING,
                 4=REACHING, 5=TRANSFERRING
        """
        body_height = features.get('body_height', 0.0)
        trunk_angle = features.get('trunk_lean_angle', 0.0)
        hip_speed = features.get('hip_speed', 0.0)
        knee_angle_l = features.get('angle_left_knee', 180.0)
        knee_angle_r = features.get('angle_right_knee', 180.0)
        avg_knee = (knee_angle_l + knee_angle_r) / 2.0

        # LYING: trunk nearly horizontal
        if trunk_angle > 60:
            return 0.0  # LYING_DOWN

        # SITTING: knees bent, not much height
        if avg_knee < 130 and body_height < 0.45:
            return 1.0  # SITTING

        # WALKING: vertical posture + motion
        if hip_speed > 0.5 and trunk_angle < 30:
            return 3.0  # WALKING

        # REACHING: arm extended
        wrist_dist = min(
            features.get('left_wrist_mouth_dist', 1.0),
            features.get('right_wrist_mouth_dist', 1.0))
        arm_elev = max(
            features.get('left_arm_elevation', 0.0),
            features.get('right_arm_elevation', 0.0))
        if arm_elev > 0.1:
            return 4.0  # REACHING

        # STANDING_IDLE
        if trunk_angle < 20 and hip_speed < 0.3:
            return 2.0  # STANDING_IDLE

        # Default: STANDING_IDLE
        return 2.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self):
        """Reset all history buffers."""
        self._kp_history.clear()
        self._com_history.clear()
        self._ankle_l_history.clear()
        self._ankle_r_history.clear()
        self._frame_count = 0
