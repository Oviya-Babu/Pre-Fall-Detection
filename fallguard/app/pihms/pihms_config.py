"""
PIHMS v2.0 Unified Configuration
=================================
Central configuration for all PIHMS modules.
Extends the existing fallguard config.py without modification.

All storage limits, thresholds, and feature toggles are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ─── Storage Allocation ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StorageConfig:
    """Storage budget for 4GB SSD constraint."""
    max_storage_gb: float = 4.0
    safety_buffer_gb: float = 0.2
    usable_gb: float = 3.8

    # Allocation in MB
    models_mb: int = 800
    active_session_mb: int = 800
    historical_compressed_mb: int = 2100
    system_overhead_mb: int = 100

    # Retention windows
    skeleton_retention_days: int = 7
    summary_retention_days: int = 7
    alert_retention_days: int = 30
    medication_retention_days: int = 7
    exercise_retention_days: int = 7

    # Compression
    gzip_level: int = 6  # Balance of speed vs compression ratio
    delta_encoding_keyframe_interval: int = 120  # Full keyframe every 120 frames (4s @ 30FPS)

    # Paths
    db_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'data', 'pihms.db')
    data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', '..', 'data')


# ─── Feature Extraction ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration for biomechanical feature extraction."""
    # Rolling window for temporal features (12 seconds @ 30 FPS)
    window_size: int = 360
    # Minimum keypoint confidence to use
    min_keypoint_confidence: float = 0.3
    # Frame rate (used for velocity/acceleration calculations)
    fps: int = 30


# ─── Pre-Fall Detection ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreFallConfig:
    """Configuration for predictive pre-fall detection."""
    # Gait instability
    gait_baseline_frames: int = 900   # 30 seconds @ 30 FPS
    gait_deviation_window: int = 90   # 3-second window
    gait_deviation_threshold: float = 0.30  # 30% deviation triggers flag

    # CoM oscillation
    com_window_seconds: float = 12.0  # 12-second FFT window
    com_sway_freq_normal_min: float = 0.5   # Hz
    com_sway_freq_normal_max: float = 2.0   # Hz
    com_sway_freq_shift_threshold: float = 0.5  # Hz shift from baseline
    com_sway_amplitude_multiplier: float = 2.0  # 2× baseline triggers flag

    # Repeated imbalance
    near_fall_ankle_velocity: float = 150.0   # °/sec
    near_fall_knee_velocity: float = 100.0    # °/sec
    near_fall_hip_acceleration: float = 2.0   # g-force
    near_fall_window_frames: int = 9000       # 5 minutes @ 30 FPS
    near_fall_count_threshold: int = 3        # 3+ near-falls triggers flag

    # Risk score weights
    weight_instability: float = 0.40
    weight_sway: float = 0.35
    weight_repeated_imbalance: float = 0.15
    weight_trend_degradation: float = 0.10

    # Alert thresholds
    yellow_threshold: int = 70
    escalated_yellow_threshold: int = 80
    red_threshold: int = 90

    # Temporal validation: signal must persist for N seconds
    temporal_validation_seconds: float = 2.0


# ─── Activity Monitoring ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActivityConfig:
    """Configuration for daily activity monitoring."""
    # State machine hysteresis: require N consecutive frames before transition
    state_hysteresis_frames: int = 3

    # Posture classification thresholds
    lying_head_below_hip_threshold: float = 0.05   # normalized y difference
    sitting_body_height_ratio: float = 0.55         # head-to-ankle height ratio
    walking_motion_threshold: float = 0.005         # cyclic limb motion amplitude

    # Anomaly detection
    anomaly_learning_days: int = 7   # First week is learning phase
    anomaly_kl_threshold: float = 0.3  # KL divergence threshold for anomaly flag
    bedbound_alert_hours: int = 18   # Alert if lying > 18 hours/day

    # Summary generation
    summary_hour_utc: int = 0   # Midnight UTC for daily summary


# ─── Eating Detection ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EatingConfig:
    """Configuration for eating activity detection."""
    # Pose-based detection thresholds
    hand_mouth_distance_cm: float = 20.0
    hand_mouth_distance_normalized: float = 0.08  # ~20cm in normalized coords
    arm_elevation_threshold: float = -0.05         # shoulder_y - torso_y
    eating_oscillation_min_hz: float = 0.5
    eating_oscillation_max_hz: float = 2.0
    min_eating_duration_frames: int = 60   # 2 seconds @ 30 FPS

    # Meal pattern detection
    meal_learning_days: int = 7
    meal_window_tolerance_hours: float = 1.0
    reduced_intake_ratio: float = 0.7  # Flag if < 70% baseline


# ─── Yoga Guidance ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class YogaConfig:
    """Configuration for yoga/exercise guidance."""
    templates_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'poses', 'yoga_templates.json')
    # Accuracy thresholds
    perfect_accuracy: float = 90.0
    acceptable_accuracy: float = 75.0
    # Session tracking
    min_session_poses: int = 3   # Minimum poses for a valid session


# ─── Medication Alerts ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MedicationConfig:
    """Configuration for medication reminders."""
    check_interval_seconds: int = 60  # Check every minute
    default_reminder_minutes_before: int = 5
    missed_dose_threshold_minutes: int = 60  # Missed if > 1 hour late
    low_adherence_threshold: float = 0.80  # 80% adherence = alert


# ─── Risk Aggregation & Alerts ──────────────────────────────────────────────────

@dataclass(frozen=True)
class AlertConfig:
    """Configuration for risk aggregation and intelligent alerts."""
    # Risk weights
    weight_pre_fall: float = 0.60
    weight_activity_anomaly: float = 0.20
    weight_wellness: float = 0.10
    weight_trend_degradation: float = 0.10

    # Severity thresholds
    critical_threshold: int = 90     # CRITICAL: <1s latency
    high_threshold: int = 75          # RED: 5-10s latency
    medium_threshold: int = 50        # YELLOW: 30-60s latency
    # Below 50: INSIGHT (batched 24h)

    # Deduplication
    dedup_window_seconds: int = 300   # 5-minute suppression window
    false_positive_threshold_24h: int = 3  # Increase threshold after 3 FP
    threshold_increase_percent: float = 5.0
    max_alerts_per_patient_per_day: int = 10

    # Staff acknowledgment suppression
    ack_suppression_minutes: int = 10

    # MQTT topics (extend existing)
    mqtt_topic_pre_fall: str = "fallguard/alert/pre_fall"
    mqtt_topic_activity: str = "fallguard/alert/activity_anomaly"
    mqtt_topic_wellness: str = "fallguard/alert/wellness"
    mqtt_topic_medication: str = "fallguard/alert/medication"
    mqtt_topic_insight: str = "fallguard/insight"


# ─── MoveNet Keypoint Indices ───────────────────────────────────────────────────

class Keypoint:
    """MoveNet Lightning keypoint indices (17 keypoints)."""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    NUM_KEYPOINTS = 17


# ─── Singleton Instances ────────────────────────────────────────────────────────

STORAGE = StorageConfig()
FEATURES = FeatureExtractionConfig()
PRE_FALL = PreFallConfig()
ACTIVITY = ActivityConfig()
EATING = EatingConfig()
YOGA = YogaConfig()
MEDICATION = MedicationConfig()
ALERT = AlertConfig()
