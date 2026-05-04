#!/usr/bin/env python3
"""
PIHMS v2.0 — Integrated Main Pipeline
=======================================
Extends the existing FallGuard main.py without breaking backward compatibility.
Supports three modes:
  - Full mode:       Camera + real models + PIHMS analytics
  - Legacy mode:     Original FallGuard only (--legacy)
  - Simulation mode: Synthetic skeleton data for testing (--simulate)

Usage:
    python3 pihms_main.py              # Full PIHMS mode (requires camera + models)
    python3 pihms_main.py --legacy     # Original FallGuard mode only
    python3 pihms_main.py --simulate   # Simulation mode (no camera/models needed)
    python3 pihms_main.py --simulate --duration 60  # Run for 60 seconds
"""

import time
import logging
import json
import argparse
import sys
import os
import signal
import numpy as np

# Ensure app directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PIHMS v2.0 modules (always available)
from pihms.feature_extraction import FeatureExtractor
from pihms.pre_fall_detection import PreFallDetector
from pihms.activity_monitoring import ActivityMonitor
from pihms.eating_detection import EatingDetector
from pihms.yoga_guidance import YogaCoach
from pihms.wellness_logic import WellnessTracker
from pihms.risk_aggregator import RiskAggregator
from pihms.alert_manager import AlertManager
from pihms.compress_skeleton import SkeletonCompressor
from pihms.database.pihms_db import PIHMSDatabase
from pihms.storage_validator import StorageValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('PIHMS')

# Graceful shutdown flag
_shutdown = False
def _signal_handler(sig, frame):
    global _shutdown
    _shutdown = True
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─── Simulation Skeleton Generators ────────────────────────────────────

def _sim_standing(noise=0.001):
    """Standing person skeleton."""
    kp = np.array([
        [0.15, 0.50, 0.95], [0.13, 0.48, 0.90], [0.13, 0.52, 0.90],
        [0.14, 0.46, 0.85], [0.14, 0.54, 0.85],
        [0.25, 0.42, 0.92], [0.25, 0.58, 0.92],
        [0.38, 0.40, 0.88], [0.38, 0.60, 0.88],
        [0.48, 0.42, 0.85], [0.48, 0.58, 0.85],
        [0.50, 0.45, 0.93], [0.50, 0.55, 0.93],
        [0.68, 0.44, 0.90], [0.68, 0.56, 0.90],
        [0.85, 0.43, 0.88], [0.85, 0.57, 0.88],
    ], dtype=np.float32)
    kp[:, :2] += np.random.randn(17, 2).astype(np.float32) * noise
    return kp

def _sim_walking(frame_idx, noise=0.002):
    """Walking person with cyclic limb motion."""
    kp = _sim_standing(noise)
    phase = frame_idx / 15.0
    kp[15, 0] += 0.02 * np.sin(phase * 2 * np.pi)
    kp[16, 0] += 0.02 * np.sin(phase * 2 * np.pi + np.pi)
    kp[15, 1] += 0.01 * np.cos(phase * 2 * np.pi)
    kp[16, 1] += 0.01 * np.cos(phase * 2 * np.pi + np.pi)
    return kp

def _sim_unstable(frame_idx, severity=1.0, noise=0.005):
    """Unstable person with sway and erratic movement."""
    kp = _sim_walking(frame_idx, noise)
    sway = severity * 0.04 * np.sin(frame_idx / 5.0)
    kp[:, 1] += sway
    kp[5:7, 0] += severity * 0.03 * np.sin(frame_idx / 3.0)
    kp[13, 0] += severity * 0.02 * np.random.randn()
    kp[14, 0] += severity * 0.02 * np.random.randn()
    return kp

def _sim_sitting(noise=0.001):
    """Sitting person skeleton."""
    kp = np.array([
        [0.25, 0.50, 0.95], [0.23, 0.48, 0.90], [0.23, 0.52, 0.90],
        [0.24, 0.46, 0.85], [0.24, 0.54, 0.85],
        [0.35, 0.42, 0.92], [0.35, 0.58, 0.92],
        [0.45, 0.38, 0.88], [0.45, 0.62, 0.88],
        [0.50, 0.40, 0.85], [0.50, 0.60, 0.85],
        [0.55, 0.45, 0.93], [0.55, 0.55, 0.93],
        [0.55, 0.44, 0.90], [0.55, 0.56, 0.90],
        [0.70, 0.43, 0.88], [0.70, 0.57, 0.88],
    ], dtype=np.float32)
    kp[:, :2] += np.random.randn(17, 2).astype(np.float32) * noise
    return kp


class SimulationScenarioRunner:
    """
    Runs through simulated daily activities for end-to-end testing.

    Timeline (compressed — 1 minute real = 1 hour simulated):
      0-5s:   Standing (morning wake-up)
      5-15s:  Walking (to bathroom)
      15-25s: Standing (bathroom)
      25-35s: Walking (to kitchen)
      35-55s: Sitting (breakfast)
      55-75s: Walking (around house)
      75-85s: Unstable walking (fatigue)
      85-95s: Standing recovery
      95+s:   Repeat with variation
    """

    def __init__(self, fps=30):
        self.fps = fps
        self._frame = 0

    def next_keypoints(self) -> np.ndarray:
        """Generate next frame of simulated skeleton."""
        t = self._frame / self.fps  # Time in seconds
        cycle = t % 95  # 95-second activity cycle

        if cycle < 5:
            kp = _sim_standing(noise=0.001)
            activity = "STANDING"
        elif cycle < 15:
            kp = _sim_walking(self._frame, noise=0.002)
            activity = "WALKING"
        elif cycle < 25:
            kp = _sim_standing(noise=0.001)
            activity = "STANDING"
        elif cycle < 35:
            kp = _sim_walking(self._frame, noise=0.002)
            activity = "WALKING"
        elif cycle < 55:
            kp = _sim_sitting(noise=0.001)
            activity = "SITTING"
        elif cycle < 75:
            kp = _sim_walking(self._frame, noise=0.002)
            activity = "WALKING"
        elif cycle < 85:
            severity = (cycle - 75) / 10.0  # Gradual increase
            kp = _sim_unstable(self._frame, severity=severity, noise=0.005)
            activity = "UNSTABLE"
        else:
            kp = _sim_standing(noise=0.001)
            activity = "RECOVERY"

        self._frame += 1
        return kp, activity


def run_simulation(duration_sec, fps=30):
    """Run the full PIHMS pipeline in simulation mode."""
    logger.info("=" * 60)
    logger.info("  PIHMS v2.0 — SIMULATION MODE")
    logger.info(f"  Duration: {duration_sec} seconds @ {fps} FPS")
    logger.info(f"  Total frames: {duration_sec * fps}")
    logger.info("=" * 60)

    # Initialize all PIHMS components
    feature_extractor = FeatureExtractor(fps=fps)
    pre_fall_detector = PreFallDetector(fps=fps)
    activity_monitor = ActivityMonitor(fps=fps)
    eating_detector = EatingDetector(fps=fps)
    wellness_tracker = WellnessTracker()
    risk_aggregator = RiskAggregator()
    alert_manager = AlertManager()
    skeleton_compressor = SkeletonCompressor(keyframe_interval=120)
    scenario = SimulationScenarioRunner(fps=fps)

    # Setup database
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    db = PIHMSDatabase()

    # Setup storage validator
    storage_validator = StorageValidator(data_dir)

    # Register sample medications
    wellness_tracker.add_medication('bp_med', 'Lisinopril 10mg', [8, 20])
    wellness_tracker.add_medication('vitamin_d', 'Vitamin D3 1000IU', [9])

    logger.info("All components initialized ✓")
    logger.info("")

    # ── Main simulation loop ──────────────────────────────────────
    total_frames = duration_sec * fps
    batch_start = time.time()
    latencies = []
    alerts_generated = []
    batches_stored = 0
    prev_activity = ""
    last_report_time = time.time()

    # Print header
    logger.info(f"{'Frame':>7} │ {'Time':>6} │ {'Scenario':>10} │ {'Activity':>15} │ "
                f"{'PreFall':>7} │ {'Total':>5} │ {'Sev':>8} │ {'Latency':>8} │ Status")
    logger.info("─" * 110)

    for frame_idx in range(total_frames):
        if _shutdown:
            logger.info("Shutdown signal received")
            break

        t_start = time.perf_counter()

        # Generate synthetic skeleton
        keypoints, sim_activity = scenario.next_keypoints()

        # Feature extraction
        features = feature_extractor.extract(keypoints)
        if features is None:
            continue

        # Pre-fall detection
        pre_fall_result = pre_fall_detector.update(features)

        # Activity monitoring
        activity_result = activity_monitor.update(features)

        # Eating detection
        eating_result = eating_detector.update(features)

        # Wellness check (every 60 simulated seconds)
        wellness_risk = wellness_tracker.get_wellness_risk()

        # Risk aggregation
        risk_result = risk_aggregator.update(
            pre_fall_result=pre_fall_result,
            activity_result=activity_result,
            wellness_risk=wellness_risk,
        )

        # Alert management
        alert = alert_manager.update(
            risk_result=risk_result,
            activity_state=int(features.get('activity_state', 2)),
            pre_fall_result=pre_fall_result,
        )

        if alert is not None:
            alerts_generated.append(alert)
            db.insert_alert(
                severity=alert['severity'],
                primary_type=alert['primary_type'],
                total_risk_score=alert['risk_breakdown']['total_risk'],
                payload=alert,
                secondary_signals=alert.get('secondary_signals', []),
                confidence=alert.get('confidence', 0),
            )

        # Skeleton compression & storage
        batch = skeleton_compressor.add_frame(keypoints)
        if batch is not None:
            db.insert_skeleton_batch(
                batch_data=batch, num_frames=120,
                timestamp_start=batch_start, timestamp_end=time.time(),
                avg_confidence=float(keypoints[:, 2].mean()),
            )
            batches_stored += 1
            batch_start = time.time()

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(elapsed_ms)

        # Print status every 30 frames (1 second) or on activity transition
        cur_activity = activity_result['state_name']
        now = time.time()
        if frame_idx % (fps) == 0 or cur_activity != prev_activity or alert is not None:
            sim_time = frame_idx / fps
            biomarkers = ','.join(pre_fall_result.get('biomarkers', []))[:15] or '—'
            status_parts = []
            if alert:
                status_parts.append(f"🚨 {alert['color']}")
            if eating_result.get('eating_detected'):
                status_parts.append("🍽️ Eating")
            if biomarkers != '—':
                status_parts.append(f"⚡ {biomarkers}")
            status = ' | '.join(status_parts) or '—'

            logger.info(
                f"{frame_idx:7d} │ {sim_time:5.1f}s │ {sim_activity:>10s} │ "
                f"{cur_activity:>15s} │ {pre_fall_result['risk_score']:>7d} │ "
                f"{risk_result['total_risk']:>5d} │ {risk_result['severity']:>8s} │ "
                f"{elapsed_ms:6.2f}ms │ {status}"
            )
            prev_activity = cur_activity

        # Proper frame timing (in simulation we run as fast as possible)
        # But cap at 10× real-time to avoid overwhelming the console
        if frame_idx % 100 == 0:
            time.sleep(0.001)

    # ── Final Summary ─────────────────────────────────────────────────
    latencies = np.array(latencies)

    # Flush remaining data
    final_batch = skeleton_compressor.flush()
    if final_batch:
        db.insert_skeleton_batch(
            batch_data=final_batch,
            num_frames=len(latencies) % 120 or 120,
            timestamp_start=batch_start, timestamp_end=time.time(),
        )
        batches_stored += 1

    # Generate daily summary
    summary = activity_monitor.get_daily_summary()
    db.upsert_daily_summary(time.strftime('%Y-%m-%d'), summary)

    # Final report
    print()
    print("═" * 70)
    print("  PIHMS v2.0 — SIMULATION REPORT")
    print("═" * 70)
    print()
    print(f"  Duration:            {duration_sec} seconds ({total_frames} frames)")
    print(f"  Frames processed:    {len(latencies)}")
    print()
    print("  ── PERFORMANCE ──────────────────────────────────────────")
    print(f"  Mean latency:        {np.mean(latencies):.2f} ms/frame")
    print(f"  Median latency:      {np.median(latencies):.2f} ms/frame")
    print(f"  P95 latency:         {np.percentile(latencies, 95):.2f} ms/frame")
    print(f"  P99 latency:         {np.percentile(latencies, 99):.2f} ms/frame")
    print(f"  Max latency:         {np.max(latencies):.2f} ms/frame")
    print(f"  30 FPS feasible:     {'✅ YES' if np.percentile(latencies, 99) < 33 else '❌ NO'}")
    print()
    print("  ── ACTIVITY SUMMARY ─────────────────────────────────────")
    print(f"  Walking:             {summary.get('walking_minutes', 0):.1f} minutes")
    print(f"  Sitting:             {summary.get('sitting_minutes', 0):.1f} minutes")
    print(f"  Standing:            {summary.get('standing_minutes', 0):.1f} minutes")
    print(f"  Lying:               {summary.get('lying_minutes', 0):.1f} minutes")
    print(f"  Transitions:         {summary.get('transfers_count', 0)}")
    print(f"  Mobility trend:      {summary.get('mobility_trend', 'unknown')}")
    print()
    print("  ── ALERTS ───────────────────────────────────────────────")
    print(f"  Total alerts:        {len(alerts_generated)}")
    if alerts_generated:
        for a in alerts_generated:
            print(f"    {a['color']} {a['primary_type']} "
                  f"(risk={a['risk_breakdown']['total_risk']})")
    else:
        print(f"    No alerts triggered (system stable)")
    print()
    print("  ── WELLNESS ─────────────────────────────────────────────")
    wsummary = wellness_tracker.get_wellness_summary()
    print(f"  Medication adherence: {wsummary['medication_adherence_7day']:.0%}")
    print(f"  Meals today:         {wsummary['meals_today']}")
    print(f"  Exercise (7d):       {wsummary['exercise_sessions_7day']} sessions")
    print(f"  Wellness risk:       {wsummary['wellness_risk_score']}")
    print()
    print("  ── STORAGE ──────────────────────────────────────────────")
    print(f"  Skeleton batches:    {batches_stored}")
    print(f"  Database size:       {db.get_db_size_mb():.3f} MB")
    print(f"  {storage_validator.generate_report()}")
    print()
    print("═" * 70)

    db.close()
    return True


def run_full_pipeline(legacy=False):
    """Run the full pipeline with real camera and models."""
    try:
        from camera import Camera
        from detector import PersonDetector
        from pose import PoseEstimator
        from signals.gait import GaitInstability
        from signals.sway import LateralSway
        from signals.trunk import TrunkLean
        from signals.bed_exit import BedExit
        from signals.freeze import FreezingOfGait
        from signals.arm_reach import ArmReach
        from risk_engine import RiskEngine
        from alert import AlertSystem
        from database import IncidentDatabase
        from ml_detector import MLPreFallDetector
        import config
    except ImportError as e:
        logger.error(f"Failed to import FallGuard modules: {e}")
        logger.error("Use --simulate mode for testing without camera/models")
        return False

    logger.info("═" * 50)
    logger.info("  PIHMS v2.0 — LIVE MODE")
    logger.info("═" * 50)

    # Initialize legacy components
    try:
        camera = Camera(config.RTSP_URL)
        detector = PersonDetector('../models/ssd_mobilenet_v2.tflite')
        pose_estimator = PoseEstimator('../models/movenet_lightning.tflite')
    except Exception as e:
        logger.error(f"Failed to initialize camera/models: {e}")
        logger.error("Models may be placeholder files. Use --simulate mode instead.")
        logger.error("To fix: download real MoveNet Lightning and MobileNet-SSD v2 TFLite models")
        return False

    gait_analyzer = GaitInstability(window_size=config.SIGNAL_WINDOW)
    sway_analyzer = LateralSway(window_size=config.SIGNAL_WINDOW,
                                sway_multiplier=config.SWAY_MULTIPLIER)
    trunk_analyzer = TrunkLean(yellow_thresh=config.TRUNK_LEAN_YELLOW,
                               red_thresh=config.TRUNK_LEAN_RED)
    bed_exit_analyzer = BedExit(bed_polygon=config.BED_POLYGON)
    freeze_analyzer = FreezingOfGait(window_size=config.SIGNAL_WINDOW,
                                     freeze_velocity=config.FREEZE_VELOCITY)
    arm_reach_analyzer = ArmReach(window_size=config.SIGNAL_WINDOW,
                                  reach_ratio=config.ARM_REACH_RATIO)
    ml_detector = MLPreFallDetector()
    risk_engine = RiskEngine()
    alert_system = AlertSystem()
    legacy_db = IncidentDatabase()

    # PIHMS v2.0 components
    if not legacy:
        feature_extractor = FeatureExtractor(fps=30)
        pre_fall_detector = PreFallDetector(fps=30)
        activity_monitor = ActivityMonitor(fps=30)
        eating_detector = EatingDetector(fps=30)
        wellness_tracker = WellnessTracker()
        risk_aggregator = RiskAggregator()
        alert_manager = AlertManager()
        skeleton_compressor = SkeletonCompressor(keyframe_interval=120)
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        pihms_db = PIHMSDatabase()
        storage_validator = StorageValidator(data_dir)
        storage_validator.alert_if_critical()
        logger.info("PIHMS v2.0 components initialized ✓")

    batch_start = time.time()
    frame_count = 0

    try:
        while not _shutdown:
            frame = camera.read_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            person_box = detector.detect(frame)
            if person_box is None:
                alert_system.update_alert_level(0)
                time.sleep(0.01)
                continue

            keypoints = pose_estimator.estimate(frame, person_box)
            if keypoints is None or len(keypoints) == 0:
                alert_system.update_alert_level(0)
                time.sleep(0.01)
                continue

            # Legacy signals
            gait_risk = gait_analyzer.analyze(keypoints)
            sway_risk = sway_analyzer.analyze(keypoints)
            trunk_risk = trunk_analyzer.analyze(keypoints)
            bed_exit_risk = bed_exit_analyzer.analyze(keypoints, person_box)
            freeze_risk = freeze_analyzer.analyze(keypoints)
            arm_reach_risk = arm_reach_analyzer.analyze(keypoints)
            ml_risk = ml_detector.analyze(keypoints)

            legacy_level = risk_engine.compute_risk(
                gait_risk, sway_risk, trunk_risk, bed_exit_risk,
                freeze_risk, arm_reach_risk, ml_risk)
            alert_system.update_alert_level(legacy_level)

            if legacy_level >= 1:
                legacy_db.log_incident(legacy_level, {
                    'gait': gait_risk, 'sway': sway_risk,
                    'trunk': trunk_risk, 'bed_exit': bed_exit_risk,
                    'freeze': freeze_risk, 'arm_reach': arm_reach_risk})

            # PIHMS v2.0
            if not legacy:
                features = feature_extractor.extract(keypoints)
                if features:
                    pf = pre_fall_detector.update(features)
                    act = activity_monitor.update(features)
                    eating_detector.update(features)
                    risk = risk_aggregator.update(
                        pre_fall_result=pf, activity_result=act,
                        wellness_risk=wellness_tracker.get_wellness_risk())
                    alert = alert_manager.update(
                        risk_result=risk,
                        activity_state=int(features.get('activity_state', 2)),
                        pre_fall_result=pf)
                    if alert:
                        pihms_db.insert_alert(
                            severity=alert['severity'],
                            primary_type=alert['primary_type'],
                            total_risk_score=alert['risk_breakdown']['total_risk'],
                            payload=alert)
                        logger.warning(f"🚨 {alert['color']} {alert['primary_type']}")

                batch = skeleton_compressor.add_frame(keypoints)
                if batch:
                    pihms_db.insert_skeleton_batch(
                        batch_data=batch, num_frames=120,
                        timestamp_start=batch_start, timestamp_end=time.time(),
                        avg_confidence=float(keypoints[:, 2].mean()))
                    batch_start = time.time()

            frame_count += 1
            if frame_count % 900 == 0:
                logger.info(f"Frame {frame_count}")

    finally:
        camera.release()
        alert_system.cleanup()
        legacy_db.close()
        if not legacy:
            summary = activity_monitor.get_daily_summary()
            pihms_db.upsert_daily_summary(time.strftime('%Y-%m-%d'), summary)
            final = skeleton_compressor.flush()
            if final:
                pihms_db.insert_skeleton_batch(
                    batch_data=final,
                    num_frames=frame_count % 120,
                    timestamp_start=batch_start, timestamp_end=time.time())
            pihms_db.close()
        logger.info("Shutdown complete")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='PIHMS v2.0 — PreFall Intelligence Health Monitoring System')
    parser.add_argument('--legacy', action='store_true',
                        help='Run in legacy FallGuard mode only')
    parser.add_argument('--simulate', action='store_true',
                        help='Run simulation mode (no camera/models needed)')
    parser.add_argument('--duration', type=int, default=30,
                        help='Simulation duration in seconds (default: 30)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS (default: 30)')
    args = parser.parse_args()

    if args.simulate:
        success = run_simulation(args.duration, args.fps)
    else:
        success = run_full_pipeline(args.legacy)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
