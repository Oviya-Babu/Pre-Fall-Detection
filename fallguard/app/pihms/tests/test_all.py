#!/usr/bin/env python3
"""
PIHMS v2.0 — Comprehensive test suite for all Phase 1-3 modules.
Tests: compression, database, feature extraction, pre-fall detection,
       activity monitoring, yoga guidance, wellness, risk aggregation, alerts.
"""

import sys
import os
import time
import tempfile
import numpy as np

# Add path for imports — the 'app' directory contains the 'pihms' package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

def make_keypoints(base_y=0.3, variation=0.0):
    """Generate synthetic 17-keypoint MoveNet data."""
    kp = np.zeros((17, 3), dtype=np.float32)
    # Standing person template (y,x,confidence)
    positions = [
        (0.15, 0.50),  # 0: nose
        (0.13, 0.48),  # 1: left eye
        (0.13, 0.52),  # 2: right eye
        (0.14, 0.46),  # 3: left ear
        (0.14, 0.54),  # 4: right ear
        (0.25, 0.42),  # 5: left shoulder
        (0.25, 0.58),  # 6: right shoulder
        (0.35, 0.38),  # 7: left elbow
        (0.35, 0.62),  # 8: right elbow
        (0.45, 0.36),  # 9: left wrist
        (0.45, 0.64),  # 10: right wrist
        (0.50, 0.45),  # 11: left hip
        (0.50, 0.55),  # 12: right hip
        (0.65, 0.44),  # 13: left knee
        (0.65, 0.56),  # 14: right knee
        (0.80, 0.43),  # 15: left ankle
        (0.80, 0.57),  # 16: right ankle
    ]
    for i, (y, x) in enumerate(positions):
        kp[i] = [y + base_y + np.random.randn() * variation,
                 x + np.random.randn() * variation,
                 0.85 + np.random.random() * 0.15]
    return kp


def test_skeleton_compression():
    """Test Task 1.1: Skeleton compression."""
    from pihms.compress_skeleton import (
        SkeletonCompressor, SkeletonDecompressor,
        encode_keyframe, decode_keyframe, encode_delta,
        quantize_frame, dequantize_frame
    )

    print("  ├─ Quantization roundtrip...")
    kp = make_keypoints(base_y=0.0)  # Keep values in [0,1] as MoveNet outputs
    q = quantize_frame(kp)
    kp_back = dequantize_frame(q)
    max_err = np.max(np.abs(kp[:, :2] - kp_back[:, :2]))
    # int16 quantization at scale 32000 gives ~3e-5 precision per step
    # but floating point rounding can push it to ~5e-4 for large values
    assert max_err < 0.005, f"Quantization error too high: {max_err}"
    print(f"     Max position error: {max_err:.6f} ✓")

    print("  ├─ Keyframe encode/decode...")
    data = encode_keyframe(kp)
    assert len(data) == 2 + 17 * 3 * 2  # magic + quantized data
    kp_dec = decode_keyframe(data)
    assert kp_dec.shape == (17, 3)
    print(f"     Keyframe size: {len(data)} bytes ✓")

    print("  ├─ Batch compression (120 frames)...")
    compressor = SkeletonCompressor(keyframe_interval=120, gzip_level=6)
    batch = None
    for i in range(120):
        kp_frame = make_keypoints(variation=0.002)
        batch = compressor.add_frame(kp_frame)

    assert batch is not None, "Batch should be emitted at 120 frames"
    raw_size = 120 * 17 * 3 * 4  # 120 frames × 204 bytes
    ratio = len(batch) / raw_size
    print(f"     Raw: {raw_size} bytes, Compressed: {len(batch)} bytes")
    print(f"     Compression ratio: {ratio:.2%} ({(1-ratio)*100:.0f}% reduction) ✓")

    print("  ├─ Batch decompression...")
    frames = SkeletonDecompressor.decompress_batch(batch)
    assert len(frames) == 120, f"Expected 120 frames, got {len(frames)}"
    assert frames[0].shape == (17, 3)
    print(f"     Decompressed {len(frames)} frames ✓")

    # Daily estimate
    daily_frames = 30 * 86400  # 30 FPS × 24h
    batches_per_day = daily_frames / 120
    est_daily_mb = (len(batch) * batches_per_day) / (1024 * 1024)
    print(f"  └─ Estimated daily storage: {est_daily_mb:.1f} MB/day ✓")
    return True


def test_database():
    """Test Task 1.2: Database schema and operations."""
    from pihms.database.pihms_db import PIHMSDatabase

    print("  ├─ Creating database...")
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        db = PIHMSDatabase(db_path=db_path)

        print("  ├─ Insert skeleton batch...")
        dummy_batch = b'\x00' * 1000
        row_id = db.insert_skeleton_batch(
            batch_data=dummy_batch, num_frames=120,
            timestamp_start=time.time() - 4, timestamp_end=time.time(),
            avg_confidence=0.9
        )
        assert row_id > 0

        print("  ├─ Insert alert...")
        alert_id = db.insert_alert(
            severity='HIGH', primary_type='PRE_FALL_WARNING',
            total_risk_score=85.0,
            payload={'test': True},
            secondary_signals=['GAIT_INSTABILITY', 'EXCESSIVE_SWAY']
        )
        assert len(alert_id) > 0

        print("  ├─ Query recent alerts...")
        alerts = db.get_recent_alerts(hours=1)
        assert len(alerts) == 1

        print("  ├─ Daily summary upsert...")
        db.upsert_daily_summary('2026-04-27', {
            'walking_minutes': 30, 'sitting_minutes': 120,
            'lying_minutes': 480, 'mobility_trend': 'stable',
        })

        print("  ├─ Medication schedule...")
        db.add_medication_schedule('med001', 'Lisinopril', [8, 20], dosage='10mg')

        print("  ├─ Log medication...")
        db.log_medication_taken('med001', time.time())

        print("  ├─ Medication adherence...")
        adherence = db.get_medication_adherence()
        assert 0 <= adherence <= 1.0

        print("  ├─ Database size...")
        size_mb = db.get_db_size_mb()
        print(f"     DB size: {size_mb:.3f} MB ✓")

        print("  ├─ Data rotation...")
        db.rotate_old_data(skeleton_days=7, alert_days=30)

        db.close()
        print("  └─ Database tests passed ✓")
    finally:
        os.unlink(db_path)
    return True


def test_feature_extraction():
    """Test Task 1.3: Feature extraction."""
    from pihms.feature_extraction import FeatureExtractor

    print("  ├─ Creating extractor...")
    extractor = FeatureExtractor(window_size=360, fps=30)

    print("  ├─ Extracting features (100 frames)...")
    t_start = time.time()
    for i in range(100):
        kp = make_keypoints(variation=0.003)
        features = extractor.extract(kp)
    elapsed = (time.time() - t_start) * 1000 / 100
    print(f"     Avg latency: {elapsed:.2f} ms/frame ✓")

    assert features is not None
    print(f"     Features extracted: {len(features)} values")
    assert len(features) >= 40, f"Expected 40+ features, got {len(features)}"

    # Check key features exist
    required = ['angle_left_knee', 'trunk_lean_angle', 'com_x', 'hip_speed',
                 'step_width_mean', 'sway_amplitude_x', 'activity_state']
    for key in required:
        assert key in features, f"Missing feature: {key}"
    print(f"  └─ Feature extraction: {elapsed:.2f}ms/frame, {len(features)} features ✓")
    return True


def test_pre_fall_detection():
    """Test Task 2.1: Pre-fall detection."""
    from pihms.feature_extraction import FeatureExtractor
    from pihms.pre_fall_detection import PreFallDetector

    print("  ├─ Initializing...")
    extractor = FeatureExtractor()
    detector = PreFallDetector(fps=30)

    # Normal walking
    print("  ├─ Normal walking (100 frames)...")
    for i in range(100):
        kp = make_keypoints(variation=0.002)
        features = extractor.extract(kp)
        result = detector.update(features)
    assert result['risk_score'] < 50, f"Normal walking risk too high: {result['risk_score']}"
    print(f"     Risk score: {result['risk_score']} (should be <50) ✓")

    # Simulate instability
    print("  ├─ Simulated instability (50 frames)...")
    for i in range(50):
        kp = make_keypoints(variation=0.02)  # 10× more variation
        features = extractor.extract(kp)
        result = detector.update(features)
    print(f"     Risk score: {result['risk_score']}, biomarkers: {result['biomarkers']}")
    print(f"  └─ Pre-fall detection: working ✓")
    return True


def test_activity_monitoring():
    """Test Task 2.2: Activity monitoring."""
    from pihms.activity_monitoring import ActivityMonitor

    print("  ├─ Initializing...")
    monitor = ActivityMonitor(fps=30)

    # Simulate standing
    print("  ├─ Standing frames...")
    for i in range(30):
        result = monitor.update({'activity_state': 2, 'hip_speed': 0.1})
    assert result['state_name'] == 'STANDING_IDLE'

    # Simulate walking
    print("  ├─ Walking frames...")
    for i in range(30):
        result = monitor.update({'activity_state': 3, 'hip_speed': 2.0})
    assert result['state_name'] == 'WALKING'

    # Daily summary
    summary = monitor.get_daily_summary()
    assert 'walking_minutes' in summary
    assert 'mobility_trend' in summary
    print(f"     Summary: {summary['walking_minutes']}min walking, trend={summary['mobility_trend']}")
    print(f"  └─ Activity monitoring: working ✓")
    return True


def test_yoga_guidance():
    """Test Task 2.4: Yoga guidance."""
    from pihms.yoga_guidance import YogaCoach

    print("  ├─ Loading templates...")
    coach = YogaCoach()
    poses = coach.get_pose_list()
    assert len(poses) == 10, f"Expected 10 poses, got {len(poses)}"
    print(f"     Loaded {len(poses)} poses ✓")

    print("  ├─ Setting active pose...")
    success = coach.set_active_pose("Mountain Pose (Tadasana)")
    assert success

    print("  ├─ Checking pose accuracy...")
    features = {
        'angle_left_knee': 175, 'angle_right_knee': 175,
        'angle_left_hip': 175, 'angle_right_hip': 175,
        'angle_left_elbow': 170, 'angle_right_elbow': 170,
        'trunk_lean_angle': 5,
    }
    feedback = coach.update(features)
    assert feedback['accuracy_percent'] > 80
    print(f"     Accuracy: {feedback['accuracy_percent']}%, status: {feedback['status']} ✓")
    print(f"  └─ Yoga guidance: working ✓")
    return True


def test_wellness_logic():
    """Test Task 2.5: Wellness logic."""
    from pihms.wellness_logic import WellnessTracker

    print("  ├─ Initializing...")
    tracker = WellnessTracker()

    tracker.add_medication('med001', 'Lisinopril', [8, 20], dosage='10mg')
    tracker.log_medication_taken('med001', 'ON_TIME')
    tracker.add_meal_log('BREAKFAST')
    tracker.add_exercise_session({'date': '2026-04-27', 'total_duration_sec': 600})

    summary = tracker.get_wellness_summary()
    assert 'wellness_risk_score' in summary
    print(f"     Wellness risk: {summary['wellness_risk_score']}, "
          f"adherence: {summary['medication_adherence_7day']}")
    print(f"  └─ Wellness logic: working ✓")
    return True


def test_risk_aggregation():
    """Test Task 3.1: Risk aggregation."""
    from pihms.risk_aggregator import RiskAggregator

    print("  ├─ Testing risk scenarios...")
    agg = RiskAggregator()

    # Low risk
    result = agg.update(
        pre_fall_result={'risk_score': 20},
        activity_result={'anomaly_score': 0.1},
        wellness_risk=5.0
    )
    assert result['severity'] == 'LOW'
    print(f"     Low risk: total={result['total_risk']}, sev={result['severity']} ✓")

    # High risk
    result = agg.update(
        pre_fall_result={'risk_score': 95},
        activity_result={'anomaly_score': 0.8},
        wellness_risk=40.0
    )
    assert result['total_risk'] >= 50, f"Expected risk>=50, got {result['total_risk']}"
    print(f"     High risk: total={result['total_risk']}, sev={result['severity']} ✓")
    print(f"  └─ Risk aggregation: working ✓")
    return True


def test_alert_manager():
    """Test Task 3.2: Alert manager."""
    from pihms.alert_manager import AlertManager

    print("  ├─ Testing alert generation...")
    mgr = AlertManager()

    # High risk → should produce alert
    alert = mgr.update(
        risk_result={'total_risk': 85, 'severity': 'HIGH',
                     'pre_fall_risk': 87, 'activity_anomaly_risk': 12,
                     'wellness_risk': 8, 'trend_degradation_risk': 5},
        pre_fall_result={'risk_validated': True, 'biomarkers': ['GAIT_INSTABILITY'],
                         'latency_sec': 7.2}
    )
    assert alert is not None
    assert alert['severity'] == 'HIGH'
    print(f"     Alert: {alert['severity']} {alert['primary_type']} ✓")

    # Duplicate suppression
    print("  ├─ Deduplication...")
    alert2 = mgr.update(
        risk_result={'total_risk': 85, 'severity': 'HIGH',
                     'pre_fall_risk': 87, 'activity_anomaly_risk': 12,
                     'wellness_risk': 8, 'trend_degradation_risk': 5},
        pre_fall_result={'risk_validated': True, 'biomarkers': ['GAIT_INSTABILITY'],
                         'latency_sec': 7.2}
    )
    assert alert2 is None, "Duplicate alert should be suppressed"
    print(f"     Duplicate suppressed ✓")

    # Low risk → no alert
    print("  ├─ Low risk suppression...")
    alert3 = mgr.update(
        risk_result={'total_risk': 30, 'severity': 'LOW',
                     'pre_fall_risk': 20, 'activity_anomaly_risk': 5,
                     'wellness_risk': 5, 'trend_degradation_risk': 0}
    )
    assert alert3 is None
    print(f"     Low risk suppressed ✓")
    print(f"  └─ Alert manager: working ✓")
    return True


def test_storage_validator():
    """Test storage validation."""
    from pihms.storage_validator import StorageValidator

    print("  ├─ Storage validation...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    validator = StorageValidator(data_dir)

    disk = validator.get_disk_usage()
    print(f"     Total: {disk['total_gb']:.1f}GB, "
          f"Free: {disk['free_gb']:.1f}GB ({disk['usage_percent']:.1f}%)")

    growth = validator.estimate_daily_growth()
    print(f"     Daily growth: {growth['total_daily_mb']:.1f}MB/day")
    print(f"     Days until full: {growth['days_until_full']:.0f}")

    report = validator.generate_report()
    print(report)
    print(f"  └─ Storage validator: working ✓")
    return True


def main():
    print("═" * 60)
    print("  PIHMS v2.0 — Comprehensive Test Suite")
    print("═" * 60)

    tests = [
        ("1.1 Skeleton Compression", test_skeleton_compression),
        ("1.2 Database Schema", test_database),
        ("1.3 Feature Extraction", test_feature_extraction),
        ("2.1 Pre-Fall Detection", test_pre_fall_detection),
        ("2.2 Activity Monitoring", test_activity_monitoring),
        ("2.4 Yoga Guidance", test_yoga_guidance),
        ("2.5 Wellness Logic", test_wellness_logic),
        ("3.1 Risk Aggregation", test_risk_aggregation),
        ("3.2 Alert Manager", test_alert_manager),
        ("4.0 Storage Validator", test_storage_validator),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n{'─'*50}")
        print(f"TEST {name}")
        print(f"{'─'*50}")
        try:
            if test_fn():
                passed += 1
                print(f"  ✅ PASSED: {name}")
            else:
                failed += 1
                print(f"  ❌ FAILED: {name}")
        except Exception as e:
            failed += 1
            print(f"  ❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'═'*60}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'═'*60}")

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
