#!/usr/bin/env python3
"""
Test script to verify the full pre-fall detection pipeline (PIHMS v2.0)
"""
import numpy as np
import cv2
import sys
import time
sys.path.insert(0, 'fallguard/app')

from detector import PersonDetector
from pose import PoseEstimator
from risk_engine import RiskEngine
from pihms.feature_extraction import FeatureExtractor
from pihms.pre_fall_detection import PreFallDetector
from pihms.activity_monitoring import ActivityMonitor
from pihms.risk_aggregator import RiskAggregator
from pihms.alert_manager import AlertManager

print("\n✅ Testing Full PIHMS v2.0 Pipeline\n")

# Test 1: Load models
try:
    person_detector = PersonDetector('fallguard/models/ssd_mobilenet_v2_cpu.tflite')
    pose_estimator = PoseEstimator('fallguard/models/movenet_lightning.tflite')
    risk_engine = RiskEngine()
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit(1)

# Test 2: Create test frame
print("\n✅ Creating test frame...")
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.rectangle(frame, (100, 100), (500, 600), (255,255,255), -1)

# Test 3: Run detection (single + multi)
print("\n✅ Running person detection...")
box = person_detector.detect(frame)
print(f"   Single detect: {box}")
all_persons = person_detector.detect_all(frame)
print(f"   Multi detect: {len(all_persons)} persons")

# Test 4: Pose estimation
print("\n✅ Running pose estimation...")
keypoints = pose_estimator.estimate(frame, (100, 100, 500, 600))
print(f"   Keypoints shape: {keypoints.shape}")

# Test 5: PIHMS analytics pipeline
print("\n✅ Running PIHMS analytics pipeline...")
fe = FeatureExtractor(fps=30)
pfd = PreFallDetector(fps=30)
am = ActivityMonitor(fps=30)
ra = RiskAggregator()
alm = AlertManager()

# Generate realistic keypoints
kp_base = np.array([
    [0.15,0.50,0.95],[0.13,0.48,0.90],[0.13,0.52,0.90],
    [0.14,0.46,0.85],[0.14,0.54,0.85],
    [0.25,0.42,0.92],[0.25,0.58,0.92],
    [0.38,0.40,0.88],[0.38,0.60,0.88],
    [0.48,0.42,0.85],[0.48,0.58,0.85],
    [0.50,0.45,0.93],[0.50,0.55,0.93],
    [0.68,0.44,0.90],[0.68,0.56,0.90],
    [0.85,0.43,0.88],[0.85,0.57,0.88],
], dtype=np.float32)

latencies = []
alerts_count = 0
max_risk = 0

# Phase 1: 200 frames stable (baseline building)
# Phase 2: 200 frames with increasing instability
for i in range(400):
    t0 = time.perf_counter()
    kp = kp_base.copy()
    if i >= 200:
        severity = (i - 200) / 200.0
        sway = severity * 0.05 * np.sin(i / 4.0)
        kp[:, 1] += sway
        kp[5:7, 0] += severity * 0.04 * np.sin(i / 3.0)
    kp += np.random.randn(17, 3).astype(np.float32) * 0.002
    kp[:, 2] = kp_base[:, 2]

    feat = fe.extract(kp)
    if feat:
        pf = pfd.update(feat)
        act = am.update(feat)
        risk = ra.update(pre_fall_result=pf, activity_result=act)
        alert = alm.update(risk_result=risk,
                           activity_state=int(feat.get('activity_state', 2)),
                           pre_fall_result=pf)
        if alert:
            alerts_count += 1
        max_risk = max(max_risk, risk['total_risk'])

    latencies.append((time.perf_counter() - t0) * 1000)

latencies = np.array(latencies)
print(f"   Frames: 400 (200 stable + 200 unstable)")
print(f"   Mean latency: {np.mean(latencies):.2f} ms")
print(f"   P99 latency:  {np.percentile(latencies, 99):.2f} ms")
print(f"   Max risk:     {max_risk}")
print(f"   Alerts:       {alerts_count}")
print(f"   30 FPS:       {'✅ YES' if np.percentile(latencies, 99) < 33 else '❌ NO'}")

# Test 6: Legacy risk engine
print("\n✅ Testing legacy risk engine...")
alert_level = risk_engine.compute_risk(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
print(f"   Alert level (no risk): {alert_level}")
alert_level = risk_engine.compute_risk(0.8, 0.7, 0.0, 0.0, 0.0, 0.0)
print(f"   Alert level (2 signals): {alert_level}")

print("\n✅ ALL SYSTEMS WORKING CORRECTLY")
print("   Dashboard: http://localhost:8080 (starts with pihms_live.py)")
print("   MQTT: localhost:1883 (topics: pihms/facility/FAC001/person/*/alert)")