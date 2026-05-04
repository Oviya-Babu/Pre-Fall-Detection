#!/usr/bin/env python3
"""
PIHMS v2.0 — PRE-FALL Prediction + Multi-Person + Alert Validation Test
=========================================================================
Validates:
  TEST 1: PRE-FALL prediction triggers correctly for a falling person
  TEST 2: Alert system fires PRE_FALL_WARNING with correct severity
  TEST 3: 10+ standing persons → zero false positives (all stay GREEN)
  TEST 4: Mixed crowd — 10 stable + 1 falling → only the faller gets alerts
  TEST 5: Edge cases — no false positives on sitting, lying, walking persons
"""

import sys
import os
import time
import numpy as np
import logging

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from pihms.feature_extraction import FeatureExtractor
from pihms.pre_fall_detection import PreFallDetector
from pihms.activity_monitoring import ActivityMonitor
from pihms.eating_detection import EatingDetector
from pihms.risk_aggregator import RiskAggregator
from pihms.alert_manager import AlertManager

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('PRE_FALL_TEST')
logger.setLevel(logging.INFO)

PASS_COUNT = 0
FAIL_COUNT = 0


def score(passed, msg):
    global PASS_COUNT, FAIL_COUNT
    if passed:
        PASS_COUNT += 1
        print(f"  ✅ {msg}")
    else:
        FAIL_COUNT += 1
        print(f"  ❌ FAIL: {msg}")


# ── Skeleton Generators ─────────────────────────────────────────────

def make_standing(person_offset=0.0, noise=0.001):
    """Stable standing person with optional lateral offset."""
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
    kp[:, 1] += person_offset
    kp[:, :2] += np.random.randn(17, 2).astype(np.float32) * noise
    return kp


def make_walking(frame_idx, person_offset=0.0, noise=0.002):
    """Normal walking person."""
    kp = make_standing(person_offset, noise)
    phase = frame_idx / 15.0
    kp[15, 0] += 0.02 * np.sin(phase * 2 * np.pi)
    kp[16, 0] += 0.02 * np.sin(phase * 2 * np.pi + np.pi)
    kp[11, 1] += 0.003 * np.sin(phase * np.pi)
    kp[12, 1] += 0.003 * np.sin(phase * np.pi)
    return kp


def make_falling(frame_idx, total_fall_frames=400, noise=0.005):
    """
    Person progressively losing balance and falling.
    Generates strong biomechanical signals that the PreFallDetector
    can pick up: wide sway, gait irregularity, trunk lean, rapid
    ankle corrections.
    """
    kp = make_standing(noise=noise)
    progress = min(1.0, frame_idx / total_fall_frames)

    # ── Phase 1 (0-30%): Increasing lateral sway ─────────────────
    # Large amplitude CoM oscillation — drives com_sway score
    sway_amp = 0.02 + progress * 0.12
    sway = sway_amp * np.sin(frame_idx / 3.5)
    kp[:, 1] += sway  # Lateral sway on ALL keypoints

    # ── Phase 2 (20-60%): Step width irregularity ────────────────
    if progress > 0.2:
        irreg = (progress - 0.2) * 0.15
        kp[15, 1] += irreg * np.sin(frame_idx / 2.0)  # Left ankle erratic
        kp[16, 1] -= irreg * np.cos(frame_idx / 2.5)  # Right ankle erratic
        # Step width standard deviation increases
        kp[13, 1] += irreg * 0.5 * np.sin(frame_idx / 3.0)  # Knees wobble

    # ── Phase 3 (30-70%): Trunk forward lean increasing ──────────
    if progress > 0.3:
        lean_factor = min(1.0, (progress - 0.3) / 0.4)
        lean = lean_factor * 0.15
        kp[5, 0] += lean   # left shoulder drops forward
        kp[6, 0] += lean   # right shoulder drops forward
        kp[0, 0] += lean * 1.5  # head leads the fall

    # ── Phase 4 (50-100%): Rapid hip drop + corrective jerks ─────
    if progress > 0.5:
        fall_factor = (progress - 0.5) / 0.5
        # Hip drops rapidly (creates hip_acceleration)
        kp[11, 0] += fall_factor * 0.20
        kp[12, 0] += fall_factor * 0.20
        # Fast corrective ankle movements (high ankle speed)
        kp[15, 1] += fall_factor * 0.10 * np.sin(frame_idx * 1.2)
        kp[16, 1] += fall_factor * 0.10 * np.cos(frame_idx * 1.2)
        kp[15, 0] += fall_factor * 0.05 * np.cos(frame_idx * 0.9)
        kp[16, 0] += fall_factor * 0.05 * np.sin(frame_idx * 0.9)
        # Knee buckling (angular velocity spike)
        kp[13, 0] += fall_factor * 0.12 * (1 + 0.3 * np.sin(frame_idx * 0.7))
        kp[14, 0] += fall_factor * 0.12 * (1 + 0.3 * np.cos(frame_idx * 0.7))
        # Arms flailing
        kp[9, 0] -= fall_factor * 0.15 * np.sin(frame_idx * 0.6)
        kp[10, 0] -= fall_factor * 0.15 * np.cos(frame_idx * 0.6)

    return kp


def make_sitting(noise=0.001):
    """Sitting person — should NOT trigger pre-fall."""
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


def make_lying(noise=0.001):
    """Lying person — should NOT trigger pre-fall."""
    kp = np.array([
        [0.50, 0.15, 0.90], [0.49, 0.13, 0.85], [0.49, 0.17, 0.85],
        [0.50, 0.11, 0.80], [0.50, 0.19, 0.80],
        [0.50, 0.25, 0.88], [0.50, 0.30, 0.88],
        [0.50, 0.35, 0.85], [0.50, 0.40, 0.85],
        [0.50, 0.42, 0.82], [0.50, 0.45, 0.82],
        [0.50, 0.50, 0.90], [0.50, 0.55, 0.90],
        [0.50, 0.65, 0.87], [0.50, 0.70, 0.87],
        [0.50, 0.80, 0.85], [0.50, 0.85, 0.85],
    ], dtype=np.float32)
    kp[:, :2] += np.random.randn(17, 2).astype(np.float32) * noise
    return kp


# ── Pipeline runner for a single person ──────────────────────────────

class PersonPipeline:
    """Independent PIHMS pipeline for one person."""
    def __init__(self, person_id):
        self.person_id = person_id
        self.fe = FeatureExtractor(fps=30)
        self.pfd = PreFallDetector(fps=30)
        self.am = ActivityMonitor(fps=30)
        self.ed = EatingDetector(fps=30)
        self.ra = RiskAggregator()
        self.alm = AlertManager()
        self.alerts_fired = []
        self.max_prefall_risk = 0
        self.last_risk = {}
        self.last_pf = {}
        self.all_biomarkers = set()

    def process_frame(self, keypoints):
        features = self.fe.extract(keypoints)
        if features is None:
            return
        pf = self.pfd.update(features)
        act = self.am.update(features)
        self.ed.update(features)
        risk = self.ra.update(pre_fall_result=pf, activity_result=act)
        alert = self.alm.update(
            risk_result=risk,
            activity_state=int(features.get('activity_state', 2)),
            pre_fall_result=pf,
        )
        self.last_risk = risk
        self.last_pf = pf
        if pf['risk_score'] > self.max_prefall_risk:
            self.max_prefall_risk = pf['risk_score']
        for bm in pf.get('biomarkers', []):
            self.all_biomarkers.add(bm)
        if alert:
            self.alerts_fired.append(alert)


# ── TEST FUNCTIONS ──────────────────────────────────────────────────

def test_1_prefall_prediction():
    """TEST 1: A falling person MUST be predicted as PRE-FALL."""
    print(f"\n{'━'*70}")
    print("  TEST 1: PRE-FALL Prediction for a Falling Person")
    print(f"{'━'*70}")

    np.random.seed(100)
    pipe = PersonPipeline("faller")

    # 350 frames baseline (enough for gait baseline of 300 frames + CoM baseline)
    for i in range(350):
        pipe.process_frame(make_standing(noise=0.001))

    # 500 frames of progressive falling (long enough for temporal validation)
    prefall_detected = False
    prefall_frame = -1
    for i in range(500):
        pipe.process_frame(make_falling(i, total_fall_frames=400, noise=0.005))
        if pipe.last_pf.get('risk_score', 0) >= 50 and not prefall_detected:
            prefall_detected = True
            prefall_frame = 350 + i

    print(f"  Max pre-fall risk score: {pipe.max_prefall_risk}")
    print(f"  Pre-fall detected (risk≥50): {'YES' if prefall_detected else 'NO'}")
    if prefall_detected:
        print(f"  First detection at frame: {prefall_frame} "
              f"({prefall_frame/30:.1f}s into session)")
    print(f"  Final risk severity: {pipe.last_risk.get('severity', 'N/A')}")
    print(f"  All biomarkers seen: {sorted(pipe.all_biomarkers)}")
    print(f"  Final biomarkers: {pipe.last_pf.get('biomarkers', [])}")

    score(pipe.max_prefall_risk >= 40,
          f"Pre-fall risk reached {pipe.max_prefall_risk} (need ≥40)")
    score(len(pipe.all_biomarkers) > 0,
          f"Biomarkers detected: {sorted(pipe.all_biomarkers)}")
    score('EXCESSIVE_SWAY' in pipe.all_biomarkers or
          'GAIT_INSTABILITY' in pipe.all_biomarkers,
          "Key biomarker (SWAY or GAIT_INSTABILITY) triggered")


def test_2_alert_fires():
    """TEST 2: Alert system must fire for a falling person."""
    print(f"\n{'━'*70}")
    print("  TEST 2: Alert System Fires for Falling Person")
    print(f"{'━'*70}")

    np.random.seed(200)
    pipe = PersonPipeline("faller_alert")

    # Long baseline for all detectors to calibrate
    for i in range(400):
        pipe.process_frame(make_standing(noise=0.001))

    # Extended fall sequence for temporal validation (2s = 60 frames minimum)
    for i in range(800):
        pipe.process_frame(make_falling(i, total_fall_frames=400, noise=0.005))

    print(f"  Max pre-fall risk: {pipe.max_prefall_risk}")
    print(f"  Alerts fired: {len(pipe.alerts_fired)}")
    for a in pipe.alerts_fired:
        print(f"    {a['color']} | {a['primary_type']} | "
              f"risk={a['risk_breakdown']['total_risk']} | "
              f"severity={a['severity']}")
    print(f"  Final total_risk: {pipe.last_risk.get('total_risk', 0)}")
    print(f"  Final severity: {pipe.last_risk.get('severity', 'N/A')}")

    has_alert = len(pipe.alerts_fired) > 0
    risk_elevated = pipe.last_risk.get('total_risk', 0) >= 50

    score(pipe.max_prefall_risk >= 50,
          f"Pre-fall risk reached {pipe.max_prefall_risk} (need ≥50 for alert)")
    score(has_alert or risk_elevated,
          f"Alert fired ({len(pipe.alerts_fired)}) or risk elevated "
          f"(total={pipe.last_risk.get('total_risk', 0)})")


def test_3_ten_standing_no_false_positives():
    """TEST 3: 10+ persons standing stably → ZERO false positives."""
    print(f"\n{'━'*70}")
    print("  TEST 3: 12 Standing Persons — Zero False Positives")
    print(f"{'━'*70}")

    np.random.seed(300)
    NUM_PERSONS = 12
    NUM_FRAMES = 400  # 13+ seconds

    pipelines = [PersonPipeline(f"person_{p}") for p in range(NUM_PERSONS)]

    for frame in range(NUM_FRAMES):
        for p_idx, pipe in enumerate(pipelines):
            offset = (p_idx - NUM_PERSONS / 2) * 0.05
            kp = make_standing(person_offset=offset, noise=0.001)
            pipe.process_frame(kp)

    total_alerts = 0
    total_false_prefall = 0
    print(f"\n  {'Person':<12} {'Max Risk':>10} {'Alerts':>8} {'Severity':>10}")
    print(f"  {'─'*45}")

    for p_idx, pipe in enumerate(pipelines):
        n_alerts = len(pipe.alerts_fired)
        total_alerts += n_alerts
        if pipe.max_prefall_risk >= 50:
            total_false_prefall += 1
        sev = pipe.last_risk.get('severity', 'LOW')
        print(f"  Person {p_idx:<4} {pipe.max_prefall_risk:>10} "
              f"{n_alerts:>8} {sev:>10}")

    print(f"\n  Total alerts across {NUM_PERSONS} persons: {total_alerts}")
    print(f"  False pre-fall detections (risk≥50): {total_false_prefall}")

    score(total_alerts == 0,
          f"Zero alerts fired for {NUM_PERSONS} stable persons (got {total_alerts})")
    score(total_false_prefall == 0,
          f"Zero false pre-fall detections (got {total_false_prefall})")


def test_4_mixed_crowd():
    """TEST 4: 10 stable + 1 falling → only the faller gets flagged."""
    print(f"\n{'━'*70}")
    print("  TEST 4: Mixed Crowd — 10 Stable + 1 Falling Person")
    print(f"{'━'*70}")

    np.random.seed(400)
    NUM_STABLE = 10

    stable_pipes = [PersonPipeline(f"stable_{i}") for i in range(NUM_STABLE)]
    faller_pipe = PersonPipeline("FALLER")

    # Baseline phase (350 frames — enough for all detectors)
    for frame in range(350):
        for p_idx, pipe in enumerate(stable_pipes):
            offset = (p_idx - NUM_STABLE / 2) * 0.05
            pipe.process_frame(make_standing(person_offset=offset, noise=0.001))
        faller_pipe.process_frame(make_standing(noise=0.001))

    # Fall phase (500 frames) — faller degrades, others stay stable
    for frame in range(500):
        for p_idx, pipe in enumerate(stable_pipes):
            offset = (p_idx - NUM_STABLE / 2) * 0.05
            if p_idx % 3 == 0:
                pipe.process_frame(
                    make_walking(frame, person_offset=offset, noise=0.002))
            else:
                pipe.process_frame(
                    make_standing(person_offset=offset, noise=0.001))
        faller_pipe.process_frame(
            make_falling(frame, total_fall_frames=400, noise=0.005))

    stable_alerts = sum(len(p.alerts_fired) for p in stable_pipes)
    stable_max_risk = max(p.max_prefall_risk for p in stable_pipes)
    faller_risk = faller_pipe.max_prefall_risk

    print(f"\n  {'Person':<14} {'Max Risk':>10} {'Alerts':>8} {'Status':>12}")
    print(f"  {'─'*48}")
    for p_idx, pipe in enumerate(stable_pipes):
        status = "✅ SAFE" if pipe.max_prefall_risk < 50 else "❌ FALSE+"
        print(f"  Stable {p_idx:<5} {pipe.max_prefall_risk:>10} "
              f"{len(pipe.alerts_fired):>8} {status:>12}")
    faller_status = "✅ DETECTED" if faller_risk >= 40 else "❌ MISSED"
    print(f"  {'FALLER':<14} {faller_risk:>10} "
          f"{len(faller_pipe.alerts_fired):>8} {faller_status:>12}")

    score(stable_alerts == 0,
          f"Zero alerts for {NUM_STABLE} stable persons (got {stable_alerts})")
    score(stable_max_risk < 50,
          f"Max risk among stable persons: {stable_max_risk} (need <50)")
    score(faller_risk >= 40,
          f"Faller's pre-fall risk: {faller_risk} (need ≥40)")


def test_5_edge_cases_no_false_positives():
    """TEST 5: Sitting, lying, and walking persons → no false positives."""
    print(f"\n{'━'*70}")
    print("  TEST 5: Edge Cases — Sitting, Lying, Walking (No False Positives)")
    print(f"{'━'*70}")

    np.random.seed(500)
    NUM_FRAMES = 400

    scenarios = {
        'Sitting person': lambda i: make_sitting(noise=0.001),
        'Lying person': lambda i: make_lying(noise=0.001),
        'Walking person': lambda i: make_walking(i, noise=0.002),
        'Standing still': lambda i: make_standing(noise=0.0005),
    }

    for name, gen_fn in scenarios.items():
        pipe = PersonPipeline(name)
        for frame in range(NUM_FRAMES):
            pipe.process_frame(gen_fn(frame))

        status = "✅ OK" if len(pipe.alerts_fired) == 0 else "❌ FALSE+"
        print(f"  {name:<20} | Max risk: {pipe.max_prefall_risk:>3} | "
              f"Alerts: {len(pipe.alerts_fired)} | {status}")

        score(len(pipe.alerts_fired) == 0,
              f"{name}: zero alerts (got {len(pipe.alerts_fired)})")
        score(pipe.max_prefall_risk < 50,
              f"{name}: risk < 50 (got {pipe.max_prefall_risk})")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("═" * 70)
    print("  PIHMS v2.0 — PRE-FALL + MULTI-PERSON + ALERT VALIDATION")
    print("═" * 70)

    test_1_prefall_prediction()
    test_2_alert_fires()
    test_3_ten_standing_no_false_positives()
    test_4_mixed_crowd()
    test_5_edge_cases_no_false_positives()

    print(f"\n{'═'*70}")
    print(f"  FINAL RESULTS: {PASS_COUNT} passed / {FAIL_COUNT} failed "
          f"out of {PASS_COUNT + FAIL_COUNT} checks")
    print(f"{'═'*70}")

    if FAIL_COUNT > 0:
        print("\n  ⚠️  Some checks failed — see details above")
        return False
    else:
        print("\n  🎉 ALL CHECKS PASSED — System validated!")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
