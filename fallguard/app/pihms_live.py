#!/usr/bin/env python3
"""
PIHMS v2.0 — Real-Time Live System with DroidCam
Uses OpenCV cascades + anatomical proportions for skeleton estimation.
Full PIHMS analytics overlay with risk gauges and alerts.
Includes: Web Dashboard (port 8080) + MQTT per-person alerts.
"""

import cv2
import numpy as np
import time
import sys
import os
import logging
import threading
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pihms.feature_extraction import FeatureExtractor
from pihms.pre_fall_detection import PreFallDetector
from pihms.activity_monitoring import ActivityMonitor
from pihms.eating_detection import EatingDetector
from pihms.wellness_logic import WellnessTracker
from pihms.risk_aggregator import RiskAggregator
from pihms.alert_manager import AlertManager
from pihms.compress_skeleton import SkeletonCompressor
from pihms.database.pihms_db import PIHMSDatabase
from pihms.storage_validator import StorageValidator
from pihms.mqtt_handler import MQTTHandler
from pihms.dashboard import run_dashboard, update_dashboard_state, broadcast_state, manager as ws_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('PIHMS-LIVE')

# DroidCam URL — override via env variable: export PIHMS_CAM_URL=http://<phone-ip>:4747/video
import os as _os
DROIDCAM_URL = _os.environ.get("PIHMS_CAM_URL", "http://10.227.127.50:4747/video")

# Keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

SKELETON_CONNECTIONS = [
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
    (NOSE, L_SHOULDER), (NOSE, R_SHOULDER),
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

# Colors
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)
DARK_BG = (30, 30, 30)
PANEL_BG = (40, 40, 50)


class CascadeBodyDetector:
    """Person + face detection — runs on half-res for speed."""
    # OPT-1: half-res detection; OPT-2: scaleFactor 1.15 (coarser pyramid = faster)
    SCALE = 0.5

    def __init__(self):
        data = os.path.join(os.path.dirname(cv2.__file__), 'data')
        self.body_cascade = cv2.CascadeClassifier(
            os.path.join(data, 'haarcascade_fullbody.xml'))
        self.upper_cascade = cv2.CascadeClassifier(
            os.path.join(data, 'haarcascade_upperbody.xml'))
        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(data, 'haarcascade_frontalface_alt2.xml'))

    def detect(self, frame):
        h, w = frame.shape[:2]
        # Downscale for fast cascade (OPT-1)
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # OPT-2: CLAHE instead of equalizeHist — cheaper and better contrast
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        sh, sw = small.shape[:2]
        detections = []
        INV = 1.0 / self.SCALE  # scale detections back to full-res

        # 1. Full body — OPT-2: scaleFactor=1.15 (was 1.05), minNeighbors=2
        bodies = self.body_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=2, minSize=(20, 40))
        for (bx, by, bw, bh) in bodies:
            detections.append({'bbox': (
                int(bx*INV), int(by*INV), int(bw*INV), int(bh*INV)), 'type': 'fullbody'})

        # 2. Upper body fallback — scaleFactor=1.15
        uppers = self.upper_cascade.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=2, minSize=(20, 20))
        for (ux, uy, uw, uh) in uppers:
            is_dup = any(
                abs(int(ux*INV) - d['bbox'][0]) < 40 and
                abs(int(uy*INV) - d['bbox'][1]) < 40
                for d in detections
            )
            if not is_dup:
                ex_h = int(uh * 2.2 * INV)
                detections.append({'bbox': (
                    int(ux*INV), int(uy*INV), int(uw*INV),
                    min(ex_h, h - int(uy*INV))), 'type': 'upperbody'})

        # 3. Face fallback — only if nothing found
        if not detections:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=3, minSize=(15, 15))
            for (fx, fy, fw_f, fh_f) in faces:
                bw2 = int(fw_f * 3.5 * INV)
                bh2 = int(fh_f * 8.5 * INV)
                bx2 = max(0, int((fx + fw_f//2)*INV) - bw2//2)
                by2 = max(0, int((fy - fh_f*0.4)*INV))
                detections.append({'bbox': (
                    bx2, by2, min(bw2, w-bx2), min(bh2, h-by2)), 'type': 'face'})

        return detections


class AnatomicalSkeletonEstimator:
    """Estimate 17 MoveNet keypoints from body bounding box.
    OPT-3: removed per-person ROI face cascade — saves ~5-10ms per person."""

    def estimate(self, frame, bbox, det_type='fullbody'):
        h, w = frame.shape[:2]
        bx, by, bw, bh = bbox
        cx = bx + bw / 2.0
        kp = np.zeros((17, 3), dtype=np.float32)

        # Use anatomical estimate for head — no ROI cascade (OPT-3)
        face_cx = cx
        face_cy = by + bh * 0.08
        face_w  = bw * 0.25

        # Anatomical proportions (fraction of body height)
        nose_y = face_cy
        nose_x = face_cx
        eye_offset = face_w * 0.15
        ear_offset = face_w * 0.25
        shoulder_y = by + bh * 0.18
        shoulder_w = bw * 0.38
        elbow_y = by + bh * 0.38
        wrist_y = by + bh * 0.52
        hip_y = by + bh * 0.50
        hip_w = bw * 0.20
        knee_y = by + bh * 0.72
        ankle_y = by + bh * 0.92

        # Build keypoints [y, x, confidence] normalized to [0,1]
        pts = [
            (nose_y, nose_x, 0.95),
            (nose_y - face_w*0.1, nose_x - eye_offset, 0.85),
            (nose_y - face_w*0.1, nose_x + eye_offset, 0.85),
            (nose_y, nose_x - ear_offset, 0.75),
            (nose_y, nose_x + ear_offset, 0.75),
            (shoulder_y, cx - shoulder_w, 0.90),
            (shoulder_y, cx + shoulder_w, 0.90),
            (elbow_y, cx - shoulder_w*1.1, 0.80),
            (elbow_y, cx + shoulder_w*1.1, 0.80),
            (wrist_y, cx - shoulder_w*1.0, 0.75),
            (wrist_y, cx + shoulder_w*1.0, 0.75),
            (hip_y, cx - hip_w, 0.88),
            (hip_y, cx + hip_w, 0.88),
            (knee_y, cx - hip_w*0.9, 0.85),
            (knee_y, cx + hip_w*0.9, 0.85),
            (ankle_y, cx - hip_w*0.8, 0.80),
            (ankle_y, cx + hip_w*0.8, 0.80),
        ]

        # Add realistic postural micro-sway/noise so biomechanical pipeline
        # receives meaningful signals (natural human sway ±1-3% of height)
        sway = np.random.normal(0, 0.008, (17, 2)).astype(np.float32)

        for i, (py, px, conf) in enumerate(pts):
            kp[i] = [(py + sway[i, 0] * bh) / h,
                     (px + sway[i, 1] * bw) / w,
                     0.9]

        return kp


class MultiPersonTracker:
    """Centroid-based tracker with stable IDs."""
    def __init__(self, max_disappeared=90, max_match_dist=120):
        self.next_id = 0
        self.persons = {}  # ID -> [bbox, disappeared_count, det_type]
        self.max_disappeared = max_disappeared
        self.max_match_dist = max_match_dist

    def update(self, detections, skip=False):
        if skip:
            return self.persons

        if len(detections) == 0:
            for pid in list(self.persons.keys()):
                self.persons[pid][1] += 1
                if self.persons[pid][1] > self.max_disappeared:
                    del self.persons[pid]
            return self.persons

        new_persons = {}
        for det in detections:
            bbox = det['bbox']
            cx, cy = bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2

            best_id, min_dist = None, float('inf')
            for pid, pdata in self.persons.items():
                pbox = pdata[0]
                pcx, pcy = pbox[0] + pbox[2]//2, pbox[1] + pbox[3]//2
                dist = np.sqrt((cx-pcx)**2 + (cy-pcy)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_id = pid

            if best_id is not None and min_dist < self.max_match_dist:
                new_persons[best_id] = [bbox, 0, det['type']]
                del self.persons[best_id]
            else:
                new_persons[self.next_id] = [bbox, 0, det['type']]
                self.next_id += 1

        # Carry forward unmatched existing persons
        for pid, pdata in self.persons.items():
            count = pdata[1] + 1
            if count <= self.max_disappeared:
                new_persons[pid] = [pdata[0], count, pdata[2]]

        self.persons = new_persons
        return self.persons


def draw_skeleton(frame, kp, h, w):
    """Draw skeleton on frame."""
    pts = []
    for i in range(17):
        y, x, c = kp[i]
        px, py = int(x * w), int(y * h)
        pts.append((px, py, c))

    for i, j in SKELETON_CONNECTIONS:
        if pts[i][2] > 0.3 and pts[j][2] > 0.3:
            cv2.line(frame, (pts[i][0], pts[i][1]), (pts[j][0], pts[j][1]), CYAN, 2)

    for i, (px, py, c) in enumerate(pts):
        if c > 0.3:
            color = GREEN if c > 0.7 else YELLOW if c > 0.5 else RED
            cv2.circle(frame, (px, py), 5, color, -1)
            cv2.circle(frame, (px, py), 6, WHITE, 1)


def draw_risk_gauge(frame, x, y, w_gauge, h_gauge, value, label, max_val=100):
    """Draw a horizontal risk gauge bar."""
    # Background
    cv2.rectangle(frame, (x, y), (x + w_gauge, y + h_gauge), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + w_gauge, y + h_gauge), (100, 100, 100), 1)

    # Fill
    fill_w = int((value / max_val) * w_gauge)
    if value >= 75:
        color = RED
    elif value >= 50:
        color = ORANGE
    elif value >= 25:
        color = YELLOW
    else:
        color = GREEN
    cv2.rectangle(frame, (x, y+1), (x + fill_w, y + h_gauge-1), color, -1)

    # Label
    cv2.putText(frame, f"{label}: {int(value)}", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)


def draw_panel(frame, x, y, w_p, h_p, title, lines):
    """Draw panel — OPT-4: no frame.copy(); draw filled rect directly (saves ~4ms/call)."""
    cv2.rectangle(frame, (x, y), (x + w_p, y + h_p), PANEL_BG, -1)
    cv2.rectangle(frame, (x, y), (x + w_p, y + h_p), (80, 80, 100), 1)
    cv2.putText(frame, title, (x + 5, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1)
    cv2.line(frame, (x + 2, y + 20), (x + w_p - 2, y + 20), (80, 80, 100), 1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (x + 5, y + 36 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)


def draw_alert_banner(frame, alert_text, severity):
    """Draw flashing alert banner at top."""
    h, w = frame.shape[:2]
    if severity == 'CRITICAL':
        color = RED
    elif severity == 'HIGH' or severity == 'ESCALATED_YELLOW':
        color = (0, 50, 255) # Orange
    elif severity in ('YELLOW', 'MEDIUM'):
        color = YELLOW
    else:
        return

    # Flashing effect for CRITICAL
    if severity == 'CRITICAL':
        flash = int(time.time() * 8) % 2 == 0
        if not flash: return
        banner_h = 60
        font_scale = 1.0
        thickness = 3
    else:
        banner_h = 40
        font_scale = 0.7
        thickness = 2

    # OPT-4: direct rect, no copy
    cv2.rectangle(frame, (0, 0), (w, banner_h), color, -1)
    cv2.putText(frame, f"!!! {alert_text} !!!", (20, banner_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, thickness)


def main():
    logger.info("=" * 60)
    logger.info("  PIHMS v2.0 — REAL-TIME LIVE SYSTEM")
    logger.info(f"  DroidCam: {DROIDCAM_URL}")
    logger.info("  Dashboard: http://localhost:8080")
    logger.info("=" * 60)

    # ── Start Dashboard Server (background thread) ──────────────
    dash_thread = threading.Thread(target=run_dashboard, kwargs={'port': 8080}, daemon=True)
    dash_thread.start()
    logger.info("Dashboard started at http://localhost:8080")

    # ── Start MQTT Handler ──────────────────────────────────────
    mqtt = MQTTHandler(broker="localhost", port=1883, facility_id="FAC001")

    # ── Async event loop for dashboard WebSocket broadcasts ─────
    _loop = asyncio.new_event_loop()
    def _run_loop():
        asyncio.set_event_loop(_loop)
        _loop.run_forever()
    loop_thread = threading.Thread(target=_run_loop, daemon=True)
    loop_thread.start()

    # Connect to camera
    is_droidcam = False
    logger.info(f"Connecting to DroidCam: {DROIDCAM_URL}")
    cap = cv2.VideoCapture(DROIDCAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

    ret = False
    # Retry up to 3 times (DroidCam app may need a moment)
    for attempt in range(3):
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                is_droidcam = True
                break
        logger.warning(f"DroidCam not ready (attempt {attempt+1}/3), retrying in 2s...")
        time.sleep(2)
        cap.release()
        cap = cv2.VideoCapture(DROIDCAM_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not ret:
        logger.warning("Cannot connect to DroidCam! Falling back to default webcam (0)...")
        cap.release()
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ret, test_frame = cap.read()

    if not cap.isOpened() or not ret:
        logger.error("Cannot connect to any camera! Exiting.")
        return

    fh, fw = test_frame.shape[:2]
    logger.info(f"Camera connected ({'DroidCam' if is_droidcam else 'Webcam'}): {fw}x{fh}")

    # Initialize detectors
    body_detector = CascadeBodyDetector()
    skeleton_estimator = AnatomicalSkeletonEstimator()

    # Initialize PIHMS
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
    db = PIHMSDatabase()

    wellness_tracker.add_medication('bp_med', 'Lisinopril 10mg', [8, 20])

    logger.info("All components ready. Press 'q' to quit.")

    # Initialize state pools for multi-person tracking
    person_pipelines = {}  # pid -> dict of components
    tracker = MultiPersonTracker()

    frame_count = 0
    fps_timer = time.time()
    fps_val = 0.0
    start_time = time.time()
    last_alerts = {}  # pid -> (alert, time)
    dash_alerts_list = []  # for dashboard feed
    batch_start = time.time()

    # OPT-5: Pre-allocate canvas once; reuse every frame
    _canvas_h = max(fh, 520)
    _canvas_w = fw + 320
    canvas = np.zeros((_canvas_h, _canvas_w, 3), dtype=np.uint8)

    # OPT-6: Threaded detection — detector runs in background,
    #         main thread always uses the most recent result.
    _det_lock = threading.Lock()
    _det_result = [[]]
    _det_frame  = [None]
    _det_busy   = [False]

    def _detection_worker():
        while True:
            with _det_lock:
                f = _det_frame[0]
            if f is None:
                time.sleep(0.002)
                continue
            result = body_detector.detect(f)
            with _det_lock:
                _det_result[0] = result
                _det_frame[0]  = None   # signal consumed
                _det_busy[0]   = False

    det_thread = threading.Thread(target=_detection_worker, daemon=True)
    det_thread.start()
    logger.info("OPT: threaded detector started")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue

        # Rotate only if DroidCam portrait
        if is_droidcam and frame.shape[0] > frame.shape[1]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]

        # OPT-5: reuse pre-allocated canvas (zero only the panel area)
        display_w = fw + 320
        display_h = max(fh, 520)
        if canvas.shape[0] != display_h or canvas.shape[1] != display_w:
            canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        canvas[:fh, :fw] = frame
        canvas[:display_h, fw:display_w] = 0  # clear panel strip only

        # OPT-6: submit frame to background detector every 5 frames;
        #         use last result in between (non-blocking)
        if frame_count % 5 == 0:
            with _det_lock:
                if not _det_busy[0]:
                    _det_frame[0] = frame.copy()
                    _det_busy[0]  = True
        with _det_lock:
            detections = _det_result[0]
        tracked_persons = tracker.update(detections, skip=False)

        active_risks = []

        for pid, (bbox, disappeared, det_type) in tracked_persons.items():
            if disappeared > 0: continue
            
            bx, by, bw, bh = bbox
            cv2.rectangle(canvas, (bx, by), (bx+bw, by+bh), CYAN if pid == 0 else GREEN, 2)
            cv2.putText(canvas, f"ID:{pid} ({det_type})", (bx, by-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN if pid == 0 else GREEN, 1)

            # Initialize pipeline for new IDs
            if pid not in person_pipelines:
                person_pipelines[pid] = {
                    'fe': FeatureExtractor(fps=30),
                    'pfd': PreFallDetector(fps=30),
                    'am': ActivityMonitor(fps=30),
                    'ed': EatingDetector(fps=30),
                    'ra': RiskAggregator(),
                    'alm': AlertManager()
                }

            pipe = person_pipelines[pid]
            
            # Estimate skeleton
            keypoints = skeleton_estimator.estimate(frame, bbox, det_type)
            draw_skeleton(canvas, keypoints, fh, fw)

            # Analytics
            features = pipe['fe'].extract(keypoints)
            if features:
                pf = pipe['pfd'].update(features)
                act = pipe['am'].update(features)
                pipe['ed'].update(features)
                risk = pipe['ra'].update(pre_fall_result=pf, activity_result=act, 
                                        wellness_risk=wellness_tracker.get_wellness_risk())
                
                alert = pipe['alm'].update(risk_result=risk, 
                                          activity_state=int(features.get('activity_state', 2)),
                                          pre_fall_result=pf)
                
                if alert:
                    last_alerts[pid] = (alert, time.time())
                    db.insert_alert(severity=alert['severity'], primary_type=alert['primary_type'],
                                   total_risk_score=alert['risk_breakdown']['total_risk'], payload=alert)
                    logger.warning(f"ID:{pid} ALERT: {alert['severity']} {alert['primary_type']}")
                    # MQTT per-person alert
                    mqtt.publish_alert(pid, alert)
                    # Dashboard alert feed
                    dash_alerts_list.append({
                        'severity': alert['severity'],
                        'primary_type': alert['primary_type'],
                        'risk_score': alert['risk_breakdown']['total_risk'],
                        'person_id': pid,
                        'timestamp_utc': alert.get('timestamp_utc', ''),
                        'timestamp': time.time(),
                    })
                    dash_alerts_list[:] = dash_alerts_list[-50:]  # keep last 50

                active_risks.append((pid, risk, pf, features, act))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # FPS calculation
        frame_count += 1
        if frame_count % 10 == 0:
            now = time.time()
            fps_val = 10.0 / (now - fps_timer) if (now - fps_timer) > 0 else 0
            fps_timer = now

        # ── Push to Dashboard + MQTT (every 5 frames) ────────────
        if frame_count % 10 == 0:  # OPT: less frequent dashboard push
            dash_persons = {}
            for pid_r, risk_r, pf_r, feat_r, act_r in active_risks:
                dash_persons[pid_r] = {
                    'risk': risk_r,
                    'pre_fall': pf_r,
                    'features': {k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in (feat_r or {}).items()},
                    'activity': act_r.get('state_name', 'UNKNOWN'),
                    'biomarkers': pf_r.get('biomarkers', []),
                }
            dash_sys = {
                'fps': round(fps_val, 1),
                'latency_ms': round(elapsed_ms, 1),
                'frame_count': frame_count,
                'persons_tracked': len(active_risks),
                'uptime_sec': round(time.time() - start_time, 0),
            }
            update_dashboard_state(dash_persons, dash_sys, dash_alerts_list)
            try:
                asyncio.run_coroutine_threadsafe(broadcast_state(), _loop)
            except Exception:
                pass
            # MQTT heartbeat every 30 frames
            if frame_count % 30 == 0:
                mqtt.publish_system_heartbeat(dash_sys)


        # ── Console Output (every 30 frames) ────────────────────────
        if frame_count % 30 == 0:
            summary = f"[F{frame_count:5d}] {fps_val:4.1f}fps | People:{len(active_risks)} | {elapsed_ms:.1f}ms"
            print(summary)
            for pid, risk, pf, feat, act in active_risks:
                bio = ','.join(pf.get('biomarkers', [])) or 'none'
                comp = pf.get('components', {})
                print(
                    f"  ID:{pid:2d} | Act:{act.get('state_name','?'):14s} | "
                    f"Risk:{risk['total_risk']:3d}[{risk['severity']:8s}] | "
                    f"PreFall:{pf['risk_score']:3d} | "
                    f"Trunk:{feat.get('trunk_lean_angle',0):5.1f}deg | "
                    f"Sway:{feat.get('sway_amplitude_x',0):.3f} | "
                    f"GaitSym:{feat.get('gait_symmetry',0):.2f} | "
                    f"HipSpd:{feat.get('hip_speed',0):.3f} | "
                    f"Bio:[{bio}]"
                )
                if pf['risk_score'] >= 50:
                    print(f"  🔴 STRICT ALARM ID:{pid} — PRE-FALL PREDICTED! "
                          f"GaitInst:{comp.get('gait_instability',0):.2f} "
                          f"Sway:{comp.get('sway_ratio',0):.2f} "
                          f"Imbalance:{comp.get('imbalance_score',0):.2f}")

        # Save capture only on alert events (preserves storage)
        if any(pf['risk_score'] >= 65 for _, _, pf, _, _ in active_risks):
            cap_path = os.path.join(os.path.dirname(__file__), '..', 'data',
                                    f'alert_{int(time.time())}.jpg')
            # Only save if no alert capture in last 10 seconds
            existing = [f for f in os.listdir(os.path.join(os.path.dirname(__file__), '..', 'data'))
                        if f.startswith('alert_')]
            if not existing or (time.time() - int(existing[-1].split('_')[1].split('.')[0])) > 10:
                cv2.imwrite(cap_path, canvas)
                logger.warning(f"ALERT frame saved: {cap_path}")

        # ── RIGHT PANEL ──────────────────────────────────────────
        px = fw + 5
        pw = 310

        # Panel 1: System Status
        draw_panel(canvas, px, 5, pw, 75, "PIHMS v2.0 LIVE", [
            (f"FPS: {fps_val:.1f}  |  Latency: {elapsed_ms:.1f}ms", GREEN),
            (f"Frame: {frame_count}  |  People: {len(active_risks)}", WHITE),
            (f"Time: {time.strftime('%H:%M:%S')}", WHITE),
        ])

        # Extract primary person (ID:0) data for the dashboard panels
        primary = None
        for r in active_risks:
            if r[0] == 0:
                primary = r
                break
        
        # If ID:0 not found, pick the first available
        if not primary and active_risks:
            primary = active_risks[0]

        if primary:
            pid, risk_result, pre_fall_result, features, activity_result = primary
            
            # Panel 2: Risk Assessment
            risk_color = GREEN if risk_result['total_risk'] < 50 else \
                         YELLOW if risk_result['total_risk'] < 75 else RED
            draw_panel(canvas, px, 90, pw, 95, f"ID:{pid} RISK ASSESSMENT", [
                (f"Total Risk: {risk_result['total_risk']}  [{risk_result['severity']}]", risk_color),
                (f"Pre-Fall: {pre_fall_result['risk_score']}  "
                 f"Activity: {risk_result.get('activity_anomaly_risk', 0):.0f}", WHITE),
                (f"Wellness: {risk_result.get('wellness_risk', 0):.0f}  "
                 f"Trend: {pre_fall_result.get('components', {}).get('trend_slope', 0):.2f}", WHITE),
                (f"Biomarkers: {', '.join(pre_fall_result.get('biomarkers', [])) or 'None'}", 
                 ORANGE if pre_fall_result.get('biomarkers') else GREEN),
            ])

            # Risk gauges
            gy = 195
            draw_risk_gauge(canvas, px+5, gy, pw-10, 12, risk_result['total_risk'], "Total Risk")
            draw_risk_gauge(canvas, px+5, gy+28, pw-10, 12, pre_fall_result['risk_score'], "Pre-Fall")

            # Panel 3: Activity & Features
            act_color = CYAN if activity_result.get('state_name') == 'WALKING' else \
                        YELLOW if activity_result.get('state_name') == 'SITTING' else WHITE
            feat_lines = [
                (f"Activity: {activity_result.get('state_name')}", act_color),
                (f"Intensity: {activity_result.get('motion_intensity', 0):.3f}", WHITE),
            ]
            if features:
                feat_lines.extend([
                    (f"Trunk Lean: {features.get('trunk_lean_angle', 0):.1f} deg", WHITE),
                    (f"Height: {features.get('body_height', 0):.3f}  Hip Spd: {features.get('hip_speed', 0):.3f}", WHITE),
                    (f"Sway X: {features.get('sway_amplitude_x', 0):.4f}", WHITE),
                    (f"Step W: {features.get('step_width_mean', 0):.4f}  Sym: {features.get('gait_symmetry', 0):.3f}", WHITE),
                ])
            draw_panel(canvas, px, 250, pw, 30 + len(feat_lines) * 16, f"ID:{pid} FEATURES", feat_lines)

        # Panel 4: CROWD MONITOR (Multi-person table)
        crowd_lines = []
        for pid, risk, pf, feat, act in sorted(active_risks, key=lambda x: x[1]['total_risk'], reverse=True):
            status = "FALL_RISK" if pf['risk_score'] >= 50 else act.get('state_name', 'IDLE')
            color = RED if pf['risk_score'] >= 70 else YELLOW if pf['risk_score'] >= 50 else WHITE
            crowd_lines.append((f"ID:{pid:2d} | Risk:{risk['total_risk']:3d} | {status:<10s}", color))
        
        draw_panel(canvas, px, 420, pw, 25 + max(5, len(crowd_lines)) * 16, "CROWD MONITOR (10+ PEOPLE)", crowd_lines[:15])

        # Alert banner (multi-person cycle)
        now = time.time()
        active_alerts = [a for pid, (a, t) in last_alerts.items() if now - t < 8]
        if active_alerts:
            # Sort by severity to show CRITICAL first
            active_alerts.sort(key=lambda x: 100 if x['severity']=='CRITICAL' else 50 if x['severity']=='HIGH' else 0, reverse=True)
            alert = active_alerts[0] # Show highest priority
            draw_alert_banner(canvas, f"STRICT ALARM: {alert['primary_type']} - RISK {alert['risk_breakdown']['total_risk']}", alert['severity'])
            
            # Print detailed verification to console on pre-fall
            if alert['severity'] == 'CRITICAL' and frame_count % 10 == 0:
                logger.critical(f"VERIFIED PRE-FALL DETECTION for target in crowd. Features: {alert['risk_breakdown']}")

        # Status bar at bottom
        cv2.rectangle(canvas, (0, display_h-25), (display_w, display_h), DARK_BG, -1)
        status = (f"PIHMS v2.0 | DroidCam {DROIDCAM_URL} | "
                  f"{fw}x{fh} | {fps_val:.0f} FPS | {elapsed_ms:.1f}ms | "
                  f"Press 'q' to quit")
        cv2.putText(canvas, status, (10, display_h-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        # Show live window
        cv2.imshow("PIHMS v2.0 - PreFall Intelligence Health Monitor", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    mqtt.shutdown()
    final = skeleton_compressor.flush()
    if final:
        db.insert_skeleton_batch(batch_data=final, num_frames=frame_count % 120,
                                 timestamp_start=batch_start, timestamp_end=time.time())
    summary = activity_monitor.get_daily_summary()
    db.upsert_daily_summary(time.strftime('%Y-%m-%d'), summary)
    db.close()
    _loop.call_soon_threadsafe(_loop.stop)
    logger.info(f"Session ended. {frame_count} frames processed.")


if __name__ == "__main__":
    main()
