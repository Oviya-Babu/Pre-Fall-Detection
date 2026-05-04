"""
Main entry point for the Pre-Fall Detection System.
Initializes components and starts the processing pipeline.
"""
import time
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Pre-Fall Detection System...")
    
    # Initialize components
    camera = Camera(config.RTSP_URL)
    detector = PersonDetector('../models/ssd_mobilenet_v2.tflite')
    pose_estimator = PoseEstimator('../models/movenet_lightning.tflite')
    
    # Initialize signal analyzers
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
    
    # Initialize risk engine and alert system
    ml_detector = MLPreFallDetector()
    risk_engine = RiskEngine()
    alert_system = AlertSystem()
    db = IncidentDatabase()
    
    logger.info("Components initialized. Starting main loop...")
    
    try:
        while True:
            # Capture frame
            frame = camera.read_frame()
            if frame is None:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Detect person
            person_box = detector.detect(frame)
            if person_box is None:
                # No person detected, skip pose estimation
                alert_system.update_alert_level(0)  # Green
                time.sleep(0.01)
                continue
            
            # Estimate pose
            keypoints = pose_estimator.estimate(frame, person_box)
            if keypoints is None or len(keypoints) == 0:
                alert_system.update_alert_level(0)
                time.sleep(0.01)
                continue
            
            # Analyze each signal
            gait_risk = gait_analyzer.analyze(keypoints)
            sway_risk = sway_analyzer.analyze(keypoints)
            trunk_risk = trunk_analyzer.analyze(keypoints)
            bed_exit_risk = bed_exit_analyzer.analyze(keypoints, person_box)
            freeze_risk = freeze_analyzer.analyze(keypoints)
            arm_reach_risk = arm_reach_analyzer.analyze(keypoints)
            ml_risk = ml_detector.analyze(keypoints)
            
            # Combine signals to get risk level
            risk_level = risk_engine.compute_risk(
                gait_risk, sway_risk, trunk_risk, bed_exit_risk,
                freeze_risk, arm_reach_risk, ml_risk
            )
            
            # Update alert system (GPIO and MQTT)
            alert_system.update_alert_level(risk_level)
            
            # Log incident if risk level is YELLOW or RED
            if risk_level >= 1:  # YELLOW or RED
                db.log_incident(risk_level, {
                    'gait': gait_risk,
                    'sway': sway_risk,
                    'trunk': trunk_risk,
                    'bed_exit': bed_exit_risk,
                    'freeze': freeze_risk,
                    'arm_reach': arm_reach_risk
                })
            
            # Small delay to control loop speed
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        camera.release()
        alert_system.cleanup()
        db.close()

if __name__ == "__main__":
    main()