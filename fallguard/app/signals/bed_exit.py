"""
Unsupported Bed Exit Signal Analyzer
Detects transitions from lying to sitting to standing without pressing nurse call.
"""
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class BedExit:
    def __init__(self, bed_polygon=None):
        """
        Initialize bed exit analyzer.
        
        Args:
            bed_polygon (list): List of (x, y) tuples defining the bed region in pixel coordinates
                               If None, we'll use a default polygon that should be set in config
        """
        # If no polygon provided, we'll set a default that can be overridden
        # But note: the polygon should be in the same coordinate system as our keypoints
        # Our keypoints are in normalized coordinates [0, 1] if we followed the pose estimator
        # However, in the pose estimator we converted to normalized coordinates in the original frame.
        # So the bed polygon should also be in normalized coordinates [0, 1]?
        # Let's assume the config provides the polygon in the same normalized coordinates.
        # We'll store the polygon and use point-in-polygon test.
        
        self.bed_polygon = bed_polygon  # List of (x, y) in normalized coordinates [0, 1]
        
        # State tracking for bed exit
        # We'll track the person's state: lying, sitting, standing, or unknown
        # Based on the position of key body parts (hips, knees, shoulders) relative to the bed
        
        # History of hip midpoint position (to detect movement out of bed)
        self.hip_history = deque(maxlen=30)  # About 1 second at 30fps
        
        # State
        self.state = 'unknown'  # Can be 'lying', 'sitting', 'standing', 'out_of_bed'
        self.last_state_change = 0  # Frame count when state last changed
        self.bed_exit_detected = False  # Whether we have detected a bed exit in the current episode
        
        # For detecting transitions, we need to define what lying, sitting, standing mean
        # We'll use the ratio of hip height to shoulder height and knee bend
        
        logger.info("BedExit analyzer initialized")
    
    def analyze(self, keypoints, person_box=None):
        """
        Analyze bed exit from pose keypoints.
        
        Args:
            keypoints (numpy.ndarray): Array of shape (17, 3) [y, x, confidence] in normalized coordinates
            person_box (tuple): (xmin, ymin, xmax, ymax) of the detected person in pixel coordinates
                               (not used if keypoints are already normalized to frame)
            
        Returns:
            float: Risk score between 0.0 and 1.0, where 1.0 indicates a bed exit is in progress
                   Note: This signal is special because bed exit is a RED alert trigger by itself.
                   We'll return 1.0 if we detect a bed exit transition (lying -> standing) without nurse call.
                   However, we don't have nurse call input, so we'll just detect the physical movement.
                   The specification says: "Unsupported Bed Exit — patient transitions from lying to sitting to standing without pressing nurse call"
                   Since we don't have nurse call input, we will treat any bed exit as a risk (assuming they didn't press the call).
                   In a real system, we would have an input from the nurse call button.
        """
        if keypoints is None or len(keypoints) < 17:
            return 0.0
        
        # Extract key keypoints for posture estimation
        # MoveNet indices: 
        # left_shoulder=5, right_shoulder=6, 
        # left_hip=11, right_hip=12,
        # left_knee=13, right_knee=14,
        # left_ankle=15, right_ankle=16
        
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Check confidence for the key points
        min_conf = 0.5
        if (left_shoulder[2] < min_conf or right_shoulder[2] < min_conf or
            left_hip[2] < min_conf or right_hip[2] < min_conf or
            left_knee[2] < min_conf or right_knee[2] < min_conf):
            # Not enough confidence to determine posture
            return 0.0
        
        # Compute midpoints
        shoulder_mid = np.array([
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0
        ])
        hip_mid = np.array([
            (left_hip[0] + right_hip[0]) / 2.0,
            (left_hip[1] + right_hip[1]) / 2.0
        ])
        knee_mid = np.array([
            (left_knee[0] + right_knee[0]) / 2.0,
            (left_knee[1] + right_knee[1]) / 2.0
        ])
        ankle_mid = np.array([
            (left_ankle[0] + right_ankle[0]) / 2.0,
            (left_ankle[1] + right_ankle[1]) / 2.0
        ])
        
        # Compute body measurements (in normalized coordinates)
        # Shoulder to hip distance (approximate torso length)
        torso_length = np.linalg.norm(shoulder_mid - hip_mid)
        # Hip to ankle distance (approximate leg length)
        leg_length = np.linalg.norm(hip_mid - ankle_mid)
        # Hip to knee distance
        thigh_length = np.linalg.norm(hip_mid - knee_mid)
        # Knee to ankle distance
        shin_length = np.linalg.norm(knee_mid - ankle_mid)
        
        # Compute angles for posture classification
        # Angle of the torso (from hip to shoulder) relative to vertical
        # In image coordinates, y increases downward, so vertical vector is [0, -1] (upward)
        torso_vector = shoulder_mid - hip_mid
        vertical_vector = np.array([0.0, -1.0])  # Upward in image coordinates
        
        # Normalize torso vector
        torso_norm = np.linalg.norm(torso_vector)
        if torso_norm > 1e-6:
            torso_unit = torso_vector / torso_norm
        else:
            torso_unit = torso_vector
        
        # Normalize vertical vector (already unit length)
        vertical_unit = vertical_vector / np.linalg.norm(vertical_vector)
        
        # Dot product for angle with vertical
        dot_product = np.clip(np.dot(torso_unit, vertical_unit), -1.0, 1.0)
        torso_angle_from_vertical = np.arccos(dot_product)  # In radians
        
        # Angle at the knee (between thigh and shin)
        thigh_vector = knee_mid - hip_mid
        shin_vector = ankle_mid - knee_mid
        
        thigh_norm = np.linalg.norm(thigh_vector)
        shin_norm = np.linalg.norm(shin_vector)
        if thigh_norm > 1e-6 and shin_norm > 1e-6:
            thigh_unit = thigh_vector / thigh_norm
            shin_unit = shin_vector / shin_norm
            knee_dot = np.clip(np.dot(thigh_unit, shin_unit), -1.0, 1.0)
            knee_angle = np.arccos(knee_dot)  # In radians (straight leg is 0, bent is >0)
        else:
            knee_angle = 0.0
        
        # Now classify posture based on these features
        # Lying: torso horizontal (torso_angle_from_vertical ~ 90 degrees), legs straight
        # Sitting: torso vertical (~0 degrees), knees bent (~90 degrees)
        # Standing: torso vertical (~0 degrees), legs straight (knee angle ~0)
        
        # Convert to degrees for easier thinking
        torso_angle_deg = np.degrees(torso_angle_from_vertical)
        knee_angle_deg = np.degrees(knee_angle)
        
        # Define thresholds (these may need tuning)
        LYING_TORSO_THRESH = 60   # degrees from vertical: if torso is more than 60 degrees from vertical, consider it horizontal
        SITTING_KNEE_THRESH = 30  # degrees: if knee is bent more than 30 degrees, consider sitting
        STANDING_KNEE_THRESH = 20 # degrees: if knee is bent less than 20 degrees, consider straight leg
        
        # Determine current posture
        if torso_angle_deg > LYING_TORSO_THRESH:
            # Torso is horizontal -> lying
            current_posture = 'lying'
        elif knee_angle_deg > SITTING_KNEE_THRESH:
            # Knees bent -> sitting
            current_posture = 'sitting'
        else:
            # Torso vertical and legs straight -> standing
            current_posture = 'standing'
        
        logger.debug(f"Posture: {current_posture} (torso angle: {torso_angle_deg:.1f}, knee angle: {knee_angle_deg:.1f})")
        
        # Update hip history for movement detection
        self.hip_history.append(hip_mid)
        
        # State machine for bed exit detection
        # We want to detect a transition from lying -> sitting -> standing (or lying -> standing) 
        # that occurs while the person is moving out of the bed region.
        
        # First, check if the person is in the bed region (if we have a polygon)
        in_bed = True
        if self.bed_polygon is not None and len(self.bed_polygon) >= 3:
            # Convert hip_mid to the same coordinate system as the polygon
            # Our hip_mid is in normalized coordinates [0, 1] if that's how we set the polygon
            # We'll assume the polygon is also in normalized coordinates [0, 1]
            in_bed = self._point_in_polygon(hip_mid, self.bed_polygon)
        
        # If the person is not in the bed, we might be tracking an exit
        # But we need to see if they just left the bed
        
        # Simple approach: we consider a bed exit detected if:
        # 1. The person was lying (or sitting) in the bed
        # 2. They transition to standing
        # 3. During the transition, they leave the bed region
        
        # We'll track state changes and bed leaving
        
        # If we are not in bed and we were recently in bed, and we are now standing, then we have exited
        if not in_bed and self.state in ['lying', 'sitting'] and current_posture == 'standing':
            # Potential bed exit
            # We'll return a risk score of 1.0 (RED) because bed exit is a RED trigger
            logger.info("Bed exit detected: transition from {} to standing while leaving bed".format(self.state))
            self.bed_exit_detected = True
            self.state = current_posture
            self.last_state_change = len(self.hip_history)  # Use frame count proxy
            return 1.0
        
        # Update state if we have a clear posture change and we are in bed
        if in_bed:
            if current_posture != self.state and (len(self.hip_history) - self.last_state_change) > 10:  # Avoid rapid toggling
                logger.debug(f"State change: {self.state} -> {current_posture}")
                self.state = current_posture
                self.last_state_change = len(self.hip_history)
        
        # If we are in bed and standing, that's not necessarily an exit (they might be standing in bed)
        # But if they transition from lying/sitting to standing while in bed, we don't count it as exit until they leave
        
        # For now, we'll return 0.0 if no bed exit detected
        return 0.0
    
    def _point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using the ray casting algorithm.
        
        Args:
            point (numpy.ndarray): [x, y] point to check
            polygon (list): List of [x, y] vertices
            
        Returns:
            bool: True if point is inside polygon, False otherwise
        """
        y, x = point  # Note: hip_mid is [y, x] order from keypoints
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
