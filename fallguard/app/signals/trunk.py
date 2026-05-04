"""
Trunk Forward Lean Signal Analyzer
Detects excessive forward lean of the trunk.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TrunkLean:
    def __init__(self, yellow_thresh=20, red_thresh=35):
        """
        Initialize trunk lean analyzer.
        
        Args:
            yellow_thresh (float): Angle threshold for YELLOW alert (degrees)
            red_thresh (float): Angle threshold for RED alert (degrees)
        """
        self.yellow_thresh = np.radians(yellow_thresh)  # Convert to radians
        self.red_thresh = np.radians(red_thresh)       # Convert to radians
        
        logger.info(f"TrunkLean analyzer initialized: yellow={yellow_thresh}°, red={red_thresh}°")
    
    def analyze(self, keypoints):
        """
        Analyze trunk lean from pose keypoints.
        
        Args:
            keypoints (numpy.ndarray): Array of shape (17, 3) [y, x, confidence] in normalized coordinates
            
        Returns:
            float: Risk score between 0.0 and 1.0, where higher means more risk
                   (0.0 = no lean, 0.5 = yellow threshold, 1.0 = red threshold)
        """
        if keypoints is None or len(keypoints) < 17:
            return 0.0
        
        # Extract shoulder and hip keypoints
        # MoveNet indices: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        # Check confidence
        min_conf = 0.5
        if (left_shoulder[2] < min_conf or right_shoulder[2] < min_conf or
            left_hip[2] < min_conf or right_hip[2] < min_conf):
            return 0.0
        
        # Compute shoulder midpoint
        shoulder_mid = np.array([
            (left_shoulder[0] + right_shoulder[0]) / 2.0,
            (left_shoulder[1] + right_shoulder[1]) / 2.0
        ])
        
        # Compute hip midpoint
        hip_mid = np.array([
            (left_hip[0] + right_hip[0]) / 2.0,
            (left_hip[1] + right_hip[1]) / 2.0
        ])
        
        # Compute vector from hip to shoulder (spine vector)
        spine_vector = shoulder_mid - hip_mid
        
        # Compute vertical vector (pointing upward in image coordinates)
        # In image coordinates, y increases downward, so upward is negative y direction
        vertical_vector = np.array([-1.0, 0.0])  # Pointing upward (y component first)
        
        # Compute angle between spine vector and vertical
        # Normalize vectors
        spine_norm = np.linalg.norm(spine_vector)
        if spine_norm < 1e-6:  # Avoid division by zero
            return 0.0
            
        spine_unit = spine_vector / spine_norm
        vertical_unit = vertical_vector / np.linalg.norm(vertical_vector)  # Already unit length
        
        # Compute dot product and clamp to [-1, 1] for arccos
        dot_product = np.clip(np.dot(spine_unit, vertical_unit), -1.0, 1.0)
        angle = np.arccos(dot_product)  # Angle in radians
        
        # Convert to degrees for easier threshold comparison
        angle_degrees = np.degrees(angle)
        
        logger.debug(f"Trunk lean angle: {angle_degrees:.1f}°")
        
        # Compute risk score using radians for correct comparison
        if angle >= self.red_thresh:
            return 1.0  # RED threshold exceeded
        elif angle >= self.yellow_thresh:
            # Linear interpolation between yellow and red thresholds
            return 0.5 + 0.5 * ((angle - self.yellow_thresh) / (self.red_thresh - self.yellow_thresh))
        else:
            # Below yellow threshold: proportional to yellow threshold
            if self.yellow_thresh > 0:
                return 0.5 * (angle / self.yellow_thresh)
            else:
                return 0.0
