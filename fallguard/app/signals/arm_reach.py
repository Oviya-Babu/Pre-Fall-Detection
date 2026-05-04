"""
Arm Reach / Balance Grab Signal Analyzer
Detects rapid outward movement of wrist beyond 0.3x body width from body midline.
"""
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ArmReach:
    def __init__(self, window_size=150, reach_ratio=0.3):
        """
        Initialize arm reach analyzer.
        
        Args:
            window_size (int): Number of frames to keep in history (about 5 seconds at 30fps)
            reach_ratio (float): Threshold ratio of wrist distance from midline to body width
        """
        self.window_size = window_size
        self.reach_ratio = reach_ratio
        
        # History of wrist positions (left and right)
        self.left_wrist_history = deque(maxlen=window_size)
        self.right_wrist_history = deque(maxlen=window_size)
        
        # History of body width (shoulder width) to compute dynamic threshold
        self.shoulder_width_history = deque(maxlen=window_size)
        
        # For detecting rapid movement: we need to detect sudden increases in wrist distance
        self.left_wrist_velocity_history = deque(maxlen=30)  # Last 1 second
        self.right_wrist_velocity_history = deque(maxlen=30)
        
        logger.info("ArmReach analyzer initialized")
    
    def analyze(self, keypoints):
        """
        Analyze arm reach from pose keypoints.
        
        Args:
            keypoints (numpy.ndarray): Array of shape (17, 3) [y, x, confidence] in normalized coordinates
            
        Returns:
            float: Risk score between 0.0 and 1.0, where higher means more risk
        """
        if keypoints is None or len(keypoints) < 17:
            return 0.0
        
        # Extract wrist and shoulder keypoints
        # MoveNet indices: left_wrist=9, right_wrist=10, left_shoulder=5, right_shoulder=6
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        
        # Check confidence
        min_conf = 0.5
        if (left_wrist[2] < min_conf or right_wrist[2] < min_conf or
            left_shoulder[2] < min_conf or right_shoulder[2] < min_conf):
            return 0.0
        
        # Compute shoulder midpoint (body midline x position)
        shoulder_mid_x = (left_shoulder[1] + right_shoulder[1]) / 2.0
        
        # Compute body width (shoulder distance)
        shoulder_width = abs(right_shoulder[1] - left_shoulder[1])
        # Add small epsilon to avoid division by zero
        shoulder_width = max(shoulder_width, 1e-6)
        
        # Store shoulder width for history
        self.shoulder_width_history.append(shoulder_width)
        
        # Compute wrist distances from midline
        left_wrist_x = left_wrist[1]
        right_wrist_x = right_wrist[1]
        
        left_distance = abs(left_wrist_x - shoulder_mid_x)
        right_distance = abs(right_wrist_x - shoulder_mid_x)
        
        # Store wrist positions
        self.left_wrist_history.append(left_wrist_x)
        self.right_wrist_history.append(right_wrist_x)
        
        # Compute reach ratios (distance from midline normalized by body width)
        left_reach_ratio = left_distance / shoulder_width
        right_reach_ratio = right_distance / shoulder_width
        
        # Check if either wrist exceeds the reach ratio threshold
        left_exceeds = left_reach_ratio > self.reach_ratio
        right_exceeds = right_reach_ratio > self.reach_ratio
        
        if not left_exceeds and not right_exceeds:
            # Neither wrist is beyond threshold
            return 0.0
        
        # If we exceed threshold, check if it's a rapid movement (potential grab)
        # Compute wrist velocity (change in position over recent frames)
        left_velocity = 0.0
        right_velocity = 0.0
        
        if len(self.left_wrist_history) >= 5:
            # Compute velocity as change in x position over last few frames
            recent_left = list(self.left_wrist_history)[-5:]
            if len(recent_left) >= 2:
                left_velocity = abs(recent_left[-1] - recent_left[0]) / len(recent_left)
        
        if len(self.right_wrist_history) >= 5:
            recent_right = list(self.right_wrist_history)[-5:]
            if len(recent_right) >= 2:
                right_velocity = abs(recent_right[-1] - recent_right[0]) / len(recent_right)
        
        # Convert velocity to a normalized measure (velocity per frame relative to shoulder width)
        # We'll use the average shoulder width from history for normalization
        avg_shoulder_width = np.mean(list(self.shoulder_width_history)) if self.shoulder_width_history else shoulder_width
        avg_shoulder_width = max(avg_shoulder_width, 1e-6)
        
        left_norm_velocity = left_velocity / avg_shoulder_width
        right_norm_velocity = right_velocity / avg_shoulder_width
        
        # Risk factors:
        # 1. How much we exceed the threshold (reach ratio)
        # 2. How rapid the movement is (velocity)
        
        left_excess = max(0, left_reach_ratio - self.reach_ratio)
        right_excess = max(0, right_reach_ratio - self.reach_ratio)
        
        # Normalize excess by the threshold to get a ratio
        left_excess_ratio = left_excess / self.reach_ratio if self.reach_ratio > 0 else 0
        right_excess_ratio = right_excess / self.reach_ratio if self.reach_ratio > 0 else 0
        
        # Risk from position (how far beyond threshold)
        position_risk = max(left_excess_ratio, right_excess_ratio)
        
        # Risk from velocity (how rapid the movement)
        # We'll consider velocity risky if it's above a threshold
        velocity_threshold = 0.1  # Adjust as needed
        left_velocity_risk = min(1.0, left_norm_velocity / velocity_threshold) if left_norm_velocity > 0 else 0
        right_velocity_risk = min(1.0, right_norm_velocity / velocity_threshold) if right_norm_velocity > 0 else 0
        velocity_risk = max(left_velocity_risk, right_velocity_risk)
        
        # Combine position and velocity risks
        # We want high risk if either we are far beyond threshold OR we moved rapidly beyond threshold
        risk = max(position_risk, velocity_risk)
        
        # But we also want to require that the wrist is actually beyond the threshold
        # If not beyond threshold, risk should be low even if velocity is high
        if not left_exceeds and not right_exceeds:
            risk = 0.0
        
        logger.debug(f"Arm reach: left_ratio={left_reach_ratio:.3f}, right_ratio={right_reach_ratio:.3f}, "
                     f"left_vel={left_norm_velocity:.3f}, right_vel={right_norm_velocity:.3f}, risk={risk:.3f}")
        
        return min(1.0, risk)
