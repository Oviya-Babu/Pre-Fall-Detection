"""
Freezing of Gait Signal Analyzer
Detects sudden drop in velocity to near-zero during motion (Parkinson's indicator).
"""
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class FreezingOfGait:
    def __init__(self, window_size=150, freeze_velocity=0.02):
        """
        Initialize freezing of gait analyzer.
        
        Args:
            window_size (int): Number of frames to keep in history (about 5 seconds at 30fps)
            freeze_velocity (float): Velocity threshold below which freezing is considered
        """
        self.window_size = window_size
        self.freeze_velocity = freeze_velocity
        
        # History of hip midpoint position to compute velocity
        self.hip_history = deque(maxlen=window_size)
        # History of velocities
        self.velocity_history = deque(maxlen=window_size)
        
        # For detecting motion: we need to know if the person was moving before freezing
        self.motion_history = deque(maxlen=window_size)  # Binary: 1 if moving, 0 if frozen
        
        # Baseline velocity during normal walking (will be updated)
        self.baseline_velocity = None
        self.baseline_established = False
        
        logger.info("FreezingOfGait analyzer initialized")
    
    def analyze(self, keypoints):
        """
        Analyze freezing of gait from pose keypoints.
        
        Args:
            keypoints (numpy.ndarray): Array of shape (17, 3) [y, x, confidence] in normalized coordinates
            
        Returns:
            float: Risk score between 0.0 and 1.0, where higher means more risk
        """
        if keypoints is None or len(keypoints) < 17:
            return 0.0
        
        # Extract hip keypoints (MoveNet indices: left hip=11, right hip=12)
        left_hip = keypoints[11]   # [y, x, confidence]
        right_hip = keypoints[12]
        
        # Check confidence
        if left_hip[2] < 0.5 or right_hip[2] < 0.5:
            # Not confident enough in hip detection
            return 0.0
        
        # Compute hip midpoint position
        hip_x = (left_hip[1] + right_hip[1]) / 2.0
        hip_y = (left_hip[0] + right_hip[0]) / 2.0
        hip_pos = np.array([hip_x, hip_y])
        
        # Add to history
        self.hip_history.append(hip_pos)
        
        # Compute velocity if we have at least 2 points
        if len(self.hip_history) >= 2:
            # Compute displacement over last 2 frames (simple velocity)
            pos_now = self.hip_history[-1]
            pos_prev = self.hip_history[-2]
            displacement = np.linalg.norm(pos_now - pos_prev)
            # Velocity is displacement per frame (we'll assume constant frame time)
            velocity = displacement
            
            self.velocity_history.append(velocity)
            
            # Determine if currently moving (above threshold)
            is_moving = velocity > self.freeze_velocity
            self.motion_history.append(1 if is_moving else 0)
            
            # Update baseline velocity during normal walking
            # We'll establish baseline when we have consistent movement
            if len(self.velocity_history) >= 50 and not self.baseline_established:
                # Use median of recent velocities when we detect movement
                recent_velocities = list(self.velocity_history)[-50:]
                # Only consider velocities that suggest movement (above a low threshold)
                moving_velocities = [v for v in recent_velocities if v > 0.005]  # Adjust as needed
                if len(moving_velocities) >= 10:
                    self.baseline_velocity = np.median(moving_velocities)
                    self.baseline_established = True
                    logger.info(f"Baseline velocity established: {self.baseline_velocity:.4f}")
            
            # Detect freezing: sudden drop in velocity during movement
            risk = 0.0
            
            # Check if we were moving recently and now velocity is very low
            if len(self.motion_history) >= 10:
                # Check if we were moving in the recent past (last 5 frames)
                recent_motion = sum(list(self.motion_history)[-5:])
                if recent_motion >= 3:  # Was moving recently
                    # And now velocity is very low
                    current_velocity = self.velocity_history[-1] if self.velocity_history else 0
                    if current_velocity < self.freeze_velocity:
                        # Compute risk based on how low the velocity is compared to baseline
                        if self.baseline_established and self.baseline_velocity > 0:
                            velocity_ratio = current_velocity / self.baseline_velocity
                            # Risk increases as velocity ratio decreases
                            # If velocity is 0, risk = 1.0
                            # If velocity is baseline, risk = 0
                            # We'll use an inverse relationship
                            risk = max(0, 1.0 - velocity_ratio)
                        else:
                            # If no baseline, use absolute threshold
                            # Risk increases as velocity decreases below freeze_velocity
                            # We'll cap the risk at 1.0 when velocity is 0
                            risk = max(0, (self.freeze_velocity - current_velocity) / self.freeze_velocity)
                        
                        # Apply a temporal component: freezing should persist for a moment
                        # Check if low velocity has persisted for a few frames
                        if len(self.velocity_history) >= 5:
                            recent_velocities = list(self.velocity_history)[-5:]
                            # Count how many recent velocities are below threshold
                            low_velocity_count = sum(1 for v in recent_velocities if v < self.freeze_velocity)
                            if low_velocity_count >= 3:  # At least 3 of last 5 frames show low velocity
                                # Boost the risk score
                                risk = min(1.0, risk * 1.5)
            
            return min(1.0, risk)
        
        return 0.0
