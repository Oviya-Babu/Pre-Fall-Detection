"""
Lateral Body Sway Signal Analyzer
Detects excessive horizontal oscillation of hip midpoint.
"""
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class LateralSway:
    def __init__(self, window_size=150, sway_multiplier=2.0):
        """
        Initialize lateral sway analyzer.
        
        Args:
            window_size (int): Number of frames to keep in history (about 5 seconds at 30fps)
            sway_multiplier (float): Threshold multiplier for personal baseline
        """
        self.window_size = window_size
        self.sway_multiplier = sway_multiplier
        
        # History of hip midpoint x position (horizontal)
        self.hip_x_history = deque(maxlen=window_size)
        
        # Personal baseline sway (standard deviation of hip x position)
        self.baseline_sway = None
        self.baseline_samples = deque(maxlen=300)  # Collect 10 seconds for baseline
        self.baseline_established = False
        
        logger.info("LateralSway analyzer initialized")
    
    def analyze(self, keypoints):
        """
        Analyze lateral sway from pose keypoints.
        
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
        
        # Compute hip midpoint x position (horizontal)
        hip_x = (left_hip[1] + right_hip[1]) / 2.0
        
        # Add to history
        self.hip_x_history.append(hip_x)
        
        # Collect baseline samples (first 10 seconds)
        if not self.baseline_established:
            self.baseline_samples.append(hip_x)
            if len(self.baseline_samples) >= 300:  # 10 seconds at 30fps
                self._establish_baseline()
        
        # If baseline not established yet, no risk
        if not self.baseline_established:
            return 0.0
        
        # Compute current sway as standard deviation over window
        if len(self.hip_x_history) >= 10:
            current_sway = np.std(list(self.hip_x_history))
        else:
            current_sway = 0.0
        
        # Compute risk based on exceeding baseline by multiplier
        if self.baseline_sway > 0:
            sway_ratio = current_sway / self.baseline_sway
            if sway_ratio > self.sway_multiplier:
                # Risk increases linearly with how much we exceed the threshold
                risk = (sway_ratio - self.sway_multiplier) / self.sway_multiplier
                return min(1.0, risk)
        
        return 0.0
    
    def _establish_baseline(self):
        """Establish personal baseline sway from initial samples."""
        if len(self.baseline_samples) >= 50:  # Need minimum samples
            self.baseline_sway = np.std(list(self.baseline_samples))
            self.baseline_established = True
            logger.info(f"Baseline sway established: {self.baseline_sway:.4f}")
        else:
            logger.warning("Not enough samples to establish baseline")
