"""
Gait Instability Signal Analyzer
Detects short, asymmetric, or slow strides over a rolling window.
"""
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class GaitInstability:
    def __init__(self, window_size=150):
        """
        Initialize gait instability analyzer.
        
        Args:
            window_size (int): Number of frames to keep in history (about 5 seconds at 30fps)
        """
        self.window_size = window_size
        # History of ankle positions (left and right) to compute stride length
        self.left_ankle_history = deque(maxlen=window_size)
        self.right_ankle_history = deque(maxlen=window_size)
        # History of stride lengths and times
        self.stride_lengths = deque(maxlen=window_size)
        self.stride_times = deque(maxlen=window_size)
        # Last heel strike times for each foot (to detect steps)
        self.last_left_heel_strike = None
        self.last_right_heel_strike = None
        
        # Baseline values (will be updated during calibration)
        self.baseline_stride_length = None
        self.baseline_stride_time = None
        self.baseline_asymmetry = None
        
        # For detecting instability: we look for significant deviation from baseline
        self.stride_length_threshold = 0.7  # Stride length < 70% of baseline is unstable
        self.stride_time_threshold = 1.3    # Stride time > 130% of baseline is slow
        self.asymmetry_threshold = 0.3      # Asymmetry > 30% is unstable
        
        logger.info("GaitInstability analyzer initialized")
    
    def analyze(self, keypoints):
        """
        Analyze gait instability from pose keypoints.
        
        Args:
            keypoints (numpy.ndarray): Array of shape (17, 3) [y, x, confidence] in normalized coordinates
            
        Returns:
            float: Risk score between 0.0 and 1.0, where higher means more risk
        """
        if keypoints is None or len(keypoints) < 17:
            return 0.0
        
        # Extract ankle keypoints (MoveNet indices: left ankle=15, right ankle=16)
        # MoveNet keypoints: [nose, left_eye, right_eye, left_ear, right_ear,
        #                    left_shoulder, right_shoulder, left_elbow, right_elbow,
        #                    left_wrist, right_wrist, left_hip, right_hip,
        #                    left_knee, right_knee, left_ankle, right_ankle]
        left_ankle = keypoints[15]  # [y, x, confidence]
        right_ankle = keypoints[16]
        
        # Check confidence
        if left_ankle[2] < 0.5 or right_ankle[2] < 0.5:
            # Not confident enough in ankle detection
            return 0.0
        
        # Store ankle positions (we'll use y-coordinate for vertical movement to detect steps)
        # In normalized coordinates, y increases downward
        self.left_ankle_history.append(left_ankle[:2])  # [y, x]
        self.right_ankle_history.append(right_ankle[:2])
        
        # Detect heel strikes (local minima in y-coordinate - ankle moving downward then upward)
        # We look for when the ankle stops going down and starts going up
        left_risk = self._detect_heel_strike_and_update_stride('left', left_ankle)
        right_risk = self._detect_heel_strike_and_update_stride('right', right_ankle)
        
        # Compute overall gait instability risk
        risk = max(left_risk, right_risk)
        
        # Also check for asymmetry between left and right stride characteristics
        asymmetry_risk = self._compute_asymmetry_risk()
        risk = max(risk, asymmetry_risk)
        
        return min(1.0, risk)  # Cap at 1.0
    
    def _detect_heel_strike_and_update_stride(self, foot, ankle_pos):
        """
        Detect heel strike and update stride metrics.
        
        Args:
            foot (str): 'left' or 'right'
            ankle_pos (numpy.ndarray): [y, x] position of ankle
            
        Returns:
            float: Risk score for this foot's gait
        """
        history = self.left_ankle_history if foot == 'left' else self.right_ankle_history
        last_strike_attr = f'last_{foot}_heel_strike'
        last_strike = getattr(self, last_strike_attr)
        
        # Need at least 3 points to detect a change in direction
        if len(history) < 3:
            return 0.0
        
        # Get recent y positions (ankle height)
        recent_y = [pos[0] for pos in list(history)[-3:]]
        
        # Detect heel strike: ankle was going down (increasing y) and now starts going up (decreasing y)
        # In image coordinates, y increases downward, so:
        # - Ankle moving down: y increasing (positive delta)
        # - Ankle moving up: y decreasing (negative delta)
        if len(recent_y) >= 3:
            y1, y2, y3 = recent_y
            # Check if we have a local minimum: y2 > y1 and y2 > y3 (ankle at bottom)
            # Actually, for heel strike we want the transition from down to up:
            # Before: y increasing (moving down)
            # After: y decreasing (moving up)
            if y2 > y1 and y2 > y3:  # Local maximum in y (highest point) - this is toe off?
                # Actually, let's think: when walking, the ankle goes:
                # 1. Heel strike: ankle at lowest point (y maximum in image coords since y increases downward)
                # 2. Ankle rolls forward
                # 3. Toe off: ankle at highest point (y minimum in image coords)
                # So heel strike is a local maximum in y (ankle lowest in image)
                pass
        
        # For simplicity, we'll use a different approach: look for peaks in y-coordinate
        # Heel strike corresponds to maximum y (ankle lowest in image)
        if len(history) >= 5:
            # Get last 5 y positions
            y_positions = [pos[0] for pos in list(history)[-5:]]
            # Check if the middle point is a local maximum
            if y_positions[2] > y_positions[1] and y_positions[2] > y_positions[3]:
                # Potential heel strike detected
                current_time = len(self.left_ankle_history)  # Using frame count as time proxy
                
                last_strike = getattr(self, last_strike_attr)
                if last_strike is not None:
                    # Compute stride length and time
                    # Stride length: distance between consecutive heel strikes of same foot
                    # We'll use ankle position at heel strike
                    prev_ankle_pos = self.left_ankle_history[last_strike] if foot == 'left' else self.right_ankle_history[last_strike]
                    curr_ankle_pos = ankle_pos
                    
                    # Stride length is the distance between heel strikes (we'll use x-distance for simplicity)
                    stride_length = abs(curr_ankle_pos[1] - prev_ankle_pos[1])
                    stride_time = current_time - last_strike  # in frames
                    
                    # Store stride metrics
                    self.stride_lengths.append(stride_length)
                    self.stride_times.append(stride_time)
                    
                    # Update baseline if we have enough data
                    if len(self.stride_lengths) >= 10:
                        self._update_baseline()
                    
                    # Compute risk based on deviation from baseline
                    risk = self._compute_stride_risk(stride_length, stride_time)
                    
                    # Update last heel strike
                    setattr(self, last_strike_attr, current_time)
                    return risk
                
                else:
                    # First heel strike detected
                    setattr(self, last_strike_attr, len(self.left_ankle_history if foot == 'left' else self.right_ankle_history))
        
        return 0.0
    
    def _update_baseline(self):
        """Update baseline gait characteristics from recent history."""
        if len(self.stride_lengths) >= 10 and len(self.stride_times) >= 10:
            # Use median to be robust to outliers
            self.baseline_stride_length = np.median(list(self.stride_lengths)[-10:])
            self.baseline_stride_time = np.median(list(self.stride_times)[-10:])
            logger.debug(f"Updated baseline: stride length={self.baseline_stride_length:.3f}, stride time={self.baseline_stride_time:.1f} frames")
    
    def _compute_stride_risk(self, stride_length, stride_time):
        """
        Compute risk based on deviation from baseline stride characteristics.
        
        Args:
            stride_length (float): Current stride length
            stride_time (float): Current stride time (in frames)
            
        Returns:
            float: Risk score
        """
        if self.baseline_stride_length is None or self.baseline_stride_time is None:
            return 0.0
        
        # Compute ratios
        length_ratio = stride_length / self.baseline_stride_length if self.baseline_stride_length > 0 else 1.0
        time_ratio = stride_time / self.baseline_stride_time if self.baseline_stride_time > 0 else 1.0
        
        # Risk factors:
        # 1. Stride too short (< 70% of baseline)
        # 2. Stride too slow (> 130% of baseline time)
        length_risk = max(0, (0.7 - length_ratio) / 0.7) if length_ratio < 0.7 else 0
        time_risk = max(0, (time_ratio - 1.3) / 0.3) if time_ratio > 1.3 else 0
        
        # Combined risk
        return max(length_risk, time_risk)
    
    def _compute_asymmetry_risk(self):
        """
        Compute risk based on asymmetry between left and right gait.
        
        Returns:
            float: Risk score
        """
        # Need history from both feet
        if (self.last_left_heel_strike is None or self.last_right_heel_strike is None or
            len(self.stride_lengths) < 2 or len(self.stride_times) < 2):
            return 0.0
        
        # Compute average stride characteristics for each foot over recent history
        # For simplicity, we'll use the last few strides
        recent_left_lengths = list(self.stride_lengths)[-5:] if len(self.stride_lengths) >= 5 else list(self.stride_lengths)
        recent_right_lengths = list(self.stride_lengths)[-5:] if len(self.stride_lengths) >= 5 else list(self.stride_lengths)
        
        # Actually, we need to separate left and right strides - this is getting complex
        # Let's simplify: we'll look at the difference in ankle phase between left and right
        # During normal walking, left and right ankles should be approximately out of phase
        
        if len(self.left_ankle_history) >= 2 and len(self.right_ankle_history) >= 2:
            # Get recent y positions (vertical movement)
            left_y = [pos[0] for pos in list(self.left_ankle_history)[-10:]]
            right_y = [pos[0] for pos in list(self.right_ankle_history)[-10:]]
            
            if len(left_y) >= 5 and len(right_y) >= 5:
                # Compute the cross-correlation to see phase relationship
                # During normal walking, there should be anti-correlation (when left is low, right is high)
                left_mean = np.mean(left_y)
                right_mean = np.mean(right_y)
                
                # Normalize
                left_norm = (np.array(left_y) - left_mean) / (np.std(left_y) + 1e-6)
                right_norm = (np.array(right_y) - right_mean) / (np.std(right_y) + 1e-6)
                
                # Compute correlation at zero lag
                corr = np.mean(left_norm * right_norm)
                
                # During normal walking, ankles move in opposite phases -> negative correlation
                # If correlation is positive (same phase), that's abnormal
                # If correlation is near zero, that's also abnormal (no clear pattern)
                # We want correlation around -0.5 to -0.8 for normal gait
                asymmetry_risk = max(0, (corr + 0.2) / 0.7)  # Risk increases as corr > -0.2
                return min(1.0, asymmetry_risk)
        
        return 0.0
