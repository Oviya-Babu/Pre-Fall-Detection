"""
Risk Engine for combining signals into alert levels.
"""
import logging

logger = logging.getLogger(__name__)

class RiskEngine:
    def __init__(self):
        """
        Initialize the risk engine.
        """
        logger.info("RiskEngine initialized")
    
    def compute_risk(self, gait_risk, sway_risk, trunk_risk, bed_exit_risk, freeze_risk, arm_reach_risk, ml_risk=0.0):
        """
        Combine individual signal risks into an overall alert level.
        
        Args:
            gait_risk (float): Risk score from gait instability (0.0-1.0)
            sway_risk (float): Risk score from lateral sway (0.0-1.0)
            trunk_risk (float): Risk score from trunk lean (0.0-1.0)
            bed_exit_risk (float): Risk score from bed exit (0.0-1.0)
            freeze_risk (float): Risk score from freezing of gait (0.0-1.0)
            arm_reach_risk (float): Risk score from arm reach (0.0-1.0)
            
        Returns:
            int: Alert level (0=GREEN, 1=YELLOW, 2=RED)
                 According to spec:
                 - YELLOW: 1 pre-fall signal detected
                 - RED: 2+ signals simultaneously OR bed exit
                 - GREEN: No risk
        """
        # Count how many signals are above a threshold (we'll use 0.5 as detection threshold)
        # But note: the trunk_risk and others are already scaled 0-1 where 0.5 is yellow threshold for trunk
        # We need to be consistent about what constitutes a "detected signal"
        
        # For most signals, we consider them "detected" if risk > 0.5
        # However, bed_exit is special: any bed_exit_risk > 0 should trigger RED immediately
        # According to spec: "RED: 2+ signals simultaneously OR bed exit"
        
        # Check for bed exit first (special case)
        if bed_exit_risk > 0:
            logger.debug(f"Bed exit detected with risk {bed_exit_risk}, triggering RED")
            return 2  # RED
        
        # Count other signals that are above detection threshold
        # We'll use 0.5 as the threshold for considering a signal "detected"
        # This works because:
        # - For gait, sway, freeze, arm_reach: 0.5 means moderate risk
        # - For trunk: 0.5 corresponds to the yellow threshold (20 degrees)
        signals_above_threshold = 0
        signal_risks = [gait_risk, sway_risk, trunk_risk, freeze_risk, arm_reach_risk, ml_risk]
        
        for i, risk in enumerate(signal_risks):
            if risk > 0.5:
                signals_above_threshold += 1
                logger.debug(f"Signal {i} above threshold: {risk}")
        
        logger.debug(f"Total signals above threshold: {signals_above_threshold}")
        
        # Determine alert level
        if signals_above_threshold >= 2:
            return 2  # RED: 2+ signals
        elif signals_above_threshold == 1:
            return 1  # YELLOW: 1 signal
        else:
            return 0  # GREEN: No signals
