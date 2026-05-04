"""
Risk Aggregator — Unified multi-signal risk scoring (0-100).
Combines pre-fall risk, activity anomaly, wellness adherence, and trend data.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RiskAggregator:
    """
    Computes a unified risk score from all PIHMS subsystem outputs.

    Weights:
      - Pre-fall risk: 60%
      - Activity anomaly: 20%
      - Wellness adherence: 10%
      - Trend degradation: 10%
    """

    def __init__(self,
                 weight_pre_fall: float = 0.60,
                 weight_activity: float = 0.20,
                 weight_wellness: float = 0.10,
                 weight_trend: float = 0.10):
        self.w_pf = weight_pre_fall
        self.w_act = weight_activity
        self.w_well = weight_wellness
        self.w_trend = weight_trend

    def update(self,
               pre_fall_result: Dict = None,
               activity_result: Dict = None,
               wellness_risk: float = 0.0,
               trend_degradation: float = 0.0) -> Dict:
        """
        Compute unified risk score.

        Args:
            pre_fall_result: Output from PreFallDetector.update()
            activity_result: Output from ActivityMonitor.update()
            wellness_risk: Score from WellnessTracker.get_wellness_risk() (0-50)
            trend_degradation: Historical trend slope (0-1)

        Returns:
            Dict with total_risk, component breakdown, severity level
        """
        # Extract component scores (normalize to 0-100)
        pf_risk = pre_fall_result.get('risk_score', 0) if pre_fall_result else 0
        act_anomaly = (activity_result.get('anomaly_score', 0) * 100
                       if activity_result else 0)
        well_risk = wellness_risk * 2  # Scale 0-50 → 0-100
        trend_risk = trend_degradation * 100  # Scale 0-1 → 0-100

        # Weighted sum
        total_risk = (
            self.w_pf * pf_risk +
            self.w_act * min(act_anomaly, 100) +
            self.w_well * min(well_risk, 100) +
            self.w_trend * min(trend_risk, 100)
        )

        # Critical-signal override: a high pre-fall risk alone is sufficient
        # to warrant an alert. Prevent dilution by inactive subsystems.
        if pf_risk >= 50:
            total_risk = max(total_risk, pf_risk)

        total_risk = int(min(100, max(0, total_risk)))

        # Determine severity
        if total_risk >= 80:
            severity = 'CRITICAL'
        elif total_risk >= 65:
            severity = 'HIGH'
        elif total_risk >= 45:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        return {
            'total_risk': total_risk,
            'severity': severity,
            'pre_fall_risk': round(pf_risk, 1),
            'activity_anomaly_risk': round(act_anomaly, 1),
            'wellness_risk': round(well_risk, 1),
            'trend_degradation_risk': round(trend_risk, 1),
        }
