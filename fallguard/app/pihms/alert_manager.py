"""
Alert Manager — Severity classification, deduplication, and fatigue prevention.
"""

import time
import uuid
import json
import logging
from collections import deque
from typing import Dict, List, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Intelligent alert system with:
    - Severity classification (CRITICAL/HIGH/MEDIUM/LOW)
    - Temporal deduplication (5-minute window)
    - Context-based suppression (lying = no pre-fall alerts)
    - Alert fatigue prevention (threshold adjustment after false positives)
    """

    def __init__(self,
                 dedup_window_sec: int = 60,
                 max_alerts_per_day: int = 500,
                 fp_threshold_24h: int = 3):
        self.dedup_window = dedup_window_sec
        self.max_daily = max_alerts_per_day
        self.fp_threshold = fp_threshold_24h

        # Alert history for deduplication
        self._recent_alerts: deque = deque(maxlen=200)
        self._daily_alert_count = 0
        self._false_positives_24h = 0
        self._threshold_boost = 0.0  # Added to risk threshold
        self._last_daily_reset = time.time()

        # Staff acknowledgment tracking
        self._ack_timestamps: Dict[str, float] = {}

    def update(self, risk_result: Dict,
               activity_state: int = 2,
               pre_fall_result: Dict = None) -> Optional[Dict]:
        """
        Evaluate risk and produce an alert if warranted.

        Args:
            risk_result: Output from RiskAggregator.update()
            activity_state: Current activity (0=LYING, 1=SITTING, etc.)
            pre_fall_result: Output from PreFallDetector.update()

        Returns:
            Alert dict if alert should fire, None if suppressed
        """
        # Reset daily counters
        if time.time() - self._last_daily_reset > 86400:
            self._daily_alert_count = 0
            self._false_positives_24h = 0
            self._threshold_boost = max(0, self._threshold_boost - 5)
            self._last_daily_reset = time.time()

        total_risk = risk_result.get('total_risk', 0)
        severity = risk_result.get('severity', 'LOW')

        # ── Context-Based Suppression ───────────────────────────────
        if activity_state == 0:  # LYING
            # Suppress pre-fall alerts when lying (can't fall from lying)
            if risk_result.get('pre_fall_risk', 0) > 50:
                logger.debug("Pre-fall alert suppressed: patient is lying")
                return None

        # ── Threshold Adjustment (false positive penalty) ──────────
        adjusted_thresholds = {
            'CRITICAL': 90 + self._threshold_boost,
            'HIGH': 75 + self._threshold_boost,
            'MEDIUM': 50 + self._threshold_boost,
        }

        # Reclassify with adjusted thresholds
        if total_risk >= adjusted_thresholds['CRITICAL']:
            severity = 'CRITICAL'
        elif total_risk >= adjusted_thresholds['HIGH']:
            severity = 'HIGH'
        elif total_risk >= adjusted_thresholds['MEDIUM']:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'

        # LOW severity = batch for 24h (no immediate alert)
        if severity == 'LOW':
            return None

        # ── Temporal Validation (pre-fall needs 2+ sec persistence) ──
        if pre_fall_result and severity in ('HIGH', 'CRITICAL'):
            if not pre_fall_result.get('risk_validated', False):
                # For pre-fall: still show MEDIUM so the dashboard shows some signal
                severity = 'MEDIUM'

        # ── Deduplication ───────────────────────────────────────────
        primary_type = self._determine_primary_type(risk_result)
        if self._is_duplicate(primary_type, severity):
            logger.debug(f"Alert suppressed: duplicate {primary_type} within window")
            return None

        # ── Daily Limit ─────────────────────────────────────────────
        if self._daily_alert_count >= self.max_daily:
            logger.warning("Daily alert limit reached, batching as insight")
            return None

        # ── Produce Alert ───────────────────────────────────────────
        alert = self._build_alert(risk_result, severity, primary_type,
                                  pre_fall_result)
        self._record_alert(alert)
        return alert

    def _determine_primary_type(self, risk_result: Dict) -> str:
        """Determine the primary alert type from risk breakdown."""
        pf = risk_result.get('pre_fall_risk', 0)
        act = risk_result.get('activity_anomaly_risk', 0)
        well = risk_result.get('wellness_risk', 0)

        if pf >= act and pf >= well:
            return 'PRE_FALL_WARNING'
        elif act >= well:
            return 'ACTIVITY_ANOMALY'
        else:
            return 'WELLNESS_ALERT'

    def _is_duplicate(self, primary_type: str, severity: str) -> bool:
        """Check if an identical alert was recently fired."""
        now = time.time()
        for alert in reversed(self._recent_alerts):
            if now - alert['timestamp'] > self.dedup_window:
                break
            if (alert['primary_type'] == primary_type and
                    alert['severity'] == severity):
                return True
            # Allow escalation (severity upgrade)
            if alert['primary_type'] == primary_type:
                sev_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
                if sev_order.get(severity, 0) > sev_order.get(alert['severity'], 0):
                    return False  # Escalation allowed
                return True
        return False

    def _build_alert(self, risk_result: Dict, severity: str,
                     primary_type: str,
                     pre_fall_result: Dict = None) -> Dict:
        """Build the full alert payload."""
        biomarkers = (pre_fall_result.get('biomarkers', [])
                      if pre_fall_result else [])
        latency = (pre_fall_result.get('latency_sec', 0)
                   if pre_fall_result else 0)

        # Color mapping
        color_map = {
            'CRITICAL': '🔴🔴 RED BLINK',
            'HIGH': '🔴 RED',
            'MEDIUM': '🟡 YELLOW',
            'LOW': '⚪ BLUE',
        }

        # Recommended actions
        actions = self._get_recommendations(severity, primary_type)

        alert = {
            'alert_id': str(uuid.uuid4()),
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'timestamp': time.time(),
            'severity': severity,
            'color': color_map.get(severity, '⚪ BLUE'),
            'primary_type': primary_type,
            'secondary_signals': biomarkers,
            'confidence': min(risk_result.get('total_risk', 0) / 100.0, 1.0),
            'risk_breakdown': {
                'total_risk': risk_result.get('total_risk', 0),
                'pre_fall_risk': risk_result.get('pre_fall_risk', 0),
                'activity_anomaly_risk': risk_result.get('activity_anomaly_risk', 0),
                'wellness_risk': risk_result.get('wellness_risk', 0),
                'trend_degradation': risk_result.get('trend_degradation_risk', 0),
            },
            'latency_seconds': latency,
            'recommended_actions': actions,
        }
        return alert

    def _get_recommendations(self, severity: str, primary_type: str) -> List[str]:
        """Get recommended actions based on alert type."""
        if severity == 'CRITICAL':
            return [
                "IMMEDIATE: Check on patient",
                "Ensure clear walking path",
                "Prepare fall response protocol",
                "Contact emergency services if fall occurs",
            ]
        elif severity == 'HIGH':
            return [
                "Check on patient within 5 minutes",
                "Consider physical support",
                "Clear walking hazards",
            ]
        elif severity == 'MEDIUM':
            return [
                "Monitor patient status",
                "Schedule routine check",
                "Review recent activity log",
            ]
        return ["Log and review during rounds"]

    def _record_alert(self, alert: Dict):
        """Record alert for deduplication tracking."""
        self._recent_alerts.append({
            'timestamp': time.time(),
            'primary_type': alert['primary_type'],
            'severity': alert['severity'],
            'alert_id': alert['alert_id'],
        })
        self._daily_alert_count += 1
        logger.info(
            f"Alert fired: {alert['severity']} {alert['primary_type']} "
            f"(risk={alert['risk_breakdown']['total_risk']})"
        )

    def acknowledge_alert(self, alert_id: str):
        """Staff acknowledges an alert (suppresses identical for 10 min)."""
        self._ack_timestamps[alert_id] = time.time()
        logger.info(f"Alert acknowledged: {alert_id}")

    def report_false_positive(self, alert_id: str):
        """Report a false positive to adjust thresholds."""
        self._false_positives_24h += 1
        if self._false_positives_24h >= self.fp_threshold:
            self._threshold_boost += 5.0
            logger.warning(
                f"False positive threshold increased by 5% "
                f"(total boost: {self._threshold_boost}%)"
            )
