"""
MQTT Handler — Per-person alert delivery with structured JSON payloads.
Connects to local Mosquitto broker.
Topic structure: pihms/facility/{fac_id}/person/{person_id}/alert
"""

import json
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logger.warning("paho-mqtt not installed, MQTT disabled")


class MQTTHandler:
    """
    Publishes PIHMS alerts and status updates to MQTT topics.
    Supports per-person topic routing and QoS levels.
    """

    def __init__(self,
                 broker: str = "localhost",
                 port: int = 1883,
                 facility_id: str = "FAC001",
                 username: str = "",
                 password: str = "",
                 keepalive: int = 60):
        self.facility_id = facility_id
        self.broker = broker
        self.port = port
        self.connected = False
        self.client = None

        if not MQTT_AVAILABLE:
            logger.warning("MQTT handler disabled: paho-mqtt not installed")
            return

        try:
            self.client = mqtt.Client(
                client_id=f"pihms-{facility_id}",
                protocol=mqtt.MQTTv311
            )
            if username:
                self.client.username_pw_set(username, password)

            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            self.client.connect(broker, port, keepalive)
            self.client.loop_start()
            logger.info(f"MQTT connecting to {broker}:{port}")
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            self.client = None

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            logger.info(f"MQTT connected to {self.broker}:{self.port}")
            # Publish online status
            self._publish_status("online")
        else:
            logger.error(f"MQTT connection failed with rc={rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc != 0:
            logger.warning(f"MQTT unexpected disconnect (rc={rc}), will auto-reconnect")

    # ── Alert Publishing ────────────────────────────────────────────

    def publish_alert(self, person_id: int, alert: Dict,
                      qos: int = 1, retain: bool = False):
        """
        Publish a per-person alert to MQTT.

        Topic: pihms/facility/{fac_id}/person/{person_id}/alert
        QoS: 1 for MEDIUM, 2 for CRITICAL/HIGH
        """
        if not self.client or not self.connected:
            return False

        # Set QoS based on severity
        severity = alert.get('severity', 'LOW')
        if severity in ('CRITICAL', 'HIGH'):
            qos = 2

        topic = f"pihms/facility/{self.facility_id}/person/{person_id}/alert"

        payload = {
            'timestamp_utc': alert.get('timestamp_utc',
                                       time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
            'facility_id': self.facility_id,
            'person_id': person_id,
            'severity': severity,
            'primary_type': alert.get('primary_type', 'UNKNOWN'),
            'risk_score': alert.get('risk_breakdown', {}).get('total_risk', 0),
            'confidence': alert.get('confidence', 0),
            'biomarkers': alert.get('secondary_signals', []),
            'recommended_actions': alert.get('recommended_actions', []),
            'latency_seconds': alert.get('latency_seconds', 0),
        }

        try:
            result = self.client.publish(topic, json.dumps(payload), qos=qos, retain=retain)
            logger.info(f"MQTT alert published: {topic} [{severity}] rc={result.rc}")
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"MQTT publish failed: {e}")
            return False

    def publish_risk_update(self, person_id: int, risk_result: Dict):
        """
        Publish periodic risk status update (low frequency, QoS 0).

        Topic: pihms/facility/{fac_id}/person/{person_id}/risk
        """
        if not self.client or not self.connected:
            return

        topic = f"pihms/facility/{self.facility_id}/person/{person_id}/risk"
        payload = {
            'timestamp': time.time(),
            'total_risk': risk_result.get('total_risk', 0),
            'severity': risk_result.get('severity', 'LOW'),
            'pre_fall_risk': risk_result.get('pre_fall_risk', 0),
            'activity_anomaly_risk': risk_result.get('activity_anomaly_risk', 0),
        }
        try:
            self.client.publish(topic, json.dumps(payload), qos=0)
        except Exception as e:
            logger.debug(f"MQTT risk update failed: {e}")

    def publish_activity(self, person_id: int, activity_result: Dict):
        """
        Publish activity state update.

        Topic: pihms/facility/{fac_id}/person/{person_id}/activity
        """
        if not self.client or not self.connected:
            return

        topic = f"pihms/facility/{self.facility_id}/person/{person_id}/activity"
        payload = {
            'timestamp': time.time(),
            'state': activity_result.get('state_name', 'UNKNOWN'),
            'motion_intensity': activity_result.get('motion_intensity', 0),
            'anomaly_score': activity_result.get('anomaly_score', 0),
        }
        try:
            self.client.publish(topic, json.dumps(payload), qos=0)
        except Exception as e:
            logger.debug(f"MQTT activity update failed: {e}")

    # ── System Status ───────────────────────────────────────────────

    def _publish_status(self, status: str):
        """Publish system status (online/offline)."""
        topic = f"pihms/facility/{self.facility_id}/status"
        payload = {
            'status': status,
            'timestamp': time.time(),
        }
        try:
            self.client.publish(topic, json.dumps(payload), qos=1, retain=True)
        except Exception:
            pass

    def publish_system_heartbeat(self, stats: Dict):
        """
        Publish system heartbeat with performance stats.

        Topic: pihms/facility/{fac_id}/heartbeat
        """
        if not self.client or not self.connected:
            return

        topic = f"pihms/facility/{self.facility_id}/heartbeat"
        payload = {
            'timestamp': time.time(),
            'fps': stats.get('fps', 0),
            'latency_ms': stats.get('latency_ms', 0),
            'persons_tracked': stats.get('persons_tracked', 0),
            'uptime_sec': stats.get('uptime_sec', 0),
        }
        try:
            self.client.publish(topic, json.dumps(payload), qos=0)
        except Exception:
            pass

    # ── Lifecycle ───────────────────────────────────────────────────

    def shutdown(self):
        """Graceful shutdown: publish offline status and disconnect."""
        if self.client:
            try:
                self._publish_status("offline")
                time.sleep(0.2)
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
            logger.info("MQTT handler shut down")
