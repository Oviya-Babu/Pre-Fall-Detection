# Configuration for Pre-Fall Detection System (PIHMS v2.0)
# All thresholds and settings are tunable
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, '..', 'models')

# Camera settings
RTSP_URL = "http://10.227.127.50:4747/video"  # DroidCam stream address

# Model paths
SSD_MODEL_PATH = os.path.join(_MODELS_DIR, 'ssd_mobilenet_v2_cpu.tflite')
MOVENET_MODEL_PATH = os.path.join(_MODELS_DIR, 'movenet_lightning.tflite')

# Bed region polygon (pixel coordinates) - define based on camera view
# Example: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] for a rectangular bed
BED_POLYGON = [(100, 200), (500, 200), (500, 400), (100, 400)]

# Signal thresholds
SWAY_MULTIPLIER = 2.0  # Sway vs baseline ratio for YELLOW
TRUNK_LEAN_YELLOW = 20  # degrees
TRUNK_LEAN_RED = 35     # degrees
FREEZE_VELOCITY = 0.02  # pixel velocity threshold for freeze
ARM_REACH_RATIO = 0.3   # wrist distance from midline ratio

# Analysis windows
SIGNAL_WINDOW = 150  # frames (~5 seconds at 30fps)
MIN_KEYPOINT_CONF = 0.5  # Ignore keypoints below this confidence
ALERT_HOLD_FRAMES = 5  # Signal must persist for 5 frames before alert

# Logging
LOG_RETENTION_DAYS = 90  # SQLite auto-purge threshold

# GPIO pins (using BOARD numbering)
GPIO_BUZZER = 18
GPIO_LED_RED = 16
GPIO_LED_GREEN = 20
GPIO_LED_AMBER = 21

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = "pihms/facility"
MQTT_LEGACY_TOPIC = "fallguard/alert"
MQTT_USERNAME = ""  # if needed
MQTT_PASSWORD = ""  # if needed

# Dashboard settings
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8080

# Facility settings
FACILITY_ID = "FAC001"