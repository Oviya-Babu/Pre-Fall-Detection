"""
Alert System for GPIO buzzer, LED, and MQTT notifications.
"""
import time
import logging
import paho.mqtt.client as mqtt
from threading import Timer
import config

# Fallback GPIO for systems without Jetson hardware
try:
    import Jetson.GPIO as GPIO
    # Test GPIO initialization
    dummy = GPIO.BOARD
    JETSON_GPIO_AVAILABLE = True
except (ImportError, Exception):
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Jetson.GPIO not available, running in simulation mode")
    # Mock GPIO class for simulation
    class MockGPIO:
        BOARD = "board"
        OUT = "out"
        LOW = 0
        HIGH = 1
        
        def setmode(self, mode):
            pass
        
        def setup(self, pin, mode):
            pass
        
        def output(self, pin, value):
            pass
        
        def cleanup(self):
            pass
    
    GPIO = MockGPIO()
    JETSON_GPIO_AVAILABLE = False

logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self):
        """
        Initialize the alert system with GPIO and MQTT.
        """
        # GPIO setup
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(config.GPIO_BUZZER, GPIO.OUT)
        GPIO.setup(config.GPIO_LED_RED, GPIO.OUT)
        GPIO.setup(config.GPIO_LED_GREEN, GPIO.OUT)
        GPIO.setup(config.GPIO_LED_AMBER, GPIO.OUT)
        
        # Initialize all outputs to OFF
        GPIO.output(config.GPIO_BUZZER, GPIO.LOW)
        GPIO.output(config.GPIO_LED_RED, GPIO.LOW)
        GPIO.output(config.GPIO_LED_GREEN, GPIO.LOW)
        GPIO.output(config.GPIO_LED_AMBER, GPIO.LOW)
        
        # MQTT setup
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
        self.mqtt_client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
        
        # Start MQTT loop in background
        self.mqtt_client.loop_start()
        
        # Alert state tracking
        self.current_level = 0  # 0=GREEN, 1=YELLOW, 2=RED
        self.buzzer_timer = None
        
        logger.info("AlertSystem initialized")
    
    def update_alert_level(self, level):
        """
        Update the alert level and trigger appropriate responses.
        
        Args:
            level (int): Alert level (0=GREEN, 1=YELLOW, 2=RED)
        """
        if level == self.current_level:
            # No change, but we might need to restart timers for persistent signals
            # For simplicity, we'll just return
            return
        
        logger.info(f"Alert level changed: {self.current_level} -> {level}")
        
        # Cancel any existing buzzer timer
        if self.buzzer_timer is not None:
            self.buzzer_timer.cancel()
            self.buzzer_timer = None
        
        # Turn off all LEDs and buzzer
        self._turn_off_all()
        
        # Set new level
        self.current_level = level
        
        # Activate appropriate outputs
        if level == 0:  # GREEN
            GPIO.output(config.GPIO_LED_GREEN, GPIO.HIGH)
            logger.debug("GREEN LED ON")
            
        elif level == 1:  # YELLOW
            GPIO.output(config.GPIO_LED_AMBER, GPIO.HIGH)
            self._publish_mqtt("YELLOW")
            logger.debug("AMBER LED ON, MQTT YELLOW published")
            
        elif level == 2:  # RED
            GPIO.output(config.GPIO_LED_RED, GPIO.HIGH)
            GPIO.output(config.GPIO_BUZZER, GPIO.HIGH)
            self._publish_mqtt("RED")
            logger.debug("RED LED ON, Buzzer ON, MQTT RED published")
            
            # Turn off buzzer after 3 seconds
            self.buzzer_timer = Timer(3.0, self._turn_off_buzzer)
            self.buzzer_timer.start()
    
    def _turn_off_all(self):
        """Turn off all GPIO outputs."""
        GPIO.output(config.GPIO_BUZZER, GPIO.LOW)
        GPIO.output(config.GPIO_LED_RED, GPIO.LOW)
        GPIO.output(config.GPIO_LED_GREEN, GPIO.LOW)
        GPIO.output(config.GPIO_LED_AMBER, GPIO.LOW)
    
    def _turn_off_buzzer(self):
        """Turn off the buzzer (called by timer)."""
        GPIO.output(config.GPIO_BUZZER, GPIO.LOW)
        logger.debug("Buzzer OFF after 3 seconds")
        self.buzzer_timer = None
    
    def _publish_mqtt(self, alert_level):
        """
        Publish alert level to MQTT topic.
        
        Args:
            alert_level (str): "YELLOW" or "RED"
        """
        try:
            self.mqtt_client.publish(config.MQTT_TOPIC, alert_level)
            logger.debug(f"Published {alert_level} to {config.MQTT_TOPIC}")
        except Exception as e:
            logger.error(f"Failed to publish MQTT message: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up AlertSystem")
        
        # Cancel buzzer timer if active
        if self.buzzer_timer is not None:
            self.buzzer_timer.cancel()
        
        # Turn off all outputs
        self._turn_off_all()
        
        # Stop MQTT loop
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        
        # Cleanup GPIO
        GPIO.cleanup()
