"""
Camera module for handling DroidCam RTSP stream using OpenCV.
"""
import cv2
import logging
import time

logger = logging.getLogger(__name__)

class Camera:
    def __init__(self, rtsp_url):
        """
        Initialize camera with RTSP URL.
        
        Args:
            rtsp_url (str): RTSP stream URL from DroidCam
        """
        self.rtsp_url = rtsp_url
        self.cap = None
        self._connect()
    
    def _connect(self):
        """Establish connection to the camera stream."""
        logger.info(f"Connecting to camera at {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Set buffer size to minimize latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to set resolution (may not work with all streams)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            logger.error("Failed to open camera stream")
            raise RuntimeError("Could not open camera stream")
        
        logger.info("Camera stream opened successfully")
    
    def read_frame(self):
        """
        Read a frame from the camera stream.
        
        Returns:
            numpy.ndarray: Frame as BGR image, or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Camera not connected, attempting to reconnect...")
            self._connect()
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            # Try to reconnect
            self.cap.release()
            self.cap = None
            time.sleep(0.5)
            self._connect()
            return None
        
        return frame
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released")