"""
Pose estimation using MoveNet Lightning TensorFlow Lite model.
"""
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import logging

logger = logging.getLogger(__name__)

class PoseEstimator:
    def __init__(self, model_path):
        """
        Initialize the pose estimator with a TFLite model.
        
        Args:
            model_path (str): Path to the TFLite model file
        """
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Model expected input shape (we assume it's known for MoveNet Lightning)
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
        logger.info(f"Pose estimator initialized with model: {model_path}")
        logger.info(f"Input shape: ({self.height}, {self.width})")
    
    def estimate(self, frame, person_box):
        """
        Estimate pose keypoints for a person in the frame.
        
        Args:
            frame (numpy.ndarray): Input frame as BGR image
            person_box (tuple): (xmin, ymin, xmax, ymax) of the detected person
            
        Returns:
            numpy.ndarray: Array of shape (17, 3) where each row is [y, x, confidence] 
                           (normalized coordinates) for each keypoint, or None if failed
        """
        if person_box is None:
            return None
        
        # Extract the person region from the frame
        xmin, ymin, xmax, ymax = person_box
        person_frame = frame[ymin:ymax, xmin:xmax]
        
        if person_frame.size == 0:
            logger.warning("Empty person frame")
            return None
        
        # Preprocess the person frame
        input_data = self._preprocess(person_frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get the pose keypoints
        # MoveNet Lightning output: a tensor of shape (1, 1, 17, 3)
        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])
        # Remove the batch and the extra dimension: (1, 1, 17, 3) -> (17, 3)
        keypoints = np.squeeze(keypoints)
        
        # The keypoints are in normalized coordinates [0, 1] relative to the input image (which is the person crop)
        # We want to convert them to be relative to the original frame? 
        # Actually, for our signals, we can work with the keypoints in the person crop coordinates.
        # But note: the signals (like trunk lean) use the relative positions, so it's okay to use the crop coordinates.
        # However, we must be consistent. Let's keep the keypoints in the original frame coordinates for simplicity in drawing? 
        # But we don't draw. We only compute distances and angles. 
        # The signals are invariant to translation? Actually, no: the trunk lean uses the absolute angle? 
        # We compute the angle between shoulder midpoint and hip midpoint. This is translation invariant.
        # Similarly, sway uses the horizontal position of the hip midpoint. This is not translation invariant if we use the crop.
        # Therefore, we must convert the keypoints back to the original frame coordinates.
        
        # Convert normalized keypoints (in the person crop) to pixel coordinates in the person crop
        crop_height = ymax - ymin
        crop_width = xmax - xmin
        keypoints[:, 0] = keypoints[:, 0] * crop_height  # y in crop
        keypoints[:, 1] = keypoints[:, 1] * crop_width   # x in crop
        
        # Then convert to original frame coordinates by adding the offset
        keypoints[:, 0] += ymin
        keypoints[:, 1] += xmin
        
        # Now keypoints are in the original frame pixel coordinates.
        # But note: the signals might expect normalized coordinates? 
        # Let's look at the signal implementations: they use the keypoints as (y, x) in pixels? 
        # We'll assume the signals work with pixel coordinates in the original frame.
        # Alternatively, we can normalize by the frame size? 
        # We'll leave it as pixel coordinates and let the signals handle it (they can use the frame size if needed).
        
        # However, the signals might be easier if we use normalized coordinates (0-1) relative to the frame.
        # Let's convert to normalized coordinates relative to the original frame.
        frame_height, frame_width = frame.shape[:2]
        keypoints[:, 0] = keypoints[:, 0] / frame_height  # y normalized
        keypoints[:, 1] = keypoints[:, 1] / frame_width   # x normalized
        
        # Now keypoints are normalized to [0, 1] in the original frame.
        # We'll return the keypoints as (y, x, confidence) in normalized coordinates.
        
        # Check confidence: if the confidence of a keypoint is low, we might want to mark it as invalid.
        # But we'll leave that to the signal processors.
        
        logger.debug(f"Pose estimated: {keypoints.shape} keypoints")
        return keypoints
    
    def _preprocess(self, frame):
        """
        Preprocess the frame for model input.
        
        Args:
            frame (numpy.ndarray): Input frame as BGR image (the person crop)
            
        Returns:
            numpy.ndarray: Preprocessed frame ready for model input
        """
        # Resize to model input size
        input_frame = cv2.resize(frame, (self.width, self.height))
        # Convert BGR to RGB
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        
        # MoveNet Lightning expects uint8 input in [0, 255]?
        # Let's check the input details. But for safety, we'll follow the common practice for MoveNet.
        # According to TensorHub, MoveNet expects uint8 input.
        if self.input_details[0]['dtype'] == np.uint8:
            input_frame = input_frame.astype(np.uint8)
        else:
            # If it's float32, we normalize to [0, 1]
            input_frame = input_frame.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_frame = np.expand_dims(input_frame, axis=0)
        
        return input_frame