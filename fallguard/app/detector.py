"""
Person detection using MobileNet-SSD TensorFlow Lite model (CPU-compatible).
Supports both single-person and multi-person detection.
"""
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import logging

logger = logging.getLogger(__name__)

# COCO class ID for person (0-indexed in standard COCO SSD models)
PERSON_CLASS_ID = 0


class PersonDetector:
    def __init__(self, model_path, confidence_threshold=0.4):
        """
        Initialize the person detector with a TFLite model.

        Args:
            model_path (str): Path to the TFLite model file
            confidence_threshold (float): Minimum confidence for detection
        """
        self.confidence_threshold = confidence_threshold
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Model expected input shape
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        logger.info(f"Person detector initialized: {model_path}")
        logger.info(f"Input: {self.height}x{self.width}, dtype={self.input_details[0]['dtype']}")

    def detect(self, frame):
        """
        Detect the primary (highest confidence) person in the frame.
        Backward-compatible: returns single bounding box.

        Args:
            frame (numpy.ndarray): Input frame as BGR image

        Returns:
            tuple: (xmin, ymin, xmax, ymax) or None
        """
        persons = self.detect_all(frame)
        if not persons:
            return None
        # Return highest confidence person
        best = max(persons, key=lambda p: p['confidence'])
        return best['bbox']

    def detect_all(self, frame, max_persons=10):
        """
        Detect all persons in the frame.

        Args:
            frame (numpy.ndarray): Input frame as BGR image
            max_persons (int): Maximum number of persons to return

        Returns:
            list[dict]: List of {'bbox': (x1,y1,x2,y2), 'confidence': float}
        """
        input_data = self._preprocess(frame)
        h, w = frame.shape[:2]

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get detection results — handle batch dimension
        boxes_raw = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes_raw = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores_raw = self.interpreter.get_tensor(self.output_details[2]['index'])
        num_raw = self.interpreter.get_tensor(self.output_details[3]['index'])

        # Squeeze batch dimension if present
        boxes = np.squeeze(boxes_raw)       # (N, 4)
        classes = np.squeeze(classes_raw)   # (N,)
        scores = np.squeeze(scores_raw)     # (N,)
        num_detections = int(np.squeeze(num_raw))

        persons = []
        for i in range(min(num_detections, len(scores))):
            if scores[i] < self.confidence_threshold:
                continue

            class_id = int(classes[i])
            # Person is class 0 in standard COCO SSD models
            if class_id != PERSON_CLASS_ID:
                continue

            # Convert normalized coords to pixel values
            ymin = int(max(0, boxes[i][0] * h))
            xmin = int(max(0, boxes[i][1] * w))
            ymax = int(min(h, boxes[i][2] * h))
            xmax = int(min(w, boxes[i][3] * w))

            # Validate box dimensions
            if xmax - xmin < 10 or ymax - ymin < 10:
                continue

            persons.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'confidence': float(scores[i]),
            })

        # Sort by confidence, limit count
        persons.sort(key=lambda p: p['confidence'], reverse=True)
        persons = persons[:max_persons]

        if persons:
            logger.debug(f"Detected {len(persons)} person(s), "
                         f"best conf={persons[0]['confidence']:.2f}")
        else:
            logger.debug("No person detected")

        return persons

    def _preprocess(self, frame):
        """
        Preprocess the frame for model input.

        Args:
            frame (numpy.ndarray): Input frame as BGR image

        Returns:
            numpy.ndarray: Preprocessed frame ready for model input
        """
        # Resize to model input size
        input_frame = cv2.resize(frame, (self.width, self.height))
        # Convert BGR to RGB
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        # Match model expected dtype
        if self.input_details[0]['dtype'] == np.float32:
            input_frame = input_frame.astype(np.float32) / 255.0
        else:
            input_frame = input_frame.astype(np.uint8)

        # Add batch dimension
        input_frame = np.expand_dims(input_frame, axis=0)

        return input_frame