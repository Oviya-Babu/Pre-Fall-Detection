#!/usr/bin/env python3
"""
Test script to verify DroidCam camera stream using OpenCV.
Based on the code snippet provided by the user.
"""
import cv2

def test_camera_stream():
    # DroidCam URL from user's example
    url = os.getenv("CAMERA_STREAM_URL")
    
    print(f"Connecting to camera at {url}")
    
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return False
    
    print("Successfully opened video stream")
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display the frame
            cv2.imshow("Phone Camera Stream", frame)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Camera stream closed")
    
    return True

if __name__ == "__main__":
    test_camera_stream()
