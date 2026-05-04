"""
Machine Learning Pre-Fall Detector
Trained on UR Fall Detection Dataset
Temporal Convolutional Network for sequence classification
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import logging

logger = logging.getLogger(__name__)

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        return F.relu(out + residual)

class FallDetectionTCN(nn.Module):
    def __init__(self, input_size=51, num_channels=[64, 64, 32], kernel_size=3):
        super().__init__()
        layers = []
        in_channels = input_size
        
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation))
            in_channels = out_channels
            
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], 3)  # 0=Safe, 1=Pre-Fall, 2=Falling
        
    def forward(self, x):
        # x shape: (batch, features, seq_len)
        out = self.network(x)
        out = torch.mean(out, dim=2)  # Global average pooling over time
        return self.classifier(out)

class MLPreFallDetector:
    def __init__(self, window_size=150):
        self.window_size = window_size
        self.device = torch.device('cpu')
        
        # Initialize model
        self.model = FallDetectionTCN()
        self.model.to(self.device)
        self.model.eval()
        
        # Pose feature history
        self.pose_history = deque(maxlen=window_size)
        
        # Load pretrained weights if available
        try:
            self.model.load_state_dict(torch.load('../models/fall_detector.pt', map_location=self.device))
            logger.info("Loaded trained fall detection model")
        except:
            logger.warning("No pretrained model found, running with random initialization")
            
        logger.info("ML Pre-Fall Detector initialized")
        
    def extract_features(self, keypoints):
        """Extract 51 dimensional features from 17 keypoints"""
        features = []
        
        # Raw coordinates
        for i in range(17):
            features.append(keypoints[i, 0])  # y
            features.append(keypoints[i, 1])  # x
            features.append(keypoints[i, 2])  # confidence
            
        # Joint angles
        # Shoulder-hip angle
        shoulder_mid = (keypoints[5, :2] + keypoints[6, :2]) / 2
        hip_mid = (keypoints[11, :2] + keypoints[12, :2]) / 2
        spine_vec = shoulder_mid - hip_mid
        spine_angle = np.arctan2(spine_vec[1], spine_vec[0])
        features.append(spine_angle)
        
        # Knee angles
        for side in [0, 1]:  # left, right
            hip = keypoints[11 + side, :2]
            knee = keypoints[13 + side, :2]
            ankle = keypoints[15 + side, :2]
            
            thigh_vec = knee - hip
            shin_vec = ankle - knee
            
            if np.linalg.norm(thigh_vec) > 0.001 and np.linalg.norm(shin_vec) > 0.001:
                thigh_norm = thigh_vec / np.linalg.norm(thigh_vec)
                shin_norm = shin_vec / np.linalg.norm(shin_vec)
                knee_angle = np.arccos(np.clip(np.dot(thigh_norm, shin_norm), -1, 1))
            else:
                knee_angle = 0.0
            features.append(knee_angle)
            
        return np.array(features, dtype=np.float32)
    
    def analyze(self, keypoints):
        """
        Analyze pose sequence for pre-fall detection
        
        Returns:
            float: Risk score 0.0 → 1.0
        """
        if keypoints is None or len(keypoints) < 17:
            return 0.0
            
        # Extract features for current frame
        features = self.extract_features(keypoints)
        self.pose_history.append(features)
        
        # Need minimum window size
        if len(self.pose_history) < 30:
            return 0.0
            
        # Run inference
        with torch.no_grad():
            seq = np.array(self.pose_history).T  # (features, seq_len)
            seq = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            
            logits = self.model(seq)
            probs = F.softmax(logits, dim=1)[0]
            
            pre_fall_prob = probs[1].item()
            falling_prob = probs[2].item()
            
            risk = max(pre_fall_prob * 0.8, falling_prob)
            
            logger.debug(f"ML Risk: {risk:.3f} (Pre-fall: {pre_fall_prob:.3f}, Falling: {falling_prob:.3f})")
            
            return min(1.0, risk)