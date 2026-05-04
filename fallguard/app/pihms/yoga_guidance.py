"""
Yoga Guidance — Template-based pose matching with real-time feedback.
Zero models. CPU-only. Target: 85%+ form matching accuracy.
"""

import json
import os
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TEMPLATES_PATH = os.path.join(os.path.dirname(__file__), 'poses', 'yoga_templates.json')

# Map template angle names to feature extractor keys
ANGLE_MAP = {
    'left_knee': 'angle_left_knee',
    'right_knee': 'angle_right_knee',
    'left_hip': 'angle_left_hip',
    'right_hip': 'angle_right_hip',
    'left_elbow': 'angle_left_elbow',
    'right_elbow': 'angle_right_elbow',
    'left_shoulder': 'angle_left_shoulder',
    'right_shoulder': 'angle_right_shoulder',
    'trunk_lean': 'trunk_lean_angle',
}


class YogaCoach:
    """
    Real-time yoga pose guidance using geometric template matching.
    Compares current joint angles against preset pose templates.
    """

    def __init__(self, templates_path: str = None):
        path = templates_path or TEMPLATES_PATH
        with open(path, 'r') as f:
            self._templates: List[Dict] = json.load(f)
        self._template_map = {t['pose_name']: t for t in self._templates}

        self._active_pose: Optional[str] = None
        self._active_template: Optional[Dict] = None
        self._pose_start_time: Optional[float] = None
        self._session_results: List[Dict] = []
        self._session_start: Optional[float] = None

        logger.info(f"YogaCoach loaded {len(self._templates)} templates")

    def get_pose_list(self) -> List[Dict]:
        """Return list of available poses with metadata."""
        return [
            {'name': t['pose_name'], 'difficulty': t['difficulty'],
             'duration_sec': t['duration_sec'], 'description': t['description']}
            for t in self._templates
        ]

    def set_active_pose(self, pose_name: str) -> bool:
        """Set the target pose for feedback."""
        template = self._template_map.get(pose_name)
        if template is None:
            logger.warning(f"Unknown pose: {pose_name}")
            return False
        self._active_pose = pose_name
        self._active_template = template
        self._pose_start_time = time.time()
        if self._session_start is None:
            self._session_start = time.time()
        logger.info(f"Active pose set: {pose_name}")
        return True

    def update(self, features: Dict[str, float]) -> Dict:
        """
        Compare current skeleton against active pose template.

        Returns:
            Dict with accuracy_percent, feedback, status, worst_joint
        """
        if self._active_template is None or features is None:
            return {'accuracy_percent': 0, 'status': 'NO_POSE_SELECTED',
                    'feedback': 'Select a pose to begin', 'worst_joint': None}

        template_angles = self._active_template.get('joint_angles', {})
        total_error = 0.0
        total_joints = 0
        worst_joint = None
        worst_error = 0.0
        joint_feedback = []

        for joint_name, spec in template_angles.items():
            feature_key = ANGLE_MAP.get(joint_name)
            if feature_key is None:
                continue

            current_angle = features.get(feature_key, 0.0)
            target_min = spec['min']
            target_max = spec['max']
            tolerance = spec.get('tolerance', 10)
            target_mid = (target_min + target_max) / 2.0

            # Error: how far outside the acceptable range
            if current_angle < target_min:
                error = (target_min - current_angle) / tolerance
            elif current_angle > target_max:
                error = (current_angle - target_max) / tolerance
            else:
                error = 0.0

            total_error += min(error, 2.0)  # Cap per-joint error
            total_joints += 1

            if error > worst_error:
                worst_error = error
                worst_joint = joint_name

            if error > 0.5:
                direction = "more" if current_angle < target_mid else "less"
                joint_feedback.append(
                    f"Adjust {joint_name.replace('_', ' ')}: "
                    f"need {direction} ({current_angle:.0f}° → {target_mid:.0f}°)"
                )

        # Accuracy score
        if total_joints > 0:
            avg_error = total_error / total_joints
            accuracy = max(0, 100 - avg_error * 50)
        else:
            accuracy = 0

        # Status
        if accuracy >= 90:
            status = 'PERFECT'
        elif accuracy >= 75:
            status = 'GOOD'
        else:
            status = 'ADJUST'

        # Duration check
        elapsed = time.time() - self._pose_start_time if self._pose_start_time else 0
        target_duration = self._active_template.get('duration_sec', 30)
        pose_complete = elapsed >= target_duration and accuracy >= 75

        if pose_complete:
            self._record_pose_result(accuracy, elapsed)

        feedback = joint_feedback[0] if joint_feedback else "Great form! Hold the pose."

        return {
            'accuracy_percent': round(accuracy, 1),
            'status': status,
            'feedback': feedback,
            'worst_joint': worst_joint,
            'elapsed_sec': round(elapsed, 1),
            'target_duration_sec': target_duration,
            'pose_complete': pose_complete,
            'tips': self._active_template.get('tips', []),
        }

    def _record_pose_result(self, accuracy: float, duration: float):
        """Record completed pose to session."""
        self._session_results.append({
            'pose': self._active_pose,
            'avg_accuracy': round(accuracy, 1),
            'duration_sec': round(duration, 1),
            'timestamp': time.time(),
        })
        logger.info(f"Pose completed: {self._active_pose}, accuracy={accuracy:.1f}%")
        self._active_pose = None
        self._active_template = None
        self._pose_start_time = None

    def get_session_summary(self) -> Dict:
        """Get current exercise session summary."""
        if not self._session_results:
            return {'poses_completed': 0, 'total_duration_min': 0}

        total_dur = sum(r['duration_sec'] for r in self._session_results)
        avg_acc = sum(r['avg_accuracy'] for r in self._session_results) / len(self._session_results)

        return {
            'date': time.strftime('%Y-%m-%d'),
            'time_start': self._session_start,
            'poses_completed': [
                {'pose': r['pose'], 'duration_sec': r['duration_sec'],
                 'avg_accuracy': r['avg_accuracy']}
                for r in self._session_results
            ],
            'total_duration_min': round(total_dur / 60, 1),
            'difficulty_level': 'easy',
            'completion_rate': round(len(self._session_results) / len(self._templates), 2),
            'avg_accuracy': round(avg_acc, 1),
        }

    def reset_session(self):
        """Reset session for a new exercise round."""
        self._session_results = []
        self._session_start = None
        self._active_pose = None
        self._active_template = None
        self._pose_start_time = None
