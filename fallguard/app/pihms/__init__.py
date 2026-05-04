"""
PIHMS - PreFall Intelligence Health Monitoring System v2.0
=========================================================
Extends the existing FallGuard pre-fall detection system with:
- Predictive pre-fall detection (rule-based biomechanical analysis)
- Daily activity monitoring and anomaly detection
- Eating activity inference and meal tracking
- Yoga/exercise guidance with pose template matching
- Medication alert and compliance tracking
- Intelligent risk aggregation and alert management
- Storage-optimized SQLite database with compression

All analytics are CPU-based (no new GPU models).
Designed for NVIDIA Jetson Orin Nano with 4GB SSD constraint.
"""

__version__ = "2.0.0"
