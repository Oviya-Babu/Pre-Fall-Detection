"""
Storage Validator — Monitors disk usage and enforces 4GB SSD constraint.
"""

import os
import shutil
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class StorageValidator:
    """Validates storage usage against the 4GB SSD constraint."""

    def __init__(self, data_dir: str, max_gb: float = 4.0, buffer_gb: float = 0.2):
        self.data_dir = os.path.abspath(data_dir)
        self.max_bytes = int(max_gb * 1024 * 1024 * 1024)
        self.buffer_bytes = int(buffer_gb * 1024 * 1024 * 1024)
        self.usable_bytes = self.max_bytes - self.buffer_bytes

    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage for the partition containing data_dir."""
        total, used, free = shutil.disk_usage(self.data_dir)
        return {
            'total_gb': total / (1024**3),
            'used_gb': used / (1024**3),
            'free_gb': free / (1024**3),
            'usage_percent': (used / total) * 100 if total > 0 else 0,
        }

    def get_project_usage(self) -> Dict[str, float]:
        """Get storage used by the PIHMS data directory."""
        total_bytes = 0
        file_count = 0
        for dirpath, _, filenames in os.walk(self.data_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_bytes += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    pass
        return {
            'data_dir_mb': total_bytes / (1024**2),
            'file_count': file_count,
        }

    def check_available(self, required_mb: float) -> bool:
        """Check if enough storage is available for an operation."""
        disk = self.get_disk_usage()
        free_mb = disk['free_gb'] * 1024
        available = free_mb > (required_mb + self.buffer_bytes / (1024**2))
        if not available:
            logger.warning(
                f"Insufficient storage: need {required_mb:.1f}MB, "
                f"free={free_mb:.1f}MB, buffer={self.buffer_bytes/(1024**2):.1f}MB"
            )
        return available

    def estimate_daily_growth(self, skeleton_fps: int = 30,
                              num_patients: int = 1) -> Dict[str, float]:
        """Estimate daily storage growth per patient."""
        # Delta-compressed skeleton: ~47 MB/day/patient (from PRD analysis)
        skeleton_mb = 47.0 * num_patients
        # Activity summaries: ~2 KB/patient/day
        summary_mb = (2.0 * num_patients) / 1024
        # Alert logs: ~1 KB/alert, ~20 alerts/day average
        alert_mb = (1.0 * 20) / 1024
        # Medication logs: ~200 bytes/med/day, 5 meds average
        med_mb = (0.2 * 5 * num_patients) / 1024
        total_mb = skeleton_mb + summary_mb + alert_mb + med_mb
        return {
            'skeleton_mb': skeleton_mb,
            'summary_mb': summary_mb,
            'alert_mb': alert_mb,
            'medication_mb': med_mb,
            'total_daily_mb': total_mb,
            'days_until_full': (self.get_disk_usage()['free_gb'] * 1024) / total_mb
                               if total_mb > 0 else float('inf'),
        }

    def alert_if_critical(self, threshold_gb: float = 0.3) -> bool:
        """Return True and log if free space is below threshold."""
        disk = self.get_disk_usage()
        if disk['free_gb'] < threshold_gb:
            logger.critical(
                f"CRITICAL: Storage critically low! "
                f"Free: {disk['free_gb']:.2f}GB < threshold {threshold_gb:.2f}GB"
            )
            return True
        return False

    def generate_report(self) -> str:
        """Generate a human-readable storage report."""
        disk = self.get_disk_usage()
        proj = self.get_project_usage()
        growth = self.estimate_daily_growth()
        lines = [
            "═══ PIHMS Storage Report ═══",
            f"Disk Total:      {disk['total_gb']:.1f} GB",
            f"Disk Used:       {disk['used_gb']:.1f} GB ({disk['usage_percent']:.1f}%)",
            f"Disk Free:       {disk['free_gb']:.1f} GB",
            f"PIHMS Data:      {proj['data_dir_mb']:.1f} MB ({proj['file_count']} files)",
            f"Daily Growth:    {growth['total_daily_mb']:.1f} MB/day",
            f"Days Until Full: {growth['days_until_full']:.0f} days",
            "═══════════════════════════",
        ]
        return '\n'.join(lines)
