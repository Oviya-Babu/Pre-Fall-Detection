"""
PIHMS Database — SQLite with compression, retention, and lifecycle management.
"""

import sqlite3
import os
import time
import gzip
import json
import uuid
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')


class PIHMSDatabase:
    """
    SQLite database for PIHMS v2.0 with automatic compression and rotation.

    Thread-safe via check_same_thread=False and WAL mode.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
            db_path = os.path.join(data_dir, 'pihms.db')

        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-2000")  # 2MB cache
        self.conn.row_factory = sqlite3.Row

        self._init_schema()
        logger.info(f"PIHMSDatabase initialized at {self.db_path}")

    def _init_schema(self):
        """Apply schema from SQL file."""
        with open(SCHEMA_PATH, 'r') as f:
            schema_sql = f.read()
        self.conn.executescript(schema_sql)
        self.conn.commit()

    # ── Skeleton Batches ────────────────────────────────────────────────

    def insert_skeleton_batch(self, batch_data: bytes, num_frames: int,
                              timestamp_start: float, timestamp_end: float,
                              patient_id: str = 'default',
                              avg_confidence: float = 0.0) -> int:
        """Insert a compressed skeleton batch."""
        cur = self.conn.execute('''
            INSERT INTO skeleton_batches
            (timestamp_start, timestamp_end, patient_id, num_frames,
             batch_data, batch_size_bytes, avg_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp_start, timestamp_end, patient_id, num_frames,
              batch_data, len(batch_data), avg_confidence))
        self.conn.commit()
        return cur.lastrowid

    def get_skeleton_batches(self, patient_id: str = 'default',
                             start_time: float = None,
                             end_time: float = None) -> List[Dict]:
        """Query skeleton batches within a time range."""
        query = "SELECT * FROM skeleton_batches WHERE patient_id = ?"
        params: list = [patient_id]
        if start_time is not None:
            query += " AND timestamp_start >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp_end <= ?"
            params.append(end_time)
        query += " ORDER BY timestamp_start"
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Alerts ──────────────────────────────────────────────────────────

    def insert_alert(self, severity: str, primary_type: str,
                     total_risk_score: float,
                     payload: Dict = None,
                     patient_id: str = 'default',
                     secondary_signals: List[str] = None,
                     confidence: float = 0.0,
                     facility_id: str = 'FAC001') -> str:
        """Insert an alert and return its UUID."""
        alert_id = str(uuid.uuid4())
        self.conn.execute('''
            INSERT INTO alerts
            (alert_id, timestamp_utc, patient_id, facility_id, severity,
             primary_type, secondary_signals, confidence, total_risk_score,
             payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (alert_id, time.time(), patient_id, facility_id, severity,
              primary_type,
              ','.join(secondary_signals) if secondary_signals else None,
              confidence, total_risk_score,
              json.dumps(payload) if payload else None))
        self.conn.commit()
        return alert_id

    def get_recent_alerts(self, patient_id: str = 'default',
                          hours: float = 24.0, limit: int = 100) -> List[Dict]:
        """Get recent alerts within the specified hours."""
        cutoff = time.time() - (hours * 3600)
        rows = self.conn.execute('''
            SELECT * FROM alerts
            WHERE patient_id = ? AND timestamp_utc >= ?
            ORDER BY timestamp_utc DESC LIMIT ?
        ''', (patient_id, cutoff, limit)).fetchall()
        return [dict(r) for r in rows]

    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        self.conn.execute('''
            UPDATE alerts SET acknowledged = 1, ack_timestamp = ?
            WHERE alert_id = ?
        ''', (time.time(), alert_id))
        self.conn.commit()

    # ── Daily Summaries ─────────────────────────────────────────────────

    def upsert_daily_summary(self, date: str, summary: Dict,
                             patient_id: str = 'default'):
        """Insert or update a daily activity summary."""
        self.conn.execute('''
            INSERT INTO daily_summaries
            (date, patient_id, walking_minutes, sitting_minutes,
             lying_minutes, standing_minutes, transfers_count,
             active_intensity_avg, anomaly_flags, mobility_trend,
             activity_timeline_json, insights_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, patient_id) DO UPDATE SET
              walking_minutes=excluded.walking_minutes,
              sitting_minutes=excluded.sitting_minutes,
              lying_minutes=excluded.lying_minutes,
              standing_minutes=excluded.standing_minutes,
              transfers_count=excluded.transfers_count,
              active_intensity_avg=excluded.active_intensity_avg,
              anomaly_flags=excluded.anomaly_flags,
              mobility_trend=excluded.mobility_trend,
              activity_timeline_json=excluded.activity_timeline_json,
              insights_json=excluded.insights_json
        ''', (date, patient_id,
              summary.get('walking_minutes', 0),
              summary.get('sitting_minutes', 0),
              summary.get('lying_minutes', 0),
              summary.get('standing_minutes', 0),
              summary.get('transfers_count', 0),
              summary.get('active_intensity_avg', 0),
              summary.get('anomaly_flags', 0),
              summary.get('mobility_trend', 'stable'),
              json.dumps(summary.get('activity_timeline', [])),
              json.dumps(summary.get('insights', {}))))
        self.conn.commit()

    # ── Medication ──────────────────────────────────────────────────────

    def add_medication_schedule(self, med_id: str, name: str,
                                schedule_hours: List[int],
                                patient_id: str = 'default',
                                dosage: str = '',
                                reminder_minutes: int = 5):
        """Add a medication schedule."""
        self.conn.execute('''
            INSERT OR REPLACE INTO medication_schedules
            (med_id, patient_id, name, dosage, schedule_hours_utc,
             reminder_minutes_before)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (med_id, patient_id, name, dosage,
              json.dumps(schedule_hours), reminder_minutes))
        self.conn.commit()

    def log_medication_taken(self, med_id: str, scheduled_time: float,
                             status: str = 'ON_TIME',
                             patient_id: str = 'default'):
        """Log a medication taken event."""
        self.conn.execute('''
            INSERT INTO medication_logs
            (med_id, patient_id, scheduled_time, actual_taken_time, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (med_id, patient_id, scheduled_time, time.time(), status))
        self.conn.commit()

    def get_medication_adherence(self, patient_id: str = 'default',
                                 days: int = 7) -> float:
        """Calculate medication adherence rate over N days."""
        cutoff = time.time() - (days * 86400)
        row = self.conn.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status IN ('ON_TIME','LATE') THEN 1 ELSE 0 END) as taken
            FROM medication_logs
            WHERE patient_id = ? AND scheduled_time >= ?
        ''', (patient_id, cutoff)).fetchone()
        total = row['total'] if row['total'] else 0
        taken = row['taken'] if row['taken'] else 0
        return taken / total if total > 0 else 1.0

    # ── Meal Logs ───────────────────────────────────────────────────────

    def log_meal(self, meal_type: str, patient_id: str = 'default',
                 duration_seconds: float = 0, source: str = 'manual',
                 notes: str = ''):
        """Log a meal event."""
        self.conn.execute('''
            INSERT INTO meal_logs
            (patient_id, meal_type, timestamp_utc, duration_seconds, source, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (patient_id, meal_type, time.time(), duration_seconds, source, notes))
        self.conn.commit()

    # ── Exercise Sessions ───────────────────────────────────────────────

    def log_exercise_session(self, session_data: Dict,
                              patient_id: str = 'default'):
        """Log an exercise/yoga session."""
        sid = str(uuid.uuid4())
        self.conn.execute('''
            INSERT INTO exercise_sessions
            (session_id, patient_id, date, time_start, time_end,
             total_duration_sec, difficulty_level, completion_rate, poses_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (sid, patient_id,
              session_data.get('date', time.strftime('%Y-%m-%d')),
              session_data.get('time_start', time.time()),
              session_data.get('time_end'),
              session_data.get('total_duration_sec', 0),
              session_data.get('difficulty_level', 'easy'),
              session_data.get('completion_rate', 0),
              json.dumps(session_data.get('poses', []))))
        self.conn.commit()
        return sid

    # ── Baselines ───────────────────────────────────────────────────────

    def update_activity_baseline(self, hour: int,
                                  distribution: Dict[str, float],
                                  patient_id: str = 'default',
                                  day_of_week: int = None):
        """Update hourly activity baseline distribution."""
        self.conn.execute('''
            INSERT INTO activity_baselines
            (patient_id, hour_of_day, day_of_week, activity_distribution_json,
             sample_count, updated_at)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(patient_id, hour_of_day, day_of_week) DO UPDATE SET
              activity_distribution_json=excluded.activity_distribution_json,
              sample_count=sample_count+1,
              updated_at=excluded.updated_at
        ''', (patient_id, hour, day_of_week,
              json.dumps(distribution), time.time()))
        self.conn.commit()

    def get_activity_baseline(self, hour: int,
                               patient_id: str = 'default') -> Optional[Dict]:
        """Get activity baseline for a specific hour."""
        row = self.conn.execute('''
            SELECT * FROM activity_baselines
            WHERE patient_id = ? AND hour_of_day = ? AND day_of_week IS NULL
        ''', (patient_id, hour)).fetchone()
        if row:
            result = dict(row)
            result['distribution'] = json.loads(result['activity_distribution_json'])
            return result
        return None

    # ── Data Lifecycle ──────────────────────────────────────────────────

    def rotate_old_data(self, skeleton_days: int = 7, alert_days: int = 30,
                        other_days: int = 7):
        """Delete data older than retention windows."""
        now = time.time()
        skel_cutoff = now - (skeleton_days * 86400)
        alert_cutoff = now - (alert_days * 86400)
        other_cutoff = now - (other_days * 86400)

        cur = self.conn.cursor()

        cur.execute('DELETE FROM skeleton_batches WHERE timestamp_start < ?',
                    (skel_cutoff,))
        skel_deleted = cur.rowcount

        cur.execute('DELETE FROM alerts WHERE timestamp_utc < ?',
                    (alert_cutoff,))
        alert_deleted = cur.rowcount

        cur.execute('DELETE FROM medication_logs WHERE scheduled_time < ?',
                    (other_cutoff,))
        cur.execute('DELETE FROM meal_logs WHERE timestamp_utc < ?',
                    (other_cutoff,))

        self.conn.commit()
        self.conn.execute("VACUUM")

        if skel_deleted or alert_deleted:
            logger.info(
                f"Data rotation: deleted {skel_deleted} skeleton batches, "
                f"{alert_deleted} alerts"
            )

    def get_db_size_mb(self) -> float:
        """Get the current database file size in MB."""
        try:
            return os.path.getsize(self.db_path) / (1024 * 1024)
        except OSError:
            return 0.0

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("PIHMSDatabase connection closed")
