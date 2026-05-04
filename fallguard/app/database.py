"""
Database module for incident logging with SQLite and automatic 90-day purge.
"""
import sqlite3
import logging
import time
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class IncidentDatabase:
    def __init__(self, db_path="./incidents.db"):
        """
        Initialize the incident database.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._init_database()
        
        logger.info(f"IncidentDatabase initialized at {db_path}")
    
    def _init_database(self):
        """Initialize the database schema."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Create incidents table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                level INTEGER NOT NULL,  -- 0=GREEN, 1=YELLOW, 2=RED
                gait_risk REAL,
                sway_risk REAL,
                trunk_risk REAL,
                bed_exit_risk REAL,
                freeze_risk REAL,
                arm_reach_risk REAL
            )
        ''')
        
        # Create index on timestamp for faster purging
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON incidents(timestamp)
        ''')
        
        self.conn.commit()
        
        # Perform initial purge of old records
        self.purge_old_incidents()
        
        logger.info("Database schema initialized")
    
    def log_incident(self, level, risks):
        """
        Log an incident to the database.
        
        Args:
            level (int): Alert level (0=GREEN, 1=YELLOW, 2=RED)
            risks (dict): Dictionary containing risk scores for each signal
        """
        if self.conn is None:
            logger.error("Database connection not initialized")
            return
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO incidents (timestamp, level, gait_risk, sway_risk, trunk_risk, bed_exit_risk, freeze_risk, arm_reach_risk)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            time.time(),
            level,
            risks.get('gait', 0.0),
            risks.get('sway', 0.0),
            risks.get('trunk', 0.0),
            risks.get('bed_exit', 0.0),
            risks.get('freeze', 0.0),
            risks.get('arm_reach', 0.0)
        ))
        
        self.conn.commit()
        
        logger.info(f"Incident logged: level={level}, risks={risks}")
    
    def purge_old_incidents(self, days_to_keep=90):
        """
        Purge incidents older than the specified number of days.
        
        Args:
            days_to_keep (int): Number of days of incidents to keep
        """
        if self.conn is None:
            logger.error("Database connection not initialized")
            return
        
        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
        
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM incidents WHERE timestamp < ?', (cutoff_time,))
        deleted_count = cursor.rowcount
        self.conn.commit()
        
        if deleted_count > 0:
            logger.info(f"Purged {deleted_count} old incidents (older than {days_to_keep} days)")
    
    def get_recent_incidents(self, limit=100):
        """
        Get recent incidents from the database.
        
        Args:
            limit (int): Maximum number of incidents to return
            
        Returns:
            list: List of incident dictionaries
        """
        if self.conn is None:
            logger.error("Database connection not initialized")
            return []
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, level, gait_risk, sway_risk, trunk_risk, bed_exit_risk, freeze_risk, arm_reach_risk
            FROM incidents
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        incidents = []
        for row in rows:
            incidents.append({
                'timestamp': row[0],
                'level': row[1],
                'gait_risk': row[2],
                'sway_risk': row[3],
                'trunk_risk': row[4],
                'bed_exit_risk': row[5],
                'freeze_risk': row[6],
                'arm_reach_risk': row[7]
            })
        
        return incidents
    
    def close(self):
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
