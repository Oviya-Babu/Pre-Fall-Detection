-- PIHMS v2.0 SQLite Schema
-- Designed for embedded edge deployment with compression

-- Skeleton frames (delta-compressed batches)
CREATE TABLE IF NOT EXISTS skeleton_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_start REAL NOT NULL,
    timestamp_end REAL NOT NULL,
    patient_id TEXT NOT NULL DEFAULT 'default',
    num_frames INTEGER NOT NULL,
    batch_data BLOB NOT NULL,          -- gzip-compressed delta-encoded batch
    batch_size_bytes INTEGER NOT NULL,
    avg_confidence REAL DEFAULT 0.0,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_skel_ts ON skeleton_batches(timestamp_start);
CREATE INDEX IF NOT EXISTS idx_skel_patient ON skeleton_batches(patient_id);

-- Daily activity summaries
CREATE TABLE IF NOT EXISTS daily_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,                 -- YYYY-MM-DD
    patient_id TEXT NOT NULL DEFAULT 'default',
    walking_minutes REAL DEFAULT 0,
    sitting_minutes REAL DEFAULT 0,
    lying_minutes REAL DEFAULT 0,
    standing_minutes REAL DEFAULT 0,
    transfers_count INTEGER DEFAULT 0,
    active_intensity_avg REAL DEFAULT 0,
    anomaly_flags INTEGER DEFAULT 0,
    mobility_trend TEXT DEFAULT 'stable',
    activity_timeline_json TEXT,        -- JSON array of hourly breakdown
    insights_json TEXT,                 -- JSON object with daily insights
    created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE(date, patient_id)
);
CREATE INDEX IF NOT EXISTS idx_summary_date ON daily_summaries(date);

-- Alerts and incidents
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT NOT NULL UNIQUE,      -- UUID
    timestamp_utc REAL NOT NULL,
    patient_id TEXT NOT NULL DEFAULT 'default',
    facility_id TEXT DEFAULT 'FAC001',
    severity TEXT NOT NULL,             -- CRITICAL, HIGH, MEDIUM, LOW
    primary_type TEXT NOT NULL,         -- PRE_FALL, ACTIVITY_ANOMALY, etc.
    secondary_signals TEXT,             -- comma-separated signal names
    confidence REAL DEFAULT 0.0,
    total_risk_score REAL DEFAULT 0.0,
    payload_json TEXT,                  -- full alert payload as JSON
    acknowledged INTEGER DEFAULT 0,
    ack_timestamp REAL,
    suppressed INTEGER DEFAULT 0,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_alert_ts ON alerts(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_alert_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alert_patient ON alerts(patient_id);

-- Medication schedules
CREATE TABLE IF NOT EXISTS medication_schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    med_id TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    dosage TEXT,
    schedule_hours_utc TEXT NOT NULL,   -- JSON array of hours e.g. [8, 20]
    start_date TEXT,
    end_date TEXT,
    reminder_minutes_before INTEGER DEFAULT 5,
    active INTEGER DEFAULT 1,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);

-- Medication compliance logs
CREATE TABLE IF NOT EXISTS medication_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    med_id TEXT NOT NULL,
    patient_id TEXT NOT NULL DEFAULT 'default',
    scheduled_time REAL NOT NULL,
    actual_taken_time REAL,
    status TEXT NOT NULL,               -- ON_TIME, LATE, MISSED, SKIPPED
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_medlog_ts ON medication_logs(scheduled_time);
CREATE INDEX IF NOT EXISTS idx_medlog_patient ON medication_logs(patient_id);

-- Exercise/yoga sessions
CREATE TABLE IF NOT EXISTS exercise_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    patient_id TEXT NOT NULL DEFAULT 'default',
    date TEXT NOT NULL,
    time_start REAL NOT NULL,
    time_end REAL,
    total_duration_sec REAL DEFAULT 0,
    difficulty_level TEXT DEFAULT 'easy',
    completion_rate REAL DEFAULT 0.0,
    poses_json TEXT,                    -- JSON array of pose results
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_exercise_date ON exercise_sessions(date);

-- Meal logs
CREATE TABLE IF NOT EXISTS meal_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL DEFAULT 'default',
    meal_type TEXT NOT NULL,            -- BREAKFAST, LUNCH, DINNER, SNACK
    timestamp_utc REAL NOT NULL,
    duration_seconds REAL DEFAULT 0,
    source TEXT DEFAULT 'manual',       -- manual or pose_detected
    notes TEXT,
    created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
CREATE INDEX IF NOT EXISTS idx_meal_ts ON meal_logs(timestamp_utc);

-- Activity baseline profiles (learned in first week)
CREATE TABLE IF NOT EXISTS activity_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL DEFAULT 'default',
    hour_of_day INTEGER NOT NULL,       -- 0-23
    day_of_week INTEGER,                -- 0=Mon, 6=Sun (NULL = all days)
    activity_distribution_json TEXT,    -- JSON: {"WALKING": 0.2, "SITTING": 0.5, ...}
    sample_count INTEGER DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT (strftime('%s','now')),
    UNIQUE(patient_id, hour_of_day, day_of_week)
);

-- Gait baseline profiles (per-patient personalization)
CREATE TABLE IF NOT EXISTS gait_baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL UNIQUE DEFAULT 'default',
    baseline_stride_length REAL,
    baseline_stride_time REAL,
    baseline_sway_amplitude REAL,
    baseline_sway_frequency REAL,
    baseline_gait_asymmetry REAL,
    known_limitations TEXT,             -- JSON array of known conditions
    alert_threshold_adjustment REAL DEFAULT 1.0,
    sample_count INTEGER DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT (strftime('%s','now'))
);
