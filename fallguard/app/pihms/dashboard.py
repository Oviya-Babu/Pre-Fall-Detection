"""
PIHMS Dashboard — Real-time Web UI using FastAPI + WebSocket.
Serves at http://localhost:8080

Features:
  - Real-time risk gauges per person
  - Alert feed with severity badges
  - Activity timeline
  - System performance metrics
  - Multi-person crowd monitor
"""

import asyncio
import json
import time
import logging
import os
from typing import Dict, List, Set
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

app = FastAPI(title="PIHMS Dashboard", version="2.0")


# ── WebSocket Connection Manager ────────────────────────────────────

class ConnectionManager:
    """Manages active WebSocket connections for real-time updates."""

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        logger.info(f"Dashboard client connected ({len(self.active)} total)")

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)
        logger.info(f"Dashboard client disconnected ({len(self.active)} total)")

    async def broadcast(self, data: dict):
        """Send JSON data to all connected clients."""
        if not self.active:
            return
        msg = json.dumps(data)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.discard(ws)


manager = ConnectionManager()

# ── Shared state (updated by the pipeline thread) ───────────────────

_dashboard_state: Dict = {
    'persons': {},       # pid -> {risk, activity, features, alert}
    'system': {          # system stats
        'fps': 0, 'latency_ms': 0, 'frame_count': 0,
        'persons_tracked': 0, 'uptime_sec': 0,
    },
    'alerts': [],        # recent alert list (last 50)
    'timestamp': 0,
}


def update_dashboard_state(persons: Dict, system: Dict, alerts: List):
    """Called from the pipeline thread to update dashboard state."""
    _dashboard_state['persons'] = persons
    _dashboard_state['system'] = system
    _dashboard_state['alerts'] = alerts[-50:]  # Keep last 50
    _dashboard_state['timestamp'] = time.time()


async def broadcast_state():
    """Send current state to all WebSocket clients."""
    await manager.broadcast(_dashboard_state)


# ── REST Endpoints ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the main dashboard HTML."""
    return DASHBOARD_HTML


@app.get("/api/status")
async def api_status():
    """Get current system status."""
    return _dashboard_state


@app.get("/api/alerts")
async def api_alerts():
    """Get recent alerts."""
    return {"alerts": _dashboard_state.get('alerts', [])}


# ── WebSocket Endpoint ──────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send initial state
        await ws.send_text(json.dumps(_dashboard_state))
        # Keep alive — the pipeline pushes updates via broadcast_state()
        while True:
            # Wait for client messages (heartbeat / commands)
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                # Handle client commands if needed
                if data == "ping":
                    await ws.send_text('{"type":"pong"}')
            except asyncio.TimeoutError:
                # Send heartbeat
                await ws.send_text('{"type":"heartbeat"}')
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ── Dashboard HTML ──────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PIHMS v2.0 — PreFall Intelligence Health Monitor</title>
<meta name="description" content="Real-time pre-fall detection and health monitoring dashboard">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg-primary: #0a0e1a;
  --bg-card: #111827;
  --bg-card-hover: #1a2332;
  --border: #1e293b;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --accent-blue: #3b82f6;
  --accent-cyan: #06b6d4;
  --accent-green: #10b981;
  --accent-yellow: #f59e0b;
  --accent-orange: #f97316;
  --accent-red: #ef4444;
  --accent-purple: #8b5cf6;
  --glow-green: 0 0 20px rgba(16,185,129,0.3);
  --glow-red: 0 0 20px rgba(239,68,68,0.4);
  --glow-yellow: 0 0 20px rgba(245,158,11,0.3);
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  font-family:'Inter',sans-serif; background:var(--bg-primary);
  color:var(--text-primary); min-height:100vh; overflow-x:hidden;
}
.header {
  background:linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
  border-bottom:1px solid var(--border); padding:12px 24px;
  display:flex; align-items:center; justify-content:space-between;
}
.header h1 { font-size:1.3rem; font-weight:700; }
.header h1 span { color:var(--accent-cyan); }
.header-stats {
  display:flex; gap:20px; font-size:0.82rem; color:var(--text-secondary);
}
.header-stats .stat-val { color:var(--accent-cyan); font-weight:600; }
.status-dot {
  width:8px; height:8px; border-radius:50%; display:inline-block;
  margin-right:6px; animation:pulse 2s infinite;
}
.status-dot.online { background:var(--accent-green); box-shadow:var(--glow-green); }
.status-dot.offline { background:var(--accent-red); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

.grid {
  display:grid; grid-template-columns:1fr 1fr 1fr;
  gap:16px; padding:16px 24px; max-width:1600px; margin:auto;
}
@media(max-width:1200px){ .grid{grid-template-columns:1fr 1fr;} }
@media(max-width:768px){ .grid{grid-template-columns:1fr;} }

.card {
  background:var(--bg-card); border:1px solid var(--border);
  border-radius:12px; padding:16px; transition:all 0.3s ease;
}
.card:hover { border-color:var(--accent-blue); background:var(--bg-card-hover); }
.card h2 {
  font-size:0.85rem; font-weight:600; color:var(--text-secondary);
  text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;
}
.card.full-width { grid-column:1/-1; }
.card.two-col { grid-column:span 2; }

/* Risk Gauge */
.gauge-container { margin:8px 0; }
.gauge-label {
  display:flex; justify-content:space-between; font-size:0.78rem;
  color:var(--text-secondary); margin-bottom:4px;
}
.gauge-bar {
  height:8px; background:#1e293b; border-radius:4px; overflow:hidden;
}
.gauge-fill {
  height:100%; border-radius:4px; transition:width 0.5s ease, background 0.5s ease;
}
.gauge-fill.low { background:linear-gradient(90deg,#10b981,#34d399); }
.gauge-fill.med { background:linear-gradient(90deg,#f59e0b,#fbbf24); }
.gauge-fill.high { background:linear-gradient(90deg,#f97316,#fb923c); }
.gauge-fill.crit { background:linear-gradient(90deg,#ef4444,#f87171); box-shadow:var(--glow-red); }

/* Person Cards */
.person-grid { display:flex; flex-direction:column; gap:12px; }
.person-row {
  display:flex; align-items:center; gap:12px; padding:10px 12px;
  background:#0f172a; border-radius:8px; border-left:3px solid var(--accent-green);
  transition:all 0.3s ease;
}
.person-row.warning { border-left-color:var(--accent-yellow); }
.person-row.high { border-left-color:var(--accent-orange); }
.person-row.critical {
  border-left-color:var(--accent-red);
  animation:flash 1s infinite alternate;
}
@keyframes flash { from{background:#0f172a} to{background:#1a0505} }
.person-id {
  font-size:1.1rem; font-weight:700; color:var(--accent-cyan);
  min-width:50px;
}
.person-info { flex:1; font-size:0.8rem; }
.person-risk {
  font-size:1.4rem; font-weight:700; min-width:60px; text-align:right;
}

/* Alert Feed */
.alert-feed { max-height:320px; overflow-y:auto; }
.alert-item {
  padding:8px 12px; border-radius:6px; margin-bottom:6px;
  border-left:3px solid var(--accent-yellow); background:#0f172a;
  font-size:0.78rem; animation:slideIn 0.3s ease;
}
@keyframes slideIn { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:none} }
.alert-item.CRITICAL { border-left-color:var(--accent-red); background:#1a0808; }
.alert-item.HIGH { border-left-color:var(--accent-orange); }
.alert-item.MEDIUM { border-left-color:var(--accent-yellow); }
.alert-time { color:var(--text-muted); font-size:0.72rem; }
.alert-badge {
  display:inline-block; padding:1px 6px; border-radius:4px;
  font-size:0.68rem; font-weight:600; margin-right:6px;
}
.alert-badge.CRITICAL { background:#7f1d1d; color:#fca5a5; }
.alert-badge.HIGH { background:#7c2d12; color:#fed7aa; }
.alert-badge.MEDIUM { background:#78350f; color:#fde68a; }

/* Metric tiles */
.metrics-grid {
  display:grid; grid-template-columns:repeat(4,1fr); gap:10px;
}
.metric-tile {
  text-align:center; padding:12px 8px; background:#0f172a;
  border-radius:8px;
}
.metric-val { font-size:1.5rem; font-weight:700; }
.metric-label { font-size:0.7rem; color:var(--text-muted); margin-top:2px; }

/* Activity bars */
.activity-bar {
  display:flex; height:24px; border-radius:6px; overflow:hidden;
  margin:8px 0; background:#1e293b;
}
.activity-bar div { transition:width 0.5s ease; }

/* No data state */
.no-data {
  text-align:center; color:var(--text-muted); padding:30px;
  font-size:0.9rem;
}
.no-data .icon { font-size:2rem; margin-bottom:8px; }

/* scrollbar */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0a0e1a; }
::-webkit-scrollbar-thumb { background:#334155; border-radius:3px; }
</style>
</head>
<body>

<div class="header">
  <h1>🛡️ <span>PIHMS</span> v2.0 — PreFall Intelligence Health Monitor</h1>
  <div class="header-stats">
    <div><span class="status-dot online" id="connDot"></span><span id="connStatus">Connecting...</span></div>
    <div>FPS: <span class="stat-val" id="hFps">-</span></div>
    <div>Latency: <span class="stat-val" id="hLatency">-</span>ms</div>
    <div>People: <span class="stat-val" id="hPeople">0</span></div>
    <div>Frame: <span class="stat-val" id="hFrame">0</span></div>
  </div>
</div>

<div class="grid">

  <!-- System Performance -->
  <div class="card">
    <h2>⚡ System Performance</h2>
    <div class="metrics-grid">
      <div class="metric-tile">
        <div class="metric-val" id="mFps" style="color:var(--accent-green)">-</div>
        <div class="metric-label">FPS</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val" id="mLatency" style="color:var(--accent-cyan)">-</div>
        <div class="metric-label">Latency (ms)</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val" id="mPeople" style="color:var(--accent-purple)">0</div>
        <div class="metric-label">Tracked</div>
      </div>
      <div class="metric-tile">
        <div class="metric-val" id="mUptime" style="color:var(--text-secondary)">0s</div>
        <div class="metric-label">Uptime</div>
      </div>
    </div>
    <div class="gauge-container" style="margin-top:12px">
      <div class="gauge-label"><span>Pipeline Load</span><span id="loadPct">0%</span></div>
      <div class="gauge-bar"><div class="gauge-fill low" id="loadBar" style="width:0%"></div></div>
    </div>
  </div>

  <!-- Primary Person Risk -->
  <div class="card" id="primaryCard">
    <h2>🎯 Primary Person Risk</h2>
    <div id="primaryContent">
      <div class="no-data"><div class="icon">👤</div>No person detected</div>
    </div>
  </div>

  <!-- Alert Feed -->
  <div class="card">
    <h2>🚨 Alert Feed</h2>
    <div class="alert-feed" id="alertFeed">
      <div class="no-data"><div class="icon">✅</div>No alerts</div>
    </div>
  </div>

  <!-- Crowd Monitor -->
  <div class="card two-col">
    <h2>👥 Crowd Monitor — All Persons</h2>
    <div class="person-grid" id="crowdGrid">
      <div class="no-data"><div class="icon">📡</div>Waiting for detections...</div>
    </div>
  </div>

  <!-- Biomarker Analysis -->
  <div class="card">
    <h2>🧬 Biomarker Analysis</h2>
    <div id="bioContent">
      <div class="no-data"><div class="icon">📊</div>Awaiting data...</div>
    </div>
  </div>

</div>

<script>
const WS_URL = `ws://${location.host}/ws`;
let ws, reconnectTimer;

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    document.getElementById('connDot').className = 'status-dot online';
    document.getElementById('connStatus').textContent = 'Connected';
    clearInterval(reconnectTimer);
  };
  ws.onclose = () => {
    document.getElementById('connDot').className = 'status-dot offline';
    document.getElementById('connStatus').textContent = 'Reconnecting...';
    reconnectTimer = setTimeout(connect, 2000);
  };
  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'heartbeat' || data.type === 'pong') return;
      updateDashboard(data);
    } catch(err) { console.error('Parse error:', err); }
  };
}

function riskColor(score) {
  if (score >= 80) return 'var(--accent-red)';
  if (score >= 60) return 'var(--accent-orange)';
  if (score >= 40) return 'var(--accent-yellow)';
  return 'var(--accent-green)';
}

function riskClass(score) {
  if (score >= 80) return 'crit';
  if (score >= 60) return 'high';
  if (score >= 40) return 'med';
  return 'low';
}

function personRowClass(score) {
  if (score >= 80) return 'critical';
  if (score >= 60) return 'high';
  if (score >= 40) return 'warning';
  return '';
}

function fmtTime(ts) {
  if (!ts) return '-';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

function fmtUptime(sec) {
  if (sec < 60) return Math.floor(sec) + 's';
  if (sec < 3600) return Math.floor(sec/60) + 'm';
  return Math.floor(sec/3600) + 'h ' + Math.floor((sec%3600)/60) + 'm';
}

function updateDashboard(data) {
  const sys = data.system || {};
  const persons = data.persons || {};
  const alerts = data.alerts || [];

  // Header stats
  document.getElementById('hFps').textContent = (sys.fps||0).toFixed(1);
  document.getElementById('hLatency').textContent = (sys.latency_ms||0).toFixed(1);
  document.getElementById('hPeople').textContent = sys.persons_tracked || 0;
  document.getElementById('hFrame').textContent = sys.frame_count || 0;

  // Performance metrics
  document.getElementById('mFps').textContent = (sys.fps||0).toFixed(1);
  document.getElementById('mFps').style.color = (sys.fps||0) >= 25 ? 'var(--accent-green)' : 'var(--accent-red)';
  document.getElementById('mLatency').textContent = (sys.latency_ms||0).toFixed(1);
  document.getElementById('mPeople').textContent = sys.persons_tracked || 0;
  document.getElementById('mUptime').textContent = fmtUptime(sys.uptime_sec || 0);

  // Pipeline load gauge
  const loadPct = Math.min(100, ((sys.latency_ms || 0) / 33) * 100);
  document.getElementById('loadPct').textContent = loadPct.toFixed(0) + '%';
  document.getElementById('loadBar').style.width = loadPct + '%';
  document.getElementById('loadBar').className = 'gauge-fill ' + riskClass(loadPct);

  // Primary person
  const pids = Object.keys(persons).sort((a,b) => a-b);
  const primaryEl = document.getElementById('primaryContent');
  if (pids.length > 0) {
    const pid = pids[0];
    const p = persons[pid];
    const r = p.risk || {};
    const pf = p.pre_fall || {};
    const feat = p.features || {};
    const score = r.total_risk || 0;
    primaryEl.innerHTML = `
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
        <div class="person-id">ID:${pid}</div>
        <div class="person-risk" style="color:${riskColor(score)}">${score}</div>
        <div style="flex:1;font-size:0.8rem;color:var(--text-secondary)">${r.severity||'LOW'}</div>
      </div>
      <div class="gauge-container">
        <div class="gauge-label"><span>Total Risk</span><span>${score}</span></div>
        <div class="gauge-bar"><div class="gauge-fill ${riskClass(score)}" style="width:${score}%"></div></div>
      </div>
      <div class="gauge-container">
        <div class="gauge-label"><span>Pre-Fall</span><span>${pf.risk_score||0}</span></div>
        <div class="gauge-bar"><div class="gauge-fill ${riskClass(pf.risk_score||0)}" style="width:${pf.risk_score||0}%"></div></div>
      </div>
      <div style="margin-top:8px;font-size:0.78rem;color:var(--text-secondary)">
        Activity: <strong>${p.activity||'-'}</strong> &nbsp;|&nbsp;
        Trunk: ${(feat.trunk_lean_angle||0).toFixed(1)}° &nbsp;|&nbsp;
        Sway: ${(feat.sway_amplitude_x||0).toFixed(4)}
      </div>
    `;
  } else {
    primaryEl.innerHTML = '<div class="no-data"><div class="icon">👤</div>No person detected</div>';
  }

  // Crowd monitor
  const crowdEl = document.getElementById('crowdGrid');
  if (pids.length > 0) {
    crowdEl.innerHTML = pids.map(pid => {
      const p = persons[pid];
      const r = p.risk || {};
      const score = r.total_risk || 0;
      return `<div class="person-row ${personRowClass(score)}">
        <div class="person-id">ID:${pid}</div>
        <div class="person-info">
          <div>${p.activity||'IDLE'} &nbsp;|&nbsp; ${r.severity||'LOW'}</div>
          <div style="color:var(--text-muted)">Biomarkers: ${(p.biomarkers||[]).join(', ')||'None'}</div>
        </div>
        <div class="person-risk" style="color:${riskColor(score)}">${score}</div>
      </div>`;
    }).join('');
  } else {
    crowdEl.innerHTML = '<div class="no-data"><div class="icon">📡</div>Waiting for detections...</div>';
  }

  // Alert feed
  const feedEl = document.getElementById('alertFeed');
  if (alerts.length > 0) {
    feedEl.innerHTML = alerts.slice(-20).reverse().map(a => {
      const sev = a.severity || 'MEDIUM';
      return `<div class="alert-item ${sev}">
        <span class="alert-badge ${sev}">${sev}</span>
        <strong>${a.primary_type||'ALERT'}</strong> — Risk: ${a.risk_score||0}
        ${a.person_id !== undefined ? '(ID:'+a.person_id+')' : ''}
        <div class="alert-time">${a.timestamp_utc || fmtTime(a.timestamp)}</div>
      </div>`;
    }).join('');
  } else {
    feedEl.innerHTML = '<div class="no-data"><div class="icon">✅</div>No alerts</div>';
  }

  // Biomarker analysis
  const bioEl = document.getElementById('bioContent');
  if (pids.length > 0) {
    let bioHtml = '';
    pids.forEach(pid => {
      const p = persons[pid];
      const pf = p.pre_fall || {};
      const comp = pf.components || {};
      bioHtml += `<div style="margin-bottom:10px">
        <div style="font-weight:600;font-size:0.82rem;color:var(--accent-cyan)">ID:${pid}</div>
        <div class="gauge-container">
          <div class="gauge-label"><span>Gait Instab.</span><span>${(comp.gait_instability||0).toFixed(3)}</span></div>
          <div class="gauge-bar"><div class="gauge-fill ${riskClass((comp.gait_instability||0)*100)}" style="width:${Math.min(100,(comp.gait_instability||0)*100)}%"></div></div>
        </div>
        <div class="gauge-container">
          <div class="gauge-label"><span>Sway Ratio</span><span>${(comp.sway_ratio||0).toFixed(3)}</span></div>
          <div class="gauge-bar"><div class="gauge-fill ${riskClass((comp.sway_ratio||0)*100)}" style="width:${Math.min(100,(comp.sway_ratio||0)*100)}%"></div></div>
        </div>
        <div class="gauge-container">
          <div class="gauge-label"><span>Postural</span><span>${(comp.postural_instability||0).toFixed(3)}</span></div>
          <div class="gauge-bar"><div class="gauge-fill ${riskClass((comp.postural_instability||0)*100)}" style="width:${Math.min(100,(comp.postural_instability||0)*100)}%"></div></div>
        </div>
      </div>`;
    });
    bioEl.innerHTML = bioHtml;
  } else {
    bioEl.innerHTML = '<div class="no-data"><div class="icon">📊</div>Awaiting data...</div>';
  }
}

connect();
</script>
</body>
</html>"""


# ── Dashboard Server Runner ─────────────────────────────────────────

def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard server (blocking). Call from a thread."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")
