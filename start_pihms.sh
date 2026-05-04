#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  PIHMS v2.0 — Master Launch Script
#  Auto-detects DroidCam IP, validates network, starts all services.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/fallguard/app"
DROIDCAM_PORT=4747
DROIDCAM_IP=""

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; WHITE='\033[1;37m'; NC='\033[0m'

banner() {
  echo -e "${CYAN}"
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║        PIHMS v2.0 — PreFall Intelligence Health Monitor      ║"
  echo "║        Real-Time DroidCam + Skeleton Analytics System        ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo -e "${NC}"
}

step()  { echo -e "${WHITE}[STEP]${NC} $1"; }
ok()    { echo -e "${GREEN}[  OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; }
info()  { echo -e "${CYAN}[INFO]${NC} $1"; }

# ── 1. Show banner ──────────────────────────────────────────────────────
banner

# ── 2. Show system network info ─────────────────────────────────────────
step "Checking network interfaces..."
MY_IP=$(ip addr show | grep 'inet ' | grep -v '127.0.0.1' | grep -v 'docker' | awk '{print $2}' | cut -d/ -f1 | head -1)
MY_SUBNET=$(echo "$MY_IP" | cut -d. -f1-3)
info "System IP: ${WHITE}$MY_IP${NC}"
info "Subnet:    ${WHITE}$MY_SUBNET.0/24${NC}"
echo ""

# ── 3. DroidCam discovery ────────────────────────────────────────────────
step "Searching for DroidCam (port ${DROIDCAM_PORT})..."
echo -e "   ${YELLOW}► Make sure your phone's DroidCam app is OPEN and on the same WiFi!${NC}"
echo ""

KNOWN_IP="10.227.127.50"
FOUND_IPS=()

# Quick check on the last known IP first
echo "   Checking known IP: $KNOWN_IP..."
if curl -s --connect-timeout 0.5 "http://${KNOWN_IP}:${DROIDCAM_PORT}/" -o /dev/null; then
  FOUND_IPS+=("$KNOWN_IP")
  echo -e "   ${GREEN}✓ DroidCam found instantly at: http://${KNOWN_IP}:${DROIDCAM_PORT}${NC}"
else
  echo "   Not found at known IP. Scanning subnet ${MY_SUBNET}.0/24..."
  for i in $(seq 1 254); do
    ip="${MY_SUBNET}.${i}"
    result=$(curl -s --connect-timeout 0.3 "http://${ip}:${DROIDCAM_PORT}/" \
             -o /dev/null -w "%{http_code}" 2>/dev/null || true)
    if [ "$result" != "000" ] && [ -n "$result" ]; then
      FOUND_IPS+=("$ip")
      echo -e "   ${GREEN}✓ DroidCam found at: http://${ip}:${DROIDCAM_PORT}${NC}"
      break # Stop after finding the first one to speed up
    fi
  done
fi

if [ ${#FOUND_IPS[@]} -eq 0 ]; then
  warn "DroidCam NOT found on the local subnet."
  echo ""
  echo -e "   ${YELLOW}Troubleshooting checklist:${NC}"
  echo "   1. Open DroidCam on your phone"
  echo "   2. Ensure phone is connected to the SAME WiFi as this computer"
  echo "       → Computer WiFi: ${MY_SUBNET}.x network"
  echo "       → Check phone IP in DroidCam app settings"
  echo "   3. Firewall: allow port ${DROIDCAM_PORT} (sudo ufw allow ${DROIDCAM_PORT}/tcp)"
  echo ""
  echo -e "   ${CYAN}Known DroidCam IP (last config): 10.236.100.192${NC}"
  echo -e "   ${CYAN}Current system subnet:           ${MY_SUBNET}.0/24${NC}"
  LAST_SUBNET="10.236.100"
  if [ "${MY_SUBNET}" != "${LAST_SUBNET}" ]; then
    echo ""
    echo -e "   ${RED}⚠  Subnets DON'T match!${NC}"
    echo "   Your phone is likely on 10.236.100.x but this PC is on ${MY_SUBNET}.x"
    echo "   → Reconnect your PHONE to the same WiFi as this computer"
    echo "   → OR use a hotspot from phone and connect PC to it"
  fi
  echo ""
  read -rp "   Enter DroidCam IP manually (or press Enter to use webcam fallback): " MANUAL_IP
  if [ -n "$MANUAL_IP" ]; then
    DROIDCAM_IP="$MANUAL_IP"
  else
    DROIDCAM_IP="WEBCAM"
  fi
else
  DROIDCAM_IP="${FOUND_IPS[0]}"
  if [ ${#FOUND_IPS[@]} -gt 1 ]; then
    echo ""
    echo "   Multiple DroidCam devices found. Select one:"
    select ip in "${FOUND_IPS[@]}"; do
      DROIDCAM_IP="$ip"; break
    done
  fi
fi

# ── 4. Set camera URL via env variable ──────────────────────────────────
if [ "$DROIDCAM_IP" == "WEBCAM" ]; then
  export PIHMS_CAM_URL="0"
  warn "Using system webcam (device 0) as fallback"
else
  export PIHMS_CAM_URL="http://${DROIDCAM_IP}:${DROIDCAM_PORT}/video"
  ok "DroidCam URL: ${WHITE}$PIHMS_CAM_URL${NC}"
fi

echo ""

# ── 5. Verify MQTT broker ────────────────────────────────────────────────
step "Checking MQTT broker (mosquitto)..."
if systemctl is-active --quiet mosquitto 2>/dev/null; then
  ok "Mosquitto MQTT broker is running"
else
  warn "Mosquitto not running, attempting to start..."
  sudo systemctl start mosquitto 2>/dev/null || true
  sleep 1
  if systemctl is-active --quiet mosquitto 2>/dev/null; then
    ok "Mosquitto started successfully"
  else
    warn "Mosquitto could not be started — MQTT will be disabled (non-fatal)"
  fi
fi

echo ""

# ── 6. Check Python deps ─────────────────────────────────────────────────
step "Verifying Python dependencies..."
python3 - <<'PYCHECK'
missing = []
try: import cv2
except: missing.append("opencv-python")
try: import numpy
except: missing.append("numpy")
try: import fastapi
except: missing.append("fastapi")
try: import uvicorn
except: missing.append("uvicorn")
try: import paho.mqtt.client
except: missing.append("paho-mqtt")
try: import websockets
except: missing.append("websockets")
if missing:
    print(f"MISSING: {', '.join(missing)}")
    print("Run: pip3 install " + " ".join(missing))
    exit(1)
else:
    print("All dependencies OK")
PYCHECK
ok "Python dependencies verified"

echo ""

# ── 7. Launch system ─────────────────────────────────────────────────────
step "Launching PIHMS v2.0..."
info "Dashboard will be available at: ${WHITE}http://localhost:8080${NC}"
info "Press 'q' in the video window to stop"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

cd "$APP_DIR"
exec python3 pihms_live.py
