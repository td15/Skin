#!/usr/bin/env bash
set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$BASE_DIR/backend"
ML_SERVICE_LOG="$BACKEND_DIR/ml_service.log"
BACKEND_LOG="$BACKEND_DIR/backend.log"
FRONTEND_LOG="$BASE_DIR/frontend.log"

cd "$BACKEND_DIR"

if [ -x "$BACKEND_DIR/venv/bin/python3" ]; then
  PYTHON="$BACKEND_DIR/venv/bin/python3"
else
  PYTHON="$(command -v python3 || true)"
fi

if [ -z "$PYTHON" ]; then
  echo "No python3 executable found. Please install Python 3 or activate your venv."
  exit 1
fi

if [ ! -f "$BACKEND_DIR/src/ml_service.py" ]; then
  echo "ML service file not found: $BACKEND_DIR/src/ml_service.py"
  exit 1
fi

echo "Starting ML service..."
nohup "$PYTHON" "$BACKEND_DIR/src/ml_service.py" > "$ML_SERVICE_LOG" 2>&1 &
ml_pid=$!
echo "ML service PID: $ml_pid"

cd "$BACKEND_DIR"
echo "Starting Node backend..."
nohup npm run start > "$BACKEND_LOG" 2>&1 &
backend_pid=$!
echo "Backend PID: $backend_pid"

cd "$BASE_DIR"
echo "Starting Frontend..."
nohup npm run dev > "$FRONTEND_LOG" 2>&1 &
frontend_pid=$!
echo "Frontend PID: $frontend_pid"

sleep 2

printf "\nServices started:\n"
printf "- ML service: http://localhost:5001 (PID %s)\n" "$ml_pid"
printf "- Node backend: http://localhost:3001 (PID %s)\n" "$backend_pid"
printf "- Frontend: http://localhost:8080 (PID %s)\n" "$frontend_pid"
printf "\nLogs:\n- %s\n- %s\n- %s\n" "$ML_SERVICE_LOG" "$BACKEND_LOG" "$FRONTEND_LOG"

echo "If the frontend does not appear immediately, wait a few seconds and refresh your browser."