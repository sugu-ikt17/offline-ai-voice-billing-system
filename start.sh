#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "🚀 Starting Offline AI Voice Billing System..."

# Start Backend
echo "🔵 Starting Backend..."
gnome-terminal -- bash -c "
cd '$PROJECT_DIR'
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
exec bash
"

# Start Frontend
echo "�� Starting Frontend..."
gnome-terminal -- bash -c "
cd '$PROJECT_DIR/frontend'
python3 -m http.server 5500
exec bash
"

echo "✅ Backend: http://localhost:8000"
echo "✅ Frontend: http://localhost:5500"
