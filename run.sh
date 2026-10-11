#!/bin/bash
# FaceCard - AI Biometric Mirror & Celebrity Matcher Launcher for Linux/WSL
cd "$(dirname "$0")"

echo "====================================================================="
echo "   FaceCard: AI Biometric Mirror & Celebrity Matcher (OpenCV)"
echo "====================================================================="
echo ""
echo "Choose execution mode:"
echo "[1] Launch Native OpenCV Desktop App (Live Webcam Window)"
echo "[2] Run Demo Mode (Synthesized Camera Test Feed)"
echo ""
read -p "Enter choice (1/2) [Default 1]: " mode
mode=${mode:-1}

if [ "$mode" = "1" ]; then
    echo "Starting native OpenCV camera app..."
    python3 app.py
elif [ "$mode" = "2" ]; then
    echo "Starting demo feed..."
    python3 app.py --demo
else
    echo "Starting native OpenCV camera app..."
    python3 app.py
fi
