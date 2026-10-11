#!/bin/bash
# AI Biometric Mirror & Celebrity Matcher Launcher for Linux/WSL
cd "$(dirname "$0")"

echo "====================================================================="
echo "   AI Biometric Mirror & Celebrity Matcher (OpenCV)"
echo "====================================================================="
echo ""
echo "Choose execution mode:"
echo "[1] Launch Native OpenCV Desktop App (Live Webcam Window)"
echo "[2] Launch Browser-Based Web Mirror (http://localhost:5000) [Recommended for WSL]"
echo "[3] Run Demo Mode (Synthesized Camera Test Feed)"
echo "[4] Push to GitHub (git push -u origin main)"
echo ""
read -p "Enter choice (1/2/3/4) [Default 2]: " mode
mode=${mode:-2}

if [ "$mode" = "1" ]; then
    echo "Starting native OpenCV camera app..."
    python3 app.py
elif [ "$mode" = "2" ]; then
    echo "Starting browser web mirror at http://localhost:5000..."
    python3 web_app.py
elif [ "$mode" = "3" ]; then
    echo "Starting demo feed..."
    python3 app.py --demo
elif [ "$mode" = "4" ]; then
    echo "Pushing commits to https://github.com/Watchdogitsme/facecard.git ..."
    git push -u origin main
else
    echo "Starting web mirror..."
    python3 web_app.py
fi
