#!/bin/bash
# FaceCard - AI Biometric Mirror & Celebrity Matcher Launcher for Linux/WSL
cd "$(dirname "$0")"

echo "====================================================================="
echo "   FaceCard: AI Biometric Mirror & Celebrity Matcher"
echo "====================================================================="
echo ""
echo "Choose execution mode:"
echo "1. Launch Native OpenCV Desktop App"
echo "2. Test Before Launching - Check File Integrity & System Readiness"
echo "3. Install / Verify Requirements"
echo ""
read -p "Enter choice (1/2/3): " mode
mode=${mode:-1}

if [ "$mode" = "1" ]; then
    echo "Starting native OpenCV camera app..."
    python3 app.py
elif [ "$mode" = "2" ]; then
    echo "Checking file integrity and testing system readiness..."
    python3 test_system.py
elif [ "$mode" = "3" ] || [ "$mode" = "6" ]; then
    echo "Installing requirements..."
    pip3 install -r requirements.txt
else
    echo "Starting native OpenCV camera app..."
    python3 app.py
fi
