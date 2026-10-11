@echo off
title FaceCard - AI Biometric Skin Rating & Celebrity Matcher
cd /d "%~dp0"

echo =====================================================================
echo    FaceCard: AI Biometric Mirror & Celebrity Matcher (OpenCV)
echo =====================================================================
echo.
echo Choose execution mode:
echo [1] Launch Native OpenCV Desktop App (Live Laptop Webcam Window)
echo [2] Run Demo Mode (Synthesized Camera Test Feed)
echo [3] Install / Verify Requirements (pip install -r requirements.txt)
echo [4] Fix / Reinstall OpenCV (Fixes 'module cv2 has no attribute CascadeClassifier')
echo.
set /p mode="Enter choice (1/2/3/4) [Default 1]: "

if "%mode%"=="" set mode=1

if "%mode%"=="1" (
    echo Starting native OpenCV camera app...
    python app.py
) else if "%mode%"=="2" (
    echo Starting demo feed...
    python app.py --demo
) else if "%mode%"=="3" (
    echo Installing requirements...
    pip install -r requirements.txt
    pause
) else if "%mode%"=="4" (
    echo Reinstalling clean opencv-python...
    pip uninstall cv2 opencv-python opencv-python-headless opencv-contrib-python -y
    pip install opencv-python
    echo Done! You can now run option 1.
    pause
) else (
    echo Invalid choice. Starting native OpenCV camera app...
    python app.py
)

pause
