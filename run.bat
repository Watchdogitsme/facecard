@echo off
title AI Biometric Mirror - OpenCV Skin Rating & Celebrity Matcher
cd /d "%~dp0"

echo =====================================================================
echo    AI Biometric Mirror & Celebrity Matcher (OpenCV)
echo =====================================================================
echo.
echo Choose execution mode:
echo [1] Launch Native OpenCV Desktop App (Live Laptop Webcam Window)
echo [2] Launch Browser-Based Web Mirror (http://localhost:5000)
echo [3] Run Demo Mode (Synthesized Camera Test Feed)
echo [4] Install / Verify Requirements (pip install -r requirements.txt)
echo [5] Fix / Reinstall OpenCV (Fixes 'module cv2 has no attribute CascadeClassifier')
echo [6] Push to GitHub (git push -u origin main)
echo.
set /p mode="Enter choice (1/2/3/4/5/6) [Default 1]: "

if "%mode%"=="" set mode=1

if "%mode%"=="1" (
    echo Starting native OpenCV camera app...
    python app.py
) else if "%mode%"=="2" (
    echo Starting browser web mirror on http://localhost:5000 ...
    start http://localhost:5000
    python web_app.py
) else if "%mode%"=="3" (
    echo Starting demo feed...
    python app.py --demo
) else if "%mode%"=="4" (
    echo Installing requirements...
    pip install -r requirements.txt
    pause
) else if "%mode%"=="5" (
    echo Reinstalling clean opencv-python...
    pip uninstall cv2 opencv-python opencv-python-headless opencv-contrib-python -y
    pip install opencv-python
    echo Done! You can now run option 1.
    pause
) else if "%mode%"=="6" (
    echo Pushing commits to https://github.com/Watchdogitsme/facecard.git ...
    git push -u origin main
    pause
) else (
    echo Invalid choice. Starting native OpenCV camera app...
    python app.py
)

pause
