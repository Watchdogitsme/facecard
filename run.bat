@echo off
title FaceCard - AI Biometric Skin Rating & Celebrity Matcher
cd /d "%~dp0"

echo =====================================================================
echo    FaceCard: AI Biometric Mirror & Celebrity Matcher
echo =====================================================================
echo.
echo Choose execution mode:
echo 1. Launch Native OpenCV Desktop App
echo 2. Test Before Launching - Check File Integrity ^& System Readiness
echo 3. Install / Verify Requirements
echo.
set /p mode="Enter choice (1/2/3): "

if "%mode%"=="" set mode=1

if "%mode%"=="1" (
    echo Starting native OpenCV camera app...
    python app.py
) else if "%mode%"=="2" (
    echo Checking file integrity and testing system readiness...
    python test_system.py
    pause
) else if "%mode%"=="3" (
    echo Installing requirements...
    pip install -r requirements.txt
    pause
) else if "%mode%"=="6" (
    echo Installing requirements...
    pip install -r requirements.txt
    pause
) else (
    echo Starting native OpenCV camera app...
    python app.py
)

pause
