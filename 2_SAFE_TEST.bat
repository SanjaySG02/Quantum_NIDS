@echo off
title QUANTUM NIDS - BENIGN CLIENT SIMULATOR
cd /d "%~dp0"
cls
echo ======================================================================
echo      LEGITIMATE VISITOR SIMULATION: BENIGN SINGLE SERVICE
echo ======================================================================
echo.
set "DEFAULT_IP=127.0.0.1"
if exist target_ip.txt set /p DEFAULT_IP=<target_ip.txt

echo Saved Target IP: %DEFAULT_IP%
set "TARGET_IP="
set /p TARGET_IP="Enter Target IP [Press ENTER to use %DEFAULT_IP%]: "
if "%TARGET_IP%"=="" set "TARGET_IP=%DEFAULT_IP%"
echo %TARGET_IP%>target_ip.txt

echo.
echo [*] Sending Legitimate Service Request to %TARGET_IP%...
echo.
python benign_client.py --target %TARGET_IP%
echo.
pause
