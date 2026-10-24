@echo off
title QUANTUM NIDS - ADVERSARIAL PORT SCAN ATTACKER
cd /d "%~dp0"
cls
echo ======================================================================
echo      MALICIOUS ACTOR SIMULATION: PORT SCAN RECONNAISSANCE
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
echo [!] Launching Multi-Port Reconnaissance Sweep against %TARGET_IP%...
echo.
if exist "simulators\port_scan_attacker.py" (
    python simulators\port_scan_attacker.py --target %TARGET_IP%
) else (
    python port_scan_attacker.py --target %TARGET_IP%
)
echo.
pause
