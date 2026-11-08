@echo off
cd /d "%~dp0"
python app3d.py
if errorlevel 1 pause
