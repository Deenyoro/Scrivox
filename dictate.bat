@echo off
title Scrivox Dictation
echo Starting Scrivox Dictation (GPU)...
echo.
python "%~dp0dictate.py" %*
pause
