@echo off
title Whisper GPU Dictation
echo Starting Whisper Dictation (GPU)...
echo.
python "%~dp0dictate.py" %*
pause
