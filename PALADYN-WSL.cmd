@echo off
setlocal
title PALADYN / V-Core

wsl.exe -- bash -lc "cd ~/PALADYN && source .venv/bin/activate && exec paladyn-ui"

if errorlevel 1 (
    echo.
    echo PALADYN nie wystartowal. Sprawdz WINDOWS.md oraz sciezke ~/PALADYN w WSL.
    pause
)
