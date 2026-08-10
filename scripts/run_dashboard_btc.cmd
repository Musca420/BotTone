@echo off
cd /d "%~dp0.."
".venv\Scripts\adaptive-bot.exe" dashboard --report data\reports\bitunix_paper.json --host 127.0.0.1 --port 8080 --research-report data\reports\research.json 1>>data\logs\dashboard-btc.out.log 2>>data\logs\dashboard-btc.err.log
