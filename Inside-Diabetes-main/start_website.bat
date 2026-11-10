@echo off
echo ========================================================
echo  Inside Diabetes - Starting Local Web Platform
echo ========================================================
echo.
cd /d "%~dp0Inside-Diabetes-main\website"
start http://localhost:8080/index.html
python -m http.server 8080
pause

