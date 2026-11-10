@echo off
echo ========================================================
echo  Inside Diabetes — Starting Local Web Platform
echo ========================================================
echo.
cd /d "%~dp0website"
start http://localhost:8080/index.html
if exist "C:\Users\HARI HARA SUDHAN V\AppData\Local\Programs\Python\Python311\python.exe" (
    "C:\Users\HARI HARA SUDHAN V\AppData\Local\Programs\Python\Python311\python.exe" -m http.server 8080
) else (
    python -m http.server 8080
)
pause

