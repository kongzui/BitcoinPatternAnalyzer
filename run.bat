@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

cd "%~dp0"
echo [Python 파일 목록]
echo.

set count=0
for %%f in (*.py) do (
    set /a count+=1
    set "file[!count!]=%%f"
    echo [!count!] %%f
)

if %count%==0 (
    echo 현재 폴더에 Python 파일이 없습니다.
    pause
    exit /b
)

echo.
set /p choice="실행할 파일 번호를 선택하세요 (1-%count%): "

if %choice% leq 0 goto invalid
if %choice% gtr %count% goto invalid

python "!file[%choice%]!"
pause
exit /b

:invalid
echo 잘못된 번호입니다. 다시 실행해주세요.
pause