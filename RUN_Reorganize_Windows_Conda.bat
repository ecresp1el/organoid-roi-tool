@echo off
setlocal
set ENV_NAME=organoid_roi_incucyte_imaging

echo === Reorganize Raw TIFFs (v2) ===
set /p RAW=Enter path to your raw_images folder: 
set /p OUT=Enter desired output project folder: 
set /p MINCOL=Only include wells with column >= (default 1; e.g., 4): 
set /p ROWS=Only include row letters (default ABCDEFGH): 
if "%MINCOL%"=="" set MINCOL=1
if "%ROWS%"=="" set ROWS=ABCDEFGH

echo [run] rows=%ROWS%  min_col>=%MINCOL%
conda run -n %ENV_NAME% python "%~dp0reorganize_v2.py" --raw "%RAW%" --out "%OUT%" --min_col %MINCOL% --rows %ROWS%

echo Done.
pause
endlocal