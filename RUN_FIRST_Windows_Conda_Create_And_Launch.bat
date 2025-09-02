@echo off
setlocal
set ENV_NAME=organoid_roi_incucyte_imaging
echo === Organoid ROI Tool: Conda setup & launch (v7) ===
where conda >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
  echo [error] Conda not found on PATH. Open "Anaconda Prompt" and run this again.
  pause
  exit /b 1
)
conda env list | findstr /R /C:" %ENV_NAME% " >nul
IF %ERRORLEVEL% NEQ 0 (
    echo [info] Creating environment %ENV_NAME% from environment.yml ...
    conda env create -n %ENV_NAME% -f "%~dp0environment.yml"
) ELSE (
    echo [ok] Environment exists: %ENV_NAME%
)
echo [check] Verifying package imports (including imagecodecs & pytest)...
conda run -n %ENV_NAME% python -c "import sys;print('python',sys.version);import napari, PySide6, numpy, tifffile, skimage, pandas, imagecodecs, pytest;print('imports ok (including imagecodecs & pytest)')"
IF %ERRORLEVEL% NEQ 0 (
  echo [error] Import check failed.
  pause
  exit /b 1
)
echo [run] Launching GUI...
conda run -n %ENV_NAME% python "%~dp0gui_app.py"
pause
endlocal
