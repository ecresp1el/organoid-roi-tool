@echo off
set ENV_NAME=organoid_roi_incucyte_imaging
conda run -n %ENV_NAME% python "%~dp0gui_app.py"
pause
