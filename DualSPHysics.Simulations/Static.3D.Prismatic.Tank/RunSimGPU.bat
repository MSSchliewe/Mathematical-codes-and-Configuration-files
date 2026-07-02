@echo off
rem #########################################################
rem Batch script to run DualSPHysics using GPU
rem #########################################################

rem 1) Navigate to the folder containing the DualSPHysics executables.
set "DSPH_PATH=X:\...\DualSPHysics\bin\windows"

rem 2) Case directory and base name (without extension)
set "CASEDIR=X:\...\Static.3D.Prismatic.Tank"
set "CASENAME=Tank6_Def"

rem 3) Output directory for the results
set "OUTDIR=%CASEDIR%\Results"

rem 4) Creates the output directory if it does not exist.
if not exist "%OUTDIR%" mkdir "%OUTDIR%"

rem 5) Change to the case directory.
cd /d "%CASEDIR%"

rem 6) Runs the simulation using the GPU and saves the summary. (-svres)
"%DSPH_PATH%\DualSPHysics5.4_win64.exe" "%CASEDIR%\%CASENAME%" "%OUTDIR%" -svres -gpu

rem 7) Checks the exit code
if errorlevel 1 (
  echo GPU simulation error!
) else (
  echo GPU simulation completed successfully!
)

rem 8) Pause to view log messages
pause