@echo off
rem #########################################################
rem Batch script to convert DualSPHysics binaries to .vtk
rem #########################################################

rem 1) Full path to PartVTK
set "PARTVTK=X:\...\DualSPHysics\bin\windows\PartVTK_win64.exe"

rem 2) Directory containing the .bi4 files (binaries)
set "DATAIN=X:\...\Results\data"

rem 3) Output directory for .vtk files
set "VTKOUT=X:\...\Results\vtk"

rem 4) Creates the output folder if it does not exist.
if not exist "%VTKOUT%" mkdir "%VTKOUT%"

rem 5) Performs the conversion, including fluid and fixed particles.
"%PARTVTK%" -dirin "%DATAIN%" -savevtk "%VTKOUT%\PartAll.vtk" -onlytype:+fluid,+fixed

rem 6) Checks exit code and notifies
if errorlevel 1 (
  echo.
  echo Error converting .bi4 to .vtk!
) else (
  echo.
  echo Conversion successfully completed in "%VTKOUT%"!
)

echo.
pause