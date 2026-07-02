@echo off
rem #########################################################
rem Batch script to generate the DualSPHysics case Gencase.exe
rem #########################################################

rem 1. Set the path to the DualSPHysics executable.
set "DSPS_PATH=X:\...\DualSPHysics\bin\windows"

rem 2. Define the input XML file. (Slump2_Def.xml)
set "CASE_XML=X:\...\...\Collapse.3D.NNPhase"

rem 3. Navigate to the executable's folder (optional, but useful for libraries)
pushd "%DSPS_PATH%"

rem 4. Call Gencase, passing the XML file.
rem    - use "-i" ou "-f" depending on the version of your Gencase.exe
"%DSPS_PATH%\GenCase_win64.exe" "%CASE_XML%"

rem 5. Checks the exit code and displays a message.
if errorlevel 1 (
    echo.
    echo Error generating the case. Check the XML file and the logs.
) else (
    echo.
    echo Case successfully generated!
)

rem 6. Waiting for a key press to close the window
echo.
pause

rem 7. Returns to the previous directory
popd