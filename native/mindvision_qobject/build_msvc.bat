@echo off
setlocal

echo Setting up MSVC environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo Building with MSVC...

echo Cleaning release directory...
if exist release rmdir /s /q release
mkdir release

echo.
echo Running qmake...
set QMAKE_MSC_VER=1944
"C:\Qt\6.10.1\msvc2022_64\bin\qmake.exe" -r mindvision_qobject.pro
if errorlevel 1 (
    echo qmake failed.
    goto error
)

echo.
echo Running nmake...
nmake
if errorlevel 1 (
    echo nmake failed.
    goto error
)

echo.
echo Build successful!
endlocal
goto :eof

:error
echo Build failed.
endlocal
exit /b 1