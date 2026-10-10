@echo off
rem Build this repository's Strata engine with the GLM-5.3-Flash port into build-engine\ (2026-10-10).
rem   build-glm5.bat            - for the GPU in this PC (compute capability from nvidia-smi)
rem   build-glm5.bat 86         - for another card (sm_86 = RTX 30; 120 = RTX 50; 89 = RTX 40; 75 = RTX 20)
rem Needs: Visual Studio 2022 Build Tools (C++), the CUDA toolkit (CUDA_PATH), cmake and ninja (START-HERE.bat's .venv
rem has them, or install them on PATH).  ggml comes from third_party\ggml's pinned commit unless STRATA_GGML_DIR is set.
setlocal
cd /d "%~dp0"
set ARCH=%1
if "%ARCH%"=="" for /f "tokens=1,2 delims=." %%a in ('nvidia-smi --query-gpu=compute_cap --format=csv^,noheader 2^>nul') do if not defined ARCH set ARCH=%%a%%b
if "%ARCH%"=="" set ARCH=120
set BDIR=%~dp0build-engine
if not "%1"=="" if not "%1"=="120" set BDIR=%~dp0build-sm%ARCH%
set CM=cmake
set NJ=ninja
if exist "%~dp0.venv\Scripts\cmake.exe" set CM=%~dp0.venv\Scripts\cmake.exe
if exist "%~dp0.venv\Scripts\ninja.exe" set NJ=%~dp0.venv\Scripts\ninja.exe
if exist "..\strata\.venv\Scripts\cmake.exe" if not exist "%~dp0.venv\Scripts\cmake.exe" set CM=%~dp0..\strata\.venv\Scripts\cmake.exe
if exist "..\strata\.venv\Scripts\ninja.exe" if not exist "%~dp0.venv\Scripts\ninja.exe" set NJ=%~dp0..\strata\.venv\Scripts\ninja.exe
set VSW=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
set VCVARS=
if exist "%VSW%" for /f "usebackq delims=" %%i in (`"%VSW%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat
if "%VCVARS%"=="" set VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" echo vcvars64.bat not found: install the Visual Studio 2022 Build Tools (Desktop development with C++) && exit /b 1
call "%VCVARS%" >nul
if "%CUDA_PATH%"=="" echo CUDA_PATH is not set: install the CUDA toolkit && exit /b 1
set GGML=
if not "%STRATA_GGML_DIR%"=="" set GGML=-DSTRATA_GGML_DIR=%STRATA_GGML_DIR%
echo building sm_%ARCH% into %BDIR% (cmake %CM%, nvcc %CUDA_PATH%\bin\nvcc.exe)
"%CM%" -G Ninja -DCMAKE_MAKE_PROGRAM=%NJ% -S "%~dp0." -B "%BDIR%" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared "-DCMAKE_CUDA_ARCHITECTURES=%ARCH%" "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%\bin\nvcc.exe" %GGML% || exit /b 1
"%CM%" --build "%BDIR%" --target strata -j 16 || exit /b 1
echo BUILD_DONE %BDIR%\strata.exe
