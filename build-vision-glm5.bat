@echo off
rem strata-vision (the image encoder) for GLM-5.3-Flash, CPU only (the GLM engine keeps the GPU full). 2026-10-11. ASCII only.
rem Needs a llama.cpp checkout that knows the glm5v projector (b11461 or newer): the first argument, else LLAMA_DIR,
rem else ..\llamacpp-glm5 beside this folder. Built for this PC (portable off). Then add a "vision" block to the
rem server settings: exe build-vision-glm5\bin\strata-vision.exe, the model's mmproj, and GLM's image markers
rem (image_start begin_of_image, image_pad image, each in the model's special-token brackets; see README.md).
setlocal
cd /d "%~dp0"
if not "%~1"=="" set LLAMA_DIR=%~1
if "%LLAMA_DIR%"=="" set LLAMA_DIR=%~dp0..\llamacpp-glm5
if not exist "%LLAMA_DIR%\CMakeLists.txt" (echo llama.cpp not found at %LLAMA_DIR% - pass its folder as the first argument & exit /b 1)
set BDIR=%~dp0build-vision-glm5
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
call "%VCVARS%" >nul
"%CM%" -G Ninja -DCMAKE_MAKE_PROGRAM=%NJ% -S "%~dp0tools\vision" -B "%BDIR%" -DCMAKE_BUILD_TYPE=Release "-DLLAMA_DIR=%LLAMA_DIR%" -DSTRATA_PORTABLE=OFF -DSTRATA_VISION_CUDA=OFF || exit /b 1
"%CM%" --build "%BDIR%" --target strata-vision -j 16 || exit /b 1
echo BUILD_DONE %BDIR%\bin\strata-vision.exe
