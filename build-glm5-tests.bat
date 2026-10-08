@echo off
rem Build the glm5 GPU kernel tests with nvcc (sm_120 = RTX 5090) into build-glm-gpu\  (2026-10-08)
rem Each test links the Strata kernel file it checks.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
set R=%~dp0
if not exist "%R%build-glm-gpu" mkdir "%R%build-glm-gpu"
call :one glm5_mhc_test fused_mhc || exit /b 1
exit /b 0

:one
echo [build] %1  (src\kernels\cuda\%2.cu)
%NVCC% -O3 -std=c++17 -arch=sm_120 -I"%R%include" "%R%tests\cuda\%1.cu" "%R%src\kernels\cuda\%2.cu" -o "%R%build-glm-gpu\%1.exe" || exit /b 1
exit /b 0
