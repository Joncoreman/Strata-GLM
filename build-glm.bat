@echo off
setlocal
rem Phase 1: CPU-only build in this worktree; never starts the engine or GPU tests.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CM=D:\AI\LLM\strata\.venv\Scripts\cmake.exe
set NJ=D:\AI\LLM\strata\.venv\Scripts\ninja.exe
"%CM%" -G Ninja -DCMAKE_MAKE_PROGRAM=%NJ% -S D:\AI\LLM\strata-glm -B D:\AI\LLM\strata-glm\build-glm -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR=D:\AI\LLM\llamacpp-glm5 || exit /b 1
if "%~1"=="" (
  "%CM%" --build D:\AI\LLM\strata-glm\build-glm --target glm5_geometry_test glm5_reader_test glm5_header_check glm5_layout_test glm5_native_expert_test -j 12
) else (
  "%CM%" --build D:\AI\LLM\strata-glm\build-glm --target %* -j 12
)
exit /b %errorlevel%
