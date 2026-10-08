@echo off
rem CPU-only test build of THIS worktree (strata-glm-gpu) into build-glm-cpu\ ; never starts the engine.
rem   build-glm-cpu.bat [targets...]   (default: the glm5 CPU tests)
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CM=D:\AI\LLM\strata\.venv\Scripts\cmake.exe
set NJ=D:\AI\LLM\strata\.venv\Scripts\ninja.exe
"%CM%" -G Ninja -DCMAKE_MAKE_PROGRAM=%NJ% -S D:\AI\LLM\strata-glm-gpu -B D:\AI\LLM\strata-glm-gpu\build-glm-cpu -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR=D:\AI\LLM\strata\third_party\llama.cpp || exit /b 1
if "%~1"=="" (
  "%CM%" --build D:\AI\LLM\strata-glm-gpu\build-glm-cpu --target glm5_expert_layout_test glm5_native_expert_test glm5_layout_test glm5_reader_test glm5_geometry_test -j 16
) else (
  "%CM%" --build D:\AI\LLM\strata-glm-gpu\build-glm-cpu --target %* -j 16
)
exit /b %errorlevel%
