@echo off
rem Build this worktree's Strata engine (sm_120 only, for the RTX 5090) into build-engine\  (2026-10-08, GLM port)
rem Same recipe as build-mn.bat (pinned ggml from D:\AI\LLM\strata\third_party\llama.cpp).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CM=D:\AI\LLM\strata\.venv\Scripts\cmake.exe
set NJ=D:\AI\LLM\strata\.venv\Scripts\ninja.exe
"%CM%" -G Ninja -DCMAKE_MAKE_PROGRAM=%NJ% -S D:\AI\LLM\strata-glm-gpu -B D:\AI\LLM\strata-glm-gpu\build-engine -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared "-DCMAKE_CUDA_ARCHITECTURES=120" "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -DSTRATA_GGML_DIR=D:\AI\LLM\strata\third_party\llama.cpp || exit /b 1
"%CM%" --build D:\AI\LLM\strata-glm-gpu\build-engine --target strata -j 16 || exit /b 1
echo BUILD_ENGINE_DONE
