@echo off
rem Build the multinode worktree engine (sm_75/86/120) into build-mn\
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set CM=D:\AI\LLM\strata\.venv\Scripts\cmake.exe
set NJ=D:\AI\LLM\strata\.venv\Scripts\ninja.exe
"%CM%" -G Ninja -DCMAKE_MAKE_PROGRAM=%NJ% -S D:\AI\LLM\strata-mn -B D:\AI\LLM\strata-mn\build-mn -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared "-DCMAKE_CUDA_ARCHITECTURES=75;86;120" "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -DSTRATA_GGML_DIR=D:\AI\LLM\strata\third_party\llama.cpp || exit /b 1
"%CM%" --build D:\AI\LLM\strata-mn\build-mn --target strata -j 12 && exit /b 0
echo (build stopped - trying once more)
"%CM%" --build D:\AI\LLM\strata-mn\build-mn --target strata -j 12 || exit /b 1
