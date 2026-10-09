# Build and run every glm5 GPU kernel test against llama.cpp's ground truth (2026-10-08).
#   powershell -ExecutionPolicy Bypass -File run-glm5-tests.ps1 [-NoBuild]
# Cases come from tools\glm5_ref (real GLM-5.3-Flash weights + llama.cpp whole-tensor dumps).
param([switch]$NoBuild)
$R = $PSScriptRoot
$py = 'D:\AI\LLM\strata\.venv\Scripts\python.exe'
# the multi-token dumps only: the -ub 1 runs (full_*_ub1_*, the layer test's ground truth) hold one token per tensor
$full = (Get-ChildItem 'D:\AI\LLM\llamacpp-glm5\runs' -Directory -Filter 'full_*' | Where-Object Name -notlike '*_ub1_*' |
         Sort-Object LastWriteTime | Select-Object -Last 1).FullName
$c = Join-Path $R 'build-glm-gpu\cases'; $b = Join-Path $R 'build-glm-gpu'
New-Item -ItemType Directory -Force $c | Out-Null
if (-not $NoBuild) { cmd /c "`"$R\build-glm5-tests.bat`" 2>&1" | Select-String -Pattern ' error|LNK' }
Push-Location (Join-Path $R 'tools\glm5_ref')
$res = [ordered]@{}
& $py make_mhc_case.py "$c\mhc_l0.bin" | Out-Null
$res['mHC'] = (& "$b\glm5_mhc_test.exe" "$c\mhc_l0.bin" | Select-Object -Last 1)
$res['router'] = (& $py check_router.py $full "$b\glm5_router_test.exe" | Select-Object -Last 1)
& $py make_ffn_case.py $full "$c\ffn.bin" | Out-Null
$res['dense FFN / shared expert'] = (& "$b\glm5_ffn_test.exe" "$c\ffn.bin" | Select-Object -Last 1)
& $py make_kda_case.py $full "$c\kda_l0.bin" | Out-Null
$res['KDA'] = (& "$b\glm5_kda_test.exe" "$c\kda_l0.bin" | Select-Object -Last 1)
& $py make_mla_case.py make $full "$c\mla_l3.bin" | Out-Null
& "$b\glm5_mla_test.exe" "$c\mla_l3.bin" "$c\mla_l3_out.bin" | Out-Null
$res['MLA'] = (& $py make_mla_case.py check $full "$c\mla_l3_out.bin" | Select-Object -Last 1)
& $py kpool_case.py make $full "$c\kpool_l3.bin" | Out-Null
& "$b\glm5_kpool_test.exe" "$c\kpool_l3.bin" "$c\kpool_l3_out.bin" | Out-Null
$res['k-pool indexer'] = (& $py kpool_case.py check $full "$c\kpool_l3.bin" "$c\kpool_l3_out.bin" | Select-Object -Last 1)
Pop-Location
$res.GetEnumerator() | ForEach-Object { '{0,-28} {1}' -f $_.Key, $_.Value }
