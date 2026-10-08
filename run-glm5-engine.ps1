# Start this worktree's strata.exe on GLM-5.3-Flash UD-IQ4_XS (2026-10-08, the GLM port).  Stderr/stdout to files.
#   powershell -ExecutionPolicy Bypass -File run-glm5-engine.ps1 [-Stdin <file with engine commands>] [-Cache 1000]
# STRATA_GLM5_GO=1 lets the start-up past the "not wired yet" stop; STRATA_CRASH_TRACE=1 prints a stack on a crash.
param([string]$Stdin = '', [int]$Cache = 1000, [int]$ResidentGiB = 60, [string]$Out = '')
$R = $PSScriptRoot
if (-not $Out) { $Out = Join-Path $R 'build-engine\glm5_run' }
$env:STRATA_GLM5_GO = '1'; $env:STRATA_CRASH_TRACE = '1'; $env:STRATA_TRACE = '1'
$pack = 'D:/AI/LLM/Strata-data/packs/glm53_iq4xs'
$a = "--serve --pack $pack --native D:/AI/LLM/GLM-5.3-Flash-GGUF/unsloth/GLM-5.3-Flash-GGUF/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf " +
     "--prefill auto --spec 2 --max-context 4096 --pcie-frac 0 --short-read 1000000000 --prompt-cache 0 --adapt-swaps 0 " +
     "--mmap-experts --resident-budget-gib $ResidentGiB --expert-profile $pack/expert-profile-uniform.bin --expert-cache $Cache"
if (-not $Stdin) { $Stdin = "$Out.stdin"; Set-Content -Path $Stdin -Value '' -NoNewline }
$p = Start-Process -FilePath "$R\build-engine\strata.exe" -ArgumentList $a -NoNewWindow -Wait -PassThru `
     -RedirectStandardError "$Out.err" -RedirectStandardOutput "$Out.out" -RedirectStandardInput $Stdin -WorkingDirectory $R
'exit 0x{0:X8} ({0})' -f $p.ExitCode
Get-Content "$Out.err" -Tail 30
