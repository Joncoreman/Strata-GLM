# Start this worktree's strata.exe on GLM-5.3-Flash UD-IQ4_XS (2026-10-08, the GLM port).  Stderr/stdout to files.
#   powershell -ExecutionPolicy Bypass -File run-glm5-engine.ps1 [-Stdin <file with engine commands>] [-Cache 1000]
# STRATA_GLM5_GO=1 lets the start-up past the "not wired yet" stop; STRATA_CRASH_TRACE=1 prints a stack on a crash.
#   -Tokens 785,6722,... : a one-shot generate (no --serve) of -MaxNew tokens, the prompt as verify windows
param([string]$Stdin = '', [int]$Cache = 1000, [int]$ResidentGiB = 60, [string]$Out = '', [string]$Tokens = '',
      [int]$MaxNew = 4, [int]$Spec = 4, [string]$Extra = '', [string]$Profile = 'expert-profile-uniform.bin')
$R = $PSScriptRoot
if (-not $Out) { $Out = Join-Path $R 'build-engine\glm5_run' }
$env:STRATA_GLM5_GO = '1'; $env:STRATA_CRASH_TRACE = '1'; $env:STRATA_TRACE = '1'
$pack = 'D:/AI/LLM/Strata-data/packs/glm53_iq4xs'
$a = "--serve --pack $pack --native D:/AI/LLM/GLM-5.3-Flash-GGUF/unsloth/GLM-5.3-Flash-GGUF/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf " +
     "--prefill auto --spec 2 --max-context 4096 --pcie-frac 0 --short-read 1000000000 --prompt-cache 0 --adapt-swaps 0 " +
     "--mmap-experts --resident-budget-gib $ResidentGiB --expert-profile $pack/$Profile --expert-cache $Cache"
if ($Tokens) { $a = $a.Replace('--serve ', "--tokens $Tokens --max-new $MaxNew ").Replace('--spec 2', "--spec $Spec") }
if ($Extra) { $a = "$a $Extra" }
if (-not $Stdin) { $Stdin = "$Out.stdin"; Set-Content -Path $Stdin -Value '' -NoNewline }
$p = Start-Process -FilePath "$R\build-engine\strata.exe" -ArgumentList $a -NoNewWindow -Wait -PassThru `
     -RedirectStandardError "$Out.err" -RedirectStandardOutput "$Out.out" -RedirectStandardInput $Stdin -WorkingDirectory $R
'exit 0x{0:X8} ({0})' -f $p.ExitCode
Get-Content "$Out.err" -Tail 30
