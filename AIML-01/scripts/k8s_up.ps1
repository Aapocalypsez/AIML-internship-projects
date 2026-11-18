param([string]$Profile = 'fraud-demo', [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$minikubeCommand = Get-Command minikube -ErrorAction SilentlyContinue
$minikubeExe = if ($minikubeCommand) { $minikubeCommand.Source } else { Join-Path $projectRoot 'data/tools/minikube.exe' }
if (-not (Test-Path -LiteralPath $minikubeExe)) {
    throw 'Install Minikube from https://minikube.sigs.k8s.io/docs/start/ or place it at data/tools/minikube.exe.'
}
if (-not (Test-Path -LiteralPath 'data/processed/features/model.ubj')) {
    throw 'Run Phase 2 training first; the evaluated model is required for deployment.'
}
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed (exit $LASTEXITCODE)" }
}
Invoke-Checked $minikubeExe @('start', '-p', $Profile, '--driver=docker', '--cpus=2', '--memory=4096')
if (-not $SkipBuild) {
    Invoke-Checked 'docker' @('build', '--target', 'runtime', '-f', 'docker/app.Dockerfile', '-t', 'fraud-platform:local', '.')
    Invoke-Checked 'docker' @('build', '--target', 'graph', '-f', 'docker/app.Dockerfile', '-t', 'fraud-platform-graph:local', '.')
}
foreach ($image in @('fraud-platform:local', 'fraud-platform-graph:local', 'redis:7.2-alpine', 'postgres:16-alpine', 'confluentinc/cp-kafka:7.5.3', 'confluentinc/cp-zookeeper:7.5.3')) {
    Invoke-Checked $minikubeExe @('-p', $Profile, 'image', 'load', $image)
}
Invoke-Checked 'docker' @('exec', $Profile, 'mkdir', '-p', '/data/fraud/raw', '/data/fraud/processed', '/data/fraud/feature_repo')
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
Invoke-Checked $projectPython @('-m', 'scripts.seed_minikube', '--profile', $Profile)
Invoke-Checked 'kubectl' @('--context', $Profile, 'apply', '-f', 'k8s/platform.yaml')
Invoke-Checked 'kubectl' @('--context', $Profile, '-n', 'fraud', 'wait', '--for=condition=complete', 'job/bootstrap', '--timeout=600s')
foreach ($name in @('api','worker','sink','dashboard')) {
    Invoke-Checked 'kubectl' @('--context', $Profile, '-n', 'fraud', 'rollout', 'status', "deployment/$name", '--timeout=300s')
}
Write-Host 'Ready. Open in two terminals:'
Write-Host "kubectl --context $Profile -n fraud port-forward service/api 8001:8000"
Write-Host "kubectl --context $Profile -n fraud port-forward service/dashboard 8502:8501"
