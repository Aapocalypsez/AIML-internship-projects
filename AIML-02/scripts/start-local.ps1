$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv-ml\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'The project ML environment is missing.' }
& $pythonExe -m tools.create_local_account
if ($LASTEXITCODE -ne 0) { throw 'Account setup failed.' }
$env:REVIEW_USERS_FILE = Join-Path $projectRoot 'data\auth\users.json'
$classifier = Join-Path $projectRoot 'models\classifier-development\best.pt'
$segmenter = Join-Path $projectRoot 'models\spleen-development\best.pt'
if (Test-Path -LiteralPath 'models\active\classifier.pt') { $classifier=Join-Path $projectRoot 'models\active\classifier.pt' }
if (Test-Path -LiteralPath 'models\active\spleen.pt') { $segmenter=Join-Path $projectRoot 'models\active\spleen.pt' }
if (Test-Path -LiteralPath $classifier) { $env:CLASSIFIER_CHECKPOINT = $classifier }
if (Test-Path -LiteralPath $segmenter) { $env:SEGMENTATION_CHECKPOINT = $segmenter }
Write-Host 'Starting the local API. In a second terminal: cd frontend; npm.cmd run dev'
Write-Host 'Open http://127.0.0.1:5173. Development checkpoints are not clinically validated.'
& $pythonExe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --no-access-log
