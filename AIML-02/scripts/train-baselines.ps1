$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv-ml\Scripts\python.exe'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
# Use new output directories for a new experiment. --resume continues the existing run.
& $pythonExe -m ml.train_classifier --manifest data/splits/classifier_development.csv --epochs 5 --batch-size 4 --output models/classifier-development --resume
if ($LASTEXITCODE -ne 0) { throw 'Classifier training failed.' }
& $pythonExe -m ml.train_segmentation --manifest data/splits/spleen.json --epochs 50 --output models/spleen-development --resume
if ($LASTEXITCODE -ne 0) { throw 'Segmentation training failed.' }
Write-Host 'Training finished. Run held-out evaluation only after validation-based model selection is final.'
