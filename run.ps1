$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimePython = '.runtime/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) {
    $runtimePython = '.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $runtimePython)) {
    Write-Host 'Create the environment first: uv venv --python 3.13 .venv'
    exit 1
}
& $runtimePython -m streamlit run app.py
