$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 x64 first.' }
}
$python = '.venv\Scripts\python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
& $python -m pip install -e '.[dev,faces]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed; build cancelled' }
& $python -m PyInstaller --noconfirm --clean --windowed --onedir --name ScreenThai `
    --collect-all rapidocr_onnxruntime --collect-all onnxruntime `
    --collect-all mediapipe `
    --collect-all keyring --hidden-import keyring.backends.Windows `
    --collect-submodules win32ctypes --copy-metadata screen-thai scripts/launch.py
if ($LASTEXITCODE -ne 0) { throw 'Packaging failed' }
Copy-Item README.md 'dist\ScreenThai\README.md'
Copy-Item docs 'dist\ScreenThai\docs' -Recurse -Force
Compress-Archive -Path 'dist\ScreenThai' -DestinationPath 'dist\ScreenThai-Windows-x64.zip' -Force
Write-Host 'Built: dist\ScreenThai-Windows-x64.zip' -ForegroundColor Green
