param([switch]$SkipTests)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw '请按 README 创建 .venv 并安装 requirements-dev.txt。'
}
if (-not (Test-Path -LiteralPath 'models/best.pt')) { throw '缺少本地模型 models/best.pt。' }
$env:YOLO_AUTOINSTALL = 'false'
if (-not $SkipTests) {
    & $taskPython -m pytest -q
    if ($LASTEXITCODE -ne 0) { throw '测试未通过，已停止打包。' }
}
& $taskPython 'tools/write_build_manifest.py' before
if ($LASTEXITCODE -ne 0) { throw '无法记录构建前的源码版本。' }
& $taskPython -m PyInstaller --noconfirm Cloudlane.spec
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller 构建失败。' }
$taskOutput = Join-Path $PSScriptRoot 'dist/云瞰街巷'
Copy-Item -LiteralPath 'README.md' -Destination (Join-Path $taskOutput '使用说明.md')
Copy-Item -LiteralPath 'SOURCE_PROVENANCE.md' -Destination (Join-Path $taskOutput '源码与模型来源.md')
& $taskPython 'tools/write_build_manifest.py' after
if ($LASTEXITCODE -ne 0) { throw '构建产物与源码版本记录未通过核对。' }
Write-Host "构建完成：$taskOutput"
Write-Host '分享时请复制整个云瞰街巷文件夹，保留 EXE 和 _internal。'
