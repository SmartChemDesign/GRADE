# Full build script for DOS-GCNN Desktop Application
# Builds both Python sidecar and Tauri app

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "DOS-GCNN Full Build" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

# Step 1: Build Python sidecar
Write-Host "`n[1/3] Building Python sidecar..." -ForegroundColor Yellow
Set-Location "$projectRoot\python-model"

# Check if sidecar already exists
$sidecarPath = "dist\dos-gcnn-sidecar.exe"
if (Test-Path $sidecarPath) {
    $rebuild = Read-Host "Sidecar already exists. Rebuild? (y/N)"
    if ($rebuild -ne "y" -and $rebuild -ne "Y") {
        Write-Host "Skipping sidecar build." -ForegroundColor Yellow
    } else {
        & ".\build_sidecar.ps1"
    }
} else {
    & ".\build_sidecar.ps1"
}

if (-not (Test-Path $sidecarPath)) {
    Write-Host "Error: Sidecar not found after build!" -ForegroundColor Red
    exit 1
}

# Step 2: Install JS dependencies
Write-Host "`n[2/3] Installing dependencies..." -ForegroundColor Yellow
Set-Location $projectRoot
if (-not (Test-Path "node_modules")) {
    yarn install
}

# Step 3: Build Tauri app
Write-Host "`n[3/3] Building Tauri application..." -ForegroundColor Yellow
yarn tauri build

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "BUILD COMPLETE!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host "`nOutput files:" -ForegroundColor Cyan
Write-Host "- MSI: src-tauri\target\release\bundle\msi\DOS-GCNN_1.0.0_x64_en-US.msi" -ForegroundColor White
Write-Host "- NSIS: src-tauri\target\release\bundle\nsis\DOS-GCNN_1.0.0_x64-setup.exe" -ForegroundColor White
