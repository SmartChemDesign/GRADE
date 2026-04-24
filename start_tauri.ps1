# Start Tauri development server
Write-Host "Starting DOS-GCNN Tauri Application..." -ForegroundColor Cyan

# Check if yarn is installed
if (-not (Get-Command yarn -ErrorAction SilentlyContinue)) {
    Write-Host "Error: yarn is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

# Check if node_modules exists
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    yarn install
}

# Start Tauri dev
Write-Host "Starting Tauri dev server..." -ForegroundColor Green
yarn tauri dev
