# Build Tauri application
Write-Host "Building DOS-GCNN Tauri Application..." -ForegroundColor Cyan

# Check if yarn is installed
if (-not (Get-Command yarn -ErrorAction SilentlyContinue)) {
    Write-Host "Error: yarn is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

# Check if Rust is installed
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Rust is not installed. Please install it from https://rustup.rs/" -ForegroundColor Red
    exit 1
}

# Install dependencies if needed
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    yarn install
}

# Build Tauri
Write-Host "Building Tauri application..." -ForegroundColor Green
yarn tauri build

Write-Host "Build complete! Check src-tauri/target/release for the executable." -ForegroundColor Green
