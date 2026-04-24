# Build the Python sidecar executable (dos-gcnn-sidecar.exe) with PyInstaller.
#
# If Miniconda / Anaconda is not installed, this script silently installs
# Miniconda per-user into %USERPROFILE%\Miniconda3 and uses it for the build.
# The rest of the build is unchanged: create the `dos_gcnn` conda env from
# environment.yml, install PyInstaller, and freeze the sidecar.

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building DOS-GCNN Sidecar Executable" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$scriptRoot   = $PSScriptRoot
$projectRoot  = Split-Path -Parent $scriptRoot
$envFile      = Join-Path $projectRoot "environment.yml"
$envName      = "dos_gcnn"
$minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
$minicondaDir = Join-Path $env:USERPROFILE "Miniconda3"

function Find-CondaExe {
    # 1. Look for the actual conda.exe on PATH. Skip PowerShell functions /
    #    aliases installed by conda-hook.ps1 — their .Source is empty, which
    #    was what broke the previous implementation.
    $cmd = Get-Command conda.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }

    # 2. If a `conda` function wrapper is loaded from a shell hook, ask it for
    #    the install root and derive conda.exe from there.
    if (Get-Command conda -ErrorAction SilentlyContinue) {
        try {
            $base = (& conda info --base 2>$null | Select-Object -Last 1)
            if ($base) {
                $base = $base.Trim()
                $exe = Join-Path $base "Scripts\conda.exe"
                if (Test-Path $exe) { return $exe }
            }
        } catch { }
    }

    # 3. Known default install locations.
    $candidates = @(
        (Join-Path $minicondaDir "Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "Anaconda3\Scripts\conda.exe"),
        "C:\ProgramData\Miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) { return $p }
    }
    return $null
}

function Install-Miniconda {
    Write-Host "`nMiniconda not found. Installing it silently into $minicondaDir ..." -ForegroundColor Yellow
    $installer = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"

    Write-Host "Downloading installer from $minicondaUrl" -ForegroundColor DarkGray
    # Prefer faster BITS transfer; fall back to Invoke-WebRequest if it is not available.
    try {
        Start-BitsTransfer -Source $minicondaUrl -Destination $installer -ErrorAction Stop
    } catch {
        Write-Host "BITS transfer failed, falling back to Invoke-WebRequest..." -ForegroundColor DarkGray
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $minicondaUrl -OutFile $installer -UseBasicParsing
    }

    Write-Host "Running silent installer (this takes a couple of minutes)..." -ForegroundColor DarkGray
    # /InstallationType=JustMe — per-user install, no admin rights needed.
    # /RegisterPython=0       — do not touch any existing Python registrations.
    # /S                      — silent mode.
    # /D=...                  — must be the last argument and unquoted (NSIS convention).
    $proc = Start-Process -FilePath $installer `
        -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/S", "/D=$minicondaDir" `
        -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "Miniconda installer exited with code $($proc.ExitCode)."
    }
    Remove-Item $installer -ErrorAction SilentlyContinue

    $condaExe = Join-Path $minicondaDir "Scripts\conda.exe"
    if (-not (Test-Path $condaExe)) {
        throw "Miniconda install reported success but conda.exe was not found at $condaExe."
    }
    Write-Host "Miniconda installed successfully." -ForegroundColor Green
    return $condaExe
}

function Initialize-CondaShell {
    param([Parameter(Mandatory)][string]$CondaExe)

    if ([string]::IsNullOrWhiteSpace($CondaExe) -or -not (Test-Path $CondaExe)) {
        throw "Initialize-CondaShell: invalid conda path '$CondaExe'."
    }

    # Load conda into the current PowerShell session so that `conda activate`
    # works without touching the user's global PowerShell profile.
    $condaRoot = Split-Path -Parent (Split-Path -Parent $CondaExe)
    $hook = Join-Path $condaRoot "shell\condabin\conda-hook.ps1"
    if (Test-Path $hook) {
        & $hook | Out-Null
    } else {
        # Fallback: expose conda.exe directly on PATH for this session.
        $env:PATH = "$condaRoot;$(Join-Path $condaRoot 'Scripts');$(Join-Path $condaRoot 'Library\bin');$env:PATH"
    }
}

# --- Ensure conda is available ------------------------------------------------

$condaExe = Find-CondaExe
if ([string]::IsNullOrWhiteSpace($condaExe) -or -not (Test-Path $condaExe)) {
    $condaExe = Install-Miniconda
}
Initialize-CondaShell -CondaExe $condaExe
Write-Host "Using conda at: $condaExe" -ForegroundColor DarkGray

# --- Ensure the dos_gcnn environment exists -----------------------------------

Write-Host "`nChecking for conda environment '$envName'..." -ForegroundColor Yellow
# `conda env list` returns lines like "dos_gcnn    C:\Users\...\envs\dos_gcnn".
# Using `-notmatch` against the array yields all non-matching lines (truthy),
# so test per-line with Where-Object instead.
$envList   = & $condaExe env list
$envExists = @($envList | Where-Object { $_ -match "^\s*$([regex]::Escape($envName))\s" }).Count -gt 0

if (-not $envExists) {
    if (-not (Test-Path $envFile)) {
        throw "environment.yml not found at $envFile"
    }
    Write-Host "Creating environment '$envName' from $envFile (this can take several minutes)..." -ForegroundColor Yellow
    & $condaExe env create -f $envFile
    if ($LASTEXITCODE -ne 0) { throw "Failed to create conda environment '$envName'." }
} else {
    Write-Host "Environment '$envName' already exists." -ForegroundColor DarkGray
}

Write-Host "`nActivating conda environment '$envName'..." -ForegroundColor Yellow
conda activate $envName
if ($LASTEXITCODE -ne 0) {
    throw "Failed to activate '$envName' environment."
}

# --- Ensure PyInstaller is installed -----------------------------------------

Write-Host "`nChecking PyInstaller..." -ForegroundColor Yellow
pip show pyinstaller *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller..." -ForegroundColor Yellow
    pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed." }
}

# --- Build --------------------------------------------------------------------

Set-Location $scriptRoot

Write-Host "`nCleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "dist")  { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

Write-Host "`nRunning PyInstaller (this may take several minutes)..." -ForegroundColor Green
pyinstaller build_sidecar.spec --clean --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

$exePath = "dist\dos-gcnn-sidecar.exe"
if (-not (Test-Path $exePath)) {
    throw "Output exe not found at $exePath."
}

$fileSize = (Get-Item $exePath).Length / 1MB
Write-Host "`n========================================" -ForegroundColor Green
Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "Output: $exePath" -ForegroundColor Cyan
Write-Host "Size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Test sidecar: .\dist\dos-gcnn-sidecar.exe path\to\test.cif" -ForegroundColor White
Write-Host "2. Build Tauri app: cd .. ; yarn tauri build" -ForegroundColor White
