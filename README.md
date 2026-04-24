# DOS-GCNN

Desktop application for predicting the electronic Density of States (DOS) of crystalline materials using a graph convolutional neural network.

The app loads a crystal structure (CIF / VASP / POSCAR), runs an ML inference pipeline, and visualises:

- a 3D interactive crystal viewer (3Dmol.js);
- per-atom and per-orbital (s / p / d / f) DOS plots (Plotly);
- total crystal DOS and selective atom-based aggregations;
- CSV / ZIP export of the computed DOS.

## Architecture

| Layer | Technology |
|-------|------------|
| Desktop shell | Tauri 2 (Rust) |
| Frontend | Vue 3 + TypeScript + Vite |
| ML sidecar | Python 3.10 + PyTorch + PyTorch Geometric, packaged with PyInstaller |

The Rust backend invokes a bundled Python executable (`dos-gcnn-sidecar.exe`) that runs inference and returns JSON to the UI.

## Requirements

To run a prebuilt release:

- Windows 10 / 11 x64

To build from source:

- Windows 10 / 11 x64 (PowerShell 5+)
- [Node.js](https://nodejs.org/) 18+ and [Yarn](https://classic.yarnpkg.com/)
- [Rust](https://rustup.rs/) (stable toolchain)
- The build scripts will install Miniconda automatically if it is not already available.

## Running the prebuilt application

1. Download `DOS-GCNN_<version>_x64_en-US.msi` or `DOS-GCNN_<version>_x64-setup.exe` from the Releases page.
2. Run the installer and launch **DOS-GCNN** from the Start menu.
3. Click **Select File**, pick a `.cif` structure, and the DOS will be predicted and rendered.

## Building and running from source

All commands are run from the repository root in PowerShell.

### Option 1 — full one-command build

Builds the Python sidecar and the Tauri application end-to-end. If Miniconda is missing, the script installs it silently before proceeding.

```powershell
.\build_all.ps1
```

Artifacts:

- `src-tauri\target\release\bundle\msi\DOS-GCNN_<version>_x64_en-US.msi`
- `src-tauri\target\release\bundle\nsis\DOS-GCNN_<version>_x64-setup.exe`

### Option 2 — development run

Starts the Tauri dev server with live reload.

```powershell
# Build the Python sidecar once (required for the Rust backend to resolve it)
cd python-model
.\build_sidecar.ps1
cd ..

# Start the dev server
.\start_tauri.ps1
```

### Option 3 — manual steps

```powershell
# 1. Create / update the Python environment
conda env create -f environment.yml        # first time only
conda activate dos_gcnn

# 2. Build the Python sidecar
cd python-model
pip install pyinstaller
pyinstaller build_sidecar.spec --clean --noconfirm
cd ..

# 3. Install JS dependencies
yarn install

# 4. Either run in dev mode...
yarn tauri dev

# ...or produce a release build
yarn tauri build
```

## Python sidecar — auto-install of Miniconda

`python-model\build_sidecar.ps1` no longer requires Miniconda / Anaconda to be pre-installed. If `conda` is not found on `PATH`, the script:

1. downloads the latest Miniconda Windows installer from `https://repo.anaconda.com/miniconda/`;
2. installs it silently per-user into `%USERPROFILE%\Miniconda3`;
3. initialises conda for the current PowerShell session;
4. creates the `dos_gcnn` environment from `environment.yml` if it does not exist;
5. activates it and continues the PyInstaller build.

Re-running the script after the first install is fast — it skips any step that is already done.

## Project layout

```
.
├── src/                      # Vue 3 + TypeScript frontend
│   ├── App.vue               # Main app, DOS aggregation, export
│   └── components/
│       ├── CrystalViewer.vue # 3Dmol.js viewer with coordination polyhedra
│       ├── DOSChart.vue      # Plotly DOS chart
│       └── FileUpload.vue    # File picker
├── src-tauri/                # Tauri / Rust backend (calls the sidecar)
├── python-model/             # Sidecar entry point + PyInstaller spec
│   └── build_sidecar.ps1     # Builds dos-gcnn-sidecar.exe
├── dos_gcnn/                 # ML package (model, data, inference)
├── bulk_new/                 # Pretrained weights, config, vocabularies
├── environment.yml           # Conda environment spec
├── build_all.ps1             # Sidecar + Tauri end-to-end build
├── build_tauri.ps1           # Tauri-only build
└── start_tauri.ps1           # Tauri dev server launcher
```

## License

The MIT License (MIT)
