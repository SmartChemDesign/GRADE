# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DOS-GCNN is a Windows desktop app that predicts the electronic Density of States (DOS) of crystalline materials from a CIF/VASP/POSCAR file using a graph convolutional neural network. It is a **three-language** project glued together at process boundaries:

- **Tauri 2 / Rust** (`src-tauri/`) — desktop shell. Owns the main window and exposes two `#[tauri::command]`s to the frontend: `predict_dos` and `check_sidecar`.
- **Vue 3 + TypeScript + Vite** (`src/`) — UI. `App.vue` orchestrates; `CrystalViewer.vue` wraps 3Dmol.js; `DOSChart.vue` wraps Plotly; `FileUpload.vue` is the picker. Frontend talks to Rust via `@tauri-apps/api` `invoke(...)`.
- **Python 3.10 + PyTorch + PyTorch Geometric** (`dos_gcnn/` + `python-model/`) — ML inference, frozen to a single `dos-gcnn-sidecar.exe` via PyInstaller.

The three parts do **not** share memory. Rust spawns `dos-gcnn-sidecar.exe <cif_path> [--force]` as a child process, captures stdout, parses it as a single JSON document, and forwards the `SidecarResult` struct to the frontend. All cross-boundary contracts live in two places: [sidecar_main.py](python-model/sidecar_main.py) (producer) and [src-tauri/src/lib.rs](src-tauri/src/lib.rs) (consumer — see `SidecarResult` / `PredictionData`). Changes to the output schema must be made in both.

### Sidecar resolution (Rust side)

`find_sidecar_exe` in [src-tauri/src/lib.rs](src-tauri/src/lib.rs:38) searches, in order: Tauri `resource_dir`, next to the main exe, then dev paths `./python-model/dist/` and `../python-model/dist/`. This is why the Python sidecar must be built **before** `yarn tauri dev` — Rust has no fallback to raw Python.

### Python package layout

`dos_gcnn/` is a regular Python package (imported both in dev and inside the frozen exe). Entry point `predict_dos()` in [dos_gcnn/inference.py](dos_gcnn/inference.py) returns a `DOSResult` dataclass. Config lives in [dos_gcnn/config.py](dos_gcnn/config.py) and is **PyInstaller-aware**: `get_base_dir()` returns `sys._MEIPASS` when frozen, so model weights and dictionaries must be bundled as data files (see `datas=[...]` in [python-model/build_sidecar.spec](python-model/build_sidecar.spec)). Model weights and vocabularies are read from `bulk_new/` — this directory is a runtime data dependency, not source.

## Common commands

All commands run from the repo root in **PowerShell** on Windows.

```powershell
# End-to-end release build (sidecar + Tauri MSI/NSIS). Auto-installs Miniconda if missing.
.\build_all.ps1

# Dev loop — requires the sidecar to exist in python-model\dist\ first.
cd python-model; .\build_sidecar.ps1; cd ..
.\start_tauri.ps1                 # wraps `yarn tauri dev`

# Sidecar only (rebuild after Python changes)
cd python-model; .\build_sidecar.ps1

# Tauri only (frontend + Rust, assumes sidecar already built)
.\build_tauri.ps1                 # release
yarn tauri dev                    # dev

# Frontend type-check + build (no Tauri)
yarn build                        # runs `vue-tsc --noEmit && vite build`

# Run the sidecar directly for debugging — bypasses Rust entirely
.\python-model\dist\dos-gcnn-sidecar.exe path\to\structure.cif
.\python-model\dist\dos-gcnn-sidecar.exe path\to\structure.cif --force
```

There is **no test suite** — neither `pytest`, nor Rust `cargo test`, nor a JS test runner is wired up. Verification is manual: run the app, load a CIF, confirm DOS renders.

## Build system notes

- `python-model/build_sidecar.ps1` will silently install Miniconda into `%USERPROFILE%\Miniconda3` and create the `dos_gcnn` conda env from `environment.yml` if neither exists. Re-runs are fast (idempotent).
- `environment.yml` installs `torch` via **pip without a version pin**, so the torch version is whatever pip resolves on first install. Mixing pip-torch's bundled MKL with conda-forge's MKL has been a source of `WinError 127` on the frozen exe — see the explicit `torch\lib` DLL collection in [python-model/build_sidecar.spec](python-model/build_sidecar.spec) and avoid also bundling conda's `mkl*.dll` into the bundle root.
- UPX is disabled in the PyInstaller spec (`upx=False`) — it corrupts torch DLLs. Don't re-enable it.
- `runtime_tmpdir=None` means the frozen exe extracts to `%TEMP%\_MEIxxxxxx` on every run; PyInstaller `_MEIPASS` paths in error messages point there.

## Sidecar JSON contract

The sidecar writes exactly one JSON object to stdout and exits. Rust tolerates leading non-JSON output by searching for the first `{`. Two success modes and one failure mode are possible:

- `success: true` with full `data` payload — normal prediction.
- `success: false` with `warning` + `min_distance` + `cif_content` — short interatomic distance detected (< 1 Å). Frontend should re-invoke with `force: true` to override.
- `success: false` with `error` (+ optional `traceback`) — hard failure.

When adding new fields to the prediction output, update all three: the Python `output` dict in [sidecar_main.py](python-model/sidecar_main.py), the `PredictionData` struct in [src-tauri/src/lib.rs](src-tauri/src/lib.rs), and the frontend consumer in `src/App.vue`.
