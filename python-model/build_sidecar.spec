# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for DOS-GCNN sidecar.
Compiles Python inference code into standalone executable.

Usage:
    conda activate dos_gcnn
    pyinstaller build_sidecar.spec --clean
"""
import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules
import glob

block_cipher = None

# Paths relative to spec file location
spec_dir = Path(SPECPATH)
project_root = spec_dir.parent
dos_gcnn_path = project_root / 'dos_gcnn'
model_data_path = project_root / 'bulk_new'

# Get site-packages path for conda environment
import torch_geometric
import torch
import ase
import scipy
import numpy

torch_geometric_path = Path(torch_geometric.__path__[0])
torch_path = Path(torch.__path__[0])
ase_path = Path(ase.__path__[0])
scipy_path = Path(scipy.__path__[0])
numpy_path = Path(numpy.__path__[0])

# Conda environment bin directory for DLLs
conda_prefix = os.environ.get('CONDA_PREFIX', '')
conda_bin = Path(conda_prefix) / 'Library' / 'bin'
conda_lib = Path(conda_prefix) / 'DLLs'

print(f"Project root: {project_root}")
print(f"Conda prefix: {conda_prefix}")
print(f"Conda bin: {conda_bin}")

# Verify paths exist
if not dos_gcnn_path.exists():
    raise FileNotFoundError(f"dos_gcnn not found: {dos_gcnn_path}")
if not model_data_path.exists():
    raise FileNotFoundError(f"bulk_new not found: {model_data_path}")

# Collect MKL and other DLLs from conda.
#
# PyTorch ships its own DLL set in torch\lib. Do not also place conda DLLs
# with the same file names at the bundle root: Windows can resolve the root
# copy first and then torch may fail with WinError 127 while loading shm.dll.
dll_binaries = []
torch_lib = torch_path / 'lib'
torch_dll_names = set()
if torch_lib.exists():
    torch_dll_names = {dll.name.lower() for dll in torch_lib.glob('*.dll')}

if conda_bin.exists():
    # MKL libraries
    for dll in conda_bin.glob('mkl*.dll'):
        if dll.name.lower() not in torch_dll_names:
            dll_binaries.append((str(dll), '.'))
    # OpenMP
    for dll in conda_bin.glob('libiomp*.dll'):
        if dll.name.lower() not in torch_dll_names:
            dll_binaries.append((str(dll), '.'))
    # Intel OpenMP
    for dll in conda_bin.glob('libomp*.dll'):
        if dll.name.lower() not in torch_dll_names:
            dll_binaries.append((str(dll), '.'))
    # BLAS/LAPACK
    for dll in conda_bin.glob('lib*blas*.dll'):
        if dll.name.lower() not in torch_dll_names:
            dll_binaries.append((str(dll), '.'))
    for dll in conda_bin.glob('lib*lapack*.dll'):
        if dll.name.lower() not in torch_dll_names:
            dll_binaries.append((str(dll), '.'))

# Explicitly bundle every DLL from torch\lib.
# collect_all('torch') misses non-module DLLs such as libuv.dll, asmjit.dll,
# uv.dll, which shm.dll / torch_cpu.dll import at runtime — their absence
# surfaces as "WinError 127: specified procedure could not be found".
if torch_lib.exists():
    for dll in torch_lib.glob('*.dll'):
        dll_binaries.append((str(dll), 'torch/lib'))

print(f"Found {len(dll_binaries)} DLLs to include")

# Collect all from problematic packages
scipy_datas, scipy_binaries, scipy_hiddenimports = collect_all('scipy')
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')
torch_datas, torch_binaries, torch_hiddenimports = collect_all('torch')

all_datas = scipy_datas + numpy_datas + torch_datas
all_binaries = scipy_binaries + numpy_binaries + torch_binaries + dll_binaries
all_binaries = [
    binary
    for binary in all_binaries
    if not str(binary[0]).lower().endswith('.lib')
]
all_hiddenimports = scipy_hiddenimports + numpy_hiddenimports + torch_hiddenimports

a = Analysis(
    ['sidecar_main.py'],
    pathex=[str(project_root)],
    binaries=all_binaries,
    datas=[
        # Include dos_gcnn module
        (str(dos_gcnn_path), 'dos_gcnn'),
        # Include model weights
        (str(model_data_path / 'saved_models' / 'model_bulk_lorentz_28_spdf_terms_new.pth'), 
         'bulk_new/saved_models'),
        # Include dictionaries
        (str(model_data_path / 'dict' / 'dictionary_default.json'), 'bulk_new/dict'),
        (str(model_data_path / 'dict' / 'terms.json'), 'bulk_new/dict'),
        # Include torch_geometric source files (needed for TorchScript)
        (str(torch_geometric_path), 'torch_geometric'),
        # Include ASE
        (str(ase_path), 'ase'),
    ] + all_datas,
    hiddenimports=[
        # PyTorch Geometric - all submodules
        'torch_geometric',
        'torch_geometric.nn',
        'torch_geometric.nn.conv',
        'torch_geometric.nn.conv.message_passing',
        'torch_geometric.nn.pool',
        'torch_geometric.nn.pool.select',
        'torch_geometric.nn.pool.select.base',
        'torch_geometric.nn.dense',
        'torch_geometric.nn.norm',
        'torch_geometric.nn.aggr',
        'torch_geometric.nn.models',
        'torch_geometric.data',
        'torch_geometric.loader',
        'torch_geometric.utils',
        'torch_geometric.transforms',
        'torch_geometric.typing',
        # PyTorch Scatter/Sparse
        'torch_scatter',
        'torch_sparse',
        'torch_cluster',
        # ASE
        'ase',
        'ase.io',
        'ase.io.cif',
        'ase.io.vasp',
        'ase.data',
        'ase.neighborlist',
        'ase.geometry',
        'ase.spacegroup',
        'ase.symbols',
        # DOS-GCNN modules
        'dos_gcnn',
        'dos_gcnn.inference',
        'dos_gcnn.config',
        'dos_gcnn.models',
        'dos_gcnn.models.dos_predict',
        'dos_gcnn.models.gc_block',
        'dos_gcnn.data',
        'dos_gcnn.data.dataset',
        'dos_gcnn.data.processing',
        'dos_gcnn.utils',
        'dos_gcnn.utils.data',
    ] + all_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'tensorboard',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='dos-gcnn-sidecar',
    debug=False,
    bootloader_ignore_signals=False,
    exclude_binaries=True,
    strip=False,
    upx=False,  # Disable UPX - can cause issues with torch
    upx_exclude=[],
    console=True,  # Console app for stdout/stderr
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='dos-gcnn-sidecar',
)
