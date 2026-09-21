#!/usr/bin/env python3
"""Sidecar entry point for DOS-GCNN.

This script is designed to be called from Tauri.
It accepts a CIF file path and returns JSON with prediction results.

Usage:
    python sidecar_main.py <cif_file_path> [--force]
    
Output:
    JSON to stdout with structure:
    {
        "success": true,
        "data": {
            "num_atoms": 4,
            "elements": [26, 8, ...],
            "element_symbols": ["Fe", "O", ...],
            "energy_grid": [...],
            "dos_s": [[...], ...],
            "dos_p": [[...], ...],
            "dos_d": [[...], ...],
            "dos_f": [[...], ...],
            "total_atomic_dos": [[...], ...],
            "total_crystal_dos": [...],
            "cif_content": "...",
            "ad": { ... } | null,
            "ad_reason": "..." | null
        }
    }
    
    Or warning if min distance < 1 Angstrom:
    {
        "success": false,
        "warning": "Short interatomic distance detected",
        "min_distance": 0.85,
        "cif_content": "..."
    }
"""

import sys
import json
import traceback
import re
from pathlib import Path

# Suppress warnings and info messages
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add parent directory to path for dos_gcnn import
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_min_distance(cif_path: str, threshold: float = 1.0) -> tuple[float, str, bool]:
    """Check minimum interatomic distance in structure.
    
    Uses neighbor list for efficient search instead of full distance matrix.
    
    Args:
        cif_path: Path to CIF file
        threshold: Minimum allowed distance in Angstrom
        
    Returns:
        Tuple of (min_distance, cif_content_clean, is_valid)
        is_valid is True if min_distance >= threshold
    """
    from ase import io
    from ase.neighborlist import neighbor_list
    
    # Read structure (simple read, no primitive cell search)
    atoms = io.read(cif_path)
    
    # Use neighbor list to find pairs within threshold - much faster than full matrix
    # Search up to threshold distance to check if any pairs are too close
    i, j, d = neighbor_list('ijd', atoms, cutoff=threshold, self_interaction=False)
    
    if len(d) > 0:
        # Found pairs closer than threshold
        min_distance = float(d.min())
        is_valid = False
    else:
        # No pairs within threshold - structure is valid
        # For display purposes, find actual min distance with slightly larger cutoff
        i, j, d = neighbor_list('ijd', atoms, cutoff=3.0, self_interaction=False)
        min_distance = float(d.min()) if len(d) > 0 else 3.0
        is_valid = True
    
    # Read and clean CIF content
    with open(cif_path, 'r') as f:
        cif_content = f.read()
    cif_content_clean = re.sub(r'(\b[A-Z][a-z]?)\d*[+-]', r'\1', cif_content)
    
    return min_distance, cif_content_clean, is_valid


def output_warning(message: str, min_distance: float, cif_content: str):
    """Output warning as JSON to stdout."""
    output = {
        "success": False,
        "warning": message,
        "min_distance": round(min_distance, 3),
        "cif_content": cif_content,
    }
    print(json.dumps(output))


def main():
    if len(sys.argv) < 2:
        output_error("No CIF file path provided")
        return 1
    
    cif_path = sys.argv[1]
    force = "--force" in sys.argv
    
    if not Path(cif_path).exists():
        output_error(f"File not found: {cif_path}")
        return 1
    
    try:
        # Check minimum distance first (unless forced)
        min_distance, cif_content_clean, is_valid = check_min_distance(cif_path, threshold=1.0)
        
        if not is_valid and not force:
            output_warning(
                f"Short interatomic distance detected: {min_distance:.3f} Å. "
                "This may indicate overlapping atoms or incorrect structure.",
                min_distance,
                cif_content_clean
            )
            return 0
        
        # Import here to avoid slow startup if there's an argument error
        from dos_gcnn import predict_dos, Config
        from ase import data as ase_data
        
        # Run prediction
        config = Config()
        result = predict_dos(cif_path, config=config)
        
        # Convert element numbers to symbols
        element_symbols = [ase_data.chemical_symbols[z] for z in result.elements]
        
        # Prepare output
        output = {
            "success": True,
            "data": {
                "num_atoms": len(result.elements),
                "elements": result.elements.tolist(),
                "element_symbols": element_symbols,
                "energy_grid": result.energy_grid.tolist(),
                "dos_s": result.dos_s.tolist(),
                "dos_p": result.dos_p.tolist(),
                "dos_d": result.dos_d.tolist(),
                "dos_f": result.dos_f.tolist(),
                "total_atomic_dos": result.total_atomic_dos.tolist(),
                "total_crystal_dos": result.total_crystal_dos[0].tolist(),
                "cif_content": cif_content_clean,
                "ad": result.ad,
                "ad_reason": result.ad_reason,
            }
        }
        
        print(json.dumps(output))
        return 0
        
    except Exception as e:
        output_error(str(e), traceback.format_exc())
        return 1


def output_error(message: str, trace: str = None):
    """Output error as JSON to stdout."""
    output = {
        "success": False,
        "data": None,
        "error": message,
    }
    if trace:
        output["traceback"] = trace
    print(json.dumps(output))


if __name__ == "__main__":
    sys.exit(main())
