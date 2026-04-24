"""Inference module for DOS-GCNN."""

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .config import Config
from .data.processing import get_dataset
from .utils.data import get_dos_value
from .models.dos_predict import model_summary


@dataclass
class DOSResult:
    """Result of DOS prediction.
    
    Attributes:
        dos_s: s-orbital DOS for each atom, shape (n_atoms, 400)
        dos_p: p-orbital DOS for each atom, shape (n_atoms, 400)
        dos_d: d-orbital DOS for each atom, shape (n_atoms, 400)
        dos_f: f-orbital DOS for each atom, shape (n_atoms, 400)
        elements: Atomic numbers for each atom
        total_atomic_dos: Total DOS for each atom (sum of s,p,d,f)
        total_crystal_dos: Total crystal DOS (sum over all atoms)
        energy_grid: Energy grid for DOS values
    """
    dos_s: np.ndarray
    dos_p: np.ndarray
    dos_d: np.ndarray
    dos_f: np.ndarray
    elements: np.ndarray
    total_atomic_dos: np.ndarray
    total_crystal_dos: np.ndarray
    energy_grid: np.ndarray
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'dos_s': self.dos_s,
            'dos_p': self.dos_p,
            'dos_d': self.dos_d,
            'dos_f': self.dos_f,
            'elements': self.elements,
            'total_atomic_dos': self.total_atomic_dos,
            'total_crystal_dos': self.total_crystal_dos,
            'energy_grid': self.energy_grid,
        }


def predict_dos(
    cif_path: Union[str, Path],
    config: Optional[Config] = None,
    plot_results: bool = False,
    output_dir: Optional[Union[str, Path]] = None,
) -> DOSResult:
    """Predict density of states from a CIF file.
    
    This is the main entry point for DOS prediction.
    
    Args:
        cif_path: Path to CIF or VASP structure file
        config: Configuration object (uses defaults if None)
        plot_results: Whether to generate plots
        output_dir: Directory to save plots (if plot_results=True)
        
    Returns:
        DOSResult object with predicted DOS for s,p,d,f orbitals
        
    Example:
        >>> from dos_gcnn import predict_dos
        >>> result = predict_dos("my_structure.cif")
        >>> print(result.total_crystal_dos.shape)
        (1, 400)
    """
    cif_path = Path(cif_path)
    if not cif_path.exists():
        raise FileNotFoundError(f"Structure file not found: {cif_path}")
    
    # Use default config if not provided
    if config is None:
        config = Config()
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as tmp_data_path:
        # Get dataset
        dataset = get_dataset(
            str(cif_path),
            tmp_data_path,
            config.get_processing_args()
        )
        
        if dataset is None:
            raise RuntimeError("Failed to process structure file")
        
        # Run prediction
        model_config = config.get_model_config()
        job_params = config.get_job_parameters()
        
        dos_s, dos_p, dos_d, dos_f, elements = _run_prediction(
            dataset,
            job_params,
            basis_expansion=model_config["basis_expansion"],
            basis_type=model_config["basis_type"],
        )
    
    # Calculate total DOS
    num_atoms = dos_s.shape[0]
    grid_size = dos_s.shape[1]
    
    total_atomic_dos = dos_s + dos_p + dos_d + dos_f
    total_crystal_dos = np.sum(total_atomic_dos, axis=0, keepdims=True)
    
    energy_grid = np.linspace(-10, 10, grid_size)
    
    result = DOSResult(
        dos_s=dos_s,
        dos_p=dos_p,
        dos_d=dos_d,
        dos_f=dos_f,
        elements=elements,
        total_atomic_dos=total_atomic_dos,
        total_crystal_dos=total_crystal_dos,
        energy_grid=energy_grid,
    )
    
    # Generate plots if requested
    if plot_results:
        from .utils.visualization import plot_all_dos
        out_dir = Path(output_dir) if output_dir else config.output_dir
        plot_all_dos(dos_s, dos_p, dos_d, dos_f, elements, out_dir)
    
    return result


def _run_prediction(
    dataset,
    job_parameters: dict,
    basis_expansion: str = "True",
    basis_type: str = "Lorentz",
) -> tuple:
    """Run model prediction on dataset.
    
    Args:
        dataset: Processed dataset
        job_parameters: Job parameters dict
        basis_expansion: Whether basis expansion is used
        basis_type: Type of basis functions
        
    Returns:
        Tuple of (dos_s, dos_p, dos_d, dos_f, elements)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = DataLoader(
        dataset[0:1],
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Load saved model
    assert os.path.exists(job_parameters["model_path"]), "Saved model not found"
    
    # Patch sys.modules so pickle can find DOSpredict class
    # (model was saved from __main__ in original script)
    from .models.dos_predict import DOSpredict
    from .models.gc_block import GC_block
    from .models.kan import KANLinear
    
    import __main__
    __main__.DOSpredict = DOSpredict
    __main__.GC_block = GC_block
    __main__.KANLinear = KANLinear
    
    if str(device) == "cpu":
        saved = torch.load(
            job_parameters["model_path"], 
            map_location=torch.device("cpu"),
            weights_only=False
        )
    else:
        saved = torch.load(
            job_parameters["model_path"], 
            map_location=torch.device("cuda"),
            weights_only=False
        )
    
    model = saved["full_model"]
    model.to(device)
    model_summary(model)

    time_start = time.time()
    model.eval()
    
    count = 0
    predict_s = predict_p = predict_d = predict_f = None
    elements = None

    for ii, data in enumerate(loader):
        print(f"Sample: {ii}")
        data = data.to(device)
        print(f"Total number of atoms in crystal: {len(data.z)}")
        
        with torch.no_grad():
            if basis_expansion == "False":
                dos_out_s, scaling_s, dos_out_p, scaling_p, \
                dos_out_d, scaling_d, dos_out_f, scaling_f = model(data)
            else:
                dos_coef_s, dos_mean_s, dos_sigma_s, scaling_s, \
                dos_coef_p, dos_mean_p, dos_sigma_p, scaling_p, \
                dos_coef_d, dos_mean_d, dos_sigma_d, scaling_d, \
                dos_coef_f, dos_mean_f, dos_sigma_f, scaling_f = model(data)
                
                # Calculate DOS values using basis expansion
                xx = torch.linspace(-10, 10, 400).to(dos_coef_s)
                dos_out_s = get_dos_value(xx, dos_coef_s, dos_mean_s, dos_sigma_s, device, basis_type)
                dos_out_p = get_dos_value(xx, dos_coef_p, dos_mean_p, dos_sigma_p, device, basis_type)
                dos_out_d = get_dos_value(xx, dos_coef_d, dos_mean_d, dos_sigma_d, device, basis_type)
                dos_out_f = get_dos_value(xx, dos_coef_f, dos_mean_f, dos_sigma_f, device, basis_type)

            # Reconstruct DOS with scaling
            scaled_dos_s = dos_out_s * scaling_s.view(-1, 1).expand_as(dos_out_s)
            scaled_dos_p = dos_out_p * scaling_p.view(-1, 1).expand_as(dos_out_p)
            scaled_dos_d = dos_out_d * scaling_d.view(-1, 1).expand_as(dos_out_d)
            scaled_dos_f = dos_out_f * scaling_f.view(-1, 1).expand_as(dos_out_f)

            if count == 0:
                predict_s = scaled_dos_s.data.cpu().numpy()
                predict_p = scaled_dos_p.data.cpu().numpy()
                predict_d = scaled_dos_d.data.cpu().numpy()
                predict_f = scaled_dos_f.data.cpu().numpy()
                elements = data.z.cpu().numpy()
            else:
                predict_s = np.concatenate((predict_s, scaled_dos_s.data.cpu().numpy()), axis=0)
                predict_p = np.concatenate((predict_p, scaled_dos_p.data.cpu().numpy()), axis=0)
                predict_d = np.concatenate((predict_d, scaled_dos_d.data.cpu().numpy()), axis=0)
                predict_f = np.concatenate((predict_f, scaled_dos_f.data.cpu().numpy()), axis=0)
                elements = np.concatenate((elements, data.z.cpu().numpy()), axis=0)

            count += dos_out_s.size(0)

    elapsed_time = time.time() - time_start
    print(f"Evaluation time (s): {elapsed_time:.5f}")

    return predict_s, predict_p, predict_d, predict_f, elements


# For backwards compatibility
def predict_spdf(*args, **kwargs):
    """Legacy function name, use predict_dos instead."""
    import warnings
    warnings.warn("predict_spdf is deprecated, use predict_dos instead", DeprecationWarning)
    return _run_prediction(*args, **kwargs)

