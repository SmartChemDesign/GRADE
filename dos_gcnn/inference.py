"""Inference module for DOS-GCNN."""

import os
import time
import tempfile
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .ad import public_ad_payload, score_structure, try_load_ad_bundle
from .config import Config
from .data.processing import get_dataset
from .utils.data import get_dos_value
from .models.dos_predict import DOSpredict, model_summary
from .models.gc_block import GC_block
from .models.kan import KANLinear


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
        ad: Optional applicability-domain payload (None if scoring is skipped)
        ad_reason: Why AD is unavailable when ad is None
    """
    dos_s: np.ndarray
    dos_p: np.ndarray
    dos_d: np.ndarray
    dos_f: np.ndarray
    elements: np.ndarray
    total_atomic_dos: np.ndarray
    total_crystal_dos: np.ndarray
    energy_grid: np.ndarray
    ad: Optional[dict] = None
    ad_reason: Optional[str] = None
    
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
            'ad': self.ad,
            'ad_reason': self.ad_reason,
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
        
        dos_s, dos_p, dos_d, dos_f, elements, ad_payload, ad_reason = _run_prediction(
            dataset,
            job_params,
            basis_expansion=model_config["basis_expansion"],
            basis_type=model_config["basis_type"],
            model_config=model_config,
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
        ad=ad_payload,
        ad_reason=ad_reason,
    )
    
    # Generate plots if requested
    if plot_results:
        from .utils.visualization import plot_all_dos
        out_dir = Path(output_dir) if output_dir else config.output_dir
        plot_all_dos(dos_s, dos_p, dos_d, dos_f, elements, out_dir)
    
    return result


def _validate_transformer_checkpoint_state(state_dict):
    """Fail loudly if the .pth is not the 3-layer Transformer Lorentz checkpoint."""
    expected_shapes = {
        "pre_lin_list.0.0.weight": (370, 205),
        "conv_list.0.lin_key.weight": (370, 370),
        "conv_list.2.lin_key.weight": (370, 370),
        "conv_list1.0.mlp.0.weight": (370, 790),
        "conv_list1.2.mlp.0.weight": (370, 790),
        "conv_list.0.lin_edge.weight": (370, 50),
        "dos_mlp_basis_coef_s.2.weight": (28, 370),
        "dos_mlp_basis_mean_s.2.weight": (28, 370),
        "dos_mlp_basis_sigma_s.2.weight": (28, 370),
        "dos_mlp_basis_coef_f.2.weight": (28, 370),
    }
    for key, expected in expected_shapes.items():
        if key not in state_dict:
            raise RuntimeError(f"Expected checkpoint tensor is absent: {key}")
        got = tuple(state_dict[key].shape)
        if got != expected:
            raise RuntimeError(
                f"Checkpoint tensor {key} has shape {got}, expected {expected}"
            )


def _validate_inference_graph(dataset):
    """Check preprocessing compatibility with the Transformer checkpoint."""
    if len(dataset) < 1:
        raise RuntimeError("Inference dataset is empty.")
    sample = dataset[0]
    if not hasattr(sample, "x") or sample.x is None:
        raise RuntimeError("Inference graph has no node features data.x.")
    if sample.x.ndim != 2 or int(sample.x.shape[1]) != 205:
        raise RuntimeError(
            f"Checkpoint expects data.x shape [n_atoms,205], got {tuple(sample.x.shape)}."
        )
    if not hasattr(sample, "edge_attr") or sample.edge_attr is None:
        raise RuntimeError("Inference graph has no edge_attr.")
    if sample.edge_attr.ndim != 2 or int(sample.edge_attr.shape[1]) != 50:
        raise RuntimeError(
            f"Checkpoint expects edge_attr shape [n_edges,50], got {tuple(sample.edge_attr.shape)}."
        )


def _load_transformer_checkpoint(dataset, checkpoint_path, device, model_config):
    """Reconstruct the 3-layer Transformer encoder and load state_dict strictly."""
    import __main__
    __main__.DOSpredict = DOSpredict
    __main__.GC_block = GC_block
    __main__.KANLinear = KANLinear

    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(saved, dict) or "state_dict" not in saved:
        raise RuntimeError("Transformer checkpoint does not contain a state_dict dictionary.")
    _validate_transformer_checkpoint_state(saved["state_dict"])

    model = DOSpredict(
        data=dataset,
        dim1=370,
        dim2=370,
        pre_fc_count=1,
        gc_count=int(model_config.get("gc_count", 3)),
        graph_conv_type=model_config.get("graph_conv_type", "Transformer"),
        batch_norm="True",
        batch_track_stats="True",
        basis_expansion=model_config.get("basis_expansion", "True"),
        num_func=int(model_config.get("num_func", 28)),
        perceptron_type=model_config.get("perceptron_type", "Regular"),
        dropout_rate=float(model_config.get("dropout_rate", 0.1)),
    ).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    if model.graph_conv_type != "Transformer":
        raise RuntimeError("Loaded model is not Transformer.")
    if len(model.conv_list) != 3 or len(model.conv_list1) != 3:
        raise RuntimeError(
            f"Unexpected encoder depth: Transformer={len(model.conv_list)}, "
            f"extra_GC={len(model.conv_list1)}"
        )
    return model


def _run_prediction(
    dataset,
    job_parameters: dict,
    basis_expansion: str = "True",
    basis_type: str = "Lorentz",
    model_config: Optional[dict] = None,
) -> tuple:
    """Run model prediction on dataset.

    Returns:
        Tuple of (dos_s, dos_p, dos_d, dos_f, elements, ad_payload, ad_reason)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if model_config is None:
        model_config = {}

    ad_bundle, ad_reason = try_load_ad_bundle(job_parameters.get("ad_model_path"))
    ad_threshold = float(job_parameters.get("ad_threshold", 0.50))
    ad_policy = str(job_parameters.get("ad_activity_policy", "element_block"))
    ad_payload = None
    if ad_bundle is not None:
        print(f"Applicability-domain model loaded: {job_parameters.get('ad_model_path')}")
        print(f"AD threshold: {ad_threshold}")
        print(f"AD sparse-channel activity policy: {ad_policy}")
    else:
        print(f"Applicability-domain skipped: {ad_reason}")

    loader = DataLoader(
        dataset[0:1],
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    assert os.path.exists(job_parameters["model_path"]), "Saved model not found"

    _validate_inference_graph(dataset)
    model = _load_transformer_checkpoint(
        dataset, job_parameters["model_path"], device, model_config
    )
    model_summary(model)

    time_start = time.time()
    model.eval()

    count = 0
    predict_s = predict_p = predict_d = predict_f = None
    elements = None

    for ii, data in enumerate(loader):
        print(f"Sample: {ii}")
        atomic_numbers = data.z.detach().cpu().numpy() if hasattr(data, "z") else None
        data = data.to(device)
        print(f"Total number of atoms in crystal: {len(data.z)}")

        with torch.no_grad():
            outputs = model(data)
            *head, embeds = outputs
            if basis_expansion == "False":
                (
                    dos_out_s, scaling_s, dos_out_p, scaling_p,
                    dos_out_d, scaling_d, dos_out_f, scaling_f,
                ) = head
            else:
                (
                    dos_coef_s, dos_mean_s, dos_sigma_s, scaling_s,
                    dos_coef_p, dos_mean_p, dos_sigma_p, scaling_p,
                    dos_coef_d, dos_mean_d, dos_sigma_d, scaling_d,
                    dos_coef_f, dos_mean_f, dos_sigma_f, scaling_f,
                ) = head
                xx = torch.linspace(-10, 10, 400).to(dos_coef_s)
                dos_out_s = get_dos_value(xx, dos_coef_s, dos_mean_s, dos_sigma_s, device, basis_type)
                dos_out_p = get_dos_value(xx, dos_coef_p, dos_mean_p, dos_sigma_p, device, basis_type)
                dos_out_d = get_dos_value(xx, dos_coef_d, dos_mean_d, dos_sigma_d, device, basis_type)
                dos_out_f = get_dos_value(xx, dos_coef_f, dos_mean_f, dos_sigma_f, device, basis_type)

            scaled_dos_s = dos_out_s * scaling_s.view(-1, 1).expand_as(dos_out_s)
            scaled_dos_p = dos_out_p * scaling_p.view(-1, 1).expand_as(dos_out_p)
            scaled_dos_d = dos_out_d * scaling_d.view(-1, 1).expand_as(dos_out_d)
            scaled_dos_f = dos_out_f * scaling_f.view(-1, 1).expand_as(dos_out_f)

            if ad_bundle is not None:
                ad_row = score_structure(
                    ad_bundle,
                    embeds,
                    atomic_numbers=atomic_numbers,
                    pred_d=scaled_dos_d.detach().cpu().numpy(),
                    pred_f=scaled_dos_f.detach().cpu().numpy(),
                    threshold=ad_threshold,
                    policy=ad_policy,
                )
                ad_payload = public_ad_payload(ad_row)
                ad_reason = None
                print(
                    "AD: p_all={:.4f} -> {} | p_s={:.4f} p_p={:.4f} p_d={:.4f} p_f={:.4f}".format(
                        ad_payload["p_ensemble_all"],
                        "INSIDE" if ad_payload["inside_ad"] else "OUTSIDE",
                        ad_payload["p_ensemble_s"],
                        ad_payload["p_ensemble_p"],
                        ad_payload["p_ensemble_d"],
                        ad_payload["p_ensemble_f"],
                    )
                )

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

    return predict_s, predict_p, predict_d, predict_f, elements, ad_payload, ad_reason


# For backwards compatibility
def predict_spdf(*args, **kwargs):
    """Legacy function name, use predict_dos instead."""
    import warnings
    warnings.warn("predict_spdf is deprecated, use predict_dos instead", DeprecationWarning)
    return _run_prediction(*args, **kwargs)

