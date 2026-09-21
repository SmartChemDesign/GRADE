"""Configuration management for DOS-GCNN."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml
import torch
import sys


def get_base_dir() -> Path:
    """Get base directory, handling PyInstaller bundle."""
    # When running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        return Path(sys._MEIPASS)
    else:
        # Normal Python execution
        return Path(__file__).parent.parent.resolve()


BASE_DIR = get_base_dir()


@dataclass
class Config:
    """Configuration for DOS-GCNN inference.
    
    Uses relative paths based on the project root directory.
    """
    # Base paths (relative to project root)
    bulk_dir: Path = field(default_factory=lambda: BASE_DIR / "bulk_new")
    
    # Model settings
    model_name: str = "model_bulk_lorentz_28_spdf_tansformer_terms_new.pth"
    graph_conv_type: str = "Transformer"
    basis_expansion: bool = True
    basis_type: str = "Lorentz"
    num_func: int = 28
    perceptron_type: str = "Regular"
    gc_count: int = 3
    
    # Processing settings
    dictionary_source: str = "default"
    graph_max_radius: float = 8.0
    graph_max_neighbors: int = 12
    graph_edge_length: int = 50
    edge_features: str = "True"
    verbose: str = "True"
    processed_path: str = "processed"
    primitive_cell: bool = True  # Use primitive cell (unique Wyckoff positions)
    
    # Output settings
    write_output: bool = False
    output_dir: Optional[Path] = None

    # Applicability-domain settings (post-encoder 370-D Transformer embeddings)
    ad_model_path: Optional[Path] = None
    ad_threshold: float = 0.50
    ad_activity_policy: str = "element_block"
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.config_path = self.bulk_dir / "config" / "config.yml"
        self.model_path = self.bulk_dir / "saved_models" / self.model_name
        self.dict_file = self.bulk_dir / "dict" / "dictionary_default.json"
        self.skipatom_file = self.bulk_dir / "dict" / "terms.json"
        self.tmp_dir = self.bulk_dir / "tmp"
        if self.ad_model_path is None:
            self.ad_model_path = self.bulk_dir / "ad" / "ad_model.pkl"
        else:
            self.ad_model_path = Path(self.ad_model_path)
        
        if self.output_dir is None:
            self.output_dir = self.bulk_dir / "outputs"
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**config_dict)
    
    def get_processing_args(self) -> dict:
        """Get processing arguments dict."""
        return {
            "dictionary_source": self.dictionary_source,
            "dict_file": str(self.dict_file),
            "skipatom_file": str(self.skipatom_file),
            "graph_max_radius": self.graph_max_radius,
            "graph_max_neighbors": self.graph_max_neighbors,
            "graph_edge_length": self.graph_edge_length,
            "edge_features": self.edge_features,
            "verbose": self.verbose,
            "processed_path": self.processed_path,
            "primitive_cell": self.primitive_cell,
        }
    
    def get_job_parameters(self) -> dict:
        """Get job parameters dict."""
        return {
            "model_path": str(self.model_path),
            "write_output": self.write_output,
            "job_name": "bulk_dos",
            "ad_model_path": str(self.ad_model_path),
            "ad_threshold": float(self.ad_threshold),
            "ad_activity_policy": self.ad_activity_policy,
        }
    
    def get_model_config(self) -> dict:
        """Get model configuration dict."""
        return {
            "graph_conv_type": self.graph_conv_type,
            "basis_expansion": "True" if self.basis_expansion else "False",
            "basis_type": self.basis_type,
            "num_func": self.num_func,
            "perceptron_type": self.perceptron_type,
            "gc_count": self.gc_count,
            "dropout_rate": 0.1,
            # KAN parameters
            "grid_size": 3,
            "spline_order": 2,
            "scale_noise": 0.1,
            "scale_base": 1.0,
            "scale_spline": 1.0,
            "base_activation": torch.nn.PReLU,
            "grid_eps": 0.02,
            "grid_range": [0, 3],
            # Scaling factor KAN
            "grid_size_w": 2,
            "spline_order_w": 2,
            "scale_noise_w": 0.1,
            "scale_base_w": 1.0,
            "scale_spline_w": 1.0,
            "base_activation_w": torch.nn.ReLU,
            "grid_eps_w": 0.02,
            "grid_range_w": [0, 3],
            # Coefficients KAN
            "grid_size_c": 2,
            "spline_order_c": 2,
            "scale_noise_c": 0.1,
            "scale_base_c": 1.0,
            "scale_spline_c": 1.0,
            "base_activation_c": torch.nn.PReLU,
            "grid_eps_c": 0.02,
            "grid_range_c": [-2, 2],
            # Mean KAN
            "grid_size_m": 2,
            "spline_order_m": 2,
            "scale_noise_m": 0.1,
            "scale_base_m": 1.0,
            "scale_spline_m": 1.0,
            "base_activation_m": torch.nn.RReLU,
            "grid_eps_m": 0.02,
            "grid_range_m": [-5, 5],
            # Dispersion KAN
            "grid_size_d": 2,
            "spline_order_d": 2,
            "scale_noise_d": 0.1,
            "scale_base_d": 1.0,
            "scale_spline_d": 1.0,
            "base_activation_d": torch.nn.ReLU,
            "grid_eps_d": 0.02,
            "grid_range_d": [0, 1],
        }

