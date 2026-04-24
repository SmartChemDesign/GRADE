"""DOS-GCNN: Graph Convolutional Neural Network for Density of States prediction."""

from .inference import predict_dos
from .config import Config

__all__ = ['predict_dos', 'Config']
__version__ = '1.0.0'

