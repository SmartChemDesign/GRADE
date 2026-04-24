"""Dataset classes for DOS-GCNN."""

import os
import torch
from torch_geometric.data import InMemoryDataset


class StructureDataset(InMemoryDataset):
    """Dataset class from pytorch/pytorch geometric; inmemory case."""
    
    def __init__(self, data_path, processed_path="processed",
                 transform=None, pre_transform=None):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], 
            weights_only=False
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        return ["data.pt"]

