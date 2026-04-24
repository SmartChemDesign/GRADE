"""Data processing functions for DOS-GCNN."""

import os
import sys
import numpy as np
import torch
import ase
from ase import io

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, add_self_loops

from ..utils.data import (
    get_dictionary,
    threshold_sort,
    OneHotDegree,
    GaussianSmearing,
    NormalizeEdge,
    Cleanup,
    GetY,
)
from .dataset import StructureDataset


def process_data_spdf(file_name, data_path, processed_path, processing_args):
    """Process structure data and create graph representations.
    
    Args:
        file_name: Path to structure file (cif, vasp, etc.)
        data_path: Path to data directory
        processed_path: Path to processed data directory
        processing_args: Processing arguments dict
    """
    # Load dictionary
    if processing_args["dictionary_source"] != "generated":
        if processing_args["dictionary_source"] == "default":
            print("Using default dictionary.")
            dict_file = processing_args["dict_file"]
            skipatom_file = processing_args["skipatom_file"]
            atom_dictionary = get_dictionary(dict_file)
            print("Loading skipatom file...")
            skipatom_dictionary = get_dictionary(skipatom_file)
            print("...Done!!")
        elif processing_args["dictionary_source"] == "blank":
            print("Using blank dictionary.")
            atom_dictionary = get_dictionary(
                os.path.join(os.path.dirname(__file__), "..", "dict", "dictionary_blank.json")
            )
        else:
            dictionary_file_path = os.path.join(
                data_path, processing_args["dictionary_path"]
            )
            if not os.path.exists(dictionary_file_path):
                print("Atom dictionary not found, exiting program...")
                sys.exit()
            else:
                print("Loading atom dictionary from file.")
                atom_dictionary = get_dictionary(dictionary_file_path)

    data_list = []
    
    for index in range(1):
        data = Data()
        
        # Read structure file using ase
        primitive_cell = processing_args.get("primitive_cell", True)
        if primitive_cell:
            try:
                ase_crystal = ase.io.read(file_name, primitive_cell=True, subtrans_included=False)
                print(len(ase_crystal), "atoms in structure (primitive)")
            except Exception as e:
                print(f"Warning: Cannot get primitive cell ({e}), using full cell")
                ase_crystal = ase.io.read(file_name)
                print(len(ase_crystal), "atoms in structure")
        else:
            ase_crystal = ase.io.read(file_name)
            print(len(ase_crystal), "atoms in structure")
        data.ase = ase_crystal

        if index == 0:
            length = [len(ase_crystal)]
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))

        # Obtain distance matrix with ase
        distance_matrix = ase_crystal.get_all_distances(mic=True)

        # Create sparse graph from distance matrix
        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]

        # Add self loops
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
        )
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        distance_matrix_mask = (
            distance_matrix_trimmed.fill_diagonal_(1) != 0
        ).int()

        data.edge_descriptor = {}
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_matrix_mask

        z = torch.LongTensor(ase_crystal.get_atomic_numbers())
        data.z = z

        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        data.u = u

        data_list.append(data)

    n_atoms_max = max(length)
    species = list(set(sum(elements, [])))
    species.sort()
    num_species = len(species)
    
    if processing_args["verbose"] == "True":
        print(
            "Max structure size:", n_atoms_max,
            "Max number of elements:", num_species,
        )
        print("Unique species:", species)
    
    crystal_length = len(ase_crystal)
    print(crystal_length, "---length---")
    data.length = torch.LongTensor([crystal_length])

    # Generate node features
    if processing_args["dictionary_source"] != "generated":
        for index in range(len(data_list)):
            # One-hot encoded atomic numbers
            atom_fea = np.vstack([
                atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                for i in range(len(data_list[index].ase))
            ]).astype(float)
            
            # Skipatom vectors
            skipatom_fea = np.vstack([
                skipatom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                for i in range(len(data_list[index].ase))
            ]).astype(float)
            
            atom_fea_tensor = torch.Tensor(atom_fea)
            skipatom_fea_tensor = torch.Tensor(skipatom_fea)
            atom_fea_total = torch.cat([atom_fea_tensor, skipatom_fea_tensor], dim=-1)
            data_list[index].x = atom_fea_total

    # Add node degree to node features
    for index in range(len(data_list)):
        data_list[index] = OneHotDegree(
            data_list[index], processing_args["graph_max_neighbors"] + 1
        )

    # Generate edge features
    if processing_args["edge_features"] == "True":
        distance_gaussian = GaussianSmearing(
            0, 1, processing_args["graph_edge_length"], 0.2
        )
        NormalizeEdge(data_list, "distance")
        for index in range(len(data_list)):
            data_list[index].edge_attr = distance_gaussian(
                data_list[index].edge_descriptor["distance"]
            )

    Cleanup(data_list, ["ase", "edge_descriptor"])

    # Save processed dataset
    print("Saving processed dataset...")
    os.makedirs(os.path.join(data_path, processed_path), exist_ok=True)
    datax, slices = InMemoryDataset.collate(data_list)
    torch.save((datax, slices), os.path.join(data_path, processed_path, "data.pt"))

    print(datax)
    print(len(data_list))
    print("...Done!")


def get_dataset(file_name, data_path, processing_args=None):
    """Fetch dataset; processes the raw data if specified.
    
    Args:
        file_name: Path to structure file
        data_path: Path to data directory
        processing_args: Processing arguments dict
        
    Returns:
        StructureDataset instance
    """
    if processing_args is None:
        processed_path = "processed"
    else:
        processed_path = processing_args.get("processed_path", "processed")
   
    process_data_spdf(file_name, data_path, processed_path, processing_args)

    transforms = GetY(index=-1)

    if os.path.exists(os.path.join(data_path, processed_path, "data.pt")):
        print("Processed dataset is found!!! Loading it!!")
        dataset = StructureDataset(
            data_path,
            processed_path,
            transforms,
        )
        print(dataset[0], "Dataset loaded")
        return dataset
    
    return None

