"""Data utility functions for DOS-GCNN."""

import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata
from torch_geometric.utils import degree


def split_data(dataset, train_ratio, val_ratio, test_ratio, seed=None, save=False):
    """Basic train, val, test split."""
    if seed is None:
        seed = np.random.randint(1, 1e6)
    
    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <= 1:
        train_length = int(dataset_size * train_ratio)
        val_length = int(dataset_size * val_ratio)
        test_length = int(dataset_size * test_ratio)
        unused_length = dataset_size - train_length - val_length - test_length
        
        train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(
            dataset,
            [train_length, val_length, test_length, unused_length],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            "train length:", train_length,
            "val length:", val_length,
            "test length:", test_length,
            "unused length:", unused_length,
            "seed:", seed,
        )
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")
        return None


def split_data_CV(dataset, num_folds=5, seed=None, save=False):
    """Split for n-fold cross validation."""
    if seed is None:
        seed = np.random.randint(1, 1e6)
    
    dataset_size = len(dataset)
    fold_length = int(dataset_size / num_folds)
    unused_length = dataset_size - fold_length * num_folds
    folds = [fold_length for _ in range(num_folds)]
    folds.append(unused_length)
    cv_dataset = torch.utils.data.random_split(
        dataset, folds, generator=torch.Generator().manual_seed(seed)
    )
    print("fold length:", fold_length, "unused length:", unused_length, "seed:", seed)
    return cv_dataset[0:num_folds]


def Cleanup(data_list, entries):
    """Deletes unnecessary data due to slow dataloader."""
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


class GetY:
    """Get specified y index from data.y."""
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        if self.index != -1:
            data.y = data.y[0][self.index]
        return data


def get_dictionary(dictionary_file):
    """Obtain dictionary file for elemental features."""
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    """Selects edges with distance threshold and limited number of neighbors."""
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    
    if not reverse:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    else:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if not adj:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    else:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr


def get_dos_features(x, dos):
    """Get DoS features."""
    dos = torch.abs(dos)
    
    center = torch.sum(x * dos, axis=1) / torch.sum(dos, axis=1)
    x_offset = torch.repeat_interleave(x[np.newaxis, :], dos.shape[0], axis=0) - center[:, None]
    width = torch.diagonal(torch.mm((x_offset**2), dos.T)) / torch.sum(dos, axis=1)
    skew = torch.diagonal(torch.mm((x_offset**3), dos.T)) / torch.sum(dos, axis=1) / (width**(1.5))
    kurtosis = torch.diagonal(torch.mm((x_offset**4), dos.T)) / torch.sum(dos, axis=1) / (width**2)
    
    zero_index = torch.abs(x - 0).argmin().long()
    ef_states = torch.sum(dos[:, zero_index-20:zero_index+20], axis=1) * abs(x[0] - x[1])
    
    return torch.stack((center, width, skew, kurtosis, ef_states), axis=1)


def get_dos_value(x, coef, mean, sigma, rank, bas_type):
    """Get DoS value using basis expansion.
    
    dos(x) = sum_i C_i * exp( -(mean_i-x)^2 / sigma_i)  for Gauss
    dos(x) = sum_i C_i * sigma / ((x-mean)^2 + sigma^2) for Lorentz
    """
    s1 = mean.shape[0]
    s2 = mean.shape[1]
    s3 = x.shape[0]

    if bas_type == "Gauss":
        mean_ex = mean.expand(s3, s1, s2).transpose(1, 0)
        sigma_ex = sigma.expand(s3, s1, s2).transpose(1, 0)
        coef_ex = coef.expand(s3, s1, s2).transpose(1, 0)
        x_ex = x.expand(s2, s3).T
        
        A = mean_ex - x_ex
        A_sq = A * A
        sigma_ex_sq = sigma_ex * sigma_ex
        B = A_sq / (sigma_ex_sq + 0.00001)
        C = torch.exp(-1.0 * B)
        D = coef_ex * C
        dos_value = D.sum(axis=2)
        return dos_value

    if bas_type == "Lorentz":
        mean_ex = mean.expand(s3, s1, s2).transpose(1, 0)
        sigma_ex = sigma.expand(s3, s1, s2).transpose(1, 0)
        coef_ex = coef.expand(s3, s1, s2).transpose(1, 0)
        x_ex = x.expand(s2, s3).T
        
        A = mean_ex - x_ex
        A_sq = A * A
        sigma_ex_sq = sigma_ex * sigma_ex
        B = A_sq + sigma_ex_sq
        B1 = torch.sqrt(sigma_ex_sq)
        C = B1 / (B + 0.00005)
        D = coef_ex * C
        D = D / 3.14159
        dos_value = D.sum(axis=2)
        return dos_value


def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    """Obtain node degree in one-hot representation."""
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data


class GaussianSmearing(torch.nn.Module):
    """Gaussian smearing for edge features."""
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def GetRanges(dataset, descriptor_label):
    """Get min/max ranges for normalized edges."""
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max


def NormalizeEdge(dataset, descriptor_label):
    """Normalizes edges."""
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)
    feature_max = 7.9998

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
            data.edge_descriptor[descriptor_label] - feature_min
        ) / (feature_max - feature_min)

