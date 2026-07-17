import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, pairwise_distances
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import random, torch, os
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data import Dataset

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def has_spatial(adata):
    return (
        "spatial" in adata.obsm
        or "X_spatial" in adata.obsm
        or "X_spatial_coords" in adata.obsm
        or {"x", "y"}.issubset(adata.obs.columns)
        or {"x_coord", "y_coord"}.issubset(adata.obs.columns)
    )

import numpy as np

def get_spatial_coords(adata, pos_key=None):
    """
    Return an (n_cells × 2) array of spatial coordinates from `adata`, by checking in this order:
      1. If pos_key is given and exists in adata.obsm, use adata.obsm[pos_key].
      2. If "spatial" exists in adata.obsm, use adata.obsm["spatial"].
      3. If "X_spatial" exists in adata.obsm, use adata.obsm["X_spatial"].
      4. If both "x" and "y" appear in adata.obs.columns, return adata.obs[["x","y"]].values.
      5. If both "x_coord" and "y_coord" appear in adata.obs.columns, return adata.obs[["x_coord","y_coord"]].values.
    Otherwise, raise an error.
    """
    # 1) If user supplied a pos_key override, try that first
    if pos_key is not None and pos_key in adata.obsm:
        coords = adata.obsm[pos_key]
        if coords.shape[1] >= 2:
            return np.asarray(coords)[:, :2]
        else:
            raise ValueError(f"adata.obsm[{pos_key!r}] exists but has shape {coords.shape}, not at least 2 columns.")

    # 2) Check the most common obsm keys
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
        if coords.shape[1] >= 2:
            return np.asarray(coords)[:, :2]
    if "X_spatial" in adata.obsm:
        coords = adata.obsm["X_spatial"]
        if coords.shape[1] >= 2:
            return np.asarray(coords)[:, :2]
    if "X_spatial_coords" in adata.obsm:
        coords = adata.obsm["X_spatial_coords"]
        if coords.shape[1] >= 2:
            return np.asarray(coords)[:, :2]

    # 3) Fall back to obs‐columns "x" & "y"
    if {"x", "y"}.issubset(adata.obs.columns):
        return adata.obs[["x", "y"]].values

    # 4) Or obs‐columns "x_coord" & "y_coord"
    if {"x_coord", "y_coord"}.issubset(adata.obs.columns):
        return adata.obs[["x_coord", "y_coord"]].values

    # 5) Nothing found
    raise ValueError(
        "No spatial coordinates found in AnnData. "
        "Tried obsm keys: "
        f"{['spatial', 'X_spatial'] + ([pos_key] if pos_key else [])}. "
        "Also looked for obs columns ['x','y'] or ['x_coord','y_coord']."
    )

# def get_edge_index_standard(pos, distance_thres):
#     # construct edge indexes in one region
#     edge_list = []
#     dists = pairwise_distances(pos)
#     dists_mask = dists < distance_thres
#     np.fill_diagonal(dists_mask, 0)
#     edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
#     return edge_list

from sklearn.neighbors import radius_neighbors_graph

def get_edge_index_standard(pos, distance_thres, method='radius'):
    """
    pos: (n,2) array
    distance_thres: radius threshold
    method: 'radius' -> tree-based radius_neighbors_graph (default, scales ~O(n log n));
            'brute'  -> upstream STELLAR's dense O(n^2) pairwise-distance matrix.

    Returns: list of [i,j] for all neighbors within threshold.
    """
    if method == 'brute':
        # Original upstream construction: full n x n distance matrix (O(n^2) mem/time).
        dists = pairwise_distances(pos)
        dists_mask = dists < distance_thres
        np.fill_diagonal(dists_mask, 0)
        return np.transpose(np.nonzero(dists_mask)).tolist()

    A = radius_neighbors_graph(pos, radius=distance_thres, mode='connectivity') # "connectivity" -> binary 0/1 edges
    A = A.maximum(A.T) # enforce symmetry
    coo = A.tocoo() # convert to coordinate form
    return list(zip(coo.row.tolist(), coo.col.tolist()))

# def get_edge_index_standard_region(pos, regions, distance_thres):
#     # construct edge indexes when there is region information
#     edge_list = []
#     regions_unique = np.unique(regions)
#     for reg in regions_unique:
#         locs = np.where(regions == reg)[0]
#         pos_region = pos[locs, :]
#         dists = pairwise_distances(pos_region)
#         dists_mask = dists < distance_thres
#         np.fill_diagonal(dists_mask, 0)
#         region_edge_list = np.transpose(np.nonzero(dists_mask)).tolist()
#         for (i, j) in region_edge_list:
#             edge_list.append([locs[i], locs[j]])
#     return edge_list

def get_edge_index_standard_region(pos, regions, distance_thres, method='radius'):
    """
    Build edges within each region using a radius-based graph.

    pos : (n,2) array of XY coordinates
    regions : length-n array of region labels (int or str)
    distance_thres : Maximum Euclidean distance for an edge.
    method : 'radius' -> tree-based radius_neighbors_graph (default);
             'brute'  -> upstream STELLAR's dense O(n^2) pairwise-distance matrix (per region).

    Returns: edge_list : List of [i, j] pairs for all neighbors within distance_thres
                (only within the same region). Both (i,j) and (j,i) are included.
    """
    edge_list = []

    for reg in np.unique(regions):
        # coords from the region
        locs = np.where(regions == reg)[0]
        coords_reg = pos[locs]

        if method == 'brute':
            # Original upstream construction: full per-region distance matrix (O(n^2)).
            dists = pairwise_distances(coords_reg)
            dists_mask = dists < distance_thres
            np.fill_diagonal(dists_mask, 0)
            for i, j in np.transpose(np.nonzero(dists_mask)):
                edge_list.append([int(locs[i]), int(locs[j])])
            continue

        # generate graph
        A = radius_neighbors_graph(coords_reg, radius=distance_thres, mode='connectivity') # "connectivity" -> binary 0/1 edges
        A = A.maximum(A.T) # enforce symmetry
        coo = A.tocoo() # convert to coordinate form

        # save edges
        for i, j in zip(coo.row, coo.col):
            edge_list.append([int(locs[i]), int(locs[j])])

    return edge_list


def calculate_optimal_accuracy_final(true_labels, cluster_labels, inverse_dict):
    clusters_to_map = {item for item in set(cluster_labels) if item not in set(inverse_dict.values())}
    true_labels_to_map = set(true_labels) - set(cluster_labels)
    
    all_labels = list(true_labels_to_map.union(clusters_to_map))
    confusion = confusion_matrix(true_labels, cluster_labels, labels=all_labels)
    
    if len(clusters_to_map) == len(true_labels_to_map):
        row_ind, col_ind = linear_sum_assignment(confusion, maximize=True)
        mapping = {all_labels[col]: all_labels[row] for row, col in zip(row_ind, col_ind)}
    else:
        mapping = {}
        assigned_clusters = set()
        for true_label in true_labels_to_map:
            true_label_index = all_labels.index(true_label)
            overlaps = confusion[true_label_index, :]
            for cluster_label in clusters_to_map:
                cluster_index = all_labels.index(cluster_label)
                if cluster_label not in assigned_clusters:
                    if overlaps[cluster_index] == overlaps.max():
                        mapping[cluster_label] = true_label
                        assigned_clusters.add(cluster_label)
                        break

    print("Mapping relationship:", mapping)
    mapped_cluster_labels = [mapping.get(label, label) for label in cluster_labels]
    return np.array(mapped_cluster_labels), mapping

class NGDataset(Dataset):
    def __init__(self, x, y):
        self.x = x  # Features
        self.y = y  # Labels

    def __len__(self):
        return len(self.x)  # Return the number of samples

    def __getitem__(self, idx):
        # Return a sample (features and corresponding label)
        sample_x = self.x[idx]
        sample_y = self.y[idx]
        return sample_x, sample_y

class Dataset(InMemoryDataset):
    def __init__(self, LX, Ly, UX, Uy):
        super().__init__('.')
        self.labeled_data   = NGDataset(x=torch.FloatTensor(LX), y=torch.LongTensor(Ly))
        self.unlabeled_data = NGDataset(x=torch.FloatTensor(UX), y=torch.LongTensor(Uy))
    def __len__(self): return 2
    def __getitem__(self, idx):
        if idx == 0: return self.labeled_data
        if idx == 1: return self.unlabeled_data
        raise IndexError(f"Index {idx} out of bounds")

class GraphDataset(InMemoryDataset):

    def __init__(self, labeled_X, labeled_y, unlabeled_X, unlabeled_y, labeled_edges, unlabeled_edges, labeled_pos, unlabeled_pos, transform=None):
        self.root = '.'
        super(GraphDataset, self).__init__(self.root, transform)
        self.labeled_data = Data(x=torch.FloatTensor(labeled_X), edge_index=torch.LongTensor(labeled_edges).T, y=torch.LongTensor(labeled_y), pos=torch.FloatTensor(labeled_pos))
        self.unlabeled_data = Data(x=torch.FloatTensor(unlabeled_X), edge_index=torch.LongTensor(unlabeled_edges).T, y=torch.LongTensor(unlabeled_y), pos=torch.FloatTensor(unlabeled_pos))

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.labeled_data, self.unlabeled_data


# ---------------------------------------------------------------------------
# Loader helper shared by the graph/embedding methods (moved here from the
# former scripts/main_functions.py). Builds a Dataset/GraphDataset from the
# pickled preprocess loader_output.
# ---------------------------------------------------------------------------
def load_complete_dataset(outputs, types, graph=False):
    if types == 'st_to_st':
        (labeled_X, labeled_y, labeled_edges, inverse_dict_reference, unlabeled_X, unlabeled_y, unlabeled_y_c, unlabeled_edges, inverse_dict_target, labeled_pos, unlabeled_pos) = outputs
    else:
        (labeled_X, labeled_y, inverse_dict_reference, unlabeled_X, unlabeled_y, unlabeled_y_c, inverse_dict_target) = outputs

    if graph:
        dataset = GraphDataset(labeled_X, labeled_y, unlabeled_X, unlabeled_y_c, labeled_edges, unlabeled_edges, labeled_pos, unlabeled_pos)
    else:
        dataset = Dataset(labeled_X, labeled_y, unlabeled_X, unlabeled_y_c)

    return dataset, labeled_y, unlabeled_y
