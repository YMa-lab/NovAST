import numpy as np
from scipy.spatial.distance import pdist, squareform
import scanpy as sc
import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from .utils import *


def select_cell_type_to_move(remove_type, adata,celltype_name="cell_type"):
    """
    Decide which cell‐type(s) to drop from the Reference based on the Target:
    
    - 'most' / 'least': by raw frequency
    - 'closest' / 'furthest': by average UMAP‐centroid distance
    """
    if remove_type in ('most', 'least'):
        freq = adata.obs[celltype_name].value_counts()
        if remove_type == 'most':
            return [freq.idxmax()]
        else:
            return [freq.idxmin()]

    elif remove_type in ('closest', 'furthest'):
        adata.obsm.pop('X_pca',  None)
        adata.obsm.pop('X_umap', None)
        sc.tl.pca(adata, n_comps=50, random_state=0)
        sc.pp.neighbors(adata, random_state=0)
        sc.tl.umap(adata, random_state=0)

        umap = adata.obsm['X_umap']
        labels = adata.obs[celltype_name].values
        df = pd.DataFrame(umap, columns=['UMAP1','UMAP2'])
        df['cell_type'] = labels

        # Compute centroids per cell_type
        centroids = df.groupby('cell_type')[['UMAP1','UMAP2']].mean()

        # Pairwise distances between centroids
        dists = squareform(pdist(centroids.values, metric='euclidean'))
        np.fill_diagonal(dists, np.nan)

        mean_dist = np.nanmean(dists, axis=1)
        ct_index = centroids.index

        if remove_type == 'closest':
            idx = np.nanargmin(mean_dist)
        else:
            idx = np.nanargmax(mean_dist)

        return [ct_index[idx]]

    return []

def read_dataset(data_type=None, reference_path=None, target_path=None):
    """
    Load an AnnData dataset (.h5ad) based on whether 'train' or 'test' data is requested.
    """
    # Determine which file path to use based on data_type
    if data_type == 'reference' and reference_path:
        file_path = reference_path
    elif data_type == 'target' and target_path:
        file_path = target_path
    else:
        # Neither a valid type nor a corresponding path was provided
        raise ValueError("You must provide a valid train/test path.")

    # Read the .h5ad dataset and return as AnnData object
    adata = sc.read_h5ad(file_path)
    return adata

def load_dataset_adata(args, adata_reference, adata_target):
    """
    Unified loader that handles:
      - controlled vs. uncontrolled (`args.uncontrolled`)
      - removing novel cell‐types from train/test (`args.rm_ref`)
      - removing specified cell‐types (`args.remove_celltype`, `args.remove_celltype_type`)
      - subsampling cells (`args.sampling_cells`)
      - selecting a subset of regions (`args.select`)
      - building a kNN graph if requested (`args.graph`, `args.region_name_reference/test`, `args.k`)
    """

    adata_reference.obs[args.celltype_name_reference] = adata_reference.obs[args.celltype_name_reference].str.lower()
    if not args.no_gt:
        adata_target.obs[args.celltype_name_target]   = adata_target.obs[args.celltype_name_target].str.lower()

    # If “controlled,” remove “novel” cell‐types from Target (and maybe Reference). ---
    if not args.uncontrolled:
        reference_cts = set(adata_reference.obs[args.celltype_name_reference].unique())
        target_cts  = set(adata_target.obs[args.celltype_name_target].unique())

        # Drop any Target cells whose label ∉ (reference_cts ∩ target_cts)
        common_cts = reference_cts & target_cts
        drop_from_target = target_cts - common_cts
        if drop_from_target:
            adata_target = adata_target[
                ~adata_target.obs[args.celltype_name_target].isin(drop_from_target)
            ].copy()

        # If rm_ref=True, also drop Reference cells whose label ∉ common_cts
        if args.rm_ref:
            drop_from_reference = reference_cts - common_cts
            if drop_from_reference:
                adata_reference = adata_reference[
                    ~adata_reference.obs[args.celltype_name_reference].isin(drop_from_reference)
                ].copy()

    # If remove_celltype=True, remove selected labels from the Reference only ---
    if args.remove_celltype:
        to_move = select_cell_type_to_move(args.remove_celltype_type, adata_target, args.celltype_name_target_select)

        adata_reference = adata_reference[
            ~adata_reference.obs[args.celltype_name_reference_select].isin(to_move)
        ].copy()
        print("Removed from the Reference:", to_move)

    # If sampling_cells is set, subsample the Reference in proportion to Target size ---
    if args.sampling_cells is not None:
        down_size = int(args.sampling_cells * adata_target.shape[0])
        comp = adata_reference.obs[args.celltype_name_reference].value_counts(normalize=True)
        downsampled_idx = []
        for ct, prop in comp.items():
            n_samples = int(prop * down_size)
            idxs = adata_reference.obs[
                adata_reference.obs[args.celltype_name_reference] == ct
            ].index
            chosen = np.random.RandomState(1).choice(idxs, size=n_samples, replace=False)
            downsampled_idx.extend(chosen)
        adata_reference = adata_reference[downsampled_idx].copy()

    # Extract reference_X, reference_y, inverse_reference
    reference_X       = adata_reference.X
    reference_y_raw   = adata_reference.obs[args.celltype_name_reference]
    reference_classes = np.sort(reference_y_raw.unique()).tolist()
    reference_map     = {ct: i for i, ct in enumerate(reference_classes)}
    inverse_reference = {i: ct for ct, i in reference_map.items()}
    reference_y       = np.array([reference_map[ct] for ct in reference_y_raw])

    # --- Step 6: Extract target_X, target_y, inverse_target
    target_X = adata_target.X

    if args.no_gt:
        # return empty as no ground truth provided
        n_target = target_X.shape[0]
        target_y   = np.zeros((n_target,), dtype=int)
        target_y_c = np.zeros((n_target,), dtype=float)
        target_y_raw  = np.zeros((n_target,), dtype=str)
        inverse_target = None
    else:
        target_y_raw   = adata_target.obs[args.celltype_name_target]
        target_classes = np.sort(target_y_raw.unique()).tolist()
        target_map     = {ct: i for i, ct in enumerate(target_classes)}
        inverse_target = {i: ct for ct, i in target_map.items()}
        target_y       = np.array([target_map[ct] for ct in target_y_raw])
        target_y_c     = np.array([target_map[ct] for ct in target_y_raw])
        target_y_raw   = target_y_raw.values

    if hasattr(reference_X, 'toarray'):
        reference_X = reference_X.toarray()
    if hasattr(target_X, 'toarray'):
        target_X  = target_X.toarray()
    return (reference_X, reference_y, inverse_reference, target_X, target_y_raw, target_y_c, inverse_target, adata_reference, adata_target)

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