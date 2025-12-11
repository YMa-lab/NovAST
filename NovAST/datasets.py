import numpy as np
from scipy.spatial.distance import pdist, squareform
import scanpy as sc
import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from .utils import *


def select_cell_type_to_move(remove_type, adata,celltype_name="cell_type"):
    """
    Decide which cell‐type(s) to drop from TRAIN based on TEST:
    
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

def read_dataset(data_type=None, train_path=None, test_path=None):
    """
    Load an AnnData dataset (.h5ad) based on whether 'train' or 'test' data is requested.
    """
    # Determine which file path to use based on data_type
    if data_type == 'train' and train_path:
        file_path = train_path
    elif data_type == 'test' and test_path:
        file_path = test_path
    else:
        # Neither a valid type nor a corresponding path was provided
        raise ValueError("You must provide a valid train/test path.")

    # Read the .h5ad dataset and return as AnnData object
    adata = sc.read_h5ad(file_path)
    return adata

def load_dataset_adata(args, adata_train, adata_test):
    """
    Unified loader that handles:
      - controlled vs. uncontrolled (`args.uncontrolled`)
      - removing novel cell‐types from train/test (`args.rm_ref`)
      - removing specified cell‐types (`args.remove_celltype`, `args.remove_celltype_type`)
      - subsampling cells (`args.sampling_cells`)
      - selecting a subset of regions (`args.select`)
      - building a kNN graph if requested (`args.graph`, `args.region_name_train/test`, `args.k`)
    """

    adata_train.obs[args.celltype_name_train] = adata_train.obs[args.celltype_name_train].str.lower()
    if not args.no_gt:
        adata_test.obs[args.celltype_name_test]   = adata_test.obs[args.celltype_name_test].str.lower()

    # If “controlled,” remove “novel” cell‐types from TEST (and maybe TRAIN). ---
    if not args.uncontrolled:
        train_cts = set(adata_train.obs[args.celltype_name_train].unique())
        test_cts  = set(adata_test.obs[args.celltype_name_test].unique())

        # Drop any TEST cells whose label ∉ (train_cts ∩ test_cts)
        common_cts = train_cts & test_cts
        drop_from_test = test_cts - common_cts
        if drop_from_test:
            adata_test = adata_test[
                ~adata_test.obs[args.celltype_name_test].isin(drop_from_test)
            ].copy()

        # If rm_ref=True, also drop TRAIN cells whose label ∉ common_cts
        if args.rm_ref:
            drop_from_train = train_cts - common_cts
            if drop_from_train:
                adata_train = adata_train[
                    ~adata_train.obs[args.celltype_name_train].isin(drop_from_train)
                ].copy()

    # If remove_celltype=True, remove selected labels from TRAIN only ---
    if args.remove_celltype:
        to_move = select_cell_type_to_move(args.remove_celltype_type, adata_test, args.celltype_name_test)
        adata_train = adata_train[
            ~adata_train.obs[args.celltype_name_train].isin(to_move)
        ].copy()
        print("Removed from TRAIN:", to_move)

    # If sampling_cells is set, subsample TRAIN in proportion to TEST size ---
    if args.sampling_cells is not None:
        down_size = int(args.sampling_cells * adata_test.shape[0])
        comp = adata_train.obs[args.celltype_name_train].value_counts(normalize=True)
        downsampled_idx = []
        for ct, prop in comp.items():
            n_samples = int(prop * down_size)
            idxs = adata_train.obs[
                adata_train.obs[args.celltype_name_train] == ct
            ].index
            chosen = np.random.RandomState(1).choice(idxs, size=n_samples, replace=False)
            downsampled_idx.extend(chosen)
        adata_train = adata_train[downsampled_idx].copy()

    # Extract train_X, train_y, inverse_train
    train_X       = adata_train.X
    train_y_raw   = adata_train.obs[args.celltype_name_train]
    train_classes = np.sort(train_y_raw.unique()).tolist()
    train_map     = {ct: i for i, ct in enumerate(train_classes)}
    inverse_train = {i: ct for ct, i in train_map.items()}
    train_y       = np.array([train_map[ct] for ct in train_y_raw])

    # --- Step 6: Extract test_X, test_y, inverse_test
    test_X = adata_test.X

    if args.no_gt:
        # return empty as no ground truth provided
        n_test = test_X.shape[0]
        test_y   = np.zeros((n_test,), dtype=int)
        test_y_c = np.zeros((n_test,), dtype=float)
        test_y_raw  = np.zeros((n_test,), dtype=str)
        inverse_test = None
    else:
        test_y_raw   = adata_test.obs[args.celltype_name_test]
        test_classes = np.sort(test_y_raw.unique()).tolist()
        test_map     = {ct: i for i, ct in enumerate(test_classes)}
        inverse_test = {i: ct for ct, i in test_map.items()}
        test_y       = np.array([test_map[ct] for ct in test_y_raw])
        test_y_c     = np.array([test_map[ct] for ct in test_y_raw])
        test_y_raw   = test_y_raw.values

    if hasattr(train_X, 'toarray'):
        train_X = train_X.toarray()
    if hasattr(test_X, 'toarray'):
        test_X  = test_X.toarray()
    return (train_X, train_y, inverse_train, test_X, test_y_raw, test_y_c, inverse_test)

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