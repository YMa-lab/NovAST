
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
import scanpy as sc
from utils import *
from sklearn.neighbors import NearestNeighbors
import os, pickle, time, json, resource
from scipy.spatial.distance import pdist, squareform

def select_cell_type_to_move(remove_type, adata, savedir, celltype_name="cell_type", explicit_name=None):
    """
    Decide which cell‐type(s) to drop from the Reference based on the Target:

    - explicit_name given: use it verbatim (reproducible, skips UMAP)
    - 'most' / 'least': by raw frequency
    - 'closest' / 'furthest': by average UMAP‐centroid distance
    """
    # Explicit override: pin the held-out type(s) so the choice is reproducible
    # instead of depending on stochastic UMAP (pynndescent is version/thread-dependent).
    if explicit_name:
        names = ([n.strip() for n in explicit_name.split(",")]
                 if isinstance(explicit_name, str) else list(explicit_name))
        present = set(adata.obs[celltype_name].astype(str).unique())
        missing = [n for n in names if n not in present]
        if missing:
            raise ValueError(
                f"remove_celltype_name {missing} not found in column '{celltype_name}'. "
                f"Available: {sorted(present)}"
            )
        print(f"Explicit remove_celltype_name override -> {names}")
        return names

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

        fig = sc.pl.umap(
            adata,
            color=celltype_name, 
            return_fig=True,
            show=False
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(savedir, "umap_target.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close(fig)

        if remove_type == 'closest':
            idx = np.nanargmin(mean_dist)
        else:
            idx = np.nanargmax(mean_dist)

        return [ct_index[idx]]

    return []


def read_dataset(dataset_name=None, data_type=None, reference_path=None, target_path=None, preprocess_skip=False):
    # If using custom path
    if data_type == 'reference' and reference_path:
        file_path = reference_path
    elif data_type == 'target' and target_path:
        file_path = target_path
    else:
        raise ValueError("You must provide either dataset_name with data_type or a valid train/test path.")

    # Read the dataset
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
        to_move = select_cell_type_to_move(args.remove_celltype_type, adata_target, args.savedir, args.celltype_name_target_select, explicit_name=getattr(args, 'remove_celltype_name', None))

        total_reference = adata_reference.n_obs
        total_target  = adata_target.n_obs

        for ct in to_move:
            ct_reference_count = (adata_reference.obs[args.celltype_name_reference_select] == ct).sum()
            ct_target_count  = (adata_target.obs[args.celltype_name_target_select]   == ct).sum()
            print(
                f"Proportion of '{ct}': "
                f"Reference {ct_reference_count/total_reference:.2%} ({ct_reference_count}/{total_reference}), "
                f"Target    {ct_target_count/total_target:.2%} ({ct_target_count}/{total_target})"
            )
        
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

    # If select is provided, subselect regions from both Reference & Target ---
    if args.select is not None:
        print(f'Select type is {args.select}')
        rng = np.random.default_rng(seed=42)

        # Reference side
        regs_tr = adata_reference.obs[args.region_name_reference].unique()
        if isinstance(args.select, str) and args.select.lower() == "half":
            pick_tr = rng.choice(regs_tr, size=len(regs_tr) // 2, replace=False)
        else:
            n_keep_tr = int(args.select)
            pick_tr = adata_reference.obs[args.region_name_reference].value_counts().head(n_keep_tr).index
        adata_reference = adata_reference[
            adata_reference.obs[args.region_name_reference].isin(pick_tr)
        ].copy()

        # Target side
        regs_te = adata_target.obs[args.region_name_target].unique()
        if isinstance(args.select, str) and args.select.lower() == "half":
            pick_te = rng.choice(regs_te, size=len(regs_te) // 2, replace=False)
        else:
            n_keep_te = int(args.select)
            pick_te = adata_target.obs[args.region_name_target].value_counts().head(n_keep_te).index
        adata_target = adata_target[
            adata_target.obs[args.region_name_target].isin(pick_te)
        ].copy()

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

    if hasattr(reference_X, 'toarray'):
        reference_X = reference_X.toarray()
    if hasattr(target_X, 'toarray'):
        target_X  = target_X.toarray()

    # save the data
    # before saving, back up processed X and swap in raw counts
    if "counts" not in adata_reference.layers or "counts" not in adata_target.layers:
        raise KeyError("You must have adata.layers['counts'] populated with raw counts")

    os.makedirs(args.savedir, exist_ok=True)

    # Stage-1 emits TWO h5ad flavours of the same (already subset / filtered /
    # celltype-removed) cells, differing only in what sits in .X:
    #   *_normalized.h5ad  -> X = normalize_total+log1p+scale  (consumed by the
    #                         installed NovAST package via run_NovAST(preprocess_skip=True))
    #   *_preprocessed.h5ad -> X = raw counts (consumed by SpatialID / Tangram,
    #                         which normalize internally, and by SingleR's prepare step)
    # Both carry layers['counts'] and layers['normalized'], so either can be
    # recovered downstream.
    adata_reference.write_h5ad(os.path.join(args.savedir, "reference_normalized.h5ad"))
    adata_target.write_h5ad( os.path.join(args.savedir, "target_normalized.h5ad"))

    adata_reference.X = adata_reference.layers["counts"]
    adata_target.X  = adata_target.layers["counts"]

    adata_reference.write_h5ad(os.path.join(args.savedir, f"reference_preprocessed.h5ad"))
    adata_target.write_h5ad( os.path.join(args.savedir, f"target_preprocessed.h5ad"))

    for suffix, d in (("reference", inverse_reference), ("target", inverse_target)):
        fp = os.path.join(args.savedir, f"inverse_dict_{suffix}.pkl")
        with open(fp, "wb") as f:
            pickle.dump(d, f)

    # load the graph as needed
    if args.training_type == 'st_to_st':
        # 'radius' (tree-based, default) vs 'brute' (upstream O(n^2) pairwise matrix)
        graph_method = getattr(args, 'graph_method', 'radius') or 'radius'

        reference_pos = get_spatial_coords(adata_reference)
        t0 = time.time()
        if args.region_name_reference is not None:
            reference_slices = adata_reference.obs[args.region_name_reference].values
            reference_edges  = get_edge_index_standard_region(reference_pos, reference_slices, args.distance_thres_reference, method=graph_method)
        else:
            reference_edges = get_edge_index_standard(reference_pos, args.distance_thres_reference, method=graph_method)
        reference_build_sec = time.time() - t0

        target_pos  = get_spatial_coords(adata_target)
        t0 = time.time()
        if args.region_name_target is not None:
            target_slices = adata_target.obs[args.region_name_target].values
            target_edges  = get_edge_index_standard_region(target_pos, target_slices, args.distance_thres_target, method=graph_method)
        else:
            target_edges = get_edge_index_standard(target_pos, args.distance_thres_target, method=graph_method)
        target_build_sec = time.time() - t0

        # record graph-construction cost (peak RSS is process-wide, in KB)
        graph_stats = {
            "graph_method": graph_method,
            "n_reference": int(reference_pos.shape[0]),
            "n_target": int(target_pos.shape[0]),
            "n_reference_edges": int(len(reference_edges)),
            "n_target_edges": int(len(target_edges)),
            "reference_build_sec": reference_build_sec,
            "target_build_sec": target_build_sec,
            "total_build_sec": reference_build_sec + target_build_sec,
            "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "distance_thres_reference": float(args.distance_thres_reference),
            "distance_thres_target": float(args.distance_thres_target),
        }
        with open(os.path.join(args.savedir, "graph_build_stats.json"), "w") as f:
            json.dump(graph_stats, f, indent=2)
        print(f"[graph build] method={graph_method} "
              f"train {graph_stats['n_reference']}c/{graph_stats['n_reference_edges']}e in {reference_build_sec:.2f}s, "
              f"test {graph_stats['n_target']}c/{graph_stats['n_target_edges']}e in {target_build_sec:.2f}s", flush=True)

        return (reference_X, reference_y, reference_edges, inverse_reference, target_X, target_y_raw.values, target_y_c, target_edges, inverse_target, reference_pos, target_pos)

    else:
        return (reference_X, reference_y, inverse_reference, target_X, target_y_raw.values, target_y_c, inverse_target)