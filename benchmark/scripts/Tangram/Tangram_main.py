import argparse
import time
import tangram as tg
import pandas as pd
import numpy as np
import scanpy as sc
import os
import gc
import random
import torch
from sklearn.cluster import KMeans


def _device():
    """Mapping device. Override with env TANGRAM_DEVICE (e.g. 'cpu')."""
    return os.environ.get("TANGRAM_DEVICE", "cuda:0")


def run_pair(at, ae, tag, savedir, seed, celltype_name_reference, celltype_name_target):
    ###Run Tangram on a train/test pair, save GT & predictions.
    # seed & preprocessing
    np.random.seed(seed)
    tg.pp_adatas(at, ae, genes=None)
    # Mapping epochs configurable via TANGRAM_EPOCHS (default 1000 = tangram default).
    # The mapping score is ~98% converged by epoch 300 on these datasets, so lowering
    # it (e.g. 300) gives a ~3x speedup with negligible change to the assignment.
    _epochs = int(os.environ.get("TANGRAM_EPOCHS", "1000"))
    ad_map = tg.map_cells_to_space(at, ae, device=_device(), num_epochs=_epochs)
    tg.project_cell_annotations(ad_map, ae, annotation=celltype_name_reference)

    # extract & save
    mat   = ae.obsm["tangram_ct_pred"].values
    names = np.array(ae.obsm["tangram_ct_pred"].columns)
    preds = names[np.argmax(mat, axis=1)]
    gt    = ae.obs[celltype_name_target].values

    os.makedirs(savedir, exist_ok=True)
    np.save(os.path.join(savedir, f"seed{seed}_gt_{tag}.npy"),   gt)
    np.save(os.path.join(savedir, f"seed{seed}_pred_{tag}.npy"), preds)

    # Free GPU memory between pairs. The mapper's tensors are dereferenced when
    # this returns, but PyTorch's caching allocator keeps the freed blocks, so
    # sequential region/chunk pairs would otherwise accumulate until CUDA OOM
    # (the failure that knocked the regions tier down to tiny chunks).
    del ad_map, mat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _kmeans_groups(adata, k, seed, slice_key=None):
    """k spatial KMeans clusters (index arrays) on obsm['spatial'], or a seeded
    shuffle split if there are no coordinates.

    Multi-slice correctness: when `slice_key` names an obs column with >1 group
    (e.g. donor_id / brain_section_label), the slides share/overlap coordinate
    frames, so a GLOBAL KMeans would put cells from physically separate sections
    into the same chunk. Instead we KMeans WITHIN each slice and split k across
    slices proportionally to size, so every chunk stays inside one slice."""
    if "spatial" not in adata.obsm:
        print("  [chunk] no obsm['spatial']; falling back to seeded shuffle split", flush=True)
        rng = np.random.RandomState(seed)
        return list(np.array_split(rng.permutation(adata.n_obs), k))

    coords = np.asarray(adata.obsm["spatial"])[:, :2].astype(float)

    if slice_key is None or slice_key not in adata.obs or adata.obs[slice_key].nunique() <= 1:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(coords)
        return [np.where(labels == j)[0] for j in range(k)]

    # multi-slice: cluster within each slice; distribute k proportionally
    groups = []
    n = adata.n_obs
    slabels = adata.obs[slice_key].astype(str).values
    for s in sorted(np.unique(slabels)):
        idx = np.where(slabels == s)[0]
        ks = max(1, int(round(k * len(idx) / n)))
        ks = min(ks, len(idx))
        if ks <= 1:
            groups.append(idx)
        else:
            sub = KMeans(n_clusters=ks, n_init=10, random_state=seed).fit_predict(coords[idx])
            for j in range(ks):
                g = idx[sub == j]
                if len(g) > 0:
                    groups.append(g)
    print(f"  [chunk] slice-aware: {adata.obs[slice_key].nunique()} slices -> "
          f"{len(groups)} within-slice chunks (target k={k})", flush=True)
    return groups


def chunk_and_run(at, ae, num_chunks, cfg_savedir, seed, celltype_reference, celltype_target,
                  slice_key=None):
    """
    Spatially partition into `num_chunks` KMeans clusters and run Tangram per
    chunk. Replaces the previous positional ``np.array_split(np.arange(n))``
    split (which inherited the .h5ad row order and, for cell-type-sorted files,
    produced label-homogeneous chunks that collapsed accuracy).

    Mode via env TANGRAM_CHUNK_MODE (default 'full_train'):
      * full_train  — KMeans the Target slide only; each test cluster is mapped
                      against the FULL train reference. Most accurate; memory
                      ~ n_reference * (n_target / k); total compute ~ n_reference * n_target.
      * split_both  — KMeans BOTH slides into k clusters and pair cluster j 1:1.
                      Memory ~ (n_reference/k) * (n_target/k); total compute ~
                      n_reference * n_target / k (k-fold cheaper, far fewer chunks
                      needed). Spatial clusters are cell-type-diverse so the
                      paired reference still covers the query's types; the
                      cross-slide pairing is the accuracy trade-off being
                      validated. If a train cluster is empty, that test chunk
                      falls back to the full train reference.

    Raises if *any* chunk fails.
    """
    mode = os.environ.get("TANGRAM_CHUNK_MODE", "full_train")
    te_groups = _kmeans_groups(ae, num_chunks, seed, slice_key=slice_key)
    tr_groups = _kmeans_groups(at, num_chunks, seed) if mode == "split_both" else None

    for j, te in enumerate(te_groups, start=1):
        if len(te) == 0:
            continue
        if mode == "split_both":
            tr = tr_groups[j - 1]
            at_sub = (at[tr] if len(tr) > 0 else at).copy()
        else:
            at_sub = at.copy()       # pp_adatas mutates in place; isolate per chunk
        ae_sub = ae[te].copy()
        tag = f"te_{j}"
        # Persist this chunk's exact test-cell indices so the plot loader can map
        # per-chunk predictions back to the right cells without re-deriving the
        # (now slice-aware) KMeans grouping.
        os.makedirs(cfg_savedir, exist_ok=True)
        np.save(os.path.join(cfg_savedir, f"seed{seed}_idx_{tag}.npy"),
                np.asarray(te, dtype=np.int64))
        # this will raise if something goes wrong
        run_pair(at_sub, ae_sub, tag, cfg_savedir, seed,
                 celltype_reference, celltype_target)