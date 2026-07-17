#!/usr/bin/env python3
"""
singler_io.py — I/O helper for run_singler.R

Two modes:
  prepare  : h5ad → .mtx + .csv  (run before R)
  convert  : .csv → .npy         (run after R)
"""
import argparse
import csv
import os
import pickle

def prepare(savedir):
    """Read preprocessed h5ad, write matrix-market + metadata CSVs."""
    import h5py
    import numpy as np
    from scipy.io import mmwrite
    from scipy import sparse

    out = os.path.join(savedir, "SingleR_input")
    os.makedirs(out, exist_ok=True)

    for tag in ("reference", "target"):
        path = os.path.join(savedir, f"{tag}_preprocessed.h5ad")
        with h5py.File(path, "r") as f:
            # Read gene and cell names (index column name varies across h5ad files)
            var_idx = f["var"].attrs.get("_index", "_index")
            obs_idx = f["obs"].attrs.get("_index", "_index")
            genes = f[f"var/{var_idx}"][()].astype(str).tolist()
            cells = f[f"obs/{obs_idx}"][()].astype(str).tolist()

            # Read X (sparse or dense)
            if "X" in f and isinstance(f["X"], h5py.Dataset):
                # dense
                X = sparse.csc_matrix(f["X"][()].T)
            else:
                # sparse (CSR or CSC stored in X/)
                data    = f["X/data"][()]
                indices = f["X/indices"][()]
                indptr  = f["X/indptr"][()]
                shape   = (len(cells), len(genes))
                encoding = f["X"].attrs.get("encoding-type", "csr_matrix")
                if encoding == "csc_matrix":
                    X_sp = sparse.csc_matrix((data, indices, indptr), shape=shape)
                else:
                    X_sp = sparse.csr_matrix((data, indices, indptr), shape=shape)
                X = X_sp.T.tocsc()  # genes x cells for R

            # Write matrix-market
            mmwrite(os.path.join(out, f"{tag}.mtx"), X)

            # Write gene names and cell barcodes
            with open(os.path.join(out, f"{tag}_genes.csv"), "w") as gf:
                gf.write("\n".join(genes) + "\n")
            with open(os.path.join(out, f"{tag}_barcodes.csv"), "w") as bf:
                bf.write("\n".join(cells) + "\n")

            # Write obs metadata as CSV
            obs_keys = [k for k in f["obs"].keys() if k != "_index"]
            obs_data = {}
            for k in obs_keys:
                item = f[f"obs/{k}"]
                if isinstance(item, h5py.Group) and "codes" in item and "categories" in item:
                    # Categorical: codes (int array) + categories (string array)
                    codes = item["codes"][()]
                    cats  = item["categories"][()].astype(str)
                    obs_data[k] = [cats[c] if 0 <= c < len(cats) else "" for c in codes]
                elif isinstance(item, h5py.Dataset):
                    vals = item[()]
                    if "categories" in item.attrs:
                        cats = f[item.attrs["categories"]][()].astype(str)
                        obs_data[k] = [cats[v] if 0 <= v < len(cats) else "" for v in vals]
                    else:
                        obs_data[k] = [v.decode() if isinstance(v, bytes) else str(v) for v in vals]
                else:
                    continue  # skip unsupported types

            with open(os.path.join(out, f"{tag}_obs.csv"), "w", newline="") as of:
                writer = csv.writer(of)
                writer.writerow([""] + obs_keys)
                for i, cell in enumerate(cells):
                    row = [obs_data[k][i] for k in obs_keys]
                    writer.writerow([cell] + row)

        print(f"{tag}: {len(cells)} cells x {len(genes)} genes")

    # Also copy inverse_dict as CSV for R
    pkl_path = os.path.join(savedir, "inverse_dict_reference.pkl")
    with open(pkl_path, "rb") as f:
        inv = pickle.load(f)
    with open(os.path.join(out, "inverse_dict_reference.csv"), "w") as f:
        for k, v in sorted(inv.items()):
            f.write(f"{k},{v}\n")

    print(f"Prepared SingleR inputs in {out}")


def convert(savedir):
    """Read R's single CSV, write prediction.npy + ground_truth.npy."""
    import csv
    import numpy as np
    singler_dir = os.path.join(savedir, "SingleR")
    csv_path = os.path.join(singler_dir, "singler_predictions.csv")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    pred_int = np.array([int(r["pred_int"]) for r in rows])
    gt = np.array([r["ground_truth"] for r in rows])

    np.save(os.path.join(singler_dir, "prediction.npy"), pred_int)
    np.save(os.path.join(singler_dir, "ground_truth.npy"), gt)
    print(f"Saved prediction.npy + ground_truth.npy ({len(pred_int)} cells)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    p = sub.add_parser("prepare")
    p.add_argument("--savedir", required=True)

    c = sub.add_parser("convert")
    c.add_argument("--savedir", required=True)

    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.savedir)
    elif args.mode == "convert":
        convert(args.savedir)
