"""
run_benchmark.py — SpatialID benchmark runner.

Extracted from the former shared scripts/main_functions.py so each method owns
its own benchmark entry point. Imported lazily by scripts/main.py
(stage='run', --method SpatialID).
"""
import os, json, resource, time
import numpy as np
from utils import set_seed
from .main_functions import spatial_classification_tool

def SpatialID_main(args, adata_sc, adata_sp):
    start_time = time.time()

    # --- vendored-code boundary -------------------------------------------
    # SpatialID/main_functions.py::spatial_classification_tool reads
    # `args.celltype_name_train`. Map ours onto it here rather than editing the
    # vendored file. (`args.dataset_train` is also referenced there, but only
    # inside a commented-out block, so it needs no mapping.)
    args.celltype_name_train = args.celltype_name_reference
    args.savedir_SpatialID = os.path.join(args.savedir, "SpatialID")
    os.makedirs(args.savedir_SpatialID, exist_ok=True)

    # Skip if results already complete (stats JSON is written last)
    stats_path = os.path.join(args.savedir_SpatialID, "SpatialID_stats.json")
    if os.path.exists(stats_path):
        print(f"SpatialID results already exist at {stats_path}, skipping.", flush=True)
        return

    ######## model training ##########
    for i in range(1, args.rounds + 1):
        seed_dir = f'{args.savedir_SpatialID}/seed{i}/'
        if (os.path.exists(os.path.join(seed_dir, "annotation.h5ad"))
                and os.path.exists(os.path.join(seed_dir, "model.csv"))):
            print(f"SpatialID seed {i} results already exist, skipping.", flush=True)
            continue

        set_seed(i)
        print(f'Seed {i} started',flush=True)
        adata_reference = adata_sc.copy()
        adata_target = adata_sp.copy()
        adata_reference.X = adata_reference.X.astype(np.float32)
        adata_target.X = adata_target.X.astype(np.float32)
        spatial_classification_tool(adata_target, adata_reference, args, seed_dir)
        print(f"Seed {i} complete",flush=True)

    
    ######## save time and memory ##########
    elapsed = time.time() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = usage.ru_maxrss

    stats = {
        "runtime_sec": elapsed,
        "peak_rss_kb": peak_rss,
        "rounds": args.rounds
    }
    with open(os.path.join(args.savedir_SpatialID, "SpatialID_stats.json"), "w") as fp:
        json.dump(stats, fp, indent=2)

    print(f"--- {elapsed:.1f}s elapsed, peak RSS={peak_rss} KB ---", flush=True)
