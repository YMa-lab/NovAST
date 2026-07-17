"""
run_benchmark.py — STELLAR benchmark runner.

Extracted from the former shared scripts/main_functions.py so each method owns
its own benchmark entry point. Imported lazily by scripts/main.py
(stage='run', --method STELLAR).
"""
import os, json, resource, time
import numpy as np
from utils import set_seed
from .STELLAR import STELLAR

def STELLAR_main(args, dataset, labeled_y, unlabeled_y):
    start_time = time.time()

    # --- vendored-code boundary -------------------------------------------
    # The benchmark says reference/target, but STELLAR.py is a VENDORED copy of
    # upstream STELLAR and reads upstream's own arg names (`args.num_parts_train`
    # / `_test`). We map onto them here instead of editing the vendored file, so
    # STELLAR_py/ stays byte-identical to upstream. This is the only place
    # 'train'/'test' survive on the STELLAR path, and it is deliberate.
    args.num_parts_train = args.num_parts_reference
    args.num_parts_test  = args.num_parts_target
    if args.num_seed_class == 0:
        args.savedir_STELLAR = os.path.join(args.savedir, "STELLAR")
    else:
        args.savedir_STELLAR = os.path.join(args.savedir, f"STELLAR_novel{args.num_seed_class}")
    os.makedirs(args.savedir_STELLAR, exist_ok=True)

    # Skip if results already complete (stats JSON is written last)
    stats_path = os.path.join(args.savedir_STELLAR, "STELLAR_stats.json")
    if os.path.exists(stats_path):
        print(f"STELLAR results already exist at {stats_path}, skipping.", flush=True)
        return

    ######## model training ##########
    for i in range(1, args.rounds + 1):
        filepath = f"{args.savedir_STELLAR}/seed{i}/"
        if (os.path.exists(os.path.join(filepath, "prediction.npy"))
                and os.path.exists(os.path.join(filepath, "ground_truth.npy"))
                and os.path.exists(os.path.join(filepath, "feat.csv"))
                and os.path.exists(os.path.join(filepath, "out_feat.csv"))):
            print(f"STELLAR seed {i} results already exist, skipping.", flush=True)
            continue

        print(f"Training started! Round {i}")
        set_seed(i)
        stellar = STELLAR(args, dataset)
        stellar.train()
        _, results, feat, out_feat  = stellar.pred()

        os.makedirs(filepath, exist_ok=True)
        np.save(f'{filepath}prediction.npy', results)
        np.save(f'{filepath}ground_truth.npy', unlabeled_y)
        np.savetxt(f'{filepath}feat.csv', feat, delimiter=",")
        np.savetxt(f'{filepath}out_feat.csv', out_feat, delimiter=",")
        print(f"Training finished! Round {i}",flush=True)

    ######## save time and memory ##########
    elapsed = time.time() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = usage.ru_maxrss

    stats = {
        "runtime_sec": elapsed,
        "peak_rss_kb": peak_rss,
        "rounds": args.rounds
    }
    with open(os.path.join(args.savedir_STELLAR, "STELLAR_stats.json"), "w") as fp:
        json.dump(stats, fp, indent=2)

    print(f"--- {elapsed:.1f}s elapsed, peak RSS={peak_rss} KB ---", flush=True)
