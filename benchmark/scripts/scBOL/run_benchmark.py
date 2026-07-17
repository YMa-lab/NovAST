"""
run_benchmark.py — scBOL benchmark runner.

Extracted from the former shared scripts/main_functions.py so each method owns
its own benchmark entry point. Imported lazily by scripts/main.py
(stage='run', --method scBOL).
"""
import os, sys, json, pickle, resource, time
import numpy as np
from utils import set_seed

def scBOL_main(args, dataset, labeled_y, unlabeled_y):
    start_time = time.time()

    # --- vendored-code boundary -------------------------------------------
    # bol_train.py is VENDORED scBOL and reads upstream's own arg names
    # (`args.num_parts_train` / `_test`). Map ours onto them here rather than
    # editing the vendored file, so scBOL/ stays faithful to upstream.
    args.num_parts_train = args.num_parts_reference
    args.num_parts_test  = args.num_parts_target
    if args.num_seed_class == 0:
        args.savedir_scBOL = os.path.join(args.savedir, "scBOL")
    else:
        args.savedir_scBOL = os.path.join(args.savedir, f"scBOL_novel{args.num_seed_class}")
    os.makedirs(args.savedir_scBOL, exist_ok=True)

    # Optional single-seed mode (env SCBOL_SINGLE_SEED=N): run only seed N so the 10
    # seeds can be split across parallel jobs. Unset -> run all seeds 1..rounds as before.
    _single_seed = os.environ.get("SCBOL_SINGLE_SEED")
    _single_seed = int(_single_seed) if _single_seed not in (None, "") else None

    # Skip if results already complete (stats JSON is written last). In single-seed mode
    # the whole-method stats guard is ignored -- the per-seed guard inside the loop decides.
    stats_path = os.path.join(args.savedir_scBOL, "scBOL_stats.json")
    if _single_seed is None and os.path.exists(stats_path):
        print(f"scBOL results already exist at {stats_path}, skipping.", flush=True)
        return

    # Load global inverse dicts
    inverse_dict_reference = pickle.load(open(os.path.join(args.savedir, "inverse_dict_reference.pkl"), "rb"))

    # Map benchmark args -> BOL-expected args
    # shared_classes = known classes from training; total_classes = shared + user-specified novel slots
    args.shared_classes = len(inverse_dict_reference)
    args.total_classes  = args.shared_classes + args.num_seed_class
    args.inverse_dict   = inverse_dict_reference
    args.labeled_pos    = dataset.labeled_data.pos.numpy()
    args.unlabeled_pos  = dataset.unlabeled_data.pos.numpy()

    # Option C: rebuild a SPARSER spatial graph for scBOL ONLY (env SCBOL_DISTANCE_THRES).
    # Some panels (e.g. cosmx) have coordinates on a much smaller scale, so the shared
    # radius-50 graph has ~460 neighbours/cell; scBOL's pred() forwards the whole graph
    # -> a 91.88 GiB CUDA OOM. Rebuilding at a smaller radius here is scBOL-internal and
    # leaves the shared graph (used by NovAST/STELLAR) untouched. Default off.
    _scbol_thres = os.environ.get("SCBOL_DISTANCE_THRES")
    if _scbol_thres not in (None, ""):
        import torch
        from utils import get_edge_index_standard
        _thr = float(_scbol_thres)
        _le = get_edge_index_standard(dataset.labeled_data.pos.numpy(),   _thr, method='radius')
        _ue = get_edge_index_standard(dataset.unlabeled_data.pos.numpy(), _thr, method='radius')
        dataset.labeled_data.edge_index   = torch.LongTensor(_le).T.contiguous()
        dataset.unlabeled_data.edge_index = torch.LongTensor(_ue).T.contiguous()
        print(f"[scBOL] Option C: rebuilt graph at distance_thres={_thr} -> "
              f"labeled {dataset.labeled_data.edge_index.shape[1]} edges, "
              f"unlabeled {dataset.unlabeled_data.edge_index.shape[1]} edges", flush=True)

    # Build scBOL-specific inverse_dict_target: shared classes keep names, novel slots get placeholder names
    inverse_dict_target_scbol = dict(inverse_dict_reference)
    for k in range(args.shared_classes, args.total_classes):
        inverse_dict_target_scbol[k] = f"novel_{k - args.shared_classes}"

    # Save scBOL's own inverse dicts into its output folder (global ones are untouched)
    with open(os.path.join(args.savedir_scBOL, "inverse_dict_reference.pkl"), "wb") as f:
        pickle.dump(inverse_dict_reference, f)
    with open(os.path.join(args.savedir_scBOL, "inverse_dict_target.pkl"), "wb") as f:
        pickle.dump(inverse_dict_target_scbol, f)

    # BOL uses args.lr / args.wd directly; map from scBOL-prefixed args
    _orig_lr, _orig_wd = args.lr, args.wd
    args.lr = args.scbol_lr
    args.wd = args.scbol_wd

    # Make scBOL subpackage importable
    import sys
    scbol_dir = os.path.dirname(__file__)
    if scbol_dir not in sys.path:
        sys.path.insert(0, scbol_dir)
    from bol_train import BOL

    ######## model training ##########
    _seed_list = [_single_seed] if _single_seed is not None else range(1, args.rounds + 1)
    for i in _seed_list:
        filepath = os.path.join(args.savedir_scBOL, f"seed{i}")
        if (os.path.exists(os.path.join(filepath, "prediction.npy"))
                and os.path.exists(os.path.join(filepath, "ground_truth.npy"))
                and os.path.exists(os.path.join(filepath, "out_feat.npy"))):
            print(f"scBOL Seed {i} results already exist, skipping.", flush=True)
            continue

        set_seed(i)
        print(f"scBOL Seed {i} started", flush=True)

        scbol = BOL(args, dataset)
        scbol.train()
        results, out_feat = scbol.pred(100)

        os.makedirs(filepath, exist_ok=True)
        np.save(f'{filepath}/prediction.npy', results)
        np.save(f'{filepath}/ground_truth.npy', unlabeled_y)
        np.save(f'{filepath}/out_feat.npy', out_feat)
        print(f"scBOL Seed {i} complete", flush=True)

    # Restore original lr/wd so other methods are not affected
    args.lr, args.wd = _orig_lr, _orig_wd

    ######## save time and memory ##########
    elapsed = time.time() - start_time
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = usage.ru_maxrss

    stats = {
        "runtime_sec": elapsed,
        "peak_rss_kb": peak_rss,
        "rounds": args.rounds
    }
    if _single_seed is None:
        # full run -> write the whole-method completion marker (skip guard for reruns)
        with open(stats_path, "w") as fp:
            json.dump(stats, fp, indent=2)
    else:
        # single-seed job -> per-seed stats only, so it does NOT prematurely mark the
        # whole method done (which would make sibling per-seed jobs skip everything).
        with open(os.path.join(args.savedir_scBOL, f"scBOL_stats_seed{_single_seed}.json"), "w") as fp:
            json.dump(stats, fp, indent=2)

    print(f"--- {elapsed:.1f}s elapsed, peak RSS={peak_rss} KB ---", flush=True)
