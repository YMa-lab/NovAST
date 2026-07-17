"""
run_benchmark.py — NovAST benchmark runner.

Unlike the other method folders, NO NovAST algorithm code is vendored here.
NovAST is a first-party *installed package* (see README.md for installation);
this file is only the thin benchmark adapter that calls its public API,
`NovAST.run_NovAST(...)`, on the shared stage-1 preprocessed data.

Dispatched by ../main.py (stage='run', --method NovAST).
"""
import os
import json
import time
import shutil
import resource


def _looks_like_raw_counts(path, n_sample=200):
    """Peek at .X and report whether it still looks like raw integer counts.

    This mirrors, on purpose, the exact test NovAST's own
    `preprocess.normalize_log_scale()` uses (`np.allclose(vals, round(vals))`),
    so it predicts what the package *would* do to this file.

    Returns True (counts), False (already normalized), or None (undetermined).
    """
    import numpy as np
    import scipy.sparse as sp
    import scanpy as sc

    try:
        adata = sc.read_h5ad(path, backed="r")
        try:
            X = adata[: min(n_sample, adata.n_obs)].to_memory().X
        finally:
            try:
                adata.file.close()
            except Exception:
                pass
        vals = X.data if sp.issparse(X) else np.asarray(X).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return bool(np.allclose(vals, np.round(vals), atol=1e-6))
    except Exception as e:  # never block a run on the check itself
        print(f"[NovAST] WARNING: could not inspect {path} ({e}); "
              "skipping the normalization sanity check.", flush=True)
        return None


def _assert_normalization_invariant(reference_path, preprocess_skip):
    """Fail loudly on either half of the normalize-once contract.

    The benchmark normalizes in stage 1 and NovAST can also normalize itself,
    so exactly one of them must do it. Both failure modes are SILENT at runtime,
    which is why they are checked here:

      * double-normalize -- normalized data + package preprocess ON.
        `normalize_log_scale` skips normalize_total/log1p on non-integer data
        but calls `sc.pp.scale` UNCONDITIONALLY, so the data gets scaled twice.
      * never-normalize  -- raw counts + package preprocess OFF: NovAST would
        train directly on counts.
    """
    raw = _looks_like_raw_counts(reference_path)
    if raw is None:
        return

    if raw and preprocess_skip is True:
        raise ValueError(
            f"{os.path.basename(reference_path)} still holds RAW COUNTS but "
            "preprocess_skip=True, so NovAST would train on un-normalized data. "
            "Pass the stage-1 '*_normalized.h5ad', or set preprocess_skip=False "
            "to let the package normalize the counts itself."
        )
    if (not raw) and preprocess_skip is not True:
        raise ValueError(
            f"{os.path.basename(reference_path)} is ALREADY normalized (stage 1) but "
            "preprocess_skip is not True, so NovAST's normalize_log_scale would "
            "run again -- it skips normalize_total/log1p on non-integer data yet "
            "always applies sc.pp.scale, silently scaling the data twice. "
            "Keep preprocess_skip=True for stage-1 normalized input."
        )
    print(f"[NovAST] normalization check OK: {os.path.basename(reference_path)} is "
          f"{'raw counts' if raw else 'already normalized'}, "
          f"preprocess_skip={preprocess_skip}.", flush=True)


def NovAST_main(args, overrides=None):
    """Run the installed NovAST package on this benchmark's stage-1 outputs.

    Method hyperparameters (gamma/beta/mmf_k/mmf_margin/latent_dim/...) are NOT
    set here on purpose: they come from the package's own `default_config.yaml`,
    so the benchmark measures exactly what a user gets from a fresh install.
    Pass `overrides` (a dict) to change any of them for an ablation.

    Only dataset-level wiring is overridden, and each override exists for a
    reason -- stage 1 has ALREADY done that work for every method, so NovAST
    must not redo it (which would silently de-synchronise it from the others):

      preprocess_skip=True   stage 1 already did normalize_total+log1p+scale;
                             we hand it `*_normalized.h5ad` (X = normalized).
      remove_celltype=False  stage 1 already dropped the held-out cell type from
                             Reference. Re-running the package's selector would drop
                             a *second* type (and its `select_cell_type_to_move`
                             has no `remove_celltype_name` pin, so closest/
                             furthest could pick a different type than the other
                             methods got).
      uncontrolled=True      stage 1 already applied the controlled filter (drop
                             Target cells whose label is absent from the Reference) BEFORE
                             removing the held-out type. Letting the package
                             re-apply it now would delete the held-out type from
                             Target -- i.e. delete the novel cells the run exists
                             to find.
    """
    from NovAST import run_NovAST  # installed package (see README.md)

    start_time = time.time()
    savedir_novast = os.path.join(args.savedir, "NovAST")
    os.makedirs(savedir_novast, exist_ok=True)

    # Skip if results already complete (stats JSON is written last)
    stats_path = os.path.join(savedir_novast, "NovAST_stats.json")
    if os.path.exists(stats_path):
        print(f"NovAST results already exist at {stats_path}, skipping.", flush=True)
        return

    # `load_paired_dataset` writes inverse_dict_reference.pkl itself, but it only
    # writes inverse_dict_target.pkl when uncontrolled == False -- and we must pass
    # uncontrolled=True (see docstring). identify_novel_cells_evaluation reads
    # inverse_dict_target.pkl, so seed it from stage 1's copy. Stage 1 derives it
    # from the same test cells, so the two agree.
    for fname in ("inverse_dict_reference.pkl", "inverse_dict_target.pkl"):
        src = os.path.join(args.savedir, fname)
        dst = os.path.join(savedir_novast, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    reference_path = os.path.join(args.savedir, "reference_normalized.h5ad")
    target_path  = os.path.join(args.savedir, "target_normalized.h5ad")
    for p in (reference_path, target_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run stage 1 (main.py --stage preprocess) first."
            )

    cfg = dict(
        # run_NovAST does savedir = join(savedir, name) -> <run>/NovAST,
        # so seeds land in <run>/NovAST/seed{i}, matching the other methods.
        savedir=args.savedir,
        name="NovAST",
        reference_path=reference_path,
        target_path=target_path,
        training_mode="evaluation",
        rounds=args.rounds,
        seed=args.seed,
        celltype_name_reference=args.celltype_name_reference,
        celltype_name_target=args.celltype_name_target,
        # stage 1 already did these -- see docstring
        preprocess_skip=True,
        remove_celltype=False,
        uncontrolled=True,
    )
    if overrides:
        cfg.update(overrides)

    # Enforce normalize-exactly-once. Checked AFTER overrides so a stray
    # {"preprocess_skip": False} (or a reference_path pointed at the counts h5ad)
    # fails loudly here instead of silently double-scaling the input.
    _assert_normalization_invariant(cfg["reference_path"], cfg.get("preprocess_skip"))

    run_NovAST(**cfg)

    ######## save time and memory ##########
    elapsed = time.time() - start_time
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    stats = {
        "runtime_sec": elapsed,
        "peak_rss_kb": peak_rss,
        "rounds": args.rounds,
    }
    with open(stats_path, "w") as fp:
        json.dump(stats, fp, indent=2)

    print(f"--- {elapsed:.1f}s elapsed, peak RSS={peak_rss} KB ---", flush=True)
