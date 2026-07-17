"""
run_benchmark.py — Tangram benchmark runner.

Extracted from the former shared scripts/main_functions.py so each method owns
its own benchmark entry point. Imported lazily by scripts/main.py
(stage='run', --method Tangram).
"""
import os, glob, json, resource, time
from utils import set_seed

def Tangram_main(args, adata_sc, adata_sp):
    args.savedir_Tangram = os.path.join(args.savedir, "Tangram")
    os.makedirs(args.savedir_Tangram, exist_ok=True)

    # Optional single-seed mode (env TANGRAM_SINGLE_SEED=N): run only seed N so the
    # 10 seeds can be split across parallel jobs. Unset -> all seeds 1..rounds.
    _single_seed = os.environ.get("TANGRAM_SINGLE_SEED")
    _single_seed = int(_single_seed) if _single_seed not in (None, "") else None

    # Skip if results already complete (stats JSON is written last). In single-seed
    # mode the whole-method stats guard is ignored -- the per-seed skip below decides.
    stats_path = os.path.join(args.savedir_Tangram, "Tangram_detailed_stats.json")
    if _single_seed is None and os.path.exists(stats_path):
        print(f"Tangram results already exist at {stats_path}, skipping.", flush=True)
        return

    from Tangram import run_pair, chunk_and_run  # lazy import: pulls in TF

    def _clear_cuda():
        # A failed tier (OOM mid-mapping) leaves ~15-18 GiB stuck in PyTorch's
        # caching allocator, so the next coarser tier starts on a nearly-full card
        # and also OOMs -> the cascade over-subdivides (20 -> 500 chunks). Clearing
        # between failed attempts lets each tier see a clean card.
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # we'll collect per-seed, per-strategy stats here
    all_stats = {}

    cfg = args.savedir_Tangram
    _seed_list = [_single_seed] if _single_seed is not None else range(1, args.rounds + 1)
    for i in _seed_list:
        # Per-seed skip: run_pair / chunk_and_run write `seed{i}_pred_{tag}.npy`.
        # Any successful strategy leaves at least one such file.
        if glob.glob(os.path.join(cfg, f"seed{i}_pred_*.npy")):
            print(f"Tangram seed {i} results already exist, skipping.", flush=True)
            all_stats[i] = {"status": "already_done"}
            continue

        set_seed(i)
        all_stats[i] = {}
        print(f"Seed {i}…", flush=True)

        at = adata_sc.copy()
        ae = adata_sp.copy()

        # Memory guard. On GPU a too-large mapping raises a catchable CUDA OOM,
        # so the cascade naturally falls back to chunking. On CPU (TANGRAM_DEVICE=cpu)
        # an over-budget allocation is an *uncatchable* SLURM OOM-kill that takes
        # the whole job down before any fallback runs. So we estimate the dense
        # mapping matrix (n_reference * n_target * 4 bytes * factor for grads/overhead)
        # and skip any strategy whose estimate exceeds TANGRAM_MEM_BUDGET_GB.
        # Defaults leave the original behaviour unchanged (budget = +inf).
        _mem_budget_gb = float(os.environ.get("TANGRAM_MEM_BUDGET_GB", "inf"))
        _mem_factor = float(os.environ.get("TANGRAM_MEM_FACTOR", "3.0"))

        def _est_gb(n_reference, n_target):
            return n_reference * n_target * 4 * _mem_factor / 1e9

        # helper to measure one run
        def _time_and_mem(fn, *args, **kwargs):
            t0 = time.time()
            usage0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result = fn(*args, **kwargs)
            t1 = time.time()
            usage1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return {
                "duration_s": t1 - t0,
                "peak_rss_kb": usage1 - usage0
            }, result

        # 1) whole‐adata
        _whole_gb = _est_gb(at.n_obs, ae.n_obs)
        if _whole_gb > _mem_budget_gb:
            print(f"  whole skipped: est {_whole_gb:.1f}GB > budget {_mem_budget_gb}GB",
                  flush=True)
            all_stats[i]["whole"] = {"skipped_est_gb": _whole_gb,
                                     "budget_gb": _mem_budget_gb}
        else:
            try:
                stats, _ = _time_and_mem(
                    run_pair,
                    at, ae, "whole", cfg, i,
                    args.celltype_name_reference, args.celltype_name_target
                )
                stats["est_gb"] = _whole_gb
                print("  whole OK", stats, flush=True)
                all_stats[i]["whole"] = stats
                continue
            except Exception as e:
                print("  whole failed:", e, flush=True)
                all_stats[i]["whole"] = {"error": str(e)}
                _clear_cuda()

        # 2) region‐by‐region (if configured). Iterate over Target regions only and
        #    map each against the FULL training reference. The previous version
        #    took the train x test region cross-product, which (a) predicted every
        #    test cell once per train region (duplication under mismatched
        #    references) and (b) blew up the pair count. Now each test cell is
        #    predicted exactly once, memory ~ n_reference * (largest test region).
        if args.region_name_reference and args.region_name_target \
                and args.region_name_target in ae.obs.columns:
            target_regions = list(ae.obs[args.region_name_target].unique())
            _max_region = max(
                int((ae.obs[args.region_name_target] == te).sum()) for te in target_regions
            )
            _region_gb = _est_gb(at.n_obs, _max_region)
            if _region_gb > _mem_budget_gb:
                print(f"  regions skipped: est {_region_gb:.1f}GB (full train x largest "
                      f"region) > budget {_mem_budget_gb}GB", flush=True)
                all_stats[i]["regions"] = {"skipped_est_gb": _region_gb,
                                           "budget_gb": _mem_budget_gb}
            else:
                try:
                    stats, _ = _time_and_mem(
                        lambda a1, a2: [
                            run_pair(
                                a1.copy(),
                                a2[a2.obs[args.region_name_target] == te].copy(),
                                f"te_{te}", cfg, i,
                                args.celltype_name_reference, args.celltype_name_target
                            )
                            for te in target_regions
                        ],
                        at, ae
                    )
                    stats["est_gb"] = _region_gb
                    stats["n_regions"] = len(target_regions)
                    print("  regions OK", stats, flush=True)
                    all_stats[i]["regions"] = stats
                    continue
                except Exception as e:
                    print("  regions failed:", e, flush=True)
                    all_stats[i]["regions"] = {"error": str(e)}
                    _clear_cuda()

        # 3) chunking fallbacks (spatial KMeans). TANGRAM_CHUNK_MODE selects the
        #    per-chunk memory model: full_train -> n_reference * (n_target/num);
        #    split_both -> (n_reference/num) * (n_target/num) (k-fold cheaper).
        _chunk_mode = os.environ.get("TANGRAM_CHUNK_MODE", "full_train")
        for num in (10, 20, 50, 100, 500, 1000):
            key = f"chunks_{num}"
            # per-chunk est uses the largest chunk (ceil division)
            _chunk_target = -(-ae.n_obs // num)
            _chunk_reference = (-(-at.n_obs // num)) if _chunk_mode == "split_both" else at.n_obs
            _chunk_gb = _est_gb(_chunk_reference, _chunk_target)
            if _chunk_gb > _mem_budget_gb:
                print(f"  {key} skipped: est {_chunk_gb:.1f}GB > budget {_mem_budget_gb}GB",
                      flush=True)
                all_stats[i][key] = {"skipped_est_gb": _chunk_gb,
                                     "budget_gb": _mem_budget_gb}
                continue
            try:
                stats, _ = _time_and_mem(
                    chunk_and_run,
                    at, ae, num, cfg, i,
                    args.celltype_name_reference, args.celltype_name_target,
                    slice_key=getattr(args, "region_name_target", None)
                )
                stats["est_gb"] = _chunk_gb
                print(f"  {key} OK", stats, flush=True)
                all_stats[i][key] = stats
                break
            except Exception as e:
                print(f"  {key} failed:", e, flush=True)
                all_stats[i][key] = {"error": str(e)}
                _clear_cuda()
        else:
            print("  all strategies failed", flush=True)
            all_stats[i]["failed_all"] = True

    # at the end write out per-seed stats. In single-seed mode write a per-seed file
    # so a parallel sibling job does NOT prematurely create the whole-method marker
    # (which would make the other per-seed jobs skip everything).
    if _single_seed is None:
        with open(os.path.join(cfg, "Tangram_detailed_stats.json"), "w") as fp:
            json.dump(all_stats, fp, indent=2)
    else:
        with open(os.path.join(cfg, f"Tangram_detailed_stats_seed{_single_seed}.json"), "w") as fp:
            json.dump(all_stats, fp, indent=2)
