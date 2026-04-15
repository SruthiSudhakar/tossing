#!/usr/bin/env python3
"""Evaluation sweep for the VLM-only loop.

Sweeps probe budgets across a catalog of objects, records per-episode results,
and optionally computes the CEM-on-true-physics ceiling for comparison.

Usage:
    MUJOCO_GL=egl PYTHONPATH=. python scripts/eval_vlm_only.py \
        --catalog catalog.json --budgets 0,3 --n-objects 10 \
        --provider claude --include-oracle \
        --output outputs/vlm_only/first_eval
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from tossing.objects.catalog import load_catalog
from tossing.env import TossEnv
from tossing.oracle.dataset import cem_optimal_throw

# Register probes.
import tossing.probes.vertical_toss  # noqa: F401
import tossing.probes.forward_toss  # noqa: F401
import tossing.probes.release_drop  # noqa: F401
import tossing.probes.wrist_flick  # noqa: F401
import tossing.probes.shake  # noqa: F401

from tossing.vlm import build_client, run_episode


def _serialize_episode(obj, distance, budget, seed, result) -> dict:
    return {
        "object": obj.name,
        "family": obj.family,
        "distance": distance,
        "budget": budget,
        "seed": seed,
        "success": bool(result.success),
        "distance_to_basket": float(result.distance_to_basket),
        "n_probes_used": int(result.n_probes_used),
        "probe_sequence": result.probe_sequence,
        "probe_params_used": result.probe_params_used,
        "probe_observations": result.probe_observations,
        "final_throw": (
            {"theta": result.final_throw.theta,
             "v": result.final_throw.v,
             "dt": result.final_throw.dt}
            if result.final_throw else None
        ),
        "parse_errors": result.parse_errors,
        "aborted": result.aborted,
        "abort_reason": result.abort_reason,
        "videos": result.videos,
    }


def _summarize(rows: list[dict]) -> dict:
    """Aggregate success rate per budget."""
    by_budget: dict[int, list[dict]] = {}
    for row in rows:
        by_budget.setdefault(row["budget"], []).append(row)
    summary = {}
    for budget, items in sorted(by_budget.items()):
        n = len(items)
        n_success = sum(1 for r in items if r["success"])
        n_aborted = sum(1 for r in items if r["aborted"])
        mean_dist = sum(r["distance_to_basket"] for r in items if not r["aborted"]) / max(
            1, n - n_aborted
        )
        summary[str(budget)] = {
            "n": n,
            "success_rate": n_success / n if n else 0.0,
            "mean_distance_to_basket": mean_dist,
            "n_aborted": n_aborted,
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=str, default="catalog.json")
    p.add_argument("--budgets", type=str, default="0,3",
                   help="Comma-separated probe budgets, e.g. '0,3' or '0,1,2,3'.")
    p.add_argument("--distance", type=float, default=2.0)
    p.add_argument("--n-objects", type=int, default=None,
                   help="Limit to the first N objects in the catalog.")
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of seeds per (object, budget). VLM runs with temperature=default "
                        "and caching — multiple seeds mainly help if you disable the cache.")
    p.add_argument("--provider", type=str, default="claude",
                   choices=["claude", "openai"])
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--output", type=str, required=True,
                   help="Output directory (will be created). Writes results.json + summary.json.")
    p.add_argument("--include-oracle", action="store_true",
                   help="Also run CEM-on-true-physics per object for a ceiling baseline.")
    p.add_argument("--oracle-cem-samples", type=int, default=200)
    p.add_argument("--oracle-cem-iters", type=int, default=5)
    p.add_argument("--record-videos", action="store_true",
                   help="Record per-probe and per-throw MP4s for every episode. "
                        "Off by default since it can produce a lot of files.")
    args = p.parse_args()

    budgets = [int(b.strip()) for b in args.budgets.split(",") if b.strip()]

    catalog = load_catalog(args.catalog)
    if args.n_objects is not None:
        catalog = catalog[: args.n_objects]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"

    client = build_client(args.provider, cache_dir=cache_dir, model=args.model)

    # Save run config.
    (out_dir / "config.json").write_text(json.dumps({
        "catalog": args.catalog,
        "budgets": budgets,
        "distance": args.distance,
        "n_objects": len(catalog),
        "seeds": args.seeds,
        "provider": args.provider,
        "model": args.model,
        "include_oracle": args.include_oracle,
        "record_videos": args.record_videos,
    }, indent=2))

    rows: list[dict] = []
    total = len(catalog) * len(budgets) * args.seeds
    pbar = tqdm(total=total, desc="Episodes")
    t0 = time.time()

    for obj in catalog:
        for budget in budgets:
            for seed in range(args.seeds):
                env = TossEnv(obj, basket_distance=args.distance)
                video_dir = None
                if args.record_videos:
                    video_dir = out_dir / "videos" / f"{obj.name}_b{budget}_s{seed}"
                result = run_episode(
                    env=env,
                    target_distance=args.distance,
                    max_probes=budget,
                    client=client,
                    video_dir=video_dir,
                )
                rows.append(_serialize_episode(obj, args.distance, budget, seed, result))
                pbar.update(1)
                # Incremental save — cheap and protects against crashes.
                (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    pbar.close()

    summary = _summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[info] Wrote {len(rows)} episodes to {out_dir}/results.json")
    print("[info] Summary (success rate by budget):")
    for k, v in summary.items():
        print(f"  budget={k}: n={v['n']} success_rate={v['success_rate']:.2%} "
              f"mean_dist={v['mean_distance_to_basket']:.3f}m aborts={v['n_aborted']}")

    # Optional oracle ceiling.
    if args.include_oracle:
        print("\n[info] Running CEM oracle per object (ceiling baseline)...")
        oracle_rows = []
        for obj in tqdm(catalog, desc="Oracle CEM"):
            params, dist = cem_optimal_throw(
                obj, args.distance,
                n_samples=args.oracle_cem_samples,
                n_iterations=args.oracle_cem_iters,
                seed=0,
            )
            # Verify with a clean env throw.
            env = TossEnv(obj, basket_distance=args.distance)
            tr = env.throw(params) if params is not None else None
            oracle_rows.append({
                "object": obj.name,
                "family": obj.family,
                "distance": args.distance,
                "throw": ({"theta": params.theta, "v": params.v, "dt": params.dt}
                          if params else None),
                "distance_to_basket": float(tr.distance_to_basket) if tr else None,
                "success": bool(tr.success) if tr else False,
                "cem_best_distance": float(dist),
            })
        (out_dir / "oracle.json").write_text(json.dumps(oracle_rows, indent=2))
        n_success = sum(1 for r in oracle_rows if r["success"])
        print(f"[info] Oracle success rate: {n_success}/{len(oracle_rows)} "
              f"({100 * n_success / max(1, len(oracle_rows)):.1f}%)")

    elapsed = time.time() - t0
    print(f"\n[info] Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
