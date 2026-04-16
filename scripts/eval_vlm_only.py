#!/usr/bin/env python3
"""Evaluation sweep for the VLM-only loop.

Sweeps probe budgets across a catalog of objects, records per-episode results,
and optionally computes the CEM-on-true-physics ceiling for comparison.

Usage:

python scripts/eval_vlm_only.py \
    --catalog catalog.json --budgets 0,3,5 --n-objects 10 \
    --provider openai --include-oracle \
    --record-videos
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


def _serialize_episode(obj, distance, budget, vlm_seed, result) -> dict:
    return {
        "object": obj.name,
        "family": obj.family,
        "distance": distance,
        "budget": budget,
        "vlm_seed": vlm_seed,
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
    """Aggregate success rate per (budget, distance) cell."""
    by_cell: dict[tuple[int, float], list[dict]] = {}
    for row in rows:
        key = (row["budget"], row["distance"])
        by_cell.setdefault(key, []).append(row)
    summary = {}
    for (budget, distance), items in sorted(by_cell.items()):
        n = len(items)
        n_success = sum(1 for r in items if r["success"])
        n_aborted = sum(1 for r in items if r["aborted"])
        mean_dist = sum(r["distance_to_basket"] for r in items if not r["aborted"]) / max(
            1, n - n_aborted
        )
        summary[f"budget={budget}|distance={distance}"] = {
            "budget": budget,
            "distance": distance,
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
    p.add_argument("--distances", type=str, default="2.0",
                   help="Comma-separated basket distances in meters, e.g. '1.5,2.0,2.5'.")
    p.add_argument("--n-objects", type=int, default=None,
                   help="Limit to the first N objects in the catalog.")
    p.add_argument("--vlm-seeds", type=int, default=1,
                   help="Number of VLM seeds per (object, distance, budget). VLM runs with "
                        "temperature=default and caching — multiple seeds mainly help if you "
                        "disable the cache.")
    p.add_argument("--provider", type=str, default="claude",
                   choices=["claude", "openai"])
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--output-root", type=str, default="outputs/vlm_only",
                   help="Parent directory for per-run output dirs. Only used when "
                        "--output is not explicitly provided.")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory (will be created). Writes results.json + summary.json. "
                        "Defaults to <output-root>/sweep_<timestamp>/ so re-runs don't collide.")
    p.add_argument("--include-oracle", action="store_true",
                   help="Also run CEM-on-true-physics per object for a ceiling baseline.")
    p.add_argument("--oracle-cem-samples", type=int, default=200)
    p.add_argument("--oracle-cem-iters", type=int, default=5)
    p.add_argument("--record-videos", action="store_true",
                   help="Record per-probe and per-throw MP4s for every episode. "
                        "Off by default since it can produce a lot of files.")
    args = p.parse_args()

    budgets = [int(b.strip()) for b in args.budgets.split(",") if b.strip()]
    distances = [float(d.strip()) for d in args.distances.split(",") if d.strip()]

    catalog = load_catalog(args.catalog)
    if args.n_objects is not None:
        catalog = catalog[: args.n_objects]

    if args.output is None:
        run_name = f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir = Path(args.output_root) / run_name
    else:
        out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Output dir: {out_dir}")

    # cache_dir is set per-episode below, so each episode's prompt/response
    # cache lives in its own directory.
    client = build_client(args.provider, cache_dir=None, model=args.model)

    # Save run config.
    (out_dir / "config.json").write_text(json.dumps({
        "catalog": args.catalog,
        "budgets": budgets,
        "distances": distances,
        "n_objects": len(catalog),
        "vlm_seeds": args.vlm_seeds,
        "provider": args.provider,
        "model": args.model,
        "include_oracle": args.include_oracle,
        "record_videos": args.record_videos,
    }, indent=2))

    rows: list[dict] = []
    total = len(catalog) * len(distances) * len(budgets) * args.vlm_seeds
    pbar = tqdm(total=total, desc="Episodes")
    t0 = time.time()

    episode_idx = 0
    for obj in catalog:
        for distance in distances:
            for budget in budgets:
                for vlm_seed in range(args.vlm_seeds):
                    episode_dir = out_dir / f"episode_{episode_idx}"
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    (episode_dir / "meta.json").write_text(json.dumps({
                        "episode_idx": episode_idx,
                        "object": obj.name,
                        "family": obj.family,
                        "budget": budget,
                        "vlm_seed": vlm_seed,
                        "distance": distance,
                    }, indent=2))

                    # Point the client at this episode's cache dir.
                    ep_cache_dir = episode_dir / "cache"
                    ep_cache_dir.mkdir(parents=True, exist_ok=True)
                    client.cache_dir = ep_cache_dir

                    env = TossEnv(obj, basket_distance=distance)
                    video_dir = (episode_dir / "videos") if args.record_videos else None
                    result = run_episode(
                        env=env,
                        target_distance=distance,
                        max_probes=budget,
                        client=client,
                        video_dir=video_dir,
                    )
                    row = _serialize_episode(obj, distance, budget, vlm_seed, result)
                    row["episode_idx"] = episode_idx
                    row["episode_dir"] = str(episode_dir)
                    rows.append(row)
                    (episode_dir / "episode.json").write_text(json.dumps(row, indent=2))
                    pbar.update(1)
                    # Incremental save — cheap and protects against crashes.
                    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
                    episode_idx += 1
    pbar.close()

    summary = _summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[info] Wrote {len(rows)} episodes to {out_dir}/results.json")
    print("[info] Summary (success rate by (budget, distance)):")
    for v in summary.values():
        print(f"  budget={v['budget']} distance={v['distance']:.2f}: "
              f"n={v['n']} success_rate={v['success_rate']:.2%} "
              f"mean_dist={v['mean_distance_to_basket']:.3f}m aborts={v['n_aborted']}")

    # Optional oracle ceiling.
    if args.include_oracle:
        print("\n[info] Running CEM oracle per (object, distance) (ceiling baseline)...")
        oracle_rows = []
        oracle_total = len(catalog) * len(distances)
        opbar = tqdm(total=oracle_total, desc="Oracle CEM")
        for obj in catalog:
            for distance in distances:
                params, dist = cem_optimal_throw(
                    obj, distance,
                    n_samples=args.oracle_cem_samples,
                    n_iterations=args.oracle_cem_iters,
                    seed=0,
                )
                # Verify with a clean env throw.
                env = TossEnv(obj, basket_distance=distance)
                oracle_video_path = None
                tr = None
                if params is not None:
                    if args.record_videos:
                        env.start_recording(fps=30)
                    tr = env.throw(params)
                    if args.record_videos:
                        video_dir = out_dir / "videos" / f"{obj.name}_d{distance:.2f}_oracle"
                        video_dir.mkdir(parents=True, exist_ok=True)
                        candidate = video_dir / "oracle_throw.mp4"
                        if env.save_recording(candidate, fps=30) > 0:
                            oracle_video_path = str(candidate)
                oracle_rows.append({
                    "object": obj.name,
                    "family": obj.family,
                    "distance": distance,
                    "throw": ({"theta": params.theta, "v": params.v, "dt": params.dt}
                              if params else None),
                    "distance_to_basket": float(tr.distance_to_basket) if tr else None,
                    "success": bool(tr.success) if tr else False,
                    "cem_best_distance": float(dist),
                    "video": oracle_video_path,
                })
                opbar.update(1)
        opbar.close()
        (out_dir / "oracle.json").write_text(json.dumps(oracle_rows, indent=2))
        n_success = sum(1 for r in oracle_rows if r["success"])
        print(f"[info] Oracle success rate: {n_success}/{len(oracle_rows)} "
              f"({100 * n_success / max(1, len(oracle_rows)):.1f}%)")

    elapsed = time.time() - t0
    print(f"\n[info] Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
