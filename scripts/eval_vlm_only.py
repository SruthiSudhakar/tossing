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
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _run_one_episode(task: dict) -> dict:
    """Worker: run a single VLM episode in its own process.

    Builds a fresh client + env so nothing is shared across workers.
    """
    obj = task["obj"]
    distance = task["distance"]
    budget = task["budget"]
    vlm_seed = task["vlm_seed"]
    episode_idx = task["episode_idx"]
    episode_dir = Path(task["episode_dir"])
    record_videos = task["record_videos"]
    provider = task["provider"]
    model = task["model"]

    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "meta.json").write_text(json.dumps({
        "episode_idx": episode_idx,
        "object": obj.name,
        "family": obj.family,
        "budget": budget,
        "vlm_seed": vlm_seed,
        "distance": distance,
    }, indent=2))

    ep_cache_dir = episode_dir / "cache"
    ep_cache_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(provider, cache_dir=ep_cache_dir, model=model)

    env = TossEnv(obj, basket_distance=distance)
    video_dir = (episode_dir / "videos") if record_videos else None
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
    (episode_dir / "episode.json").write_text(json.dumps(row, indent=2))
    return row


def _run_one_oracle(task: dict) -> dict:
    """Worker: run CEM oracle for one (object, distance) in its own process."""
    obj = task["obj"]
    distance = task["distance"]
    n_samples = task["n_samples"]
    n_iterations = task["n_iterations"]
    record_videos = task["record_videos"]
    out_dir = Path(task["out_dir"])

    params, dist = cem_optimal_throw(
        obj, distance,
        n_samples=n_samples,
        n_iterations=n_iterations,
        seed=0,
    )
    env = TossEnv(obj, basket_distance=distance)
    oracle_video_path = None
    tr = None
    if params is not None:
        if record_videos:
            env.start_recording(fps=30)
        tr = env.throw(params)
        if record_videos:
            video_dir = out_dir / "videos" / f"{obj.name}_d{distance:.2f}_oracle"
            video_dir.mkdir(parents=True, exist_ok=True)
            candidate = video_dir / "oracle_throw.mp4"
            if env.save_recording(candidate, fps=30) > 0:
                oracle_video_path = str(candidate)
    return {
        "object": obj.name,
        "family": obj.family,
        "distance": distance,
        "throw": ({"theta": params.theta, "v": params.v, "dt": params.dt}
                  if params else None),
        "distance_to_basket": float(tr.distance_to_basket) if tr else None,
        "success": bool(tr.success) if tr else False,
        "cem_best_distance": float(dist),
        "video": oracle_video_path,
    }


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
    p.add_argument("--n-workers", type=int, default=10,
                   help="Number of episodes to run in parallel via "
                        "ProcessPoolExecutor. Set to 1 for sequential execution.")
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

    # Clients are built per-worker inside _run_one_episode so nothing is
    # shared across processes.

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
        "n_workers": args.n_workers,
    }, indent=2))

    tasks: list[dict] = []
    episode_idx = 0
    for obj in catalog:
        for distance in distances:
            for budget in budgets:
                for vlm_seed in range(args.vlm_seeds):
                    tasks.append({
                        "obj": obj,
                        "distance": distance,
                        "budget": budget,
                        "vlm_seed": vlm_seed,
                        "episode_idx": episode_idx,
                        "episode_dir": str(out_dir / f"episode_{episode_idx}"),
                        "record_videos": args.record_videos,
                        "provider": args.provider,
                        "model": args.model,
                    })
                    episode_idx += 1

    rows: list[dict] = []
    pbar = tqdm(total=len(tasks), desc="Episodes")
    t0 = time.time()

    n_workers = max(1, args.n_workers)
    if n_workers == 1:
        for t in tasks:
            row = _run_one_episode(t)
            rows.append(row)
            pbar.update(1)
            (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_run_one_episode, t) for t in tasks]
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                pbar.update(1)
                # Incremental save — cheap and protects against crashes.
                (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
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
        oracle_tasks = [
            {
                "obj": obj,
                "distance": distance,
                "n_samples": args.oracle_cem_samples,
                "n_iterations": args.oracle_cem_iters,
                "record_videos": args.record_videos,
                "out_dir": str(out_dir),
            }
            for obj in catalog
            for distance in distances
        ]
        oracle_rows: list[dict] = []
        opbar = tqdm(total=len(oracle_tasks), desc="Oracle CEM")
        if n_workers == 1:
            for t in oracle_tasks:
                oracle_rows.append(_run_one_oracle(t))
                opbar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_run_one_oracle, t) for t in oracle_tasks]
                for fut in as_completed(futures):
                    oracle_rows.append(fut.result())
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
