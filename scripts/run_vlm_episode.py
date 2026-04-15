#!/usr/bin/env python3
"""Single-object smoke test for the VLM-only loop.

Usage:
    MUJOCO_GL=egl PYTHONPATH=. python scripts/run_vlm_episode.py \
        --object dense_compacts_000 --distance 2.0 --max-probes 3 \
        --provider claude
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

from tossing.objects.catalog import load_catalog, generate_catalog
from tossing.env import TossEnv
from tossing.oracle.dataset import cem_optimal_throw

# Register all probes so run_probe(name) works.
import tossing.probes.vertical_toss  # noqa: F401
import tossing.probes.forward_toss  # noqa: F401
import tossing.probes.release_drop  # noqa: F401
import tossing.probes.wrist_flick  # noqa: F401
import tossing.probes.shake  # noqa: F401

from tossing.vlm import build_client, run_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalog", type=str, default="catalog.json",
                   help="Path to catalog.json.")
    p.add_argument("--object", type=str, required=True,
                   help="Object name (e.g., dense_compacts_000).")
    p.add_argument("--distance", type=float, default=2.0,
                   help="Basket distance in meters.")
    p.add_argument("--max-probes", type=int, default=3,
                   help="Probe budget.")
    p.add_argument("--provider", type=str, default="claude",
                   choices=["claude", "openai"],
                   help="VLM provider.")
    p.add_argument("--model", type=str, default=None,
                   help="Override default model (e.g., claude-opus-4-5, gpt-4o).")
    p.add_argument("--output-root", type=str, default="outputs/vlm_only",
                   help="Parent directory for per-run output dirs.")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory for this run. Defaults to "
                        "<output-root>/<object>_<timestamp>/. "
                        "Writes episode.json and a cache/ subdir here.")
    p.add_argument("--include-oracle", action="store_true",
                   help="Also run CEM-on-true-physics for a ceiling baseline.")
    p.add_argument("--oracle-cem-samples", type=int, default=200)
    p.add_argument("--oracle-cem-iters", type=int, default=5)
    args = p.parse_args()

    if args.output is None:
        run_name = f"{args.object}_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir = Path(args.output_root) / run_name
    else:
        out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    videos_dir = out_dir / "videos"
    episode_path = out_dir / "episode.json"

    catalog_path = Path(args.catalog)
    if catalog_path.exists():
        catalog = load_catalog(catalog_path)
    else:
        print(f"[info] {catalog_path} not found; generating default catalog in-memory.")
        catalog = generate_catalog()

    obj_map = {o.name: o for o in catalog}
    if args.object not in obj_map:
        print(f"[error] Unknown object: {args.object!r}")
        print(f"Available (first 10): {list(obj_map)[:10]}")
        sys.exit(1)
    obj = obj_map[args.object]

    print(f"[info] Object: {obj.name} ({obj.family})")
    print(f"[info] Hidden physics: mass={obj.mass:.3f} drag={obj.drag_coeff:.3f} "
          f"com_offset={obj.com_offset} inertia={obj.inertia:.5f}")
    print(f"[info] Target distance: {args.distance} m")
    print(f"[info] Probe budget: {args.max_probes}")
    print(f"[info] VLM provider: {args.provider}")

    print(f"[info] Output dir: {out_dir}")

    env = TossEnv(obj, basket_distance=args.distance)
    client = build_client(args.provider, cache_dir=cache_dir, model=args.model)

    result = run_episode(
        env=env,
        target_distance=args.distance,
        max_probes=args.max_probes,
        client=client,
        video_dir=videos_dir,
    )

    print("\n" + "=" * 60)
    print("EPISODE RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Target distance: {args.distance:.3f} m")
    if result.final_landing_x is not None:
        direction = "long" if result.final_signed_error > 0 else "short"
        print(f"Landing x: {result.final_landing_x:.4f} m  "
              f"(signed error: {result.final_signed_error:+.4f} m, "
              f"|err|={abs(result.final_signed_error):.4f} m, {direction})")
    print(f"Distance to basket (2D miss): {result.distance_to_basket:.4f} m")
    print(f"Probes used: {result.n_probes_used} ({result.probe_sequence})")
    if result.final_throw:
        t = result.final_throw
        print(f"Throw: theta={t.theta:.2f} v={t.v:.3f} dt={t.dt:.3f}")
    if result.aborted:
        print(f"ABORTED: {result.abort_reason}")
    if result.parse_errors:
        print(f"Parse errors: {result.parse_errors}")

    print("\n" + "-" * 60)
    print("VLM REASONING TRACES")
    print("-" * 60)
    for i, trace in enumerate(result.reasoning_traces):
        print(f"\n--- Turn {i + 1} ---")
        print(trace)

    record = {
        "object": obj.name,
        "family": obj.family,
        "distance": args.distance,
        "max_probes": args.max_probes,
        "provider": args.provider,
        "model": args.model,
        "success": bool(result.success),
        "distance_to_basket": float(result.distance_to_basket),
        "target_distance": float(args.distance),
        "final_landing_x": result.final_landing_x,
        "final_signed_error": result.final_signed_error,
        "final_abs_error": (abs(result.final_signed_error)
                            if result.final_signed_error is not None else None),
        "n_probes_used": int(result.n_probes_used),
        "probe_sequence": result.probe_sequence,
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
        "reasoning_traces": result.reasoning_traces,
        "videos": result.videos,
    }
    if args.include_oracle:
        print("\n" + "-" * 60)
        print("CEM ORACLE (true-physics ceiling)")
        print("-" * 60)
        params, cem_best = cem_optimal_throw(
            obj, args.distance,
            n_samples=args.oracle_cem_samples,
            n_iterations=args.oracle_cem_iters,
            seed=0,
        )
        oracle_env = TossEnv(obj, basket_distance=args.distance)
        oracle_video_path = None
        tr = None
        if params is not None:
            oracle_env.start_recording(fps=30)
            tr = oracle_env.throw(params)
            videos_dir.mkdir(parents=True, exist_ok=True)
            oracle_video_path = videos_dir / "oracle_throw.mp4"
            if oracle_env.save_recording(oracle_video_path, fps=30) > 0:
                print(f"[info] Oracle throw video: {oracle_video_path}")
            else:
                oracle_video_path = None
        oracle_landing_x = None
        oracle_signed_error = None
        if tr is not None and tr.trajectory is not None and len(tr.trajectory) > 0:
            oracle_landing_x = float(tr.trajectory[-1][0])
            oracle_signed_error = oracle_landing_x - args.distance
        oracle_record = {
            "throw": ({"theta": params.theta, "v": params.v, "dt": params.dt}
                      if params else None),
            "distance_to_basket": float(tr.distance_to_basket) if tr else None,
            "success": bool(tr.success) if tr else False,
            "landing_x": oracle_landing_x,
            "signed_error": oracle_signed_error,
            "abs_error": abs(oracle_signed_error) if oracle_signed_error is not None else None,
            "cem_best_distance": float(cem_best),
            "cem_samples": args.oracle_cem_samples,
            "cem_iters": args.oracle_cem_iters,
            "video": str(oracle_video_path) if oracle_video_path else None,
        }
        record["oracle"] = oracle_record
        if params is not None:
            print(f"Throw: theta={params.theta:.2f} v={params.v:.3f} dt={params.dt:.3f}")
        print(f"Success: {oracle_record['success']}")
        print(f"Distance to basket: {oracle_record['distance_to_basket']}")
        print(f"\n[compare] VLM dist={result.distance_to_basket:.4f}m  "
              f"oracle dist={oracle_record['distance_to_basket']}")

    episode_path.write_text(json.dumps(record, indent=2))
    print(f"\n[info] Wrote episode to {episode_path}")
    if result.videos:
        print(f"[info] Videos ({len(result.videos)}) in {videos_dir}")


if __name__ == "__main__":
    main()
