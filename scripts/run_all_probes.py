#!/usr/bin/env python3
"""Run all probes on all catalog objects and save results."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tossing.objects.catalog import load_catalog, generate_catalog
from tossing.env import TossEnv

# Register all probes
import tossing.probes.vertical_toss
import tossing.probes.forward_toss
import tossing.probes.release_drop
import tossing.probes.wrist_flick
import tossing.probes.shake
from tossing.probes import list_probes

from tqdm import tqdm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=str, default=None)
    parser.add_argument("--output", type=str, default="probe_results.json")
    parser.add_argument("--n-objects", type=int, default=None)
    args = parser.parse_args()

    if args.catalog:
        catalog = load_catalog(args.catalog)
    else:
        catalog = generate_catalog()

    if args.n_objects:
        catalog = catalog[:args.n_objects]

    probe_types = list_probes()
    results = []

    for obj in tqdm(catalog, desc="Objects"):
        env = TossEnv(obj, basket_distance=2.0)
        obj_results = {"object": obj.name, "family": obj.family, "probes": {}}

        for pt in probe_types:
            result = env.run_probe(pt)
            obj_results["probes"][pt] = result.observations

        results.append(obj_results)

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Saved {len(results)} objects × {len(probe_types)} probes → {args.output}")


if __name__ == "__main__":
    main()
