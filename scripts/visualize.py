#!/usr/bin/env python3
"""Visualize the tossing environment: render objects, probes, throws."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tossing.types import ObjectSpec, ThrowParams
from tossing.env import TossEnv
from tossing.objects.catalog import generate_catalog, generate_test_objects

# Register probes
import tossing.probes.vertical_toss
import tossing.probes.forward_toss
import tossing.probes.release_drop
import tossing.probes.wrist_flick
import tossing.probes.shake


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/viz")
    parser.add_argument("--n-objects", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog = generate_catalog()[:args.n_objects]
    test_objs = generate_test_objects()

    # Render each training object
    for obj in catalog:
        env = TossEnv(obj, basket_distance=2.0)
        img = env.render_pair()
        img.save(output_dir / f"{obj.name}.png")
        print(f"Rendered {obj.name}")

    # Render test objects
    for obj in test_objs:
        env = TossEnv(obj, basket_distance=2.0)
        img = env.render_pair()
        img.save(output_dir / f"test_{obj.name}.png")
        print(f"Rendered test_{obj.name}")

    print(f"\nSaved renders to {output_dir}")


if __name__ == "__main__":
    main()
