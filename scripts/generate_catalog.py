#!/usr/bin/env python3
"""Generate the 100-object training catalog."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tossing.objects.catalog import generate_catalog, save_catalog


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="catalog.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    catalog = generate_catalog(seed=args.seed)
    save_catalog(catalog, args.output)
    print(f"Generated {len(catalog)} objects → {args.output}")

    for fam in sorted(set(o.family for o in catalog)):
        objs = [o for o in catalog if o.family == fam]
        masses = [o.mass for o in objs]
        drags = [o.drag_coeff for o in objs]
        print(f"  {fam}: n={len(objs)}, "
              f"mass=[{min(masses):.3f}, {max(masses):.3f}], "
              f"drag=[{min(drags):.3f}, {max(drags):.3f}]")


if __name__ == "__main__":
    main()
