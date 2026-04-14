#!/usr/bin/env python3
"""Train the oracle throw MLP.

Usage:
    python scripts/train_oracle.py [--small] [--catalog catalog.json]

    --small: Use a small subset (10 objects, 5 distances) for quick testing
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tossing.objects.catalog import generate_catalog, load_catalog
from tossing.oracle.dataset import generate_dataset, save_dataset, load_dataset
from tossing.oracle.train import train_oracle, evaluate_oracle


def main():
    parser = argparse.ArgumentParser(description="Train oracle throw MLP")
    parser.add_argument("--small", action="store_true",
                        help="Quick test with small subset")
    parser.add_argument("--catalog", type=str, default=None,
                        help="Path to catalog JSON (generates if not provided)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to pre-generated dataset JSON")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="outputs/oracle")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate catalog
    if args.catalog:
        catalog = load_catalog(args.catalog)
    else:
        catalog = generate_catalog()

    if args.small:
        catalog = catalog[:2]
        n_distances = 1
        cem_samples = 200
        cem_iters = 3
    else:
        n_distances = 20
        cem_samples = 300
        cem_iters = 5

    # Generate or load dataset
    if args.dataset:
        dataset = load_dataset(args.dataset)
        print(f"Loaded {len(dataset)} samples from {args.dataset}")
    else:
        print(f"Generating dataset: {len(catalog)} objects × {n_distances} distances")
        dataset = generate_dataset(
            catalog,
            n_distances=n_distances,
            cem_samples=cem_samples,
            cem_iters=cem_iters,
        )
        save_dataset(dataset, output_dir / "dataset.json")
        print(f"Saved dataset to {output_dir / 'dataset.json'}")

    # Train
    print(f"\nTraining oracle MLP for {args.epochs} epochs...")
    model, info = train_oracle(
        dataset,
        n_epochs=args.epochs,
        save_path=str(output_dir / "oracle_best.pt"),
    )
    print(f"Best val loss: {info['best_val_loss']:.6f}")

    # Evaluate
    print("\nEvaluating on training catalog...")
    eval_catalog = catalog[:10] if args.small else catalog[:20]
    eval_results = evaluate_oracle(
        model, eval_catalog, info["input_stats"],
        n_distances=5,
    )
    print(f"Success rate: {eval_results['success_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
