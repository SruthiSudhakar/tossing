"""Train and evaluate the oracle throw MLP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tossing.oracle.mlp import OracleMLP, normalize_input, normalize_output, INPUT_STATS
from tossing.types import ObjectSpec, ThrowParams
from tossing.env import TossEnv


def prepare_tensors(dataset: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dataset dicts to input/output tensors."""
    inputs = []
    outputs = []

    for rec in dataset:
        inputs.append([
            rec["mass"], rec["drag_coeff"],
            rec["com_offset_x"], rec["com_offset_z"],
            rec["inertia"], rec["basket_distance"],
        ])
        outputs.append([rec["theta"], rec["v"], rec["dt"]])

    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(outputs, dtype=torch.float32)
    return X, Y


def compute_input_stats(X: torch.Tensor) -> dict:
    """Compute normalization statistics from training data."""
    keys = ["mass", "drag_coeff", "com_offset_x", "com_offset_z",
            "inertia", "basket_distance"]
    stats = {}
    for i, k in enumerate(keys):
        stats[k] = {
            "mean": float(X[:, i].mean()),
            "std": float(X[:, i].std()),
        }
    return stats


def train_oracle(
    dataset: list[dict],
    n_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = "cpu",
    save_path: str | None = None,
) -> tuple[OracleMLP, dict]:
    """Train the oracle MLP.

    Returns (model, training_info dict).
    """
    # Filter to successful throws only for training
    successful = [r for r in dataset if r.get("success", False)]
    if len(successful) < 10:
        print(f"Warning: only {len(successful)} successful throws. Using all {len(dataset)} samples.")
        successful = dataset

    X, Y = prepare_tensors(successful)
    stats = compute_input_stats(X)

    # Normalize inputs
    X_norm = normalize_input(X, stats)

    # Normalize outputs to [0, 1]
    Y_norm = normalize_output(Y[:, 0], Y[:, 1], Y[:, 2])

    # Train/val split
    n = len(X_norm)
    n_val = int(n * val_split)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    X_train, Y_train = X_norm[train_idx], Y_norm[train_idx]
    X_val, Y_val = X_norm[val_idx], Y_norm[val_idx]

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size, shuffle=True,
    )

    model = OracleMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(X_train)
        history["train_loss"].append(epoch_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device))
            val_loss = criterion(val_pred, Y_val.to(device)).item()
        history["val_loss"].append(val_loss)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save({
                    "model_state": model.state_dict(),
                    "stats": stats,
                }, save_path)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: train_loss={epoch_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, lr={scheduler.get_last_lr()[0]:.6f}")

    info = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "best_val_loss": best_val_loss,
        "input_stats": stats,
        "history": history,
    }

    return model, info


def evaluate_oracle(
    model: OracleMLP,
    catalog: list[ObjectSpec],
    stats: dict,
    distances: list[float] | None = None,
    n_distances: int = 10,
    device: str = "cpu",
) -> dict:
    """Evaluate oracle MLP by simulating predicted throws.

    Returns evaluation metrics dict.
    """
    if distances is None:
        distances = np.linspace(1.0, 3.0, n_distances).tolist()

    model.eval()
    successes = 0
    total = 0
    all_distances = []

    for obj in catalog:
        for d in distances:
            total += 1

            # Build input
            x = torch.tensor([[
                obj.mass, obj.drag_coeff,
                obj.com_offset[0], obj.com_offset[2],
                obj.inertia, d,
            ]], dtype=torch.float32)

            x_norm = normalize_input(x, stats).to(device)

            with torch.no_grad():
                params_tensor = model.predict_params(x_norm)

            theta = float(params_tensor[0, 0])
            v = float(params_tensor[0, 1])
            dt = float(params_tensor[0, 2])

            # Simulate
            env = TossEnv(obj, basket_distance=d)
            result = env.throw(ThrowParams(theta=theta, v=v, dt=dt))

            if result.success:
                successes += 1
            all_distances.append(result.distance_to_basket)

    success_rate = successes / max(total, 1)
    mean_dist = float(np.mean(all_distances))

    print(f"Oracle eval: {successes}/{total} = {100*success_rate:.1f}% success, "
          f"mean dist to basket: {mean_dist:.3f}m")

    return {
        "success_rate": success_rate,
        "successes": successes,
        "total": total,
        "mean_distance": mean_dist,
    }
