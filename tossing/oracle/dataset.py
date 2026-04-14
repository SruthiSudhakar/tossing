"""CEM-based optimal throw dataset generation.

For each (object, basket_distance) pair, find the optimal throw parameters
(theta, v, dt) using the Cross-Entropy Method (CEM).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tossing.types import ObjectSpec, ThrowParams, ThrowResult
from tossing.env import TossEnv


def cem_optimal_throw(
    obj: ObjectSpec,
    basket_distance: float,
    n_samples: int = 500,
    n_elite: int = 50,
    n_iterations: int = 5,
    seed: int | None = None,
) -> tuple[ThrowParams | None, float]:
    """Find optimal throw parameters using Cross-Entropy Method.

    Returns (best_params, best_distance) or (None, inf) if no feasible throw found.
    """
    rng = np.random.default_rng(seed)

    # Initialize distribution: theta ~ N(45, 15), v ~ N(4.5, 1.5), dt ~ N(0, 0.05)
    mu = np.array([45.0, 4.5, 0.0])
    sigma = np.array([15.0, 1.5, 0.05])

    # Param bounds
    lo = np.array([20.0, 1.0, -0.1])
    hi = np.array([80.0, 8.0, 0.1])

    best_params = None
    best_dist = float("inf")

    env = TossEnv(obj, basket_distance=basket_distance)

    for iteration in range(n_iterations):
        # Sample candidates
        samples = rng.normal(mu, sigma, size=(n_samples, 3))
        samples = np.clip(samples, lo, hi)

        # Evaluate each candidate
        distances = np.full(n_samples, float("inf"))
        for j in range(n_samples):
            theta, v, dt = samples[j]
            result = env.throw(ThrowParams(theta=theta, v=v, dt=dt))
            distances[j] = result.distance_to_basket
            if result.success:
                distances[j] = 0.0  # bonus for in-basket

        # Select elites
        elite_idx = np.argsort(distances)[:n_elite]
        elites = samples[elite_idx]

        # Update distribution
        mu = elites.mean(axis=0)
        sigma = elites.std(axis=0) + 1e-4  # prevent collapse

        # Track best
        if distances[elite_idx[0]] < best_dist:
            best_dist = distances[elite_idx[0]]
            best_params = ThrowParams(
                theta=float(elites[0, 0]),
                v=float(elites[0, 1]),
                dt=float(elites[0, 2]),
            )

    return best_params, best_dist


def generate_dataset(
    catalog: list[ObjectSpec],
    distances: list[float] | None = None,
    n_distances: int = 20,
    cem_samples: int = 300,
    cem_elite: int = 30,
    cem_iters: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Generate (physics, target) → optimal throw dataset.

    Args:
        catalog: List of ObjectSpecs to generate data for.
        distances: Specific basket distances, or None to sample uniformly.
        n_distances: Number of distances per object (if distances is None).

    Returns list of dicts with keys:
        mass, drag_coeff, com_offset_x, com_offset_z, inertia,
        basket_distance, theta, v, dt, success
    """
    rng = np.random.default_rng(seed)

    if distances is None:
        distances = np.linspace(1.0, 3.0, n_distances).tolist()

    dataset = []
    n_success = 0
    n_total = 0

    for obj in tqdm(catalog, desc="Objects"):
        for d in distances:
            n_total += 1
            params, dist = cem_optimal_throw(
                obj, d,
                n_samples=cem_samples,
                n_elite=cem_elite,
                n_iterations=cem_iters,
                seed=rng.integers(0, 2**31),
            )

            if params is None:
                continue

            # Verify the found throw
            env = TossEnv(obj, basket_distance=d)
            result = env.throw(params)

            record = {
                "object_name": obj.name,
                "mass": obj.mass,
                "drag_coeff": obj.drag_coeff,
                "com_offset_x": obj.com_offset[0],
                "com_offset_z": obj.com_offset[2],
                "inertia": obj.inertia,
                "basket_distance": d,
                "theta": params.theta,
                "v": params.v,
                "dt": params.dt,
                "distance_to_basket": float(result.distance_to_basket),
                "success": bool(result.success),
            }
            dataset.append(record)

            if result.success:
                n_success += 1

    print(f"Dataset: {len(dataset)} samples, {n_success}/{n_total} successful "
          f"({100*n_success/max(n_total,1):.1f}%)")
    return dataset


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_dataset(dataset: list[dict], path: str | Path):
    """Save dataset to JSON."""
    Path(path).write_text(json.dumps(dataset, indent=2, cls=NumpyEncoder))


def load_dataset(path: str | Path) -> list[dict]:
    """Load dataset from JSON."""
    return json.loads(Path(path).read_text())
