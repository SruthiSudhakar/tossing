"""Oracle MLP: maps ground-truth physics + target distance → throw parameters."""

from __future__ import annotations

import torch
import torch.nn as nn


class OracleMLP(nn.Module):
    """MLP that predicts throw parameters from physics properties and target distance.

    Input: [mass, drag_coeff, com_offset_x, com_offset_z, inertia, basket_distance]
    Output: [theta, v, dt] (normalized to [0, 1], mapped to physical ranges externally)
    """

    def __init__(self, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()

        layers = []
        in_dim = 6
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())  # output in [0, 1]

        self.net = nn.Sequential(*layers)

        # Output ranges for denormalization
        self.theta_range = (20.0, 80.0)
        self.v_range = (1.0, 8.0)
        self.dt_range = (-0.1, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input and output are normalized."""
        return self.net(x)

    def predict_params(self, x: torch.Tensor) -> torch.Tensor:
        """Predict denormalized throw parameters."""
        raw = self.forward(x)
        theta = raw[:, 0] * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
        v = raw[:, 1] * (self.v_range[1] - self.v_range[0]) + self.v_range[0]
        dt = raw[:, 2] * (self.dt_range[1] - self.dt_range[0]) + self.dt_range[0]
        return torch.stack([theta, v, dt], dim=1)


# Input normalization stats (computed from training data, set during training)
INPUT_STATS = {
    "mass": {"mean": 0.5, "std": 0.5},
    "drag_coeff": {"mean": 0.5, "std": 0.5},
    "com_offset_x": {"mean": 0.0, "std": 0.05},
    "com_offset_z": {"mean": 0.0, "std": 0.01},
    "inertia": {"mean": 0.01, "std": 0.01},
    "basket_distance": {"mean": 2.0, "std": 0.6},
}


def normalize_input(data: torch.Tensor, stats: dict = INPUT_STATS) -> torch.Tensor:
    """Normalize input features using mean/std."""
    means = torch.tensor([stats[k]["mean"] for k in
                          ["mass", "drag_coeff", "com_offset_x", "com_offset_z",
                           "inertia", "basket_distance"]])
    stds = torch.tensor([stats[k]["std"] for k in
                         ["mass", "drag_coeff", "com_offset_x", "com_offset_z",
                          "inertia", "basket_distance"]])
    return (data - means) / (stds + 1e-8)


def normalize_output(theta: torch.Tensor, v: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Normalize throw params to [0, 1] for MSE loss."""
    theta_norm = (theta - 20.0) / 60.0
    v_norm = (v - 1.0) / 7.0
    dt_norm = (dt + 0.1) / 0.2
    return torch.stack([theta_norm, v_norm, dt_norm], dim=1)
