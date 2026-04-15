"""Shared dataclasses for the tossing simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class ObjectSpec:
    """Specification for a single tossable object."""

    name: str
    family: str  # e.g., "dense_compact", "light_elongated", ...

    # MuJoCo geometry (visual + collision)
    geom_type: str  # "sphere", "box", "cylinder", "capsule"
    geom_size: list[float]  # MuJoCo size params (depends on geom_type)

    # Hidden physics properties (the 4 things the VLM must discover)
    mass: float  # kg
    drag_coeff: float  # dimensionless
    com_offset: list[float]  # [dx, dy, dz] in meters from geometric center
    inertia: float  # kg*m^2 (scalar, moment about y-axis for 2D)

    # Derived / auxiliary
    cross_section_area: float  # m^2 (for drag force computation)
    rgba: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ObjectSpec:
        return cls(**d)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ObjectSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class ProbeResult:
    """Structured observations returned by a probe action."""

    probe_type: str  # "vertical_toss" | "forward_toss" | "release_drop" | "wrist_flick" | "shake"
    observations: dict[str, float]  # named numerical observations
    trajectory: np.ndarray | None = None  # (T, 6) pos+vel sampled at 30Hz, optional
    cost: int = 1


@dataclass
class ThrowParams:
    """Parameters for a throw action."""

    theta: float  # release angle in degrees, [20, 80]
    v: float  # release speed in m/s, [1, 8]
    dt: float = 0.0  # release timing offset in s, [-0.1, 0.1]


@dataclass
class ThrowResult:
    """Result of a throw attempt."""

    success: bool  # did the object land in the basket?
    distance_to_basket: float  # meters from basket center at landing
    trajectory: np.ndarray | None = None  # (T, 6) pos+vel at 30Hz
    flight_time: float = 0.0  # seconds from release to landing/stop
