"""Object family definitions with parameter sampling ranges.

Each family specifies distributions for the 4 hidden physics properties
plus visual appearance (geom type, size, color).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FamilyDef:
    """Definition of an object family with parameter ranges."""

    name: str
    description: str

    # Physics ranges [min, max]
    mass_range: tuple[float, float]
    drag_coeff_range: tuple[float, float]
    com_offset_range: tuple[float, float]  # magnitude of offset (±)
    inertia_range: tuple[float, float]

    # Geometry options
    geom_types: list[str]
    geom_size_range: tuple[list[float], list[float]]  # [min_sizes, max_sizes] per geom dim

    # Cross-section area range (for drag computation)
    cross_section_range: tuple[float, float]

    # RGBA color range for visual variety
    rgba_base: list[float]  # base color, will be jittered


FAMILIES: dict[str, FamilyDef] = {
    "dense_compact": FamilyDef(
        name="dense_compact",
        description="Heavy, compact objects: steel ball, rock, clay ball",
        mass_range=(0.3, 2.0),
        drag_coeff_range=(0.05, 0.2),
        com_offset_range=(-0.005, 0.005),
        inertia_range=(0.0001, 0.001),
        geom_types=["sphere", "box"],
        geom_size_range=([0.02], [0.06]),
        cross_section_range=(0.002, 0.012),
        rgba_base=[0.5, 0.5, 0.5, 1.0],
    ),
    "light_elongated": FamilyDef(
        name="light_elongated",
        description="Light, long objects: pencil, straw, chopstick",
        mass_range=(0.01, 0.1),
        drag_coeff_range=(0.1, 0.5),
        com_offset_range=(0.02, 0.08),
        inertia_range=(0.0005, 0.005),
        geom_types=["capsule", "cylinder"],
        geom_size_range=([0.005, 0.05], [0.015, 0.15]),
        cross_section_range=(0.001, 0.01),
        rgba_base=[0.8, 0.7, 0.3, 1.0],
    ),
    "draggy_flat": FamilyDef(
        name="draggy_flat",
        description="Flat, high-drag objects: paper plate, cardboard, leaf",
        mass_range=(0.05, 0.3),
        drag_coeff_range=(0.8, 3.0),
        com_offset_range=(-0.01, 0.01),
        inertia_range=(0.001, 0.01),
        geom_types=["box"],
        geom_size_range=([0.04, 0.04, 0.002], [0.1, 0.1, 0.008]),
        cross_section_range=(0.01, 0.05),
        rgba_base=[0.9, 0.85, 0.7, 1.0],
    ),
    "asymmetric": FamilyDef(
        name="asymmetric",
        description="Asymmetric objects: hammer, wrench, ladle",
        mass_range=(0.2, 1.0),
        drag_coeff_range=(0.1, 0.5),
        com_offset_range=(0.03, 0.1),
        inertia_range=(0.005, 0.05),
        geom_types=["capsule", "box"],
        geom_size_range=([0.015, 0.04], [0.03, 0.1]),
        cross_section_range=(0.005, 0.03),
        rgba_base=[0.4, 0.4, 0.6, 1.0],
    ),
    "uniform_moderate": FamilyDef(
        name="uniform_moderate",
        description="Moderate, uniform objects: tennis ball, orange, stress ball",
        mass_range=(0.1, 0.5),
        drag_coeff_range=(0.2, 0.6),
        com_offset_range=(-0.01, 0.01),
        inertia_range=(0.0005, 0.005),
        geom_types=["sphere"],
        geom_size_range=([0.025], [0.06]),
        cross_section_range=(0.005, 0.015),
        rgba_base=[0.3, 0.7, 0.3, 1.0],
    ),
}


def sample_object_from_family(family_name: str, rng: np.random.Generator,
                               index: int = 0):
    """Sample a random ObjectSpec from the given family.

    Returns an ObjectSpec (imported lazily to avoid circular imports).
    """
    from tossing.types import ObjectSpec

    fam = FAMILIES[family_name]

    mass = rng.uniform(*fam.mass_range)
    drag_coeff = rng.uniform(*fam.drag_coeff_range)
    inertia = rng.uniform(*fam.inertia_range)

    # CoM offset: for elongated/asymmetric, offset is along x-axis (the long axis)
    com_mag = rng.uniform(*sorted(fam.com_offset_range))
    com_offset = [com_mag, 0.0, 0.0]

    cross_section = rng.uniform(*fam.cross_section_range)

    # Geometry
    geom_type = rng.choice(fam.geom_types)
    min_sizes = fam.geom_size_range[0]
    max_sizes = fam.geom_size_range[1]

    if geom_type == "sphere":
        size = [rng.uniform(min_sizes[0], max_sizes[0])]
    elif geom_type in ("cylinder", "capsule"):
        r = rng.uniform(min_sizes[0], max_sizes[0])
        h = rng.uniform(min_sizes[1], max_sizes[1])
        size = [r, h]
    elif geom_type == "box":
        if len(min_sizes) == 3:
            size = [rng.uniform(mn, mx) for mn, mx in zip(min_sizes, max_sizes)]
        else:
            s = rng.uniform(min_sizes[0], max_sizes[0])
            size = [s, s, s]
    else:
        size = [0.05]

    # Color: jitter the base color
    rgba = [
        np.clip(fam.rgba_base[0] + rng.normal(0, 0.1), 0, 1),
        np.clip(fam.rgba_base[1] + rng.normal(0, 0.1), 0, 1),
        np.clip(fam.rgba_base[2] + rng.normal(0, 0.1), 0, 1),
        1.0,
    ]

    name = f"{family_name}_{index:03d}"

    return ObjectSpec(
        name=name,
        family=family_name,
        geom_type=geom_type,
        geom_size=[float(s) for s in size],
        mass=float(mass),
        drag_coeff=float(drag_coeff),
        com_offset=[float(c) for c in com_offset],
        inertia=float(inertia),
        cross_section_area=float(cross_section),
        rgba=[float(c) for c in rgba],
    )
