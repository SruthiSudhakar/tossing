"""Object catalog: generate and manage the 100-object training set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tossing.types import ObjectSpec
from tossing.objects.families import FAMILIES, sample_object_from_family


OBJECTS_PER_FAMILY = 20
DEFAULT_SEED = 42


def generate_catalog(seed: int = DEFAULT_SEED,
                     objects_per_family: int = OBJECTS_PER_FAMILY) -> list[ObjectSpec]:
    """Generate the full training catalog (100 objects by default)."""
    rng = np.random.default_rng(seed)
    catalog = []

    for family_name in sorted(FAMILIES.keys()):
        for i in range(objects_per_family):
            obj = sample_object_from_family(family_name, rng, index=i)
            catalog.append(obj)

    return catalog


def save_catalog(catalog: list[ObjectSpec], path: str | Path):
    """Save catalog to JSON file."""
    data = [obj.to_dict() for obj in catalog]
    Path(path).write_text(json.dumps(data, indent=2))


def load_catalog(path: str | Path) -> list[ObjectSpec]:
    """Load catalog from JSON file."""
    data = json.loads(Path(path).read_text())
    return [ObjectSpec.from_dict(d) for d in data]


def generate_test_objects(seed: int = 123) -> list[ObjectSpec]:
    """Generate held-out test objects with appearance-physics mismatches."""
    rng = np.random.default_rng(seed)

    test_objects = [
        # Hollow metal sphere: looks heavy (gray sphere), actually light
        ObjectSpec(
            name="hollow_metal_sphere", family="test_mismatch",
            geom_type="sphere", geom_size=[0.05],
            mass=0.05, drag_coeff=0.15,
            com_offset=[0.0, 0.0, 0.0], inertia=0.0008,
            cross_section_area=0.008,
            rgba=[0.6, 0.6, 0.65, 1.0],
        ),
        # Lead-filled plastic bottle: looks light, actually heavy
        ObjectSpec(
            name="lead_filled_bottle", family="test_mismatch",
            geom_type="cylinder", geom_size=[0.025, 0.08],
            mass=1.5, drag_coeff=0.3,
            com_offset=[0.0, 0.0, -0.03], inertia=0.003,
            cross_section_area=0.01,
            rgba=[0.8, 0.85, 0.9, 1.0],
        ),
        # Foam airplane: looks rigid, extreme drag
        ObjectSpec(
            name="foam_airplane", family="test_mismatch",
            geom_type="box", geom_size=[0.08, 0.06, 0.003],
            mass=0.03, drag_coeff=2.5,
            com_offset=[0.02, 0.0, 0.0], inertia=0.002,
            cross_section_area=0.04,
            rgba=[1.0, 1.0, 1.0, 1.0],
        ),
        # Weighted dart (tail-heavy): symmetric look, asymmetric CoM
        ObjectSpec(
            name="weighted_dart", family="test_mismatch",
            geom_type="capsule", geom_size=[0.01, 0.06],
            mass=0.15, drag_coeff=0.2,
            com_offset=[0.04, 0.0, 0.0], inertia=0.001,
            cross_section_area=0.005,
            rgba=[0.2, 0.2, 0.8, 1.0],
        ),
        # Crumpled aluminum foil ball: ambiguous everything
        ObjectSpec(
            name="crumpled_foil_ball", family="test_mismatch",
            geom_type="sphere", geom_size=[0.04],
            mass=0.08, drag_coeff=1.0,
            com_offset=[0.01, 0.0, 0.005], inertia=0.0015,
            cross_section_area=0.02,
            rgba=[0.75, 0.75, 0.78, 1.0],
        ),
    ]

    return test_objects
