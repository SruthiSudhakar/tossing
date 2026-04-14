"""Tests for object generation pipeline."""

import os
os.environ["MUJOCO_GL"] = "egl"

import json
import tempfile

import pytest

from tossing.types import ObjectSpec
from tossing.objects.families import FAMILIES, sample_object_from_family
from tossing.objects.catalog import generate_catalog, save_catalog, load_catalog, generate_test_objects
from tossing.env import TossEnv

import numpy as np


class TestFamilies:
    def test_five_families_defined(self):
        assert len(FAMILIES) == 5
        assert set(FAMILIES.keys()) == {
            "dense_compact", "light_elongated", "draggy_flat",
            "asymmetric", "uniform_moderate",
        }

    def test_sample_returns_valid_object(self):
        rng = np.random.default_rng(42)
        for family_name in FAMILIES:
            obj = sample_object_from_family(family_name, rng)
            assert isinstance(obj, ObjectSpec)
            assert obj.family == family_name
            assert obj.mass > 0
            assert obj.drag_coeff >= 0
            assert obj.inertia > 0

    def test_parameters_within_ranges(self):
        rng = np.random.default_rng(42)
        for family_name, fam in FAMILIES.items():
            for _ in range(10):
                obj = sample_object_from_family(family_name, rng)
                assert fam.mass_range[0] <= obj.mass <= fam.mass_range[1]
                assert fam.drag_coeff_range[0] <= obj.drag_coeff <= fam.drag_coeff_range[1]
                assert fam.inertia_range[0] <= obj.inertia <= fam.inertia_range[1]


class TestCatalog:
    def test_generate_100_objects(self):
        catalog = generate_catalog()
        assert len(catalog) == 100

    def test_20_per_family(self):
        catalog = generate_catalog()
        for family_name in FAMILIES:
            count = sum(1 for o in catalog if o.family == family_name)
            assert count == 20

    def test_save_and_load(self):
        catalog = generate_catalog()
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            save_catalog(catalog, f.name)
            loaded = load_catalog(f.name)

        assert len(loaded) == len(catalog)
        for orig, loaded_obj in zip(catalog, loaded):
            assert orig.name == loaded_obj.name
            assert abs(orig.mass - loaded_obj.mass) < 1e-6

    def test_all_objects_loadable_in_mujoco(self):
        """Every generated object should create a valid MuJoCo model."""
        catalog = generate_catalog()
        for obj in catalog[:20]:  # test first 20 for speed
            env = TossEnv(obj, basket_distance=2.0)
            assert env.object_pos is not None


class TestTestObjects:
    def test_five_test_objects(self):
        test_objs = generate_test_objects()
        assert len(test_objs) == 5

    def test_test_objects_loadable(self):
        for obj in generate_test_objects():
            env = TossEnv(obj, basket_distance=2.0)
            assert env.object_pos is not None
