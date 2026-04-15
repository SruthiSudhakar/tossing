"""Tests for probe controllers."""

import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import pytest

from tossing.types import ObjectSpec
from tossing.env import TossEnv
from tossing.physics import ballistic_apex, ballistic_hang_time

# Import probes to register them
import tossing.probes.vertical_toss
import tossing.probes.forward_toss
import tossing.probes.release_drop
import tossing.probes.wrist_flick
import tossing.probes.shake
from tossing.probes import list_probes, get_probe


PROBE_NAMES = ["vertical_toss", "forward_toss", "release_drop", "wrist_flick", "shake"]


def _make_obj(**overrides) -> ObjectSpec:
    defaults = dict(
        name="test", family="test",
        geom_type="sphere", geom_size=[0.05],
        mass=0.3, drag_coeff=0.0, com_offset=[0.0, 0.0, 0.0],
        inertia=0.001, cross_section_area=0.008,
        rgba=[0.8, 0.2, 0.2, 1.0],
    )
    defaults.update(overrides)
    return ObjectSpec(**defaults)


class TestProbeRegistry:
    def test_all_probes_registered(self):
        probes = list_probes()
        assert set(probes) == set(PROBE_NAMES)

    def test_get_probe_returns_controller(self):
        for pid in PROBE_NAMES:
            ctrl = get_probe(pid)
            assert ctrl.probe_type == pid

    def test_unknown_probe_raises(self):
        with pytest.raises(ValueError):
            get_probe("nonexistent_probe")


class TestVerticalToss:
    def test_returns_correct_keys(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        result = env.run_probe("vertical_toss")
        assert "apex_height" in result.observations
        assert "hang_time" in result.observations
        assert "landing_offset" in result.observations

    def test_apex_matches_ballistic(self):
        """With zero drag, apex should match v^2/(2g) within 5%."""
        env = TossEnv(_make_obj(drag_coeff=0.0), basket_distance=2.0)
        result = env.run_probe("vertical_toss")
        expected = ballistic_apex(2.0)  # launch speed is 2 m/s
        actual = result.observations["apex_height"]
        assert abs(actual - expected) / expected < 0.05

    def test_hang_time_matches_ballistic(self):
        env = TossEnv(_make_obj(drag_coeff=0.0), basket_distance=2.0)
        result = env.run_probe("vertical_toss")
        expected = ballistic_hang_time(2.0)
        actual = result.observations["hang_time"]
        assert abs(actual - expected) / expected < 0.05

    def test_drag_reduces_apex(self):
        no_drag = _make_obj(drag_coeff=0.0, cross_section_area=0.04)
        hi_drag = _make_obj(drag_coeff=2.0, cross_section_area=0.04, mass=0.1)

        env1 = TossEnv(no_drag, basket_distance=2.0)
        env2 = TossEnv(hi_drag, basket_distance=2.0)

        r1 = env1.run_probe("vertical_toss")
        r2 = env2.run_probe("vertical_toss")

        assert r1.observations["apex_height"] > r2.observations["apex_height"]


class TestForwardToss:
    def test_returns_correct_keys(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        result = env.run_probe("forward_toss")
        assert "landing_distance" in result.observations
        assert "flight_time" in result.observations
        assert "lateral_drift" in result.observations

    def test_no_drag_drift_near_zero(self):
        env = TossEnv(_make_obj(drag_coeff=0.0), basket_distance=2.0)
        result = env.run_probe("forward_toss")
        assert abs(result.observations["lateral_drift"]) < 0.1

    def test_drag_causes_negative_drift(self):
        env = TossEnv(_make_obj(drag_coeff=2.0, cross_section_area=0.04, mass=0.1),
                       basket_distance=2.0)
        result = env.run_probe("forward_toss")
        assert result.observations["lateral_drift"] < -0.05


class TestReleaseDrop:
    def test_returns_correct_keys(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        result = env.run_probe("release_drop")
        assert "fall_time" in result.observations
        assert "coefficient_of_restitution" in result.observations
        assert "bounce_count" in result.observations

    def test_fall_time_positive(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        result = env.run_probe("release_drop")
        assert result.observations["fall_time"] > 0


class TestWristFlick:
    def test_returns_correct_keys(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        result = env.run_probe("wrist_flick")
        assert "angular_velocity_response" in result.observations
        assert "angular_deceleration" in result.observations
        assert "precession_detected" in result.observations

    def test_inertia_sensitivity(self):
        """Higher inertia → lower angular velocity from same torque impulse."""
        lo_I = _make_obj(inertia=0.0005)
        hi_I = _make_obj(inertia=0.05)

        env1 = TossEnv(lo_I, basket_distance=2.0)
        env2 = TossEnv(hi_I, basket_distance=2.0)

        r1 = env1.run_probe("wrist_flick")
        r2 = env2.run_probe("wrist_flick")

        assert r1.observations["angular_velocity_response"] > r2.observations["angular_velocity_response"]


class TestShake:
    def test_returns_correct_keys(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        result = env.run_probe("shake")
        assert "peak_force" in result.observations
        assert "damping_ratio" in result.observations
        assert "perceived_resistance" in result.observations

    def test_heavier_object_more_resistance(self):
        light = _make_obj(mass=0.05)
        heavy = _make_obj(mass=2.0)

        env1 = TossEnv(light, basket_distance=2.0)
        env2 = TossEnv(heavy, basket_distance=2.0)

        r1 = env1.run_probe("shake")
        r2 = env2.run_probe("shake")

        assert r2.observations["peak_force"] > r1.observations["peak_force"]


class TestAllProbesResetProperly:
    def test_env_resets_after_each_probe(self):
        """After running any probe, object should be back at gripper."""
        env = TossEnv(_make_obj(), basket_distance=2.0)
        initial_z = env.object_pos[2]

        for probe_type in list_probes():
            env.run_probe(probe_type)
            assert abs(env.object_pos[2] - initial_z) < 0.2, \
                f"After {probe_type}, z={env.object_pos[2]} vs initial {initial_z}"
