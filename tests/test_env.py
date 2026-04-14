"""Tests for the TossEnv simulator."""

import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import pytest

from tossing.types import ObjectSpec, ThrowParams
from tossing.env import TossEnv, LAUNCHER_HEIGHT
from tossing.physics import ballistic_apex, ballistic_hang_time, ballistic_fall_time


def _make_obj(**overrides) -> ObjectSpec:
    """Create a test ObjectSpec with sensible defaults."""
    defaults = dict(
        name="test", family="test",
        geom_type="sphere", geom_size=[0.05],
        mass=0.3, drag_coeff=0.0, com_offset=[0.0, 0.0, 0.0],
        inertia=0.001, cross_section_area=0.008,
        rgba=[0.8, 0.2, 0.2, 1.0],
    )
    defaults.update(overrides)
    return ObjectSpec(**defaults)


class TestEnvCreation:
    def test_env_creates(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        assert env.object_pos is not None

    def test_object_starts_at_launcher_height(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        assert abs(env.object_pos[2] - LAUNCHER_HEIGHT) < 0.05

    def test_object_starts_at_origin_x(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        assert abs(env.object_pos[0]) < 0.01

    def test_different_basket_distances(self):
        for d in [1.0, 2.0, 3.0]:
            env = TossEnv(_make_obj(), basket_distance=d)
            basket_x = env.data.xpos[env.basket_body_id][0]
            assert abs(basket_x - d) < 0.01


class TestPhysics:
    def test_vertical_launch_apex_no_drag(self):
        """Object launched upward should reach expected apex (within 5%)."""
        env = TossEnv(_make_obj(drag_coeff=0.0), basket_distance=2.0)
        vz = 2.0
        env._release(linear_vel=np.array([0.0, 0.0, vz]))

        max_z = env.object_pos[2]
        release_z = max_z
        for _ in range(2000):
            env._step()
            z = env.object_pos[2]
            if z > max_z:
                max_z = z

        apex = max_z - release_z
        expected = ballistic_apex(vz)
        assert abs(apex - expected) / expected < 0.05, f"Apex {apex:.4f} vs expected {expected:.4f}"

    def test_2d_enforcement(self):
        """Y position should stay near zero during flight."""
        env = TossEnv(_make_obj(), basket_distance=2.0)
        env._release(linear_vel=np.array([3.0, 0.0, 3.0]))

        for _ in range(1000):
            env._step()
            y = env.object_pos[1]
            assert abs(y) < 0.01, f"Y position drifted to {y}"

    def test_drag_reduces_range(self):
        """Object with drag should travel shorter distance than without."""
        no_drag = _make_obj(drag_coeff=0.0, cross_section_area=0.01)
        hi_drag = _make_obj(drag_coeff=2.0, cross_section_area=0.04)

        env1 = TossEnv(no_drag, basket_distance=5.0)
        env2 = TossEnv(hi_drag, basket_distance=5.0)

        for env in [env1, env2]:
            env._release(linear_vel=np.array([3.0, 0.0, 3.0]))
        for _ in range(3000):
            env1._step()
            env2._step()

        # No-drag should have traveled farther
        assert env1.object_pos[0] > env2.object_pos[0]


class TestGraspRelease:
    def test_soft_reset_returns_to_start(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        initial_pos = env.object_pos.copy()

        env._release(linear_vel=np.array([2.0, 0.0, 2.0]))
        for _ in range(500):
            env._step()

        env._soft_reset()
        reset_pos = env.object_pos
        assert np.allclose(reset_pos, initial_pos, atol=0.1)


class TestThrow:
    def test_throw_returns_result(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        result = env.throw(ThrowParams(theta=45, v=4.0))
        assert result.trajectory is not None
        assert result.trajectory.shape[1] == 6
        assert result.flight_time > 0

    def test_throw_can_succeed(self):
        """With right parameters, throw should land in basket."""
        env = TossEnv(_make_obj(drag_coeff=0.0), basket_distance=2.0)
        # These params were validated earlier
        result = env.throw(ThrowParams(theta=35, v=3.5))
        assert result.distance_to_basket < 0.3


class TestRender:
    def test_render_returns_images(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        imgs = env.render()
        assert "side" in imgs and "top" in imgs
        assert imgs["side"].shape == (480, 640, 3)
        assert imgs["top"].shape == (480, 640, 3)

    def test_render_pair(self):
        env = TossEnv(_make_obj(), basket_distance=2.0)
        pair = env.render_pair()
        assert pair.size == (1280, 480)
