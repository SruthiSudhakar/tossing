"""Tests for the VLM-only tossing loop.

Uses FakeVLMClient so nothing here hits a real API. The loop still instantiates
a real TossEnv because probe and throw execution are not part of the VLM layer
being tested — they're exercised by the existing probe/env tests.
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import pytest

from tossing.types import ObjectSpec, ThrowParams
from tossing.env import TossEnv

# Register probes
import tossing.probes.vertical_toss  # noqa: F401
import tossing.probes.forward_toss  # noqa: F401
import tossing.probes.release_drop  # noqa: F401
import tossing.probes.wrist_flick  # noqa: F401
import tossing.probes.shake  # noqa: F401

from tossing.vlm import (
    FakeVLMClient,
    VLMParseError,
    parse_vlm_output,
    run_episode,
)


def _make_obj() -> ObjectSpec:
    return ObjectSpec(
        name="test", family="test",
        geom_type="sphere", geom_size=[0.05],
        mass=0.3, drag_coeff=0.0, com_offset=[0.0, 0.0, 0.0],
        inertia=0.001, cross_section_area=0.008,
        rgba=[0.8, 0.2, 0.2, 1.0],
    )


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

class TestParser:
    def test_probe_action(self):
        action = parse_vlm_output("Let me think...\nACTION: release_drop")
        assert action.kind == "PROBE"
        assert action.probe_id == "release_drop"

    def test_probe_action_case_insensitive(self):
        action = parse_vlm_output("action: VERTICAL_TOSS")
        assert action.kind == "PROBE"
        assert action.probe_id == "vertical_toss"

    def test_throw_action(self):
        text = "Reasoning goes here.\nACTION: THROW\nTHETA: 45.0\nV: 4.5\nDT: 0.0"
        action = parse_vlm_output(text)
        assert action.kind == "THROW"
        assert isinstance(action.throw, ThrowParams)
        assert action.throw.theta == 45.0
        assert action.throw.v == 4.5
        assert action.throw.dt == 0.0

    def test_throw_with_negative_dt(self):
        text = "ACTION: THROW\nTHETA: 60\nV: 3.2\nDT: -0.05"
        action = parse_vlm_output(text)
        assert action.throw.dt == -0.05

    def test_throw_clamps_slightly_out_of_range(self):
        # theta=85 is outside [20,80] but within the 50% tolerance band — should clamp to 80.
        text = "ACTION: THROW\nTHETA: 85\nV: 5.0\nDT: 0.0"
        action = parse_vlm_output(text)
        assert action.throw.theta == 80.0

    def test_throw_raises_for_wildly_out_of_range(self):
        text = "ACTION: THROW\nTHETA: 500\nV: 5.0\nDT: 0.0"
        with pytest.raises(VLMParseError):
            parse_vlm_output(text)

    def test_uses_last_occurrence(self):
        # VLM may restate mid-reasoning; we keep the last.
        text = "I might do ACTION: vertical_toss\nbut actually\nACTION: wrist_flick"
        action = parse_vlm_output(text)
        assert action.probe_id == "wrist_flick"

    def test_missing_action_line(self):
        with pytest.raises(VLMParseError):
            parse_vlm_output("I think I should probe but I forgot the action format.")

    def test_throw_missing_params(self):
        with pytest.raises(VLMParseError):
            parse_vlm_output("ACTION: THROW\nTHETA: 45\nV: 5")  # no DT


# ------------------------------------------------------------------
# Loop
# ------------------------------------------------------------------

@pytest.fixture
def env():
    return TossEnv(_make_obj(), basket_distance=2.0)


class TestLoop:
    def test_zero_shot_throws_immediately(self, env):
        client = FakeVLMClient(responses=[
            "ACTION: THROW\nTHETA: 45\nV: 5.0\nDT: 0.0",
        ])
        result = run_episode(env, target_distance=2.0, max_probes=0, client=client)
        assert result.n_probes_used == 0
        assert result.final_throw is not None
        assert result.final_throw.theta == 45.0
        assert not result.aborted
        assert len(client.calls) == 1

    def test_probe_then_throw(self, env):
        client = FakeVLMClient(responses=[
            "I'll probe first.\nACTION: vertical_toss",
            "Now I have info.\nACTION: THROW\nTHETA: 50\nV: 4.0\nDT: 0.0",
        ])
        result = run_episode(env, target_distance=2.0, max_probes=3, client=client)
        assert result.n_probes_used == 1
        assert result.probe_sequence == ["vertical_toss"]
        assert len(result.probe_observations) == 1
        assert "apex_height" in result.probe_observations[0]
        assert result.final_throw.theta == 50.0

    def test_budget_forces_throw(self, env):
        # Budget = 2, VLM tries to probe 3 times then throws.
        # After 2 probes, the loop must force a throw.
        client = FakeVLMClient(responses=[
            "ACTION: vertical_toss",
            "ACTION: forward_toss",
            # Third response: even though VLM tries another probe, force_throw
            # was set to True, so the prompt says "throw now". The FakeVLM
            # doesn't see the prompt changes, but it should just return a THROW.
            "ACTION: THROW\nTHETA: 45\nV: 5.0\nDT: 0.0",
        ])
        result = run_episode(env, target_distance=2.0, max_probes=2, client=client)
        assert result.n_probes_used == 2
        assert result.final_throw is not None
        assert not result.aborted

    def test_vlm_over_probes_ignored_and_forced(self, env):
        # VLM asks for a 3rd probe when budget is 2. Loop should ignore it
        # and continue requesting a decision until THROW.
        client = FakeVLMClient(responses=[
            "ACTION: vertical_toss",
            "ACTION: forward_toss",
            "ACTION: release_drop",  # over-budget, loop sets force_throw and re-asks
            "ACTION: THROW\nTHETA: 45\nV: 5.0\nDT: 0.0",
        ])
        result = run_episode(env, target_distance=2.0, max_probes=2, client=client)
        assert result.n_probes_used == 2
        assert result.probe_sequence == ["vertical_toss", "forward_toss"]
        assert not result.aborted

    def test_parse_error_retry_succeeds(self, env):
        client = FakeVLMClient(responses=[
            "I am not going to output an action. Oops.",
            "ACTION: THROW\nTHETA: 45\nV: 5.0\nDT: 0.0",
        ])
        result = run_episode(env, target_distance=2.0, max_probes=0, client=client)
        assert result.parse_errors == 1
        assert not result.aborted
        assert result.final_throw is not None

    def test_parse_error_twice_aborts(self, env):
        client = FakeVLMClient(responses=[
            "no action",
            "still no action",
        ])
        result = run_episode(env, target_distance=2.0, max_probes=0, client=client)
        assert result.aborted
        assert result.parse_errors == 2
        assert not result.success

    def test_video_dir_writes_probe_and_throw_mp4s(self, env, tmp_path):
        client = FakeVLMClient(responses=[
            "ACTION: vertical_toss",
            "ACTION: THROW\nTHETA: 45\nV: 5.0\nDT: 0.0",
        ])
        result = run_episode(env, target_distance=2.0, max_probes=2,
                             client=client, video_dir=tmp_path)
        assert len(result.videos) == 2
        probe_mp4 = tmp_path / "01_vertical_toss.mp4"
        throw_mp4 = tmp_path / "02_throw.mp4"
        assert probe_mp4.exists() and probe_mp4.stat().st_size > 0
        assert throw_mp4.exists() and throw_mp4.stat().st_size > 0
