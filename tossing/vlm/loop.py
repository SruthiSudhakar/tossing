"""Per-episode VLM-only loop.

The VLM decides at each turn whether to run another probe or commit to a throw.
When the probe budget is exhausted, the loop forces a THROW turn. Parse errors
get one retry (with the error appended to the user message); a second failure
aborts the episode with success=False.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tossing.types import ProbeResult, ThrowParams, ThrowResult
from tossing.vlm.client import VLMClient
from tossing.vlm.parser import VLMAction, VLMParseError, parse_vlm_output
from tossing.vlm.prompts import SYSTEM_PROMPT, build_user_message


@dataclass
class EpisodeResult:
    success: bool
    distance_to_basket: float
    n_probes_used: int
    probe_sequence: list[str] = field(default_factory=list)
    probe_observations: list[dict] = field(default_factory=list)
    final_throw: ThrowParams | None = None
    reasoning_traces: list[str] = field(default_factory=list)
    parse_errors: int = 0
    aborted: bool = False
    abort_reason: str | None = None
    videos: list[str] = field(default_factory=list)
    final_landing_x: float | None = None
    final_signed_error: float | None = None


def _render_image(env):
    """Prefer render_pair() for side+top PIL image."""
    if hasattr(env, "render_pair"):
        return env.render_pair()
    # Fallback: build PIL from first dict-render view.
    from PIL import Image
    views = env.render()
    return Image.fromarray(views["side"])


def run_episode(
    env,
    target_distance: float,
    max_probes: int,
    client: VLMClient,
    video_dir: str | Path | None = None,
    video_fps: int = 30,
) -> EpisodeResult:
    """Run a single VLM-only tossing episode.

    Args:
        env: TossEnv already constructed with the object and target distance.
        target_distance: meters (should match env.basket_distance).
        max_probes: hard cap on probes before a throw is forced.
        client: VLMClient to drive decisions.
        video_dir: if set, save one MP4 per probe and one for the final throw
            into this directory. Files are named `<turn>_<probe_id>.mp4` and
            `<turn>_throw.mp4`.
        video_fps: capture framerate when recording.
    """
    history: list[tuple[str, ProbeResult]] = []
    reasoning_traces: list[str] = []
    probe_sequence: list[str] = []
    probe_observations: list[dict] = []
    parse_errors = 0
    videos: list[str] = []

    if video_dir is not None:
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

    # Cap total VLM turns so a persistently-probing VLM can't loop forever.
    # max_probes probes + 1 throw + a small buffer for forced throws.
    max_turns = max_probes + 2
    force_throw = max_probes == 0  # zero-shot

    for turn in range(max_turns):
        image = _render_image(env)
        probes_remaining = max_probes - len(history)
        user_msg = build_user_message(
            target_distance=target_distance,
            history=history,
            probes_remaining=max(0, probes_remaining),
            force_throw=force_throw,
        )

        # Call VLM with one retry on parse failure.
        try:
            raw = client.complete(image, SYSTEM_PROMPT, user_msg)
            reasoning_traces.append(raw)
            action = parse_vlm_output(raw)
        except VLMParseError as e:
            parse_errors += 1
            retry_msg = (
                user_msg
                + "\n\nYOUR PREVIOUS RESPONSE COULD NOT BE PARSED: "
                + str(e)
                + "\nOutput the structured ACTION block exactly as specified."
            )
            try:
                raw = client.complete(image, SYSTEM_PROMPT, retry_msg)
                reasoning_traces.append(raw)
                action = parse_vlm_output(raw)
            except VLMParseError as e2:
                parse_errors += 1
                return EpisodeResult(
                    success=False,
                    distance_to_basket=float("inf"),
                    n_probes_used=len(history),
                    probe_sequence=probe_sequence,
                    probe_observations=probe_observations,
                    final_throw=None,
                    reasoning_traces=reasoning_traces,
                    parse_errors=parse_errors,
                    aborted=True,
                    abort_reason=f"parse_error_after_retry: {e2}",
                    videos=videos,
                )

        if action.kind == "PROBE":
            if probes_remaining <= 0:
                # VLM ignored the budget. Force a throw next turn.
                force_throw = True
                continue
            probe_id = action.probe_id
            if video_dir is not None:
                env.start_recording(fps=video_fps)
            result = env.run_probe(probe_id)
            if video_dir is not None:
                path = video_dir / f"{turn + 1:02d}_{probe_id}.mp4"
                if env.save_recording(path, fps=video_fps) > 0:
                    videos.append(str(path))
            history.append((probe_id, result))
            probe_sequence.append(probe_id)
            probe_observations.append(dict(result.observations))
            # After this probe, if we're now out of budget, force a throw.
            if len(history) >= max_probes:
                force_throw = True
            continue

        # action.kind == "THROW"
        throw = action.throw
        if video_dir is not None:
            env.start_recording(fps=video_fps)
        throw_result: ThrowResult = env.throw(throw)
        if video_dir is not None:
            path = video_dir / f"{turn + 1:02d}_throw.mp4"
            if env.save_recording(path, fps=video_fps) > 0:
                videos.append(str(path))
        landing_x = None
        signed_error = None
        if throw_result.trajectory is not None and len(throw_result.trajectory) > 0:
            landing_x = float(throw_result.trajectory[-1][0])
            signed_error = landing_x - target_distance
        return EpisodeResult(
            success=bool(throw_result.success),
            distance_to_basket=float(throw_result.distance_to_basket),
            n_probes_used=len(history),
            probe_sequence=probe_sequence,
            probe_observations=probe_observations,
            final_throw=throw,
            reasoning_traces=reasoning_traces,
            parse_errors=parse_errors,
            aborted=False,
            abort_reason=None,
            videos=videos,
            final_landing_x=landing_x,
            final_signed_error=signed_error,
        )

    # Ran out of turns without a throw.
    return EpisodeResult(
        success=False,
        distance_to_basket=float("inf"),
        n_probes_used=len(history),
        probe_sequence=probe_sequence,
        probe_observations=probe_observations,
        final_throw=None,
        reasoning_traces=reasoning_traces,
        parse_errors=parse_errors,
        aborted=True,
        abort_reason=f"max_turns ({max_turns}) reached without THROW",
        videos=videos,
    )
