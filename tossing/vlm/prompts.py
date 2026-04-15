"""Prompt templates for the VLM-only tossing loop."""

from __future__ import annotations

from tossing.types import ProbeResult


SYSTEM_PROMPT = """You are a physics experiment designer controlling a robot tossing system.

Your job is to land a novel object in a basket at a known distance. You do not know
the object's hidden physics (mass, drag, center-of-mass offset, moment of inertia).
You may run up to a small number of diagnostic probes before committing to a throw.

PROBE MENU (each costs nothing and returns structured numerical observations):
  vertical_toss = Launch straight up at 2 m/s.
       Observations: apex_height (m), hang_time (s), landing_offset (m).
       Diagnostic: mass (hang time), drag (apex shortfall vs ballistic).
  forward_toss = Launch at 45 degrees, 3 m/s.
       Observations: landing_distance (m), flight_time (s), lateral_drift (m).
       Diagnostic: drag (range shortfall), CoM offset (lateral drift).
  release_drop = Open gripper, let fall from 1.5 m.
       Observations: fall_time (s), coefficient_of_restitution, bounce_count.
       Diagnostic: mass vs drag (fall time), restitution.
  wrist_flick = Apply a known torque impulse.
       Observations: angular_velocity_response (rad/s), angular_deceleration (rad/s^2),
                     precession_detected (0 or 1), translational_drift (m).
       Diagnostic: moment of inertia (angular response ~ 1/I), CoM offset (precession, drift).
  shake = Oscillate gripper +/- 5 cm at 4 Hz.
       Observations: peak_force (N), damping_ratio, perceived_resistance.
       Diagnostic: mass, internal inertia distribution.
  THROW = Commit to a final throw.

THROW PARAMETERS (when committing):
  theta: release angle in degrees, in [20, 80]
  v:     release speed in m/s, in [1, 8]
  dt:    release timing offset in seconds, in [-0.1, 0.1]

Each turn you will see the object (side + top view), the target basket distance,
observations from every probe you have already run, and the probes remaining.

Reason step by step. At the END of your response, emit exactly one structured block.
If probing: one line with just the probe name.
  ACTION: vertical_toss
If throwing: four lines with the action and numeric parameters.
  ACTION: THROW
  THETA: 45.0
  V: 4.5
  DT: 0.0

Only the final structured block is parsed; reason freely before it but keep the
tail of your response clean.
"""


def format_observations(result: ProbeResult) -> str:
    """Render a ProbeResult's observations as key=value text."""
    parts = []
    for key, value in result.observations.items():
        if isinstance(value, bool):
            parts.append(f"{key}={int(value)}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def build_user_message(
    target_distance: float,
    history: list[tuple[str, ProbeResult]],
    probes_remaining: int,
    force_throw: bool = False,
) -> str:
    """Build the per-turn user message.

    Args:
        target_distance: basket distance in meters.
        history: list of (probe_id, ProbeResult) for probes already run this episode.
        probes_remaining: how many probes the VLM can still choose before being
            forced to throw.
        force_throw: if True, instruct the VLM to output ACTION: THROW only.
    """
    lines = [
        f"TARGET: basket at x = {target_distance:.2f} m.",
        "",
    ]

    if history:
        lines.append(f"PROBES COMPLETED THIS EPISODE ({len(history)}):")
        for probe_id, result in history:
            lines.append(f"  {probe_id}: {format_observations(result)}")
    else:
        lines.append("PROBES COMPLETED THIS EPISODE: none.")
    lines.append("")

    lines.append(f"PROBES REMAINING: {probes_remaining}.")
    lines.append("")

    if force_throw:
        lines.append(
            "BUDGET EXHAUSTED. You MUST commit to a throw now. "
            "Output only an ACTION: THROW block (with THETA, V, DT)."
        )
    elif probes_remaining == 0:
        lines.append(
            "You have no probes remaining. You must commit to a throw now. "
            "Output an ACTION: THROW block (with THETA, V, DT)."
        )
    else:
        lines.append(
            "Decide: run one more probe, or commit to a throw. Think about which "
            "hidden physics parameters are still most uncertain and whether another "
            "probe would meaningfully reduce that uncertainty given the budget."
        )

    return "\n".join(lines)
