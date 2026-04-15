"""Parse structured VLM output.

The VLM is instructed to emit exactly one structured tail block per turn:
either `ACTION: <probe_name>` for a probe, or `ACTION: THROW` followed by
THETA, V, DT lines. The parser is lenient about surrounding reasoning text
and uses the LAST occurrence of each key, so the VLM may restate things
mid-thought without breaking the parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from tossing.types import ThrowParams


PROBE_IDS = ("vertical_toss", "forward_toss", "release_drop", "wrist_flick", "shake")

THETA_MIN, THETA_MAX = 20.0, 80.0
V_MIN, V_MAX = 1.0, 8.0
DT_MIN, DT_MAX = -0.1, 0.1


class VLMParseError(ValueError):
    """Raised when the VLM output cannot be parsed into a valid action."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


@dataclass
class VLMAction:
    kind: Literal["PROBE", "THROW"]
    probe_id: str | None = None
    throw: ThrowParams | None = None


_ACTION_RE = re.compile(
    r"ACTION\s*:\s*(vertical_toss|forward_toss|release_drop|wrist_flick|shake|THROW)\b",
    re.IGNORECASE,
)
_THETA_RE = re.compile(r"THETA\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_V_RE = re.compile(r"\bV\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_DT_RE = re.compile(r"\bDT\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def _last_match(pattern: re.Pattern, text: str) -> re.Match | None:
    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def parse_vlm_output(text: str) -> VLMAction:
    """Extract the structured action from a VLM response.

    Raises VLMParseError if no valid action is found.
    """
    m = _last_match(_ACTION_RE, text)
    if m is None:
        raise VLMParseError(
            "No 'ACTION: <probe_name|THROW>' line found in VLM output.",
            raw=text,
        )
    token = m.group(1).lower()

    if token in PROBE_IDS:
        return VLMAction(kind="PROBE", probe_id=token)

    # token == "throw"
    theta_m = _last_match(_THETA_RE, text)
    v_m = _last_match(_V_RE, text)
    dt_m = _last_match(_DT_RE, text)

    missing = []
    if theta_m is None:
        missing.append("THETA")
    if v_m is None:
        missing.append("V")
    if dt_m is None:
        missing.append("DT")
    if missing:
        raise VLMParseError(
            f"ACTION: THROW missing parameters: {', '.join(missing)}.",
            raw=text,
        )

    try:
        theta = float(theta_m.group(1))
        v = float(v_m.group(1))
        dt = float(dt_m.group(1))
    except ValueError as e:
        raise VLMParseError(f"Could not parse throw parameter as float: {e}", raw=text)

    # Validate ranges; the loop clamps rather than rejecting so a slightly
    # out-of-bounds number still yields a throw, but we raise if wildly off.
    def _clamp(x: float, lo: float, hi: float, name: str) -> float:
        if x < lo - 0.5 * (hi - lo) or x > hi + 0.5 * (hi - lo):
            raise VLMParseError(
                f"{name}={x} is far outside valid range [{lo}, {hi}].",
                raw=text,
            )
        return max(lo, min(hi, x))

    theta = _clamp(theta, THETA_MIN, THETA_MAX, "THETA")
    v = _clamp(v, V_MIN, V_MAX, "V")
    dt = _clamp(dt, DT_MIN, DT_MAX, "DT")

    return VLMAction(
        kind="THROW",
        throw=ThrowParams(theta=theta, v=v, dt=dt),
    )
