"""VLM-only loop for tossing experiments.

Level 3 of the simplification ladder: no learned belief net, no learned throw MLP.
The VLM reads the rendered image + raw probe observations as text, picks the next
probe or commits to a throw, and emits throw parameters directly.
"""

from tossing.vlm.client import (
    VLMClient,
    AnthropicClient,
    OpenAIClient,
    FakeVLMClient,
    build_client,
)
from tossing.vlm.loop import run_episode, EpisodeResult
from tossing.vlm.parser import VLMAction, VLMParseError, parse_vlm_output

__all__ = [
    "VLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "FakeVLMClient",
    "build_client",
    "run_episode",
    "EpisodeResult",
    "VLMAction",
    "VLMParseError",
    "parse_vlm_output",
]
