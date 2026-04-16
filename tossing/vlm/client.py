"""VLM client abstractions with a shared JSON cache.

Two real backends (Claude via anthropic, GPT via openai) plus a FakeVLMClient
that returns canned responses for tests. All clients share a common cache keyed
on (provider, model, prompt_hash) so repeat eval runs are deterministic and cheap.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from PIL import Image


def _encode_image_png_b64(image: Image.Image) -> str:
    """Encode a PIL image as base64 PNG."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _cache_key(provider: str, model: str, system: str, user: str, image_b64: str) -> str:
    """Stable hash for a (provider, model, prompt, image) tuple."""
    h = hashlib.sha256()
    h.update(provider.encode())
    h.update(b"\0")
    h.update(model.encode())
    h.update(b"\0")
    h.update(system.encode())
    h.update(b"\0")
    h.update(user.encode())
    h.update(b"\0")
    h.update(image_b64.encode())
    return h.hexdigest()


class VLMClient(ABC):
    """Abstract base for VLM backends.

    Concrete subclasses implement `_call_api`. The public `complete` method
    wraps it with a disk cache.
    """

    provider: str = "abstract"
    model: str = "abstract"

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, image: Image.Image, system: str, user: str) -> str:
        image_b64 = _encode_image_png_b64(image)
        key = _cache_key(self.provider, self.model, system, user, image_b64)

        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{key}.json"
            if cache_path.exists():
                return json.loads(cache_path.read_text())["response"]

        response = self._call_api(image_b64, system, user)

        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{key}.json"
            cache_path.write_text(json.dumps({
                "provider": self.provider,
                "model": self.model,
                "system": system,
                "user": user,
                "response": response,
            }, indent=2))

        return response

    @abstractmethod
    def _call_api(self, image_b64: str, system: str, user: str) -> str:
        ...


class AnthropicClient(VLMClient):
    """Claude via the anthropic SDK."""

    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-4-5",
                 cache_dir: str | Path | None = None,
                 max_tokens: int = 2048):
        super().__init__(cache_dir=cache_dir)
        self.model = model
        self.max_tokens = max_tokens
        import anthropic  # lazy import
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def _call_api(self, image_b64: str, system: str, user: str) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user},
                ],
            }],
        )
        # Concatenate all text blocks (Claude sometimes splits).
        parts = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)


class OpenAIClient(VLMClient):
    """GPT via the openai SDK."""

    provider = "openai"

    def __init__(self, model: str = "gpt-5.4-mini-2026-03-17",
                 cache_dir: str | Path | None = None,
                 max_tokens: int = 2048):
        super().__init__(cache_dir=cache_dir)
        self.model = model
        self.max_tokens = max_tokens
        import openai  # lazy import
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY

    def _call_api(self, image_b64: str, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        )
        return resp.choices[0].message.content or ""


class FakeVLMClient(VLMClient):
    """Test double that returns scripted responses.

    Pass either a list (consumed in order) or a callable
    (response_fn(image, system, user) -> str).
    """

    provider = "fake"
    model = "fake"

    def __init__(self,
                 responses: list[str] | Callable[[Image.Image, str, str], str] | None = None):
        super().__init__(cache_dir=None)
        self._responses = responses
        self._idx = 0
        self.calls: list[tuple[str, str]] = []  # (system, user) per call

    def complete(self, image: Image.Image, system: str, user: str) -> str:
        # Bypass the cache entirely so tests are hermetic.
        self.calls.append((system, user))
        return self._call_api("", system, user)

    def _call_api(self, image_b64: str, system: str, user: str) -> str:
        if callable(self._responses):
            return self._responses(None, system, user)
        if isinstance(self._responses, list):
            if self._idx >= len(self._responses):
                raise RuntimeError(
                    f"FakeVLMClient ran out of canned responses at call #{self._idx + 1}"
                )
            r = self._responses[self._idx]
            self._idx += 1
            return r
        raise RuntimeError("FakeVLMClient was not given responses")


def build_client(provider: str,
                 cache_dir: str | Path | None = None,
                 model: str | None = None) -> VLMClient:
    """Factory: instantiate a VLM client by provider name."""
    provider = provider.lower()
    if provider in ("claude", "anthropic"):
        kwargs = {"cache_dir": cache_dir}
        if model is not None:
            kwargs["model"] = model
        return AnthropicClient(**kwargs)
    if provider in ("openai", "gpt-4o", "gpt4o", "gpt-5.4-mini-2026-03-17"):
        kwargs = {"cache_dir": cache_dir}
        if model is not None:
            kwargs["model"] = model
        return OpenAIClient(**kwargs)
    raise ValueError(f"Unknown VLM provider: {provider!r}. Use 'claude' or 'openai'.")
