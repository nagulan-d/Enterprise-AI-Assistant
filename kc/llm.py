"""Thin LLM wrapper. Uses Claude through the `anthropic` SDK when an API key is present;
otherwise `available` is False and the copilot switches to its offline extractive mode."""

from __future__ import annotations

import os

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMError(RuntimeError):
    """Raised when the model call fails, so callers can degrade gracefully."""


class LLM:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self.model = model or os.environ.get("COPILOT_MODEL") or DEFAULT_MODEL
        self._client = None
        self.reason = ""
        if not self.api_key:
            self.reason = "No ANTHROPIC_API_KEY set"
            return
        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            self.reason = "The 'anthropic' package is not installed (pip install anthropic)"

    @property
    def available(self) -> bool:
        return self._client is not None

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 1500,
        temperature: float = 0.2,
    ) -> str:
        if not self.available:
            raise LLMError(self.reason or "LLM not configured")
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
        except Exception as exc:  # network, auth, rate limit...
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        if not text:
            raise LLMError("Model returned an empty response")
        return text
