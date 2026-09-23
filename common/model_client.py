"""Thin wrapper around an OpenAI-compatible chat endpoint (vLLM, or any other
server exposing /v1/chat/completions). Shared by every experiment so retry/
error handling for one flaky problem doesn't kill an entire run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class ModelConfig:
    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.2
    max_tokens: int = 1536
    retries: int = 2
    retry_backoff: float = 2.0
    # Per-request ceiling. The SDK default is 10 minutes, which turns a dead
    # server into a multi-hour hang instead of a fast, visible failure.
    timeout: float = 180.0
    # Passed straight through to the server; used for vendor-specific knobs
    # such as gpt-oss's {"reasoning_effort": "low"}.
    extra_body: Dict[str, Any] = field(default_factory=dict)


def make_client(cfg: ModelConfig) -> OpenAI:
    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                  timeout=cfg.timeout, max_retries=0)  # retries handled in chat_messages


def chat(client: OpenAI, cfg: ModelConfig, prompt: str, system: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Returns (raw_text, error). Never raises -- a single bad response
    shouldn't abort a batch run; callers check `error`."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat_messages(client, cfg, messages)


def chat_messages(client: OpenAI, cfg: ModelConfig,
                  messages: List[Dict[str, Any]]) -> tuple[str, Optional[str]]:
    """Multi-turn variant of `chat`: caller owns the full message list. Used by
    conversational experiments where the model must see its own prior turns."""
    sent = messages
    for attempt in range(cfg.retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=sent,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                **({"extra_body": cfg.extra_body} if cfg.extra_body else {}),
            )
            msg = resp.choices[0].message
            text = msg.content or ""
            if not text.strip():
                # Reasoning models (gpt-oss, and anything else vLLM parses into a
                # separate channel) put the chain of thought in a side field and
                # can leave `content` null when the answer budget runs out.
                # vLLM has used both spellings depending on version/model, so try
                # both -- otherwise these turns silently become empty strings.
                text = (getattr(msg, "reasoning_content", None)
                        or getattr(msg, "reasoning", None) or "")
            return text, None
        except Exception as exc:  # network / server hiccup for this one call
            # Some chat templates (Gemma, among others) have no system role and
            # reject the request outright. Fold the system prompt into the first
            # user message and try again, rather than losing the whole run.
            if _is_system_role_error(exc) and sent is messages and _has_system(messages):
                sent = _merge_system(messages)
                continue
            if attempt == cfg.retries:
                return "", f"{type(exc).__name__}: {exc}"
            time.sleep(cfg.retry_backoff * (2 ** attempt))
    return "", "unreachable"


def _has_system(messages: List[Dict[str, Any]]) -> bool:
    return bool(messages) and messages[0].get("role") == "system"


def _is_system_role_error(exc: Exception) -> bool:
    blob = str(exc).lower()
    return "system role" in blob or "system instruction" in blob or (
        "system" in blob and "not supported" in blob)


def _merge_system(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepend the system prompt to the first user turn."""
    system, rest = messages[0]["content"], list(messages[1:])
    for i, m in enumerate(rest):
        if m.get("role") == "user":
            rest[i] = {**m, "content": f"{system}\n\n---\n\n{m['content']}"}
            return rest
    return [{"role": "user", "content": system}] + rest
