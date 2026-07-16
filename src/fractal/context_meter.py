from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fractal.agent.service import load_workspace_instructions
from fractal.agent.signature import build_edit_workspace_signature

# Measured with ``litellm.token_counter(model="gpt-5.5", ...)`` against the
# PredictRLM 0.7.2 base instruction template after substituting Fractal's fixed
# input and output fields. Keep this static: the meter must not call
# PredictRLM's private prompt-construction APIs.
PREDICT_RLM_BASE_PROMPT_TOKEN_ALLOWANCE = 3364


@dataclass(frozen=True)
class ContextEstimateCacheKey:
    workspace_path: str
    included_paths: tuple[str, ...]
    session_digest: str
    model_label: str
    max_iterations: int
    instructions_mtime_ns: int | None
    instructions_size: int | None


def context_estimate_cache_key(runtime: object) -> ContextEstimateCacheKey:
    workspace_path = Path(getattr(runtime, "workspace_path")).resolve()
    included_paths = tuple(
        str(Path(path).resolve()) for path in getattr(runtime, "included_paths", []) or []
    )
    instructions_mtime_ns: int | None = None
    instructions_size: int | None = None
    try:
        stat = (workspace_path / "AGENTS.md").stat()
    except OSError:
        pass
    else:
        instructions_mtime_ns = stat.st_mtime_ns
        instructions_size = stat.st_size

    return ContextEstimateCacheKey(
        workspace_path=str(workspace_path),
        included_paths=included_paths,
        session_digest=_session_digest(getattr(runtime, "session")),
        model_label=_model_label(runtime),
        max_iterations=_max_iterations(runtime),
        instructions_mtime_ns=instructions_mtime_ns,
        instructions_size=instructions_size,
    )


def estimate_next_context_tokens(runtime: object, *, user_message: str = "") -> int | None:
    messages = build_next_context_messages(runtime, user_message=user_message)
    tokens = count_messages_tokens(_model_label(runtime), messages)
    if tokens is None:
        return None
    return PREDICT_RLM_BASE_PROMPT_TOKEN_ALLOWANCE + tokens


def build_next_context_messages(
    runtime: object,
    *,
    user_message: str = "",
) -> list[dict[str, Any]]:
    """Format only Fractal-owned context for the next-turn token estimate.

    PredictRLM constructs its own action prompt and runtime input adapter
    instructions during ``acall``. Its measured fixed instruction allowance is
    added separately by ``estimate_next_context_tokens``.
    """

    workspace_path = Path(getattr(runtime, "workspace_path")).resolve()
    included_paths = [
        str(Path(path).resolve()) for path in getattr(runtime, "included_paths", []) or []
    ]
    session = getattr(runtime, "session")
    signature = build_edit_workspace_signature(
        session.summary(),
        workspace_instructions=load_workspace_instructions(workspace_path),
    )
    turn_context = "\n".join(
        (
            "## Fractal-managed next-turn inputs",
            f"Primary workspace: {workspace_path}",
            f"Included workspaces: {included_paths or 'none'}",
            f"User message: {user_message}",
            f"Session history: {json.dumps(session.session_history_payload(), default=str)}",
            f"Iteration: 1/{_max_iterations(runtime)}",
        )
    )
    return [
        {"role": "system", "content": signature.instructions},
        {"role": "user", "content": turn_context},
    ]


def count_messages_tokens(
    model_label: str,
    messages: list[dict[str, Any]],
) -> int | None:
    try:
        import litellm

        tokens = litellm.token_counter(model=model_label, messages=messages)
        if isinstance(tokens, int) and tokens > 0:
            return tokens
    except Exception:
        pass
    return _count_messages_with_tiktoken(model_label, messages)


def _count_messages_with_tiktoken(
    model_label: str,
    messages: list[dict[str, Any]],
) -> int | None:
    try:
        import tiktoken
    except Exception:
        return None

    normalized_model = model_label.removeprefix("openai/")
    try:
        encoding = tiktoken.encoding_for_model(normalized_model)
    except Exception:
        try:
            encoding = tiktoken.get_encoding("o200k_base")
        except Exception:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None

    tokens = 2
    for message in messages:
        tokens += 4
        tokens += len(encoding.encode(str(message.get("role", ""))))
        tokens += len(encoding.encode(_message_content_text(message.get("content", ""))))
        name = message.get("name")
        if name is not None:
            tokens += 1 + len(encoding.encode(str(name)))
    return tokens


def _message_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_message_content_text(item) for item in content)
    if isinstance(content, dict):
        for key in ("text", "content", "url", "image_url"):
            value = content.get(key)
            if value is not None:
                return _message_content_text(value)
        try:
            return json.dumps(content, sort_keys=True, default=str)
        except TypeError:
            return str(content)
    return str(content)


def _session_digest(session: object) -> str:
    summary = getattr(session, "summary_model")
    history = getattr(session, "history", [])
    raw = {
        "summary": _model_json(summary),
        "history": [
            {
                "turn_id": getattr(turn, "turn_id", ""),
                "status": getattr(turn, "status", ""),
                "updated_at": getattr(turn, "updated_at", ""),
            }
            for turn in history
        ],
    }
    payload = json.dumps(raw, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_json(value: object) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return json.dumps(value, sort_keys=True, default=str)


def _model_label(runtime: object) -> str:
    model_label = getattr(runtime, "model_label", "")
    return str(model_label or "")


def _max_iterations(runtime: object) -> int:
    agent = getattr(runtime, "agent", None)
    value = getattr(agent, "max_iterations", 30)
    try:
        max_iterations = int(value)
    except (TypeError, ValueError):
        return 30
    return max(max_iterations, 1)
