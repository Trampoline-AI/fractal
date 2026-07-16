from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip(
    "predict_rlm",
    reason="predict-rlm is required for Fractal context estimation",
)


class FakeAgent:
    max_iterations = 12


class FakeRuntime:
    def __init__(self, tmp_path: Path) -> None:
        from fractal.session import FractalSession

        self.workspace_path = tmp_path
        self.included_paths: list[Path] = []
        self.session = FractalSession(session_id="context-test")
        self.agent = FakeAgent()
        self.model_label = "gpt-5.5"


def test_build_next_context_messages_formats_initial_action_prompt(tmp_path: Path) -> None:
    from fractal.context_meter import build_next_context_messages

    runtime = FakeRuntime(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Always run `uv run pytest`.\n", encoding="utf-8")
    turn_id = runtime.session.add_user_message("old request")
    runtime.session.add_agent_turn(
        status="succeeded",
        response="old answer",
        turn_id=turn_id,
    )

    messages = build_next_context_messages(runtime)

    assert [message["role"] for message in messages] == ["system", "user"]
    prompt_text = "\n\n".join(str(message["content"]) for message in messages)
    assert "Always run `uv run pytest`." in prompt_text
    assert "old request" in prompt_text
    assert "old answer" in prompt_text
    assert "Primary workspace:" in prompt_text
    assert "Session history:" in prompt_text
    assert "Iteration: 1/12" in prompt_text
    assert "current draft" not in prompt_text


def test_estimate_next_context_tokens_adds_fixed_predict_rlm_allowance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fractal import context_meter

    runtime = FakeRuntime(tmp_path)
    monkeypatch.setattr(context_meter, "count_messages_tokens", lambda *_: 123)

    assert context_meter.estimate_next_context_tokens(runtime) == (
        context_meter.PREDICT_RLM_BASE_PROMPT_TOKEN_ALLOWANCE + 123
    )


def test_context_estimate_cache_key_changes_with_session(tmp_path: Path) -> None:
    from fractal.context_meter import context_estimate_cache_key

    runtime = FakeRuntime(tmp_path)
    before = context_estimate_cache_key(runtime)

    runtime.session.add_user_message("new request")
    after = context_estimate_cache_key(runtime)

    assert after != before


def test_count_messages_tokens_falls_back_to_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal import context_meter

    class BrokenLiteLLM:
        @staticmethod
        def token_counter(**kwargs: object) -> int:
            raise RuntimeError("unsupported model")

    monkeypatch.setitem(__import__("sys").modules, "litellm", BrokenLiteLLM)

    tokens = context_meter.count_messages_tokens(
        "not-a-real-model",
        [{"role": "user", "content": "hello"}],
    )

    assert tokens is None or tokens > 0
