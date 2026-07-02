from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

pytest.importorskip(
    "predict_rlm",
    reason="predict-rlm is required for Fractal runtime tests",
)


def test_runtime_submit_persists_success_and_exposes_pending_state(tmp_path: Path) -> None:
    from fractal.agent.schema import FractalResult
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    pending_seen: list[str] = []
    calls: list[dict[str, object]] = []

    class FakeAgent:
        async def aforward(self, **kwargs: object) -> FractalResult:
            calls.append(kwargs)
            return FractalResult(response="updated docs", changed_files=["README.md"])

    session = FractalSession()
    included_path = tmp_path / "included"
    included_path.mkdir()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        included_paths=[included_path],
        session=session,
        agent=FakeAgent(),
    )

    async def on_pending() -> None:
        pending_seen.append(session.turns[-1].agent.status if session.turns[-1].agent else "pending")

    result = asyncio.run(runtime.submit("update docs", on_pending=on_pending))

    assert result.response == "updated docs"
    assert pending_seen == ["pending"]
    assert session.turns[-1].user.message == "update docs"
    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.response == "updated docs"
    assert session.turns[-1].agent.files_read_count == 0
    assert session.turns[-1].agent.files_changed_count == 1
    assert session.turns[-1].agent.commands_run_count == 0
    assert session.history[-1].files_modified == ["README.md"]
    assert calls[0]["workspace_path"] == tmp_path
    assert calls[0]["included_paths"] == [included_path.resolve()]
    assert calls[0]["user_message"] == "update docs"
    assert "update docs" in str(calls[0]["rendered_session_summary"])
    assert callable(calls[0]["on_runtime_event"])
    assert callable(calls[0]["on_iteration_event"])
    assert session.history[-1].status == "succeeded"


def test_runtime_submit_surfaces_runtime_events_and_persists_safe_facts(
    tmp_path: Path,
) -> None:
    from fractal.agent.schema import FractalResult
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    surfaced: list[str] = []

    class EventAgent:
        async def aforward(self, **kwargs: object) -> FractalResult:
            on_runtime_event = kwargs["on_runtime_event"]
            assert callable(on_runtime_event)
            on_runtime_event({
                "target": "builtins.open",
                "phase": "before",
                "args": ["README.md", "r"],
                "timestamp": 0.0,
            })
            on_runtime_event({
                "target": "builtins.open",
                "phase": "after",
                "args": ["README.md", "r"],
                "timestamp": 0.0,
            })
            on_runtime_event({
                "target": "pathlib.Path.write_text",
                "phase": "before",
                "args": ["src/app.py", "updated"],
                "timestamp": 0.0,
            })
            on_runtime_event({
                "target": "pathlib.Path.write_text",
                "phase": "after",
                "args": ["src/app.py", "updated"],
                "timestamp": 0.0,
            })
            on_runtime_event({
                "target": "subprocess.run",
                "phase": "before",
                "args": [["uv", "run", "pytest"]],
                "timestamp": 0.0,
            })
            return FractalResult(response="done")

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=EventAgent(),
    )

    result = asyncio.run(
        runtime.submit(
            "update docs",
            on_runtime_event=lambda event: surfaced.append(event.message),
        )
    )

    assert result.changed_files == []
    assert surfaced == [
        "opening README.md",
        "editing src/app.py",
        "running uv run pytest",
    ]
    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.files_read_count == 1
    assert session.turns[-1].agent.files_changed_count == 0
    assert session.turns[-1].agent.commands_run_count == 1
    assert session.history[-1].files_read == ["README.md"]
    assert session.history[-1].files_modified == []
    assert session.history[-1].commands_run == ["uv run pytest"]


def test_runtime_submit_surfaces_iteration_events(tmp_path: Path) -> None:
    from predict_rlm.trace import IterationStep

    from fractal.agent.schema import FractalIterationEvent, FractalResult
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    step = IterationStep(
        iteration=1,
        reasoning="Inspect the workspace.",
        code="print('ok')",
        output="ok",
        untruncated_output="ok",
        duration_ms=5,
    )
    surfaced: list[FractalIterationEvent] = []

    class IterationAgent:
        async def aforward(self, **kwargs: object) -> FractalResult:
            on_iteration_event = kwargs["on_iteration_event"]
            assert callable(on_iteration_event)
            on_iteration_event(
                FractalIterationEvent(
                    step=step,
                    max_iterations=3,
                    is_final=False,
                )
            )
            return FractalResult(response="done")

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(),
        agent=IterationAgent(),
    )

    asyncio.run(
        runtime.submit(
            "update docs",
            on_iteration_event=surfaced.append,
        )
    )

    assert len(surfaced) == 1
    assert surfaced[0].step is step
    assert surfaced[0].max_iterations == 3
    assert surfaced[0].is_final is False


def test_runtime_submit_persists_failure_before_reraising(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    class FailingAgent:
        async def aforward(self, **kwargs: object) -> object:
            raise RuntimeError("model failed")

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=FailingAgent(),
    )

    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(runtime.submit("run tests"))

    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.status == "failed"
    assert session.turns[-1].agent.error == "model failed"
    assert session.history[-1].status == "failed"
    assert session.history[-1].error == "model failed"


def test_runtime_submit_persists_user_facing_sbx_auth_failure(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    class FailingAgent:
        async def aforward(self, **kwargs: object) -> object:
            called_process_error = subprocess.CalledProcessError(
                1,
                ["sbx", "create", "shell", "/tmp/sbx", "/workspace"],
                stderr=(
                    "ERROR: request failed: 401 Unauthorized: "
                    "user is not authenticated to Docker\n"
                    "no valid user session found, please sign in to Docker to proceed"
                ),
            )
            raise RuntimeError("Failed to create sbx sandbox") from called_process_error

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=FailingAgent(),
    )

    with pytest.raises(RuntimeError, match="Failed to create sbx sandbox"):
        asyncio.run(runtime.submit("run tests"))

    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.error == (
        "Your sbx CLI is not logged in to Docker. "
        "Run `sbx login`, then try Fractal again."
    )
    assert session.history[-1].error == session.turns[-1].agent.error


def test_runtime_submit_does_not_mask_failure_with_malformed_trace(
    tmp_path: Path,
) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    class FailingAgent:
        async def aforward(self, **kwargs: object) -> object:
            exc = RuntimeError("model failed")
            exc.trace = {"not": "a RunTrace"}
            raise exc

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=FailingAgent(),
    )

    with pytest.raises(RuntimeError, match="model failed"):
        asyncio.run(runtime.submit("run tests"))

    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.status == "failed"
    assert session.turns[-1].agent.error == "model failed"
    assert session.history[-1].status == "failed"
    assert session.history[-1].trace is None


def test_runtime_submit_persists_interruption_before_reraising(tmp_path: Path) -> None:
    from predict_rlm import RunTrace

    from fractal.runtime import FractalRuntime
    from fractal.session import INTERRUPTED_ERROR, FractalSession

    trace = RunTrace(
        status="error",
        model="test-model",
        iterations=1,
        max_iterations=3,
        duration_ms=10,
    )

    class SlowAgent:
        async def aforward(self, **kwargs: object) -> object:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                exc.trace = trace
                raise

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=SlowAgent(),
    )

    interrupt_requested = False

    async def cancel_submit() -> None:
        nonlocal interrupt_requested
        task = asyncio.create_task(
            runtime.submit(
                "stop",
                interrupt_requested=lambda: interrupt_requested,
            )
        )
        await asyncio.sleep(0)
        interrupt_requested = True
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_submit())

    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.status == "interrupted"
    assert session.turns[-1].agent.error == INTERRUPTED_ERROR
    assert session.history[-1].status == "interrupted"
    assert session.history[-1].trace == trace


def test_runtime_submit_propagates_external_cancellation(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    class SlowAgent:
        async def aforward(self, **kwargs: object) -> object:
            await asyncio.Event().wait()

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=SlowAgent(),
    )

    async def cancel_submit() -> None:
        task = asyncio.create_task(runtime.submit("shutdown"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_submit())

    assert session.turns[-1].agent is None
    assert session.history[-1].status == "pending"


def test_runtime_reclassifies_interrupt_shutdown_error(tmp_path: Path) -> None:
    from predict_rlm import RunTrace

    from fractal.runtime import FractalRuntime
    from fractal.session import INTERRUPTED_ERROR, FractalSession

    trace = RunTrace(
        status="error",
        model="test-model",
        iterations=0,
        max_iterations=3,
        duration_ms=10,
    )
    interrupted = False

    class InterruptedShutdownAgent:
        async def aforward(self, **kwargs: object) -> object:
            nonlocal interrupted
            interrupted = True
            exc = RuntimeError("Deno exited (code -2) during health check")
            exc.trace = trace
            raise exc

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=InterruptedShutdownAgent(),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            runtime.submit(
                "stop",
                interrupt_requested=lambda: interrupted,
            )
        )

    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.status == "interrupted"
    assert session.turns[-1].agent.error == INTERRUPTED_ERROR
    assert session.history[-1].status == "interrupted"
    assert session.history[-1].trace == trace


def test_runtime_submit_persists_max_iterations_as_incomplete(tmp_path: Path) -> None:
    from predict_rlm import RunTrace

    from fractal.agent.schema import FractalResult
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    trace = RunTrace(
        status="max_iterations",
        model="test-model",
        iterations=2,
        max_iterations=2,
        duration_ms=10,
    )

    class MaxIterationAgent:
        async def aforward(self, **kwargs: object) -> FractalResult:
            return FractalResult(
                response="fallback answer",
                changed_files=["README.md"],
                trace=trace,
            )

    session = FractalSession()
    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=session,
        agent=MaxIterationAgent(),
    )

    result = asyncio.run(runtime.submit("finish task"))

    assert result.response == "fallback answer"
    assert session.turns[-1].agent is not None
    assert session.turns[-1].agent.status == "max_iterations"
    assert session.turns[-1].agent.response == "fallback answer"
    assert session.turns[-1].agent.files_changed_count == 1
    assert session.history[-1].files_modified == ["README.md"]
    assert session.history[-1].status == "max_iterations"
    assert session.history[-1].trace == trace


def test_runtime_create_and_resume_load_session_ids(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    class FakeAgent:
        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError("agent should not run")

    existing = FractalSession(session_id="existing")
    existing.add_user_message("prior work")
    existing.save(tmp_path)
    included_path = tmp_path / "included"
    included_path.mkdir()

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(),
        agent=FakeAgent(),
    )
    runtime.resume("existing")

    assert runtime.session_id == "existing"
    assert runtime.turns[-1].user.message == "prior work"

    created = FractalRuntime.create(
        workspace_path=tmp_path,
        included_paths=[included_path],
        lm=None,
        sub_lm=None,
        max_iterations=1,
        verbose=False,
        debug=False,
        session_id="existing",
    )

    assert created.session_id == "existing"
    assert created.included_paths == [included_path.resolve()]
    assert created.turns[-1].user.message == "prior work"


def test_runtime_create_reuses_one_interpreter_until_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fractal import runtime as runtime_module
    from fractal.runtime import FractalRuntime

    included_path = tmp_path / "included"
    included_path.mkdir()
    interpreter = object()
    created_with: list[tuple[Path, list[Path]]] = []

    def fake_create_sbx_interpreter(
        workspace_path: str | Path,
        included_paths: list[str | Path] | None = None,
        *,
        reuse: bool = True,
    ) -> object:
        created_with.append((
            Path(workspace_path),
            [Path(path) for path in included_paths or []],
        ))
        return interpreter

    monkeypatch.setattr(
        runtime_module,
        "create_sbx_interpreter",
        fake_create_sbx_interpreter,
    )

    runtime = FractalRuntime.create(
        workspace_path=tmp_path,
        included_paths=[included_path],
        lm=None,
        sub_lm=None,
        max_iterations=1,
        verbose=False,
        debug=False,
    )

    assert created_with == [(tmp_path.resolve(), [included_path])]
    assert runtime.agent.interpreter is interpreter


def test_runtime_close_closes_agent(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    closed: list[bool] = []

    class FakeAgent:
        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError("agent should not run")

        def close(self) -> None:
            closed.append(True)

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(),
        agent=FakeAgent(),
    )

    runtime.close()

    assert closed == [True]


def test_runtime_apply_provider_selection_updates_main_and_following_sub_lm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.providers import ProviderSelection
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")

    class FakeLM:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    import dspy

    monkeypatch.setattr(dspy, "LM", FakeLM)

    class FakeAgent:
        lm = "old-main"
        sub_lm = "old-main"

        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError("agent should not run")

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(),
        agent=FakeAgent(),
        sub_lm_follows_main=True,
    )

    runtime.apply_provider_selection(
        ProviderSelection(
            provider="openai-api",
            model="gpt-5.4",
            api_key_env="OPENAI_API_KEY",
            auth_source="env",
        )
    )

    assert runtime.provider_label == "openai-api"
    assert runtime.model_label == "gpt-5.4"
    assert runtime.agent.lm.kwargs == {
        "model": "openai/gpt-5.4",
        "api_key": "sk-secret-value",
    }
    assert runtime.agent.sub_lm is runtime.agent.lm


def test_runtime_apply_provider_selection_preserves_configured_num_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.providers import ProviderSelection
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")

    class FakeLM:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    import dspy

    monkeypatch.setattr(dspy, "LM", FakeLM)

    class FakeAgent:
        lm = "old-main"
        sub_lm = "old-sub"

        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError("agent should not run")

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(),
        agent=FakeAgent(),
        lm_num_retries=7,
    )
    selection = ProviderSelection(
        provider="openai-api",
        model="gpt-5.4",
        api_key_env="OPENAI_API_KEY",
        auth_source="env",
    )

    runtime.apply_provider_selection(selection, sub_model="gpt-5.4-mini")

    assert runtime.agent.lm.num_retries == 7
    assert runtime.agent.sub_lm.num_retries == 7


def test_runtime_apply_provider_selection_sets_and_clears_sub_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.providers import ProviderSelection
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")

    class FakeLM:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    import dspy

    monkeypatch.setattr(dspy, "LM", FakeLM)

    class FakeAgent:
        lm = "old-main"
        sub_lm = "old-sub"

        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError("agent should not run")

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(),
        agent=FakeAgent(),
        sub_lm_follows_main=False,
    )
    selection = ProviderSelection(
        provider="openai-api",
        model="gpt-5.4",
        api_key_env="OPENAI_API_KEY",
        auth_source="env",
    )

    runtime.apply_provider_selection(selection, sub_model="gpt-5.4-mini")

    assert runtime.agent.lm.kwargs == {
        "model": "openai/gpt-5.4",
        "api_key": "sk-secret-value",
    }
    assert runtime.agent.sub_lm.kwargs == {
        "model": "openai/gpt-5.4-mini",
        "api_key": "sk-secret-value",
    }
    assert runtime.sub_lm_follows_main is False
    assert runtime.sub_model_label == "gpt-5.4-mini"

    # Choosing "same as main" resets the sub-LM to follow the main model.
    runtime.apply_provider_selection(selection)

    assert runtime.agent.sub_lm == runtime.agent.lm
    assert runtime.sub_lm_follows_main is True
    assert runtime.sub_model_label == "gpt-5.4"


def test_runtime_prewarm_prewarms_agent(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    prewarmed: list[bool] = []

    class FakeAgent:
        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError("agent should not run")

        def close(self) -> None:
            pass

        def prewarm(self) -> None:
            prewarmed.append(True)

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(),
        agent=FakeAgent(),
    )

    runtime.prewarm()

    assert prewarmed == [True]


def test_runtime_resume_requires_existing_session(tmp_path: Path) -> None:
    from fractal.runtime import FractalRuntime
    from fractal.session import FractalSession

    class FakeAgent:
        async def aforward(self, **kwargs: object) -> object:
            raise AssertionError("agent should not run")

    runtime = FractalRuntime(
        workspace_path=tmp_path,
        session=FractalSession(session_id="current"),
        agent=FakeAgent(),
    )

    with pytest.raises(FileNotFoundError, match="missing"):
        runtime.resume("missing")

    with pytest.raises(FileNotFoundError, match="missing"):
        FractalRuntime.create(
            workspace_path=tmp_path,
            lm=None,
            sub_lm=None,
            max_iterations=1,
            verbose=False,
            debug=False,
            session_id="missing",
        )

    assert runtime.session_id == "current"
