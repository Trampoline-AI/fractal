from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

import pytest

pytest.importorskip(
    "predict_rlm",
    reason="predict-rlm is required for Fractal integration tests",
)


pytestmark = pytest.mark.integration


def _resolve_lms() -> tuple[object, object]:
    """Build the main/sub LMs from the saved Fractal config.

    Mirrors what the CLI does via ``resolve_runtime_lms`` so the integration
    test uses the user's configured provider instead of relying on a global
    ``dspy.configure``. Skips if no usable config is present.
    """
    from fractal.config import FractalConfigError, load_layered_config
    from fractal.providers import ProviderError, build_lm
    from fractal.runtime_lms import selection_from_config, sub_selection_from_config

    try:
        result = load_layered_config(workspace=None)
    except FractalConfigError as exc:
        pytest.skip(f"no usable Fractal config for integration test: {exc}")
    if result.config is None:
        pytest.skip(
            "no Fractal config found; run `fractal config setup` to run this test"
        )

    try:
        lm = build_lm(selection_from_config(result.config, path=result.path))
        sub_selection = sub_selection_from_config(result.config, path=result.path)
        sub_lm = build_lm(sub_selection) if sub_selection is not None else lm
    except ProviderError as exc:
        pytest.skip(f"Fractal provider not ready for integration test: {exc}")
    return lm, sub_lm


async def _wait_for_path(path: Path, *, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        await asyncio.sleep(0.25)
    return False


async def _cancel_interrupted_turn(task: asyncio.Task[object]) -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=30.0)


def test_runtime_recovers_rlm_and_sandbox_after_user_interrupt(tmp_path: Path) -> None:
    async def run() -> None:
        from fractal.runtime import FractalRuntime

        workspace = tmp_path / "repo"
        workspace.mkdir()
        (workspace / "README.md").write_text("interrupt recovery integration fixture\n")

        lm, sub_lm = _resolve_lms()
        runtime = FractalRuntime.create(
            workspace_path=workspace,
            included_paths=None,
            lm=lm,
            sub_lm=sub_lm,
            max_iterations=8,
            verbose=False,
            debug=False,
        )

        entered = workspace / "entered_sandbox.txt"
        recovered = workspace / "recovered.txt"
        interrupted = False
        first_turn: asyncio.Task[object] | None = None

        try:
            first_turn = asyncio.create_task(
                runtime.submit(
                    (
                        "Use Python code in the REPL to write entered_sandbox.txt with "
                        "exactly entered, then sleep for 120 seconds before responding. "
                        "Do not skip the Python code execution."
                    ),
                    interrupt_requested=lambda: interrupted,
                )
            )

            reached_sandbox = await _wait_for_path(entered)
            interrupted = True
            if not reached_sandbox:
                await _cancel_interrupted_turn(first_turn)
                pytest.fail("first turn never reached sandbox execution")

            await _cancel_interrupted_turn(first_turn)

            assert runtime.session.turns[-1].agent is not None
            assert runtime.session.turns[-1].agent.status == "interrupted"
            assert runtime.session.history[-1].status == "interrupted"

            result = await runtime.submit(
                (
                    "Use Python code in the REPL to write recovered.txt with exactly ok. "
                    "Then respond with exactly recovered."
                )
            )

            assert "recovered" in result.response.lower()
            assert recovered.read_text().strip() == "ok"
        finally:
            if first_turn is not None and not first_turn.done():
                interrupted = True
                first_turn.cancel()
                with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(first_turn, timeout=5.0)
            runtime.close()

    asyncio.run(run())
