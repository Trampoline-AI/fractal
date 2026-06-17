from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fractal.agent.schema import FractalResult
from fractal.runtime import FractalRuntime
from fractal.session import FractalSession


class _StubInterpreter:
    """Stand-in for the sbx interpreter; identity is what the test cares about."""


class _StubAgent:
    """Minimal FractalAgentLike whose per-turn behavior the test scripts.

    Models the predict-rlm#42 contract from Fractal's side: a cancelled turn
    raises ``CancelledError`` out of ``aforward`` (the cancellation-safe
    ``aexecute`` having already interrupted the sandbox and left it healthy), so
    the next turn reuses the same interpreter with no rebuild.
    """

    def __init__(self) -> None:
        self.interpreter = _StubInterpreter()
        self.behaviors: list = []
        self.calls = 0

    async def aforward(self, **_: object) -> FractalResult:
        self.calls += 1
        return self.behaviors.pop(0)()

    def close(self) -> None: ...

    def prewarm(self) -> None: ...


def test_interrupt_persists_turn_and_next_submit_reuses_interpreter(
    tmp_path: Path,
) -> None:
    async def run() -> None:
        agent = _StubAgent()
        interpreter = agent.interpreter
        runtime = FractalRuntime(
            workspace_path=tmp_path,
            session=FractalSession(),
            agent=agent,
        )

        # The interrupt arrives *during* the first turn: the flag is false at the
        # top-of-submit pre-check, then flips as the turn is cancelled mid-run.
        state = {"interrupted": False}

        def interrupt_mid_turn() -> FractalResult:
            state["interrupted"] = True
            raise asyncio.CancelledError()

        agent.behaviors = [
            interrupt_mid_turn,
            lambda: FractalResult(response="recovered", changed_files=[]),
        ]

        # The runtime persists the interrupted turn distinctly and re-raises.
        with pytest.raises(asyncio.CancelledError):
            await runtime.submit(
                "long task",
                interrupt_requested=lambda: state["interrupted"],
            )

        assert runtime.session.turns[-1].agent is not None
        assert runtime.session.turns[-1].agent.status == "interrupted"
        assert runtime.session.history[-1].status == "interrupted"

        # The follow-up turn runs cleanly against the SAME interpreter — no
        # sandbox rebuild. predict-rlm#42 leaves the interpreter quiescent after
        # an interrupt, so reuse is correct and a rebuild bandaid is unnecessary.
        result = await runtime.submit("recover")

        assert result.response == "recovered"
        assert runtime.agent is agent
        assert runtime.agent.interpreter is interpreter
        assert agent.calls == 2

    asyncio.run(run())
