from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "predict_rlm",
    reason="predict-rlm is required for Fractal RLM smoke tests",
)


def workspace_available() -> bool:
    import predict_rlm

    return hasattr(predict_rlm, "Workspace")


def test_cli_parser_defaults_to_cwd() -> None:
    from fractal.cli import build_parser

    args = build_parser().parse_args([])

    assert args.workspace == Path.cwd()
    assert args.include == []
    assert args.max_iterations is None
    assert args.lm is None
    assert args.sub_lm is None
    assert args.quiet is False
    assert args.verbose is False
    assert args.resume is None


def test_cli_parser_accepts_repeated_include_paths(tmp_path: Path) -> None:
    from fractal.cli import build_parser

    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()

    args = build_parser().parse_args([
        "--include",
        str(first),
        "--include",
        str(second),
    ])

    assert args.include == [first.resolve(), second.resolve()]


def test_cli_parser_rejects_invalid_include_paths(tmp_path: Path) -> None:
    from fractal.cli import build_parser

    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory", encoding="utf-8")
    link_path = tmp_path / "link"
    link_path.symlink_to(tmp_path, target_is_directory=True)
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--include", str(tmp_path / "missing")])
    with pytest.raises(SystemExit):
        parser.parse_args(["--include", str(file_path)])
    with pytest.raises(SystemExit):
        parser.parse_args(["--include", str(link_path)])


def test_cli_parser_accepts_resume_session_id() -> None:
    from fractal.cli import build_parser

    args = build_parser().parse_args(["--resume", "session-123"])

    assert args.resume == "session-123"


def test_cli_parser_accepts_non_interactive_prompt() -> None:
    from fractal.cli import build_parser

    args = build_parser().parse_args(["-p", "update docs"])

    assert args.prompt == "update docs"


def test_cli_parser_accepts_verbose() -> None:
    from fractal.cli import build_parser

    args = build_parser().parse_args(["--verbose"])

    assert args.verbose is True


def test_cli_main_dispatches_non_interactive_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from fractal import cli

    calls: list[object] = []

    def fake_run_non_interactive(args: object, **kwargs: object) -> int:
        calls.append(args)
        return 23

    def fake_run_tui(args: object, **kwargs: object) -> int:
        raise AssertionError("TUI should not run")

    monkeypatch.setattr(cli, "run_non_interactive", fake_run_non_interactive)
    monkeypatch.setattr(cli, "run_tui", fake_run_tui)

    assert cli.main(["-p", "update docs"]) == 23
    assert len(calls) == 1


def test_build_non_interactive_message_appends_stdin_context() -> None:
    from fractal.cli import build_non_interactive_message

    message = build_non_interactive_message("explain this", "line 1\nline 2\n")

    assert message.startswith("explain this\n\n<Fractal stdin context>")
    assert "line 1\nline 2\n" in message
    assert message.endswith("\n</Fractal stdin context>")


def test_build_non_interactive_message_uses_stdin_as_prompt() -> None:
    from fractal.cli import build_non_interactive_message

    assert build_non_interactive_message("-", "do the task") == "do the task"


def test_read_non_interactive_stdin_skips_open_pipe_with_no_data() -> None:
    import os
    import threading

    from fractal.cli import read_non_interactive_stdin

    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "r")
    result: list[str | None] = []
    try:
        worker = threading.Thread(
            target=lambda: result.append(read_non_interactive_stdin("fix it", stdin)),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive(), "read_non_interactive_stdin blocked on idle stdin"
        assert result == [None]
    finally:
        os.close(write_fd)
        stdin.close()


def test_read_non_interactive_stdin_reads_pipe_with_data() -> None:
    import os

    from fractal.cli import read_non_interactive_stdin

    read_fd, write_fd = os.pipe()
    with os.fdopen(write_fd, "w") as writer:
        writer.write("piped context\n")
    with os.fdopen(read_fd, "r") as stdin:
        assert read_non_interactive_stdin("fix it", stdin) == "piped context\n"


def test_run_non_interactive_prints_response_and_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fractal import cli
    from fractal.agent.schema import FractalResult
    from fractal.runtime import FractalRuntime

    calls: dict[str, object] = {}

    class FakeRuntime:
        workspace_path = tmp_path
        session_id = "session-123"

        def close(self) -> None:
            calls["closed"] = True

        async def submit(self, message: str, **kwargs: object) -> FractalResult:
            calls["message"] = message
            on_runtime_event = kwargs.get("on_runtime_event")
            if callable(on_runtime_event):
                on_runtime_event(SimpleNamespace(message="opening README.md"))
            return FractalResult(response="done", changed_files=["README.md"])

    def fake_create(**kwargs: object) -> FakeRuntime:
        calls["create_kwargs"] = kwargs
        return FakeRuntime()

    monkeypatch.setattr(FractalRuntime, "create", fake_create)
    args = cli.build_parser().parse_args(
        ["--workspace", str(tmp_path), "--lm", "test-lm", "-p", "update docs"]
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_non_interactive(
        args,
        stdin=StringIO("extra context"),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue() == "done\n"
    assert calls["message"] == (
        "update docs\n\n"
        "<Fractal stdin context>\n"
        "extra context"
        "\n</Fractal stdin context>"
    )
    assert "______              _        _" in stderr.getvalue()
    assert "|_|  |_|  \\__,_|\\___|\\__\\__,_|_|" in stderr.getvalue()
    assert "fractal: session session-123" in stderr.getvalue()
    assert "fractal: opening README.md" in stderr.getvalue()
    assert "fractal: changed files README.md" in stderr.getvalue()
    create_kwargs = calls["create_kwargs"]
    assert isinstance(create_kwargs, dict)
    assert create_kwargs["workspace_path"] == tmp_path
    assert create_kwargs["verbose"] is False


def test_run_non_interactive_closes_runtime_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fractal import cli
    from fractal.agent.schema import FractalResult
    from fractal.runtime import FractalRuntime

    calls: dict[str, object] = {}

    class FakeRuntime:
        workspace_path = tmp_path
        session_id = "session-closed"
        fail = False

        def close(self) -> None:
            calls["closed"] = calls.get("closed", 0) + 1

        async def submit(self, message: str, **kwargs: object) -> FractalResult:
            if self.fail:
                raise RuntimeError("boom")
            return FractalResult(response="done", changed_files=[])

    monkeypatch.setattr(FractalRuntime, "create", lambda **kwargs: FakeRuntime())
    args = cli.build_parser().parse_args(
        ["--workspace", str(tmp_path), "--lm", "test-lm", "-p", "task"]
    )

    assert cli.run_non_interactive(args, stdin=StringIO(), stdout=StringIO(), stderr=StringIO()) == 0
    assert calls["closed"] == 1

    FakeRuntime.fail = True
    stderr = StringIO()
    assert cli.run_non_interactive(args, stdin=StringIO(), stdout=StringIO(), stderr=stderr) == 1
    assert calls["closed"] == 2
    assert "fractal: failed: boom" in stderr.getvalue()


def test_run_non_interactive_prints_iteration_trace_to_stderr_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from predict_rlm.trace import IterationStep

    from fractal import cli
    from fractal.agent.schema import FractalIterationEvent, FractalResult
    from fractal.runtime import FractalRuntime

    calls: dict[str, object] = {}
    step = IterationStep(
        iteration=1,
        reasoning="Inspect the files.",
        code="if True:\n    print('ok')",
        output="ok\n",
        untruncated_output="ok\nextra hidden text",
        duration_ms=5,
    )

    class FakeRuntime:
        workspace_path = tmp_path
        session_id = "session-123"

        def close(self) -> None:
            calls["closed"] = True

        async def submit(self, message: str, **kwargs: object) -> FractalResult:
            on_iteration_event = kwargs.get("on_iteration_event")
            calls["on_iteration_event"] = on_iteration_event
            if callable(on_iteration_event):
                on_iteration_event(
                    FractalIterationEvent(
                        step=step,
                        max_iterations=2,
                    )
                )
            return FractalResult(response="done")

    def fake_create(**kwargs: object) -> FakeRuntime:
        calls["create_kwargs"] = kwargs
        return FakeRuntime()

    monkeypatch.setattr(FractalRuntime, "create", fake_create)
    args = cli.build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--lm",
            "test-lm",
            "-p",
            "update docs",
        ]
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_non_interactive(
        args,
        stdin=StringIO(),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue() == "done\n"
    assert args.verbose is False
    assert callable(calls["on_iteration_event"])
    create_kwargs = calls["create_kwargs"]
    assert isinstance(create_kwargs, dict)
    assert create_kwargs["verbose"] is False
    stderr_text = stderr.getvalue()
    assert "RLM turn 1/2" in stderr_text
    assert "reasoning: Inspect the files." in stderr_text
    assert "python: 2 lines" in stderr_text
    assert "output: 20 chars" in stderr_text
    assert "code:" in stderr_text
    assert "if True:" in stderr_text
    assert "    print('ok')" in stderr_text
    assert "output:" in stderr_text
    assert "ok" in stderr_text
    assert "extra hidden text" not in stderr_text


def test_run_non_interactive_quiet_suppresses_verbose_iteration_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fractal import cli
    from fractal.agent.schema import FractalResult
    from fractal.runtime import FractalRuntime

    calls: dict[str, object] = {}

    class FakeRuntime:
        workspace_path = tmp_path
        session_id = "session-123"

        def close(self) -> None:
            calls["closed"] = True

        async def submit(self, message: str, **kwargs: object) -> FractalResult:
            calls["on_iteration_event"] = kwargs.get("on_iteration_event")
            return FractalResult(response="done")

    monkeypatch.setattr(
        FractalRuntime,
        "create",
        lambda **kwargs: FakeRuntime(),
    )
    args = cli.build_parser().parse_args(
        [
            "--workspace",
            str(tmp_path),
            "--lm",
            "test-lm",
            "--quiet",
            "--verbose",
            "-p",
            "update docs",
        ]
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_non_interactive(
        args,
        stdin=StringIO(),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stdout.getvalue() == "done\n"
    assert stderr.getvalue() == ""
    assert calls["on_iteration_event"] is None


def test_run_non_interactive_reads_dash_prompt_from_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fractal import cli
    from fractal.agent.schema import FractalResult
    from fractal.runtime import FractalRuntime

    calls: dict[str, object] = {}

    class FakeRuntime:
        workspace_path = tmp_path
        session_id = "session-123"

        def close(self) -> None:
            calls["closed"] = True

        async def submit(self, message: str, **kwargs: object) -> FractalResult:
            calls["message"] = message
            return FractalResult(response="done")

    monkeypatch.setattr(
        FractalRuntime,
        "create",
        lambda **kwargs: FakeRuntime(),
    )
    args = cli.build_parser().parse_args(["--workspace", str(tmp_path), "-p", "-"])
    args.lm = "test-lm"

    exit_code = cli.run_non_interactive(
        args,
        stdin=StringIO("full prompt"),
        stdout=StringIO(),
        stderr=StringIO(),
    )

    assert exit_code == 0
    assert calls["message"] == "full prompt"


def test_run_non_interactive_returns_distinct_code_for_max_iterations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from predict_rlm import RunTrace

    from fractal import cli
    from fractal.agent.schema import FractalResult
    from fractal.runtime import FractalRuntime

    calls: dict[str, object] = {}

    trace = RunTrace(
        status="max_iterations",
        model="test-model",
        iterations=2,
        max_iterations=2,
        duration_ms=10,
    )

    class FakeRuntime:
        workspace_path = tmp_path
        session_id = "session-123"

        def close(self) -> None:
            calls["closed"] = True

        async def submit(self, message: str, **kwargs: object) -> FractalResult:
            return FractalResult(response="partial", trace=trace)

    monkeypatch.setattr(
        FractalRuntime,
        "create",
        lambda **kwargs: FakeRuntime(),
    )
    args = cli.build_parser().parse_args(
        ["--workspace", str(tmp_path), "--lm", "test-lm", "-p", "finish"]
    )
    stdout = StringIO()
    stderr = StringIO()

    exit_code = cli.run_non_interactive(
        args,
        stdin=StringIO(),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == cli.MAX_ITERATIONS_EXIT_CODE
    assert stdout.getvalue() == "partial\n"
    assert "fractal: max iterations reached" in stderr.getvalue()


def test_run_tui_shows_shutdown_status_and_closes_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import rich.console as rich_console

    import fractal.runtime as runtime_module
    import fractal.tui as tui_module
    from fractal.cli import run_tui

    events: list[str] = []

    class FakeStatus:
        def __enter__(self) -> None:
            events.append("status_enter")

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            events.append("status_exit")

    class FakeConsole:
        def print(self, message: str, *, style: str | None = None) -> None:
            events.append(f"print:{message}:{style}")

        def status(self, message: str, *, spinner: str) -> FakeStatus:
            events.append(f"status:{message}:{spinner}")
            return FakeStatus()

    class FakeRuntime:
        def prewarm(self) -> None:
            events.append("prewarm")

        def close(self) -> None:
            events.append("close")

    class FakeFractalRuntime:
        @classmethod
        def create(cls, **kwargs: object) -> FakeRuntime:
            events.append("create")
            return FakeRuntime()

    class FakeTerminalFractalApp:
        def __init__(
            self,
            runtime: FakeRuntime,
            *,
            console: FakeConsole,
            verbose_iterations: bool,
            banner: str | None = None,
            update_notice: str | None = None,
        ) -> None:
            events.append(f"app:{verbose_iterations}:banner={banner is not None}")

        async def run(self) -> None:
            events.append("run")

    monkeypatch.setattr(rich_console, "Console", FakeConsole)
    monkeypatch.setattr(runtime_module, "FractalRuntime", FakeFractalRuntime)
    monkeypatch.setattr(tui_module, "TerminalFractalApp", FakeTerminalFractalApp)

    result = run_tui(
        SimpleNamespace(
            workspace=tmp_path,
            include=[],
            lm="test-lm",
            sub_lm=None,
            max_iterations=1,
            debug=False,
            resume=None,
            verbose=True,
            fresh=False,
            ephemeral=False,
        )
    )

    assert result == 0
    assert events == [
        "create",
        "status:[dim]starting sandbox (reusing hot sandbox if available)...[/dim]:dots",
        "status_enter",
        "prewarm",
        "status_exit",
        "app:True:banner=True",
        "run",
        "status:[dim]shutting down sandbox... press Ctrl-C again to force exit without cleaning up the sandbox[/dim]:dots",
        "status_enter",
        "close",
        "status_exit",
    ]


def test_run_tui_allows_force_exit_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import rich.console as rich_console

    import fractal.runtime as runtime_module
    import fractal.tui as tui_module
    from fractal.cli import run_tui

    events: list[str] = []

    class FakeStatus:
        def __enter__(self) -> None:
            events.append("status_enter")

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            events.append("status_exit")

    class FakeConsole:
        def print(self, message: str, *, style: str | None = None) -> None:
            events.append(f"print:{message}:{style}")

        def status(self, message: str, *, spinner: str) -> FakeStatus:
            events.append(f"status:{message}:{spinner}")
            return FakeStatus()

    class FakeRuntime:
        def prewarm(self) -> None:
            events.append("prewarm")

        def close(self) -> None:
            events.append("close")
            raise KeyboardInterrupt

    class FakeFractalRuntime:
        @classmethod
        def create(cls, **kwargs: object) -> FakeRuntime:
            events.append("create")
            return FakeRuntime()

    class FakeTerminalFractalApp:
        def __init__(
            self,
            runtime: FakeRuntime,
            *,
            console: FakeConsole,
            verbose_iterations: bool,
            banner: str | None = None,
            update_notice: str | None = None,
        ) -> None:
            events.append("app")

        async def run(self) -> None:
            events.append("run")

    monkeypatch.setattr(rich_console, "Console", FakeConsole)
    monkeypatch.setattr(runtime_module, "FractalRuntime", FakeFractalRuntime)
    monkeypatch.setattr(tui_module, "TerminalFractalApp", FakeTerminalFractalApp)

    result = run_tui(
        SimpleNamespace(
            workspace=tmp_path,
            include=[],
            lm="test-lm",
            sub_lm=None,
            max_iterations=1,
            debug=False,
            resume=None,
            verbose=False,
            fresh=False,
            ephemeral=False,
        )
    )

    assert result == 130
    assert events[-2:] == [
        "status_exit",
        (
            "print:sandbox shutdown interrupted; a sandbox may still be running. "
            "Run `sbx ls` and `sbx rm --force <name>` to clean it up.:yellow"
        ),
    ]


def test_signature_fields() -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent.signature import build_edit_workspace_signature
    from fractal.session import SessionHistoryTurn

    signature = build_edit_workspace_signature("User: fix tests")
    fields = signature.model_fields

    assert {
        "workspace",
        "included_paths",
        "user_message",
        "session_history",
        "response",
        "changed_files",
    } <= set(fields)
    assert "session_summary" not in fields
    assert fields["session_history"].annotation == list[SessionHistoryTurn]
    assert "User: fix tests" in signature.instructions


def test_signature_workspace_instructions() -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent.signature import build_edit_workspace_signature

    signature = build_edit_workspace_signature(
        "User: fix tests",
        workspace_instructions="Always run `make lint` before finishing.",
    )
    instructions = signature.instructions
    assert "## Workspace instructions (AGENTS.md)" in instructions
    assert "Always run `make lint` before finishing." in instructions
    # Static AGENTS.md content must precede the dynamic per-turn summary so the
    # prompt keeps a stable cacheable prefix.
    assert instructions.index("## Workspace instructions (AGENTS.md)") < instructions.index(
        "## Always-visible session summary"
    )

    without = build_edit_workspace_signature("User: fix tests")
    assert "## Workspace instructions (AGENTS.md)" not in without.instructions


def test_load_workspace_instructions(tmp_path) -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent.service import (
        _MAX_WORKSPACE_INSTRUCTIONS_CHARS,
        load_workspace_instructions,
    )

    assert load_workspace_instructions(tmp_path) == ""

    (tmp_path / "AGENTS.md").write_text("Use uv for everything.\n", encoding="utf-8")
    assert load_workspace_instructions(tmp_path) == "Use uv for everything."

    (tmp_path / "AGENTS.md").write_text(
        "x" * (_MAX_WORKSPACE_INSTRUCTIONS_CHARS + 1), encoding="utf-8"
    )
    truncated = load_workspace_instructions(tmp_path)
    assert truncated.endswith("[AGENTS.md truncated — read the full file from the workspace.]")
    assert len(truncated) < _MAX_WORKSPACE_INSTRUCTIONS_CHARS + 200


def test_service_construction() -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent.service import FractalAgent

    agent = FractalAgent(max_iterations=7, verbose=False, debug=True)

    assert agent.max_iterations == 7
    assert agent.verbose is False
    assert agent.debug is True


def test_agent_aforward_constructs_rlm_and_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent import service
    from fractal.session import SessionHistoryTurn

    calls: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, signature: object, **kwargs: object) -> None:
            calls["signature"] = signature
            calls["kwargs"] = kwargs

        async def acall(self, **kwargs: object) -> object:
            calls["acall_count"] = int(calls.get("acall_count", 0)) + 1
            calls["acall_kwargs"] = kwargs
            return SimpleNamespace(
                response="done",
                changed_files=["README.md"],
                trace=None,
            )

    monkeypatch.setattr(service, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(service, "build_predict_runtime_hooks", lambda: [])

    history_turn = SessionHistoryTurn(
        turn_id="turn-1",
        user_message="previous",
        status="succeeded",
        created_at="2026-05-19T00:00:00+00:00",
        updated_at="2026-05-19T00:00:00+00:00",
    )

    agent = service.FractalAgent(max_iterations=7, verbose=False, debug=True)
    included_path = tmp_path / "included"
    included_path.mkdir()
    result = asyncio.run(
        agent.aforward(
            tmp_path,
            "update the README",
            rendered_session_summary="previous context",
            session_history=[history_turn],
            included_paths=[included_path],
        )
    )

    signature = calls["signature"]
    assert isinstance(signature, type)
    assert "previous context" in signature.instructions
    assert "session_summary" not in signature.model_fields
    assert "session_history" in signature.model_fields
    assert "included_paths" in signature.model_fields
    assert calls["kwargs"] == {
        "lm": None,
        "sub_lm": None,
        "skills": [service.filesystem_coding_skill, service.spreadsheet, service.pdf, service.docx],
        "max_iterations": 7,
        "verbose": False,
        "debug": True,
        "sandbox_backend": "sbx",
    }
    assert calls["acall_count"] == 1
    acall_kwargs = calls["acall_kwargs"]
    assert isinstance(acall_kwargs, dict)
    workspace = acall_kwargs["workspace"]
    assert isinstance(workspace, service.Workspace)
    assert workspace.path == str(tmp_path.resolve())
    assert "mount_path" not in workspace.model_fields_set
    assert workspace.mode is service.WorkspaceMode.DIRECT
    assert ".fractal" in workspace.exclude
    included_paths = acall_kwargs["included_paths"]
    assert isinstance(included_paths, list)
    assert len(included_paths) == 1
    assert included_paths[0].path == str(included_path.resolve())
    assert "mount_path" not in included_paths[0].model_fields_set
    assert included_paths[0].mode is service.WorkspaceMode.DIRECT
    assert acall_kwargs["user_message"] == "update the README"
    assert acall_kwargs["session_history"] == [history_turn]
    assert result.changed_files == ["README.md"]


def test_agent_aforward_passes_runtime_hooks_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent import service

    calls: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, signature: object, **kwargs: object) -> None:
            calls["kwargs"] = kwargs

        async def acall(self, **kwargs: object) -> object:
            return SimpleNamespace(response="done", changed_files=[], trace=None)

    def on_runtime_event(event: object) -> None:
        calls["event"] = event

    monkeypatch.setattr(service, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(service, "build_predict_runtime_hooks", lambda: ["hook"])

    agent = service.FractalAgent(max_iterations=7, verbose=False, debug=True)
    asyncio.run(
        agent.aforward(
            tmp_path,
            "update the README",
            on_runtime_event=on_runtime_event,
        )
    )

    predictor_kwargs = calls["kwargs"]
    assert isinstance(predictor_kwargs, dict)
    assert predictor_kwargs["runtime_hooks"] == ["hook"]
    assert predictor_kwargs["on_runtime_hook_event"] is on_runtime_event


def test_agent_aforward_attaches_iteration_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from predict_rlm.trace import IterationStep

    from fractal.agent import service
    from fractal.agent.schema import FractalIterationEvent

    events: list[FractalIterationEvent] = []
    step = IterationStep(
        iteration=1,
        reasoning="Inspect the files.",
        code="print('ok')",
        output="ok",
        untruncated_output="ok",
        duration_ms=5,
    )

    class FakePredictRLM:
        def __init__(self, signature: object, **kwargs: object) -> None:
            self.callbacks: list[object] = []

        async def acall(self, **kwargs: object) -> object:
            for callback in self.callbacks:
                callback.on_rlm_iteration_end(
                    call_id="call-1",
                    instance=self,
                    iteration=1,
                    step=step,
                    is_final=False,
                    exception=None,
                )
            return SimpleNamespace(response="done", changed_files=[], trace=None)

    monkeypatch.setattr(service, "PredictRLM", FakePredictRLM)
    monkeypatch.setattr(service, "build_predict_runtime_hooks", lambda: [])

    agent = service.FractalAgent(max_iterations=7, verbose=False, debug=True)
    asyncio.run(
        agent.aforward(
            tmp_path,
            "update the README",
            on_iteration_event=events.append,
        )
    )

    assert len(events) == 1
    assert events[0].step is step
    assert events[0].max_iterations == 7
    assert events[0].is_final is False


def test_agent_iteration_callback_satisfies_dspy_base_handlers() -> None:
    from dspy.utils.callback import BaseCallback

    from fractal.agent import service

    callback = service._FractalIterationCallback(
        max_iterations=3,
        on_iteration_event=lambda event: None,
    )

    assert isinstance(callback, BaseCallback)
    callback.on_module_start(
        call_id="call-1",
        instance=object(),
        inputs={},
    )
    callback.on_module_end(
        call_id="call-1",
        outputs=None,
        exception=None,
    )


def test_agent_aforward_uses_reusable_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent import service

    calls: dict[str, object] = {}
    interpreter = MagicMock()

    class FakePredictRLM:
        def __init__(self, signature: object, **kwargs: object) -> None:
            calls["kwargs"] = kwargs

        async def acall(self, **kwargs: object) -> object:
            return SimpleNamespace(response="done", changed_files=[], trace=None)

    monkeypatch.setattr(service, "PredictRLM", FakePredictRLM)

    agent = service.FractalAgent(interpreter=interpreter)
    asyncio.run(agent.aforward(tmp_path, "update the README"))

    predictor_kwargs = calls["kwargs"]
    assert isinstance(predictor_kwargs, dict)
    assert predictor_kwargs["interpreter"] is interpreter
    assert "sandbox_backend" not in predictor_kwargs


def test_agent_aforward_omits_empty_included_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent import service

    calls: dict[str, object] = {}

    class FakePredictRLM:
        def __init__(self, signature: object, **kwargs: object) -> None:
            pass

        async def acall(self, **kwargs: object) -> object:
            calls["acall_kwargs"] = kwargs
            return SimpleNamespace(response="done", changed_files=[], trace=None)

    monkeypatch.setattr(service, "PredictRLM", FakePredictRLM)

    agent = service.FractalAgent(max_iterations=7, verbose=False, debug=True)
    asyncio.run(agent.aforward(tmp_path, "update the README"))

    acall_kwargs = calls["acall_kwargs"]
    assert isinstance(acall_kwargs, dict)
    assert acall_kwargs["included_paths"] is None


def test_build_direct_workspace_mounts_uses_absolute_paths(tmp_path: Path) -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent.service import build_direct_workspace_mounts

    included_path = tmp_path / "included"
    included_path.mkdir()

    mounts = build_direct_workspace_mounts(tmp_path, [included_path])

    assert [
        (mount.host_path, mount.sandbox_path)
        for mount in mounts
    ] == [
        (str(tmp_path.resolve()), str(tmp_path.resolve())),
        (str(included_path.resolve()), str(included_path.resolve())),
    ]


def test_agent_prewarm_prewarms_interpreter() -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from fractal.agent.service import FractalAgent

    interpreter = MagicMock()
    agent = FractalAgent(interpreter=interpreter)

    agent.prewarm()

    interpreter.prewarm.assert_called_once_with()


def test_predict_rlm_sees_absolute_workspace_paths(tmp_path: Path) -> None:
    if not workspace_available():
        pytest.skip("predict_rlm.Workspace is not exported by the local branch yet")

    from predict_rlm import PredictRLM

    from fractal.agent.service import Workspace, WorkspaceMode
    from fractal.agent.signature import build_edit_workspace_signature

    included_path = tmp_path / "included"
    included_path.mkdir()
    workspace = Workspace(path=str(tmp_path), mode=WorkspaceMode.DIRECT)
    included = Workspace(path=str(included_path), mode=WorkspaceMode.DIRECT)
    rlm = PredictRLM(
        build_edit_workspace_signature("test"),
        sub_lm=MagicMock(),
        max_iterations=1,
        sandbox_backend="sbx",
    )

    plan, args = rlm._prepare_file_io({
        "workspace": workspace,
        "included_paths": [included],
    })

    assert plan is not None
    assert args["workspace"] == str(tmp_path.resolve())
    assert args["included_paths"] == [str(included_path.resolve())]
    assert [
        (mount.host_path, mount.sandbox_path)
        for mount in plan["direct_workspace_mounts"]
    ] == [
        (str(tmp_path.resolve()), str(tmp_path.resolve())),
        (str(included_path.resolve()), str(included_path.resolve())),
    ]


def test_prediction_to_result_rejects_string_changed_files() -> None:
    from fractal.agent.service import _prediction_to_result

    prediction = SimpleNamespace(response="done", changed_files="README.md", trace=None)

    with pytest.raises(TypeError, match="changed_files"):
        _prediction_to_result(prediction)
