from __future__ import annotations

import argparse
import asyncio
import select
import sys
from pathlib import Path
from typing import Any, TextIO

from .config_commands import run_config_command
from .runtime_lms import resolve_runtime_lms

MAX_STDIN_BYTES = 10 * 1024 * 1024
STDIN_CONTEXT_GRACE_SECONDS = 1.0
MAX_ITERATIONS_EXIT_CODE = 2
DEFAULT_MAX_ITERATIONS = 30


FRACTAL_BANNER = r"""

               --
              ----
             ----
            ----
           ---- +++++
          ----    ++++
         ----      ++++
          --        ++++
      ++++++++++++++ ++++
     +++++++++++++++  ++++

 ______              _        _
|  ____|            | |      | |
| |__ _ __ __ _  ___| |_ __ _| |
|  __| '__/ _` |/ __| __/ _` | |
| |  | | | (_| | (__| || (_| | |
|_|  |_|  \__,_|\___|\__\__,_|_|
""".strip("\n")


def _print_startup_banner(output: TextIO) -> None:
    print(FRACTAL_BANNER, file=output)
    print(file=output)


def _emit_headless_json(stdout: TextIO, result: Any) -> None:
    stdout.write(result.to_json())
    stdout.write("\n")


def _effective_max_iterations(args: argparse.Namespace, lm_config: Any) -> int:
    if args.max_iterations is not None:
        return args.max_iterations
    defaults = getattr(lm_config, "defaults", None)
    if defaults is not None and defaults.max_iterations is not None:
        return defaults.max_iterations
    return DEFAULT_MAX_ITERATIONS


def _effective_verbose(args: argparse.Namespace, lm_config: Any) -> bool:
    if getattr(args, "verbose", False):
        return True
    if getattr(args, "prompt", None) is not None:
        return True
    defaults = getattr(lm_config, "defaults", None)
    return bool(defaults is not None and defaults.verbose)


def include_path(value: str) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise argparse.ArgumentTypeError(f"included path cannot be a symlink: {value}")
    resolved = path.resolve()
    if not resolved.exists():
        raise argparse.ArgumentTypeError(f"included path does not exist: {value}")
    if not resolved.is_dir():
        raise argparse.ArgumentTypeError(f"included path is not a directory: {value}")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fractal")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="workspace directory to edit; defaults to the current directory",
    )
    parser.add_argument("--lm", help="override the configured main DSPy LM")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        type=include_path,
        help=(
            "additional directory to mount into the sandbox at its absolute path; "
            "may be passed multiple times"
        ),
    )
    parser.add_argument("--sub-lm", help="override the configured DSPy sub-LM")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="max RLM iterations per turn; defaults to the configured value or 30",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="resume an existing workspace-local session by id",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="reserved for quieter terminal output"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show generated code and model-visible output for each RLM iteration",
    )
    parser.add_argument(
        "--debug", action="store_true", help="enable PredictRLM debug mode"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "tear down the directory's hot sandbox before starting so a clean "
            "one is created"
        ),
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help=(
            "do not reuse or keep a hot sandbox; create a throwaway one and "
            "remove it on exit"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "with -p, print one JSON result object to stdout instead of plain "
            "text (pair with --quiet for machine-only output)"
        ),
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help=(
            "run one Fractal turn non-interactively with this prompt; use '-' "
            "to read the full prompt from stdin"
        ),
    )

    subparsers = parser.add_subparsers(dest="command")
    config_parser = subparsers.add_parser(
        "config",
        help="inspect or repair global Fractal provider/model configuration",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
        required=True,
    )
    config_subparsers.add_parser(
        "show",
        help="show effective global config with credential references redacted",
    )
    status_parser = config_subparsers.add_parser(
        "status",
        help="validate configured provider, model, and auth availability",
    )
    status_parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the live provider connectivity check",
    )
    setup_parser = config_subparsers.add_parser(
        "setup",
        help="run interactive global provider/model/auth setup",
    )
    setup_parser.add_argument(
        "--offline",
        action="store_true",
        help="skip the live provider connectivity check after setup",
    )
    get_parser = config_subparsers.add_parser(
        "get",
        help="print one effective config value by dotted key",
    )
    get_parser.add_argument("key", help="dotted key, e.g. defaults.max_iterations")
    set_parser = config_subparsers.add_parser(
        "set",
        help="set one config value by dotted key",
    )
    set_parser.add_argument("key", help="dotted key, e.g. active_model")
    set_parser.add_argument("value", help="TOML literal or bare string")
    set_parser.add_argument(
        "--project",
        action="store_true",
        help="write to the workspace project config instead of the global config",
    )
    reset_parser = config_subparsers.add_parser(
        "reset",
        help="delete the global config so setup can start fresh",
    )
    reset_parser.add_argument(
        "--credentials",
        action="store_true",
        help="also delete locally stored API keys",
    )
    reset_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="skip the confirmation prompt",
    )
    unset_parser = config_subparsers.add_parser(
        "unset",
        help="remove one config value by dotted key",
    )
    unset_parser.add_argument("key", help="dotted key, e.g. active_sub_model")
    unset_parser.add_argument(
        "--project",
        action="store_true",
        help="edit the workspace project config instead of the global config",
    )
    return parser


def run_tui(args: argparse.Namespace, notifier: Any | None = None) -> int:
    from rich.console import Console

    console = Console()

    def show_setup_banner() -> None:
        console.print(FRACTAL_BANNER, style="bold #8b5cf6")
        console.print("")

    lm_config = resolve_runtime_lms(
        args,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        auto_setup=_stdin_is_tty(sys.stdin),
        on_setup_start=show_setup_banner,
    )
    if lm_config is None:
        return 1

    from .runtime import FractalRuntime
    from .tui import TerminalFractalApp

    workspace = args.workspace.resolve()
    display_verbose = _effective_verbose(args, lm_config)
    reuse_sandbox = not args.ephemeral
    if args.fresh and reuse_sandbox:
        from .agent.service import remove_sandbox_for

        remove_sandbox_for(workspace, args.include)
    runtime = FractalRuntime.create(
        workspace_path=workspace,
        included_paths=args.include,
        lm=lm_config.lm,
        sub_lm=lm_config.sub_lm,
        max_iterations=_effective_max_iterations(args, lm_config),
        verbose=False,
        debug=args.debug,
        session_id=args.resume,
        provider_selection=lm_config.provider_selection,
        sub_lm_follows_main=lm_config.sub_lm_follows_main,
        sub_model=lm_config.sub_model,
        reuse_sandbox=reuse_sandbox,
    )
    status_text = (
        "[dim]starting sandbox...[/dim]"
        if args.ephemeral
        else "[dim]starting sandbox (reusing hot sandbox if available)...[/dim]"
    )
    try:
        with console.status(status_text, spinner="dots"):
            runtime.prewarm()
        asyncio.run(
            TerminalFractalApp(
                runtime,
                console=console,
                verbose_iterations=display_verbose,
                banner=FRACTAL_BANNER,
                update_notice=notifier.notice() if notifier is not None else None,
            ).run()
        )
    finally:
        try:
            with console.status(
                "[dim]shutting down sandbox... press Ctrl-C again to force exit without cleaning up the sandbox[/dim]",
                spinner="dots",
            ):
                runtime.close()
        except KeyboardInterrupt:
            console.print(
                "sandbox shutdown interrupted; a sandbox may still be running. "
                "Run `sbx ls` and `sbx rm --force <name>` to clean it up.",
                style="yellow",
            )
            return 130
    return 0


def run_non_interactive(
    args: argparse.Namespace,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    notifier: Any | None = None,
) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr

    try:
        stdin_text = read_non_interactive_stdin(args.prompt, stdin)
        message = build_non_interactive_message(args.prompt, stdin_text)
    except ValueError as exc:
        print(f"fractal: {exc}", file=stderr)
        return 1

    # An empty prompt has nothing to run; skip the turn before importing the
    # runtime or spending a model call to answer nothing.
    if not message.strip():
        if not args.quiet:
            print("fractal: empty prompt; nothing to do", file=stderr)
        return 0

    from .session import HeadlessResult

    workspace = args.workspace.resolve()

    lm_config = resolve_runtime_lms(
        args,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        auto_setup=_stdin_is_tty(stdin),
    )
    if lm_config is None:
        if args.json:
            _emit_headless_json(
                stdout,
                HeadlessResult(
                    workspace=str(workspace),
                    status="failed",
                    error="could not resolve provider/model configuration",
                ),
            )
        return 1

    display_verbose = _effective_verbose(args, lm_config)
    from .runtime import FractalRuntime

    reuse_sandbox = not args.ephemeral
    if args.fresh and reuse_sandbox:
        from .agent.service import remove_sandbox_for

        remove_sandbox_for(workspace, args.include)
    try:
        runtime = FractalRuntime.create(
            workspace_path=workspace,
            included_paths=args.include,
            lm=lm_config.lm,
            sub_lm=lm_config.sub_lm,
            max_iterations=_effective_max_iterations(args, lm_config),
            verbose=False,
            debug=args.debug,
            session_id=args.resume,
            provider_selection=lm_config.provider_selection,
            sub_lm_follows_main=lm_config.sub_lm_follows_main,
            sub_model=lm_config.sub_model,
            reuse_sandbox=reuse_sandbox,
        )
    except Exception as exc:
        if args.json:
            _emit_headless_json(
                stdout,
                HeadlessResult(
                    workspace=str(workspace), status="failed", error=str(exc)
                ),
            )
        else:
            print(f"fractal: {exc}", file=stderr)
        return 1

    try:
        return _run_non_interactive_turn(
            args,
            runtime=runtime,
            message=message,
            display_verbose=display_verbose,
            stdout=stdout,
            stderr=stderr,
            notifier=notifier,
        )
    finally:
        try:
            runtime.close()
        except KeyboardInterrupt:
            print(
                "fractal: sandbox shutdown interrupted; a sandbox may still be "
                "running. Run `sbx ls` and `sbx rm --force <name>` to clean it up.",
                file=stderr,
            )
        except Exception as exc:
            print(f"fractal: sandbox cleanup failed: {exc}", file=stderr)


def _run_non_interactive_turn(
    args: argparse.Namespace,
    *,
    runtime: Any,
    message: str,
    display_verbose: bool,
    stdout: TextIO,
    stderr: TextIO,
    notifier: Any | None = None,
) -> int:
    from rich.console import Console

    from .session import HeadlessResult, headless_result_from_turn
    from .tui.app import render_iteration_event_log, render_trace_summary

    if not args.quiet:
        _print_startup_banner(stderr)
        if notifier is not None:
            notice = notifier.notice()
            if notice:
                print(notice, file=stderr)
        print(f"fractal: workspace {runtime.workspace_path}", file=stderr)
        print(f"fractal: session {runtime.session_id}", file=stderr)
        print("fractal: running RLM...", file=stderr)

    def print_runtime_event(event: Any) -> None:
        if not args.quiet:
            print(f"fractal: {event.message}", file=stderr)

    trace_console = Console(file=stderr, force_terminal=False, color_system=None)
    live_iteration_events_seen = 0

    def print_iteration_event(event: object) -> None:
        nonlocal live_iteration_events_seen
        if args.quiet or not display_verbose:
            return
        live_iteration_events_seen += 1
        trace_console.print(render_iteration_event_log(event, verbose=True))

    try:
        result = asyncio.run(
            runtime.submit(
                message,
                on_runtime_event=print_runtime_event if not args.quiet else None,
                on_iteration_event=(
                    print_iteration_event
                    if display_verbose and not args.quiet
                    else None
                ),
            )
        )
    except KeyboardInterrupt:
        if args.json:
            _emit_headless_json(
                stdout,
                HeadlessResult(
                    session_id=runtime.session_id,
                    workspace=str(runtime.workspace_path),
                    status="interrupted",
                    error="interrupted by user",
                ),
            )
        else:
            print("fractal: interrupted", file=stderr)
        return 130
    except Exception as exc:
        if args.json:
            _emit_headless_json(
                stdout,
                HeadlessResult(
                    session_id=runtime.session_id,
                    workspace=str(runtime.workspace_path),
                    status="failed",
                    error=str(exc),
                ),
            )
        else:
            print(f"fractal: failed: {exc}", file=stderr)
        return 1

    if (
        display_verbose
        and not args.quiet
        and live_iteration_events_seen == 0
        and result.trace is not None
    ):
        trace_console.print(render_trace_summary(result.trace, verbose=True))

    if args.json:
        _emit_headless_json(
            stdout,
            headless_result_from_turn(
                session_id=runtime.session_id,
                workspace=str(runtime.workspace_path),
                response=result.response,
                changed_files=result.changed_files,
                trace=result.trace,
            ),
        )
    elif result.response:
        stdout.write(result.response)
        if not result.response.endswith("\n"):
            stdout.write("\n")

    if result.changed_files and not args.quiet:
        print(
            "fractal: changed files " + ", ".join(result.changed_files),
            file=stderr,
        )

    if not args.quiet:
        from .session import turn_usage_from_trace

        usage = turn_usage_from_trace(result.trace)
        if usage is not None and (usage.input_tokens or usage.output_tokens):
            print(
                f"fractal: usage {usage.input_tokens:,} in / "
                f"{usage.output_tokens:,} out tokens, ${usage.cost:.4f}",
                file=stderr,
            )

    if result.trace is not None and result.trace.status == "max_iterations":
        print("fractal: max iterations reached before completion", file=stderr)
        return MAX_ITERATIONS_EXIT_CODE

    if not args.quiet:
        print("fractal: complete", file=stderr)
    return 0


def read_non_interactive_stdin(prompt: str, stdin: TextIO) -> str | None:
    if prompt == "-":
        stdin_text = _read_limited_stdin(stdin)
        if not stdin_text:
            raise ValueError("-p - requires stdin input")
        return stdin_text

    if _stdin_is_tty(stdin):
        return None
    if not _stdin_has_data(stdin):
        return None
    stdin_text = _read_limited_stdin(stdin)
    return stdin_text or None


def build_non_interactive_message(prompt: str, stdin_text: str | None) -> str:
    if prompt == "-":
        return stdin_text or ""
    if stdin_text is None:
        return prompt
    return (
        f"{prompt}\n\n<Fractal stdin context>\n{stdin_text}\n</Fractal stdin context>"
    )


def _read_limited_stdin(stdin: TextIO) -> str:
    chunks: list[str] = []
    total_bytes = 0
    while True:
        chunk = stdin.read(8192)
        if chunk == "":
            break
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > MAX_STDIN_BYTES:
            raise ValueError(
                f"stdin input exceeds {MAX_STDIN_BYTES // (1024 * 1024)} MiB limit"
            )
        chunks.append(chunk)
    return "".join(chunks)


def _stdin_has_data(stdin: TextIO) -> bool:
    """Whether implicit stdin context should be read.

    A non-TTY stdin that stays open without delivering data (CI runners,
    process supervisors, agent harnesses) must not block the turn, so wait at
    most STDIN_CONTEXT_GRACE_SECONDS for the first byte. Streams without a
    selectable fd (e.g. StringIO) cannot block and are always read.
    """
    try:
        fd = stdin.fileno()
    except (OSError, ValueError, AttributeError):
        return True
    try:
        ready, _, _ = select.select([fd], [], [], STDIN_CONTEXT_GRACE_SECONDS)
    except (OSError, ValueError):
        return True
    return bool(ready)


def _stdin_is_tty(stdin: TextIO) -> bool:
    try:
        return stdin.isatty()
    except AttributeError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config":
        return run_config_command(args)

    # Kick off the stealth PyPI update check now so it overlaps with config
    # resolution and sandbox prewarm; the notice is read back once we're ready
    # to render. Skipped for --json so machine-readable output stays clean.
    notifier = None
    if not getattr(args, "json", False):
        from .version_check import UpdateNotifier

        notifier = UpdateNotifier.start()

    if args.prompt is not None:
        return run_non_interactive(args, notifier=notifier)
    return run_tui(args, notifier=notifier)


if __name__ == "__main__":
    raise SystemExit(main())
