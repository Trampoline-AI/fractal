from __future__ import annotations

import asyncio
import re
import signal
import sys
from pathlib import Path
from typing import Protocol, TextIO

from predict_rlm import IterationStep, RunTrace
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from fractal.agent.schema import FractalIterationEvent, FractalResult
from fractal.context_meter import (
    ContextEstimateCacheKey,
    context_estimate_cache_key,
    estimate_next_context_tokens,
)
from fractal.events import FractalRuntimeEvent
from fractal.session import (
    SessionSummary,
    SummaryTurn,
    TurnUsage,
    list_sessions,
    summarize_usage,
    turn_usage_from_trace,
)

INPUT_PROMPT = HTML("<prompt>fractal</prompt><session>›</session> ")

PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "bold #8b5cf6",
        "session": "ansibrightblack",
        "bottom-toolbar": "noreverse",
        "bottom-toolbar.label": "#6b7280",
        "bottom-toolbar.value": "ansicyan",
    }
)
SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/sessions": "List resumable sessions in this workspace",
    "/resume": "Resume an existing session by id",
    "/new": "Start a fresh session",
    "/model": "Change the main model and sub-model",
    "/provider": "Change provider, model, and auth setup",
    "/usage": "Show token usage and cost for this session",
    "/verbose": "Toggle verbose RLM iteration output",
    "/exit": "Exit Fractal",
    "/quit": "Exit Fractal",
}
RUNNING_STATUS = "[dim]running RLM... (Ctrl-C to interrupt)[/dim]"
INTERRUPTING_STATUS = "[yellow]interrupting RLM...[/yellow] [dim](waiting for shutdown)[/dim]"
RUNTIME_EVENT_STYLES = {
    "file_read": "cyan",
    "file_write": "yellow",
    "command": "magenta",
}
MARKDOWN_STYLE_OVERRIDES = {
    # Rich defaults inline code to "cyan on black", which reads as a
    # highlight block inside Fractal's already framed response panel.
    # Keep Markdown emphasis visible without adding another background.
    "markdown.code": "bold cyan",
    "markdown.code_block": "cyan",
    "markdown.strong": "bold",
    "markdown.item.bullet": "bright_black",
    "markdown.list": "none",
}
MARKDOWN_THEME = Theme(MARKDOWN_STYLE_OVERRIDES)


class FractalMarkdown(Markdown):
    def __rich_console__(self, console: Console, options: object) -> object:
        with console.use_theme(MARKDOWN_THEME):
            yield from super().__rich_console__(console, options)


class SlashCommandCompleter(Completer):
    def get_completions(self, document: Document, complete_event: object) -> object:
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for command, description in SLASH_COMMANDS.items():
            if command.startswith(text):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display_meta=description,
                )


def slash_command_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event: object) -> None:
        buffer = event.current_buffer
        complete_state = buffer.complete_state
        completion = (
            complete_state.current_completion if complete_state is not None else None
        )
        if completion is not None:
            buffer.apply_completion(completion)
            if not buffer.document.text_before_cursor.endswith(" "):
                buffer.insert_text(" ")
            return
        buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _(event: object) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-j")
    def _(event: object) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


def _prompt_continuation(width: int, line_number: int, is_soft_wrap: bool) -> str:
    return "        "


class _FooterPromptSession(PromptSession[str]):
    """PromptSession whose bottom toolbar hugs the input box.

    prompt_toolkit claims every row between the cursor and the bottom of the
    screen and renders the bottom toolbar on the region's last row, leaving a
    blank gap whenever the prompt sits above the last screen row. A
    filler window appended below the toolbar absorbs that extra space instead,
    so the toolbar stays glued to the input.
    """

    def _create_layout(self) -> Layout:
        layout = super()._create_layout()
        container = layout.container
        if isinstance(container, HSplit):
            container.children.append(
                Window(height=Dimension(preferred=0, weight=1))
            )
        return layout


def _format_token_count(tokens: int) -> str:
    if tokens < 1000:
        return str(tokens)
    if tokens < 1_000_000:
        return f"{tokens / 1000:.1f}k"
    return f"{tokens / 1_000_000:.2f}M"


class SessionLike(Protocol):
    @property
    def summary_model(self) -> SessionSummary: ...


class FractalRuntimeLike(Protocol):
    workspace_path: Path

    @property
    def session_id(self) -> str: ...

    @property
    def session(self) -> SessionLike: ...

    def resume(self, session_id: str) -> None: ...

    def new_session(self) -> None: ...

    @property
    def provider_label(self) -> str: ...

    @property
    def model_label(self) -> str: ...

    @property
    def sub_model_label(self) -> str: ...

    def apply_provider_selection(
        self, selection: object, *, sub_model: str | None = None
    ) -> None: ...

    async def submit(self, user_message: str, **kwargs: object) -> FractalResult: ...


class TerminalFractalApp:
    """Terminal-native Fractal interface using the user's normal scrollback."""

    def __init__(
        self,
        runtime: FractalRuntimeLike,
        *,
        console: Console | None = None,
        input_stream: TextIO | None = None,
        prompt_session: PromptSession[str] | None = None,
        verbose_iterations: bool = False,
        banner: str | None = None,
        update_notice: str | None = None,
        config_stdin: TextIO | None = None,
        config_stdout: TextIO | None = None,
        config_stderr: TextIO | None = None,
    ) -> None:
        self.runtime = runtime
        self.console = console or Console()
        self.input_stream = input_stream
        self.verbose_iterations = verbose_iterations
        self.banner = banner
        self.update_notice = update_notice
        self.config_stdin = config_stdin or input_stream or sys.stdin
        self.config_stdout = config_stdout or getattr(self.console, "file", sys.stdout)
        self.config_stderr = config_stderr or getattr(self.console, "file", sys.stderr)
        self.prompt_session = prompt_session or _FooterPromptSession(
            style=PROMPT_STYLE,
            completer=SlashCommandCompleter(),
            # Only auto-complete (and reserve rows for the menu) while typing
            # a slash command; otherwise the input box stays one row tall with
            # the footer glued below it.
            complete_while_typing=Condition(self._typing_slash_command),
            key_bindings=slash_command_key_bindings(),
            multiline=True,
            prompt_continuation=_prompt_continuation,
            bottom_toolbar=self._render_bottom_toolbar,
        )
        self._rendered_turn_ids: set[str] = set()
        self._pending_turn_ids: set[str] = set()
        self._prompt_echo_turn_ids: set[str] = set()
        self._sigint_mode = "prompt"
        self._active_submit_task: asyncio.Task[FractalResult] | None = None
        self._turn_interrupt_requested = False
        self._active_status: Status | None = None
        self._last_turn_live_iteration_count = 0
        self._context_estimate_cache: (
            tuple[ContextEstimateCacheKey, int | None] | None
        ) = None

    async def run(self) -> None:
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)
        try:
            if self.input_stream is None and self.console.is_terminal:
                self._pad_to_bottom()
            if self.banner:
                self.console.print(Text(self.banner, style="bold #8b5cf6"))
                self.console.print()
            self.render_header()
            self.render_new_turns()

            while True:
                self._sigint_mode = "prompt"
                self._active_submit_task = None
                self._turn_interrupt_requested = False

                message = await self.read_message()
                if message is None or message in {"/exit", "/quit"}:
                    return
                if not message:
                    continue
                if await self.handle_slash_command(message):
                    self._sigint_mode = "prompt"
                    continue

                self._sigint_mode = "turn"
                await self._execute_turn(message)
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

    async def _execute_turn(self, message: str) -> None:
        result = await self.run_turn(message)
        if result is None:
            return
        self.console.print(render_turn_footer(result))
        if result.changed_files:
            self.console.print(render_changed_files(result.changed_files))
        if result.trace is not None and self._last_turn_live_iteration_count == 0:
            self.console.print(
                render_trace_summary(
                    result.trace,
                    verbose=self.verbose_iterations,
                )
            )
        self.render_new_turns()

    def _pad_to_bottom(self) -> None:
        # Land the first prompt at the bottom of the screen so history flows
        # upward and the input box sits directly on the status footer. The
        # prompt claims every row below the cursor (prompt_toolkit renders its
        # bottom toolbar at the end of that region), so leaving fewer rows
        # below the header means a smaller gap. Header (3) + the blank line
        # printed before the prompt + input + footer = 6 rows, plus the banner
        # and its trailing blank line when one is set.
        rows = 6
        if self.banner:
            rows += len(self.banner.splitlines()) + 1
        pad = max(self.console.height - rows, 0)
        if pad:
            self.console.print("\n" * (pad - 1))

    async def run_turn(self, message: str) -> FractalResult | None:
        def mark_pending() -> None:
            self.mark_latest_turn_as_prompt_echoed()

        loop = asyncio.get_running_loop()

        def show_runtime_event(event: FractalRuntimeEvent) -> None:
            loop.call_soon_threadsafe(self._show_runtime_event_status, event)

        live_iteration_events_seen = 0

        def show_iteration_event(event: FractalIterationEvent) -> None:
            nonlocal live_iteration_events_seen
            live_iteration_events_seen += 1
            loop.call_soon_threadsafe(self._show_iteration_event_status, event)

        status = self.console.status(RUNNING_STATUS, spinner="dots")
        status.start()
        self._active_status = status
        status_running = True
        self._last_turn_live_iteration_count = 0
        if self._turn_interrupt_requested:
            self._show_interrupting_status()

        def stop_status() -> None:
            nonlocal status_running
            if status_running:
                status.stop()
                status_running = False

        submit_task = asyncio.create_task(
            self.runtime.submit(
                message,
                on_pending=mark_pending,
                on_runtime_event=show_runtime_event,
                on_iteration_event=show_iteration_event,
                interrupt_requested=lambda: self._turn_interrupt_requested,
            )
        )
        self._active_submit_task = submit_task
        try:
            result = await submit_task
        except asyncio.CancelledError:
            if not self._turn_interrupt_requested:
                raise
            stop_status()
            self.render_new_turns()
            return None
        except Exception:
            stop_status()
            self.console.print(Text("✗ failed", style="red"))
            self.render_new_turns()
            return None
        finally:
            self._last_turn_live_iteration_count = live_iteration_events_seen
            self._active_submit_task = None
            self._active_status = None
            self._sigint_mode = "prompt"
            stop_status()
        return result

    def _handle_sigint(self, signum: int, frame: object) -> None:
        if self._sigint_mode != "turn":
            # A second Ctrl-C can arrive after the interrupted turn has already
            # returned control to the prompt. Raising from the process signal
            # handler escapes prompt_toolkit/asyncio and crashes the CLI.
            return
        self._turn_interrupt_requested = True
        self._show_interrupting_status()
        task = self._active_submit_task
        if task is not None and not task.done():
            task.cancel()

    def _typing_slash_command(self) -> bool:
        buffer = getattr(self.prompt_session, "default_buffer", None)
        if buffer is None:
            return False
        return buffer.document.text.startswith("/")

    def _render_bottom_toolbar(self) -> list[tuple[str, str]]:
        try:
            return self._bottom_toolbar_fragments()
        except Exception:
            return [("class:bottom-toolbar.label", " fractal")]

    def _bottom_toolbar_fragments(self) -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        model_label = getattr(self.runtime, "model_label", None)
        if model_label is None:
            lm = getattr(self.runtime, "lm", None)
            model_label = str(lm) if lm is not None else None
        if model_label:
            sub_label = getattr(self.runtime, "sub_model_label", None) or model_label
            fragments.append(("class:bottom-toolbar.label", " model "))
            fragments.append(("class:bottom-toolbar.value", str(model_label)))
            fragments.append(("class:bottom-toolbar.label", " · sub "))
            fragments.append(("class:bottom-toolbar.value", str(sub_label)))
            fragments.append(("class:bottom-toolbar.label", " · verbose "))
        else:
            fragments.append(("class:bottom-toolbar.label", " verbose "))
        fragments.append(
            (
                "class:bottom-toolbar.value",
                "on" if self.verbose_iterations else "off",
            )
        )

        tokens = self._next_context_tokens()
        if tokens:
            fragments.append(("class:bottom-toolbar.label", " · "))
            fragments.append(
                ("class:bottom-toolbar.value", f"~{_format_token_count(tokens)} ctx")
            )
        return fragments

    def _next_context_tokens(self) -> int | None:
        try:
            key = context_estimate_cache_key(self.runtime)
        except Exception:
            return None
        if self._context_estimate_cache is not None:
            cached_key, cached_tokens = self._context_estimate_cache
            if cached_key == key:
                return cached_tokens
        try:
            tokens = estimate_next_context_tokens(self.runtime)
        except Exception:
            tokens = None
        self._context_estimate_cache = (key, tokens)
        return tokens

    def _show_interrupting_status(self) -> None:
        status = self._active_status
        if status is None:
            return
        status.update(INTERRUPTING_STATUS)

    def _show_runtime_event_status(self, event: FractalRuntimeEvent) -> None:
        if self._turn_interrupt_requested:
            return
        self.console.print(render_runtime_event_log(event))

    def _show_iteration_event_status(self, event: FractalIterationEvent) -> None:
        if self._turn_interrupt_requested:
            return
        self.console.print(
            render_iteration_event_log(
                event,
                verbose=self.verbose_iterations,
            )
        )

    def render_header(self) -> None:
        self.console.print(
            Text.assemble(
                ("Fractal", "bold"),
                " | ",
                (str(self.runtime.workspace_path), "dim"),
                " | session ",
                (self.runtime.session_id, "cyan"),
            )
        )
        model_label = getattr(self.runtime, "model_label", None)
        if model_label is None:
            lm = getattr(self.runtime, "lm", None)
            model_label = str(lm) if lm is not None else None
        verbose_state = "on" if self.verbose_iterations else "off"
        if model_label:
            sub_label = getattr(self.runtime, "sub_model_label", None) or model_label
            self.console.print(
                Text.assemble(
                    ("model ", "dim"),
                    (model_label, "dim cyan"),
                    (" | sub ", "dim"),
                    (str(sub_label), "dim cyan"),
                    (" | verbose ", "dim"),
                    (verbose_state, "dim cyan"),
                )
            )
        else:
            self.console.print(
                Text.assemble(
                    ("verbose ", "dim"),
                    (verbose_state, "dim cyan"),
                )
            )
        self.console.print(
            Text(
                "Type /help for commands, /exit to quit. "
                "Alt+Enter inserts a newline.",
                style="dim",
            )
        )
        if self.update_notice:
            self.console.print(Text(self.update_notice, style="yellow"))

    async def handle_slash_command(self, message: str) -> bool:
        command, _, rest = message.partition(" ")
        rest = rest.strip()
        if command == "/resume":
            self._handle_resume(rest)
            return True
        if command == "/help":
            self._handle_help()
            return True
        if command == "/sessions":
            self._handle_sessions()
            return True
        if command == "/new":
            self._handle_new_session()
            return True
        if command == "/usage":
            self._handle_usage()
            return True
        if command == "/provider":
            return await self.handle_provider_command(rest)
        if command == "/model":
            return await self.handle_model_command(rest)
        if command == "/verbose":
            return self.handle_verbose_command(rest)
        if _looks_like_slash_command(message):
            self.console.print(
                Text(f"unknown command: {command} (try /help)", style="yellow")
            )
            return True
        return False

    def _handle_resume(self, session_id: str) -> None:
        if not session_id:
            self.console.print(Text("usage: /resume <session-id>", style="yellow"))
            return
        try:
            self.runtime.resume(session_id)
        except FileNotFoundError as exc:
            self.console.print(Text(str(exc), style="red"))
            return
        self._reset_rendered_state()
        self.console.print(Text(f"resumed session {self.runtime.session_id}", style="dim"))
        self.render_new_turns()

    def _handle_help(self) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True)
        table.add_column()
        for command, description in SLASH_COMMANDS.items():
            table.add_row(Text(command, style="cyan"), Text(description, style="dim"))
        self.console.print(table)

    def _handle_sessions(self) -> None:
        sessions = list_sessions(self.runtime.workspace_path)
        if not sessions:
            self.console.print(Text("No stored sessions in this workspace.", style="dim"))
            return
        table = Table.grid(padding=(0, 2))
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True, justify="right")
        table.add_column(overflow="ellipsis", max_width=48)
        for info in sessions:
            marker = " (current)" if info.session_id == self.runtime.session_id else ""
            table.add_row(
                Text.assemble((info.session_id, "cyan"), (marker, "dim")),
                Text(f"{info.turn_count} turns", style="dim"),
                Text(info.first_message or "(empty)", style="dim"),
            )
        self.console.print(table)
        self.console.print(Text("Resume one with /resume <session-id>.", style="dim"))

    def _handle_new_session(self) -> None:
        self.runtime.new_session()
        self._reset_rendered_state()
        self.console.print(
            Text(f"started new session {self.runtime.session_id}", style="dim")
        )

    def _handle_usage(self) -> None:
        totals = summarize_usage(self.runtime.session.summary_model)
        self.console.print(render_usage_report(totals))

    def _reset_rendered_state(self) -> None:
        self._rendered_turn_ids.clear()
        self._pending_turn_ids.clear()
        self._prompt_echo_turn_ids.clear()

    async def handle_provider_command(self, rest: str) -> bool:
        if rest.strip():
            self.console.print(Text("usage: /provider", style="yellow"))
            return True

        if await self.run_provider_setup():
            self.console.print(
                Text(
                    "Provider updated for this session and saved as the default.",
                    style="dim",
                )
            )
            self._warn_project_override(
                ("active_provider", "active_model", "active_sub_model")
            )
        return True

    async def handle_model_command(self, rest: str) -> bool:
        if rest.strip():
            self.console.print(Text("usage: /model", style="yellow"))
            return True

        if await self.run_model_setup():
            self.console.print(
                Text(
                    f"Model updated to {self.runtime.model_label} "
                    f"(sub {self.runtime.sub_model_label}) for this session.",
                    style="dim",
                )
            )
            self._warn_project_override(("active_model", "active_sub_model"))
        return True

    def _warn_project_override(self, keys: tuple[str, ...]) -> None:
        """Warn when a project config will mask a change saved globally."""
        from fractal.config import load_project_config, project_config_path

        try:
            project = load_project_config(self.runtime.workspace_path)
        except Exception:
            return
        if project is None:
            return
        overridden = [key for key in keys if getattr(project, key, None) is not None]
        if not overridden:
            return
        path = project_config_path(self.runtime.workspace_path)
        self.console.print(
            Text(
                f"note: {path} overrides {', '.join(overridden)}; the saved "
                "default applies now but the project value wins on next launch.",
                style="yellow",
            )
        )

    def handle_verbose_command(self, rest: str) -> bool:
        mode = rest.strip().lower()
        if mode in {"", "toggle"}:
            self.verbose_iterations = not self.verbose_iterations
        elif mode == "on":
            self.verbose_iterations = True
        elif mode == "off":
            self.verbose_iterations = False
        else:
            self.console.print(Text("usage: /verbose [on|off]", style="yellow"))
            return True
        state = "on" if self.verbose_iterations else "off"
        self.console.print(Text(f"verbose iteration output {state}", style="dim"))
        self.console.print(
            Text(
                "applies to this session only; persist with "
                "`fractal config set defaults.verbose "
                f"{'true' if self.verbose_iterations else 'false'}`",
                style="dim",
            )
        )
        return True

    async def run_provider_setup(self) -> bool:
        from fractal.config import FractalConfigError, write_config
        from fractal.onboarding import SetupInputError, async_prompt_for_config
        from fractal.providers import ProviderError
        from fractal.runtime_lms import selection_from_config, sub_selection_from_config

        try:
            existing = _existing_config()
            config = await async_prompt_for_config(
                stdin=self.config_stdin,
                stdout=self.config_stdout,
                existing=existing,
            )
            selection = selection_from_config(config)
            sub_selection = sub_selection_from_config(config)
            self.runtime.apply_provider_selection(
                selection,
                sub_selection=sub_selection,
            )
            path = write_config(config)
        except (FractalConfigError, ProviderError, SetupInputError, ValueError) as exc:
            print(f"fractal provider setup: {exc}", file=self.config_stderr)
            print(
                "No config was written. Fix the issue, then run "
                "`/provider` again.",
                file=self.config_stderr,
            )
            return False

        print(f"Fractal config written to {path}", file=self.config_stdout)
        return True

    async def run_model_setup(self) -> bool:
        from fractal.config import FractalConfigError, load_config, write_config
        from fractal.onboarding import (
            SetupInputError,
            async_prompt_for_model,
            async_prompt_for_sub_model,
        )
        from fractal.providers import ProviderError, get_provider
        from fractal.runtime_lms import selection_from_config, sub_selection_from_config

        try:
            result = load_config()
            if result.config is None:
                raise SetupInputError("no config found; run `/provider` first")
            provider = get_provider(result.config.active_provider)
            model = await async_prompt_for_model(
                provider=provider,
                stdin=self.config_stdin,
                stdout=self.config_stdout,
            )
            # The sub-model may live on its own provider; /model changes the
            # models only, /provider changes the providers.
            sub_provider_id = (
                result.config.active_sub_provider or result.config.active_provider
            )
            sub_model = await async_prompt_for_sub_model(
                provider=get_provider(sub_provider_id),
                main_model=model,
                stdin=self.config_stdin,
                stdout=self.config_stdout,
                current=result.config.active_sub_model,
                allow_same=result.config.active_sub_provider is None,
            )
            config = result.config.model_copy(
                update={"active_model": model, "active_sub_model": sub_model}
            )
            selection = selection_from_config(config, path=result.path)
            sub_selection = sub_selection_from_config(config, path=result.path)
            self.runtime.apply_provider_selection(
                selection, sub_selection=sub_selection
            )
            path = write_config(config, path=result.path)
        except (FractalConfigError, ProviderError, SetupInputError, ValueError) as exc:
            print(f"fractal model setup: {exc}", file=self.config_stderr)
            print(
                "No config was written. Fix the issue, then run `/model` again.",
                file=self.config_stderr,
            )
            return False

        print(f"Fractal config written to {path}", file=self.config_stdout)
        return True

    def render_new_turns(self) -> None:
        for turn in self.runtime.session.summary_model.turns:
            if turn.turn_id in self._rendered_turn_ids:
                continue
            if turn.agent is None:
                if turn.turn_id not in self._pending_turn_ids:
                    if turn.turn_id not in self._prompt_echo_turn_ids:
                        self.render_turn(turn, pending=True)
                    self._pending_turn_ids.add(turn.turn_id)
                continue
            if turn.turn_id in self._pending_turn_ids:
                self.console.print(Rule(style="dim"))
                self.console.print(render_agent_response(turn))
                self._pending_turn_ids.remove(turn.turn_id)
            elif turn.turn_id in self._prompt_echo_turn_ids:
                self.console.print(Rule(style="dim"))
                self.console.print(render_agent_response(turn))
            else:
                self.render_turn(turn)
            self._rendered_turn_ids.add(turn.turn_id)

    def render_turn(self, turn: SummaryTurn, *, pending: bool = False) -> None:
        self.console.print(render_user_message(turn.user.message))
        self.console.print(Rule(style="dim"))
        self.console.print(render_agent_response(turn, pending=pending))

    def mark_latest_turn_as_prompt_echoed(self) -> None:
        if self.runtime.session.summary_model.turns:
            self._prompt_echo_turn_ids.add(
                self.runtime.session.summary_model.turns[-1].turn_id
            )

    async def read_message(self) -> str | None:
        self.console.print()
        if self.input_stream is None:
            try:
                message = await self.prompt_session.prompt_async(
                    INPUT_PROMPT,
                    handle_sigint=False,
                    wrap_lines=True,
                )
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return None
            message = message.strip()
            if _will_submit_turn(message):
                self._sigint_mode = "turn"
            return message

        try:
            message = await asyncio.to_thread(self._readline)
        except EOFError:
            self.console.print()
            return None
        message = message.strip()
        if _will_submit_turn(message):
            self._sigint_mode = "turn"
        return message

    def _readline(self) -> str:
        assert self.input_stream is not None
        self.console.print(render_prompt_label(), end="")
        line = self.input_stream.readline()
        if line == "":
            raise EOFError
        return line


def render_summary(summary: SessionSummary) -> Group:
    rendered: list[object] = []
    for index, turn in enumerate(summary.turns):
        if index > 0:
            rendered.append(Rule(style="dim"))
        rendered.append(render_user_message(turn.user.message))
        rendered.append(Rule(style="dim"))
        rendered.append(render_agent_response(turn))
    return Group(*rendered)


def render_trace_summary(trace: RunTrace, *, verbose: bool = False) -> Group:
    rendered: list[object] = []
    if trace.steps:
        rendered.append("")
    for index, step in enumerate(trace.steps):
        if index > 0:
            rendered.append("")
        rendered.append(
            render_trace_step(
                step,
                max_iterations=trace.max_iterations,
                verbose=verbose,
            )
        )
    if not rendered:
        rendered.append(Text("No RLM iteration trace captured.", style="dim italic"))
    return Group(*rendered)


def render_iteration_event_log(
    event: FractalIterationEvent,
    *,
    verbose: bool = False,
) -> Group:
    return Group(
        "",
        render_trace_step(
            event.step,
            max_iterations=event.max_iterations,
            verbose=verbose,
        ),
    )


def render_trace_step(
    step: IterationStep,
    *,
    max_iterations: int,
    verbose: bool = False,
) -> Group:
    code = step.code
    output = step.untruncated_output or step.output
    model_output = step.output
    reasoning = step.reasoning.strip()
    iteration = step.iteration
    status = "error" if step.error else "ok"

    text = Text()
    text.append(
        f"RLM turn {iteration}/{max_iterations} ",
        style="bold bright_black",
    )
    text.append(f"({status})", style="red" if status == "error" else "dim")
    rendered: list[object] = [text]
    if reasoning:
        rendered.append(render_reasoning(reasoning))
    rendered.append(
        Padding(
            Text(
                f"python: {_line_count(code)} lines\noutput: {len(output)} chars",
                style="dim",
            ),
            (0, 0, 0, 2),
        )
    )
    if verbose:
        rendered.append(render_trace_detail("code:", code, syntax="python"))
        rendered.append(render_trace_detail("output:", model_output))
    return Group(*rendered)


def render_trace_detail(label: str, body: str, *, syntax: str | None = None) -> Group:
    if body:
        content: Text | Syntax = (
            Syntax(
                body,
                syntax,
                background_color="default",
                line_numbers=False,
                word_wrap=True,
            )
            if syntax is not None
            else Text(body, style="dim")
        )
    else:
        content = Text("(empty)", style="dim italic")
    return Group(
        Padding(Text(label, style="dim italic"), (0, 0, 0, 2)),
        Padding(content, (0, 0, 0, 4)),
    )


def render_reasoning(reasoning: str) -> Padding:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(no_wrap=True)
    table.add_column(ratio=1)
    table.add_row(
        Text("reasoning:", style="dim italic"),
        Text(reasoning, style="dim italic"),
    )
    return Padding(table, (0, 0, 0, 2))


def render_runtime_event_log(event: FractalRuntimeEvent) -> Text:
    style = RUNTIME_EVENT_STYLES.get(event.kind, "cyan")
    text = Text.assemble(
        ("  ", "dim"),
        (event.message, style),
    )
    return text


def render_user_message(message: str) -> Group:
    return Group(
        "",
        Text.assemble(render_prompt_label(), (message, "bold")),
    )


def render_prompt_label() -> Text:
    return Text.assemble(("fractal", "bold #8b5cf6"), ("›", "bright_black"), " ")


def _existing_config() -> object | None:
    from fractal.config import FractalConfigError, load_config

    try:
        return load_config().config
    except FractalConfigError:
        return None


def _will_submit_turn(message: str) -> bool:
    if not message:
        return False
    return not _looks_like_slash_command(message)


def _looks_like_slash_command(message: str) -> bool:
    command, _, _ = message.partition(" ")
    if command in SLASH_COMMANDS:
        return True
    # A leading "/word" reads as a command attempt; absolute paths and other
    # slash-containing prompts fall through to the agent.
    return bool(re.fullmatch(r"/[A-Za-z][A-Za-z0-9_-]*", command))


def render_agent_message(turn: SummaryTurn, *, pending: bool = False) -> object:
    if pending or turn.agent is None:
        return Text("Running...", style="italic dim")
    elif turn.agent.status == "failed":
        return Text(turn.agent.error or "Turn failed.", style="red")
    elif turn.agent.status == "interrupted":
        return Text(turn.agent.error or "Turn interrupted by user.", style="yellow")
    elif turn.agent.status == "max_iterations":
        # PredictRLM's fallback can contain useful work, but the agent did not
        # explicitly SUBMIT it. Make that state visible in scrollback.
        response: Text | Markdown
        if turn.agent.response:
            response = FractalMarkdown(turn.agent.response)
        else:
            response = Text("No fallback response.", style="dim")
        body = Group(
            Text("Reached max iterations; showing fallback response.", style="yellow"),
            "",
            response,
        )
        return body
    else:
        return FractalMarkdown(turn.agent.response)


def render_agent_response(turn: SummaryTurn, *, pending: bool = False) -> Padding:
    return Padding(render_agent_message(turn, pending=pending), (0, 0, 0, 2))


def render_turn_footer(result: FractalResult) -> Text:
    if result.trace is not None and result.trace.status == "max_iterations":
        footer = Text("! max iterations", style="yellow")
    else:
        footer = Text("✓ complete")
    usage = turn_usage_from_trace(result.trace)
    if usage is None:
        return footer
    parts: list[str] = []
    if usage.iterations:
        unit = "iteration" if usage.iterations == 1 else "iterations"
        parts.append(f"{usage.iterations} {unit}")
    if usage.duration_ms:
        parts.append(_format_duration(usage.duration_ms))
    if usage.input_tokens or usage.output_tokens:
        parts.append(
            f"{_format_tokens(usage.input_tokens)} in / "
            f"{_format_tokens(usage.output_tokens)} out"
        )
    if usage.context_tokens:
        parts.append(f"{_format_tokens(usage.context_tokens)} ctx")
    for part in parts:
        footer.append(" · ", style="dim")
        footer.append(part, style="dim")
    return footer


def render_changed_files(changed_files: list[str]) -> Text:
    text = Text("  changed: ", style="dim")
    text.append(", ".join(changed_files), style="yellow")
    return text


def render_usage_report(totals: TurnUsage) -> Group:
    if not (totals.input_tokens or totals.output_tokens or totals.iterations):
        return Group(Text("No recorded usage for this session yet.", style="dim"))
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(justify="right")
    table.add_row(Text("input tokens", style="dim"), Text(f"{totals.input_tokens:,}"))
    table.add_row(Text("output tokens", style="dim"), Text(f"{totals.output_tokens:,}"))
    if totals.context_tokens:
        table.add_row(
            Text("current context", style="dim"),
            Text(f"~{totals.context_tokens:,} tokens"),
        )
    table.add_row(Text("iterations", style="dim"), Text(f"{totals.iterations:,}"))
    table.add_row(
        Text("agent time", style="dim"), Text(_format_duration(totals.duration_ms))
    )
    table.add_row(Text("cost", style="dim"), Text(f"${totals.cost:.4f}"))
    return Group(table)


def _format_tokens(count: int) -> str:
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}k"
    return str(count)


def _format_duration(duration_ms: int) -> str:
    if duration_ms < 1_000:
        return f"{duration_ms}ms"
    seconds = duration_ms / 1_000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m {seconds - minutes * 60:.0f}s"


def _line_count(text: str) -> int:
    return len(text.splitlines()) if text else 0
