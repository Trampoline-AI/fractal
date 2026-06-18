<br/>

<p align="center">
  <a href="https://fractal.trampoline.ai">
    <img src="https://raw.githubusercontent.com/Trampoline-AI/fractal/main/assets/logo-mark.png" alt="Fractal" width="132" height="132"/>
  </a>
</p>

<h1 align="center">fractal</h1>

<p align="center">
  <em>the recursive language model CLI agent</em>
</p>

<p align="center">
  A terminal agent that <strong>is</strong> an RLM. Powered by
  <a href="https://github.com/Trampoline-AI/predict-rlm">predict-rlm</a> —
  Trampoline's self-harnessed Recursive Language Model runtime.<br/>
  The easiest way to see an RLM in action on your own work.
</p>

<p align="center">
  <a href="https://github.com/Trampoline-AI/fractal/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/Trampoline-AI/fractal/tests.yml?label=Tests" alt="Tests"></a>
  <a href="https://pypi.org/project/fractal-rlm/"><img src="https://img.shields.io/pypi/v/fractal-rlm?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/fractal-rlm/"><img src="https://img.shields.io/pypi/pyversions/fractal-rlm" alt="Python"></a>
  <a href="https://github.com/Trampoline-AI/fractal/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Trampoline-AI/fractal?color=brightgreen" alt="License"></a>
  <a href="https://discord.gg/BAkd288sGN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?style=flat&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/Trampoline-AI/fractal"><img src="https://img.shields.io/github/stars/Trampoline-AI/fractal?cacheSeconds=3600" alt="GitHub stars"></a>
</p>

<p align="center">
  <a href="https://fractal.trampoline.ai"><strong>Website</strong></a> ·
  <a href="https://github.com/Trampoline-AI/predict-rlm"><strong>predict-rlm</strong></a> ·
  <a href="https://discord.gg/BAkd288sGN"><strong>Discord</strong></a>
</p>

<p align="center">
  <sub>crafted with ♥ in MTL · NYC · FLP · by <a href="https://www.trampoline.ai/">Trampoline AI</a></sub>
</p>

## Quick start

```bash
curl -LsSf https://fractal.trampoline.ai/install.sh | sh
```

This installs the `fractal` command (and [uv](https://docs.astral.sh/uv/) if you
don't have it). Make sure you meet the [Requirements](#requirements) below, then
drop into any project and run `fractal`.

## Requirements

- **Python 3.11+**.
- **[uv](https://docs.astral.sh/uv/)** to install and run Fractal.
- **Docker**, running. Every Fractal turn executes generated code inside a
  Docker Sandbox, so the Docker daemon must be up.
- **The `sbx` CLI, logged in.** Fractal uses predict-rlm's `sbx` (Docker
  Sandboxes) backend for code execution:

  ```bash
  brew install docker/tap/sbx
  sbx login
  ```

  If Docker is not running or `sbx` is not logged in, the first turn fails. You
  can verify the rest of your setup (provider, model, auth) ahead of time with
  `fractal config status`.
- **A model provider.** One of the providers in the
  [configuration table](#configuration), with its API key available (or
  `codex login` for `openai-codex`, or a local Ollama server). Setup walks you
  through this on first run.

## Installation

[Quick start](#quick-start) covers the one-line installer, which installs
[uv](https://docs.astral.sh/uv/) if needed, installs Fractal as an isolated
tool, and checks your Docker/`sbx` prerequisites. To pin a version, set
`FRACTAL_VERSION`.

If you already use uv or pipx, install the tool directly instead:

```bash
uv tool install fractal-rlm  # or: pipx install fractal-rlm
fractal --help
```

Both create an isolated environment and put `fractal` on your PATH, so it is
callable from any directory.

## Use it with your coding agent

Fractal shines as a tool your main agent (Claude Code, Cursor, …) defers to for
heavy, large-context jobs: it hands Fractal the work in
[headless mode](#headless--ci-use) and gets back a distilled answer. The bundled
**`fractal` skill** teaches an agent when and how to do this.

Install it in Claude Code, Codex, Cursor, or any compatible coding agent:

```bash
npx skills add https://github.com/Trampoline-AI/fractal/tree/main/.agents/skills/fractal
```

See [Headless / CI use](#headless--ci-use) for the full non-interactive
interface.

## What is Fractal?

Most agents call a model in a loop that humans hand-engineer — the control flow,
the context management, the tool routing. **Fractal's loop _is_ the model.**

It's a thin terminal UI over
[predict-rlm](https://github.com/Trampoline-AI/predict-rlm), Trampoline's
self-harnessed Recursive Language Model runtime. The model writes and runs its
own code, calls a sub-model when it needs to, and manages its own context as it
works — so capability scales with the underlying model instead of with harness
engineering, and
[without context rot](https://github.com/Trampoline-AI/predict-rlm). (It's an
implementation of the
[Recursive Language Models](https://arxiv.org/abs/2512.24601v1) work from MIT
CSAIL.)

Fractal adds exactly one thing on top: **session management** — multi-turn
conversation and history, which predict-rlm doesn't do on its own. That's the
whole product. Each turn is a single RLM call over your workspace, mounted into
a Docker sandbox so the model's own code and project commands run against the
real files.

It's an early, intentionally bare-bones proof of concept — released to see what
people build with it, and to be **the easiest way to get started with an RLM**
and actually understand how one works, by experimenting on your own tasks.

## How it works

Every Fractal turn runs fully inside a [Docker Sandbox](https://hub.docker.com/u/dockersbx)
(`sbx`) — an isolated container with no network access by default. Your workspace
is mounted directly into the sandbox via an SBX passthrough mount, so the agent
reads and edits your actual files in place; changes appear on the host
immediately, with no copy or sync step needed.

The agent recurses. predict-rlm spawns sub-LMs to work the shards of a task that
won't fit one context, then folds their results back up:

```
fractal› go through this 123 page contract, build a timeline, set reminders for the deadlines

RLM turn 1/30 (ok)
  reasoning: 123 pages won't fit one context — split it
  python:
    │ class DatedItem(BaseModel):
    │     date: datetime.date
    │     description: str
    │
    │ results = await asyncio.gather(*[
    │     predict("page: dspy.Image -> items: list[DatedItem]", page=render(page, dpi=80))
    │     for page in doc
    │ ])
        ↳ sub-lm 47/123 · page 47
          2 items · 2026-04-01 renewal notice · 2026-06-30 term end
          ↳ returning items to parent

RLM turn 2/30 (ok)
  reasoning: 31 items collected — sort them, then write the deliverable
  python:
    │ items = sorted((i for r in results for i in r.items), key=lambda i: i.date)
    │ write_file("timeline.md", to_markdown(items))
    │ for i in upcoming(items):
    │     add_reminder(i.date, i.description)

  Contract timeline · Acme MSA
  Across all 123 pages I found 31 dated items. Full timeline written to
  timeline.md; the 4 upcoming deadlines are on your calendar.
```

> This is a pseudo trace, to help you understand what goes on inside the RLM.

A single line can stand in for a million sub-calls — in direct contrast to
agents that must mechanically emit each sub-agent call one at a time. And every
peek, chunk, sub-call, and verification step is fully readable in the trace.

## Where it shines

Because the RLM reasons over context programmatically instead of stuffing
everything into one prompt, Fractal is strongest on **analysis- and
context-heavy work**: reading across a large or deep codebase, synthesizing an
answer from many files, audits, and open-ended investigation — anything where
the context is the hard part. Two ways to use it:

- **Directly**, as your own terminal agent — ask questions, edit code, run
  tasks.
- **As a tool other agents defer to** — your main agent (Claude Code, Cursor,
  etc.) can hand a heavy analysis or large-context job to Fractal in
  [headless mode](#headless--ci-use) and get back a distilled answer. The
  bundled [`fractal` skill](.agents/skills/fractal/SKILL.md)
  teaches an agent when and how to do this.

Fractal is not trying to replace your daily coding agent — more mature tools
exist for that. It's a window onto what a self-harnessed RLM can do.

## What you get

- **Powered by predict-rlm** — recursive and self-harnessed. The runtime is
  the agent; there's no orchestration to assemble.
- **Model-agnostic** — OpenAI, Anthropic, Gemini, Groq, Ollama, OpenRouter,
  or any OpenAI-compatible endpoint.
- **Sandboxed by default** — every turn runs in an isolated Docker sandbox.
  Point it at real work without flinching.
- **Headless & scriptable** — drive it from CI or another agent with
  `fractal -p "…"`.

## Why RLMs?

- [**Recursive Language Models**](https://arxiv.org/abs/2512.24601v1) — the MIT CSAIL paper introducing RLMs: self-harnessed models that write and run their own inference code.
- [**LongCoT: a benchmark worthy of an RLM's attention**](https://raw.works/longcot-a-benchmark-worthy-of-a-rlms-attention/) — why standard long-context benchmarks miss what RLMs are actually good at.
- [**RLMs on the AppWorld benchmark**](https://x.com/GabLesperance/status/2060754345247863075) — early results and observations from using an RLM on real tasks.

## First run

```bash
cd your-project
fractal                       # first run launches provider/model setup, then a session
```

On first interactive run with no global config, Fractal runs setup
automatically: pick a provider, a model, an optional cheaper sub-model, and how
to supply the API key. After setup you land in an interactive session in the
current directory. Type a request, and Fractal edits the workspace and reports
what it changed. Use `/help` to list slash commands and `/exit` to quit.

## Usage

```bash
fractal                       # interactive session in the current directory
fractal -p "fix the tests"    # one non-interactive turn
fractal --resume <session-id> # resume a stored workspace session
```

Interactive slash commands: `/help`, `/sessions`, `/resume <id>`, `/new`,
`/model`, `/provider`, `/usage`, `/verbose`, `/exit`. The header always shows
both the main model and the sub-model.

### Workspace and included directories

By default Fractal mounts the directory it was launched from as the workspace —
that's what the sandbox sees and what the agent reads and edits.

- **`--workspace DIR`** changes which directory is mounted as the workspace,
  so you can run Fractal against a project other than the current directory.
- **`--include DIR`** (repeatable) mounts an *additional* directory into the
  sandbox at its absolute path. Use it when the agent needs files that live
  outside the workspace. Common cases:
  - **Local path dependencies** — a sibling package or editable install your
    project points at by path; include it so the agent can read and run it.
  - **Git worktrees** — a worktree's `.git` lives in the main checkout, so
    include that directory to give Fractal access to the real git history.

### Command-line options

| Flag | Description |
| --- | --- |
| `--workspace DIR` | Workspace directory to edit; defaults to the current directory. |
| `--include DIR` | Additional directory to mount into the sandbox at its absolute path. Repeatable. |
| `-p`, `--prompt TEXT` | Run one turn non-interactively with `TEXT`; use `-` to read the prompt from stdin. |
| `--resume SESSION_ID` | Resume an existing workspace-local session by id. |
| `--max-iterations N` | Max RLM iterations per turn; defaults to the configured value or 30. |
| `--lm MODEL` | Override the configured main model for this run (bypasses config resolution). |
| `--sub-lm MODEL` | Override the configured sub-model for this run. |
| `--verbose` | Show generated code and model-visible output for each RLM iteration; enabled by default with `-p`. |
| `--quiet` | Suppress progress chatter (non-interactive runs). |
| `--debug` | Enable PredictRLM debug mode. |

Subcommands: `fractal config <show|status|setup|get|set|unset|reset>` manage
configuration (see [Configuration](#configuration)).

### Headless / CI use

`-p`/`--prompt` runs a single turn without the interactive UI — use it from
scripts, hooks, CI, or to hand Fractal heavy work from another agent:

```bash
fractal -p "fix the failing tests"          # one turn, prompt as an argument
git diff | fractal -p -                      # read the entire prompt from stdin
echo "summarize recent changes" | fractal -p "review this diff"  # prompt + stdin context
```

Headless runs show generated code and model-visible output for each RLM
iteration on stderr by default. Use `--quiet` for stdout-only scripts.

See [docs/headless.md](docs/headless.md) for the full output contract, exit codes, and CI patterns.

## Configuration

On first run, Fractal walks you through setup interactively — pick a provider,
model, and how to supply your API key. In an interactive session, use `/provider`
to change provider and `/model` to switch models.

Supported providers:

| Provider | Default API key env var |
| --- | --- |
| `openai-codex` | `codex login --device-auth` |
| `openai-api` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `xai` | `XAI_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `ollama` | local server, no key |
| `custom-openai-compatible` | `CUSTOM_OPENAI_API_KEY` |

See [docs/config.md](docs/config.md) for credentials, non-interactive access, environment variable overrides, and the full config schema.

## Development

To work on Fractal itself, clone the repository and use uv:

```bash
git clone git@github.com:Trampoline-AI/fractal.git
cd fractal
uv sync                # install dependencies
uv run fractal --help
uv run pytest          # 200+ tests
```

When running from a checkout, prefix commands with `uv run` (e.g. `uv run
fractal`); an installed tool just uses `fractal`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[CHANGELOG.md](CHANGELOG.md) for release notes.


Fractal is a fully open-source proof of concept we're putting out to see what
people build with it. It's early, and moving fast.
