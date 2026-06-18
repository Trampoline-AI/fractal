# Headless / CI use

`-p`/`--prompt` runs a single turn without the interactive UI, which is the
mode to use from scripts, hooks, and CI — and how another agent hands Fractal
the heavy lifting:

```bash
fractal -p "fix the failing tests"          # one turn, prompt as an argument
git diff | fractal -p -                      # read the entire prompt from stdin
echo "summarize recent changes" | fractal -p "review this diff"  # prompt + stdin context
```

## Output contract

| Channel | Content |
| --- | --- |
| stdout | Final response text only |
| stderr | Banner, progress, verbose iteration detail, changed files, usage/cost, completion status |

- Headless runs show generated code and model-visible output for each RLM
  iteration on stderr by default.
- Add `--quiet` to silence everything but the final stdout response.
- Add `--json` for a machine-readable result object on stdout (`session_id`,
  `status`, `response`, `changed_files`, `usage`, `error`); pair with
  `--quiet` for stdout-only JSON.
- An empty prompt is a no-op: Fractal exits `0` without making a model call.
- Stdin input is capped at 10 MiB.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Turn completed successfully |
| `1` | Error (bad input, setup or runtime failure) |
| `2` | Hit `--max-iterations` before completing; best-effort response still on stdout |
| `130` | Interrupted (Ctrl-C / SIGINT) |

## Prerequisites

- A provider must already be configured. Headless runs do **not** trigger
  interactive setup when stdin is not a TTY. Configure first with
  `fractal config setup`, or pin a model inline with `--lm`. Environment
  variables (`FRACTAL_PROVIDER`, `FRACTAL_MODEL`, …) are convenient for CI —
  see [config.md](config.md).
- Docker must be running and `sbx` logged in on the runner, exactly as for
  interactive use.

## Usage stats

After each turn Fractal writes to stderr: iterations, wall time, tokens in/out,
current RLM context size, billed cost, and changed files. Because the RLM loop
re-summarizes between turns, "context" is the prompt size of the latest
main-LM call rather than a cumulative count. `/usage` reports session totals,
which persist in `.fractal/sessions/<session-id>.json` and survive `--resume`.

## CI patterns

```bash
# Capture response, keep diagnostics visible
out=$(fractal -p "audit this codebase for security issues" 2>fractal.log)

# Structured output — parse with jq
result=$(fractal -p "summarize changes" --json --quiet)
echo "$result" | jq '.response'

# Resume a session across CI steps (pass SESSION_ID as an env var)
fractal -p "run the tests and fix any failures" --resume "$SESSION_ID"
```
