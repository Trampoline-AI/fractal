# Configuration

Fractal uses a global TOML config for non-secret provider and model settings.
On first interactive run, if no global config exists, Fractal starts setup
automatically. Setup uses inline keyboard menus for provider and model
selection:

```bash
fractal
```

You can also run setup directly. Use Up/Down to move through highlighted
choices, Space to select, and Enter to confirm:

```bash
fractal config setup
fractal config status
fractal config show
```

Inside an interactive session, `/provider` re-runs provider setup, `/model`
switches models for the configured providers, and `/verbose` toggles trace
display.

## Providers

For API-key providers, setup defaults to **pasting the key** — Fractal stores it
locally (`~/.config/fractal/credentials.toml`, `0600`), never in config — with an
**environment variable** as the alternative. The last column is that env var's
default name:

| Provider | Credential | Default credential reference |
| --- | --- | --- |
| `openai-codex` | Official Codex CLI login | `codex login --device-auth` |
| `openai-api` | Paste (stored) or env var | `OPENAI_API_KEY` |
| `anthropic` | Paste (stored) or env var | `ANTHROPIC_API_KEY` |
| `gemini` | Paste (stored) or env var | `GEMINI_API_KEY` |
| `xai` | Paste (stored) or env var | `XAI_API_KEY` |
| `zai` | Paste (stored) or env var | `ZAI_API_KEY` |
| `deepseek` | Paste (stored) or env var | `DEEPSEEK_API_KEY` |
| `mistral` | Paste (stored) or env var | `MISTRAL_API_KEY` |
| `groq` | Paste (stored) or env var | `GROQ_API_KEY` |
| `openrouter` | Paste (stored) or env var | `OPENROUTER_API_KEY` |
| `ollama` | Local server, no credential | `http://localhost:11434` |
| `custom-openai-compatible` | Paste (stored) or env var, plus base URL | `CUSTOM_OPENAI_API_KEY` |

`openai-codex` requires the official `codex` CLI and an existing Codex login.
Fractal reads Codex CLI auth through PredictRLM's `dspy_codex_lm.CodexLM`
adapter and does not copy Codex OAuth tokens into Fractal config. Fractal only
offers the Codex `gpt-5.5` family during setup right now.

Setup model menus are curated starting points, not exhaustive provider
catalogs. Every provider except `openai-codex` also accepts a free-form model
id (the "Custom model..." entry in menus), so newly released models work
without a Fractal update. `ollama` talks to a local Ollama server and needs no
API key; setup asks for the server URL (default `http://localhost:11434`) and
queries `/api/tags` so models you have actually pulled are listed first,
marked "(installed)", falling back to static suggestions when the server is
not running.

## Credentials

For API-key providers, setup asks how to provide the key: paste it directly
(the default), or reference an environment variable. Pasted keys are stored in
`~/.config/fractal/credentials.toml` with `0600` permissions, next to the
config but never inside it; the config records only `auth_source = "stored"`.

If setup uses an environment variable that is currently unset, it still writes
the config (which never contains secrets) and prints the exact variable to
export; `fractal config status` verifies readiness afterwards.

Setup and `config status` also make one cheap authenticated request when
Fractal knows a suitable provider endpoint (a models-list endpoint, or
`/api/tags` for Ollama) so a typo'd or revoked key is caught immediately
instead of on the first agent turn. Providers without a known cheap check are
reported as "not checked for this provider". Pass `--offline` to skip the live
check; network failures during setup only warn and never discard a finished
setup.

## Non-interactive access

For scripts and quick edits there is non-interactive dotted-key access. `set`
parses TOML literals (`12`, `true`) and falls back to strings; values are
validated against the schema before anything is written, and raw secrets are
rejected. `--project` targets the workspace config instead of the global one:

```bash
fractal config get active_model
fractal config set active_model gpt-5.4-mini
fractal config set defaults.max_iterations 12
fractal config set active_model gpt-5.4 --project
fractal config unset active_sub_model
```

Setting `active_model` or `active_sub_model` warns when the model is not in
the provider's known catalog, and refuses ids the provider restricts.

To start over, `config reset` deletes the global config after confirmation
(`--yes` skips the prompt); add `--credentials` to also delete locally stored
API keys. Project configs are never touched by reset:

```bash
fractal config reset
fractal config reset --credentials --yes
```

## Config file, layering, and overrides

The default config path is `~/.config/fractal/config.toml`, or
`$XDG_CONFIG_HOME/fractal/config.toml` when `XDG_CONFIG_HOME` is set. The config
stores provider ids, model names, auth source metadata, API-key environment
variable names, and custom OpenAI-compatible base URLs. It must not store raw
API keys, OAuth tokens, or other secrets.

Config is resolved in layers: the global file, then per-workspace overrides in
`<workspace>/.fractal/config.toml` (same schema, every field optional), then
`FRACTAL_PROVIDER` / `FRACTAL_MODEL` / `FRACTAL_SUB_PROVIDER` /
`FRACTAL_SUB_MODEL` / `FRACTAL_MAX_ITERATIONS` / `FRACTAL_VERBOSE`
environment variables, with CLI
flags on top. A repo can pin its model without touching anyone's global
config, and CI can override via env. `fractal config show` lists which layers
contributed. Environment overrides apply only once some config file exists,
so first-run onboarding still triggers.

Beyond the active provider and model, the config supports:

```toml
# optional: a cheaper model for RLM sub-calls; chosen during setup and /model,
# defaults to the main model
active_sub_model = "gpt-5.4-mini"
# optional: run the sub-model on a different provider (its auth is collected
# during setup too); defaults to the main provider
active_sub_provider = "groq"

[defaults]            # optional run defaults, overridden by CLI flags
max_iterations = 30   # --max-iterations
verbose = false       # --verbose
```

The config can hold several provider profiles at once. Setup merges into the
existing `providers` table instead of replacing it, marks already-configured
providers in the menu, defaults to the active one, and offers to keep their
saved auth — so switching back to a configured provider is just two prompts
(provider, model), and `fractal config set active_provider <id>` switches
non-interactively. Switching providers clears `active_sub_model`; run
defaults are preserved.

Setup walks main provider → main model → sub-model provider (defaulting to
"same as main provider") → sub-model, then collects auth for each distinct
provider; `/model` changes only the two models within their providers.

For one-off runs or tests, `--lm` bypasses global config resolution:

```bash
fractal --lm openai/gpt-5.5 -p "summarize this repo"
```
