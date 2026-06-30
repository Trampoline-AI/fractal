from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest


def install_fake_dspy_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLM:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    import dspy

    monkeypatch.setattr(dspy, "LM", FakeLM)


GLOBAL_CONFIG = """
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.5"

[providers.openai-api]
auth_source = "env"
api_key_env = "OPENAI_API_KEY"
""".strip()


def write_global_config(config_home: Path, text: str = GLOBAL_CONFIG) -> Path:
    path = config_home / "fractal" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_project_config(workspace: Path, text: str) -> Path:
    path = workspace / ".fractal" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip(), encoding="utf-8")
    return path


def test_project_config_overrides_model_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.config import load_layered_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    write_global_config(tmp_path / "home")
    workspace = tmp_path / "repo"
    write_project_config(
        workspace,
        """
active_model = "gpt-5.4-mini"

[defaults]
max_iterations = 8
""",
    )

    result = load_layered_config(workspace=workspace)

    assert result.config is not None
    assert result.config.active_provider == "openai-api"
    assert result.config.active_model == "gpt-5.4-mini"
    assert result.config.defaults.max_iterations == 8
    assert result.project_path == workspace / ".fractal" / "config.toml"


def test_project_config_can_switch_provider_with_own_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.config import load_layered_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    write_global_config(tmp_path / "home")
    workspace = tmp_path / "repo"
    write_project_config(
        workspace,
        """
active_provider = "anthropic"
active_model = "claude-sonnet-4-6"

[providers.anthropic]
auth_source = "env"
api_key_env = "ANTHROPIC_API_KEY"
""",
    )

    result = load_layered_config(workspace=workspace)

    assert result.config is not None
    assert result.config.active_provider == "anthropic"
    # The global provider entry survives the merge.
    assert "openai-api" in result.config.providers


def test_env_overrides_beat_project_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.config import load_layered_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    write_global_config(tmp_path / "home")
    workspace = tmp_path / "repo"
    write_project_config(workspace, 'active_model = "gpt-5.4-mini"')
    monkeypatch.setenv("FRACTAL_MODEL", "gpt-5.4")
    monkeypatch.setenv("FRACTAL_MAX_ITERATIONS", "5")
    monkeypatch.setenv("FRACTAL_VERBOSE", "true")

    result = load_layered_config(workspace=workspace)

    assert result.config is not None
    assert result.config.active_model == "gpt-5.4"
    assert result.config.defaults.max_iterations == 5
    assert result.config.defaults.verbose is True
    assert set(result.env_overrides) == {
        "FRACTAL_MODEL",
        "FRACTAL_MAX_ITERATIONS",
        "FRACTAL_VERBOSE",
    }


def test_env_overrides_ignored_without_any_config_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.config import load_layered_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FRACTAL_MODEL", "gpt-5.4")

    result = load_layered_config(workspace=tmp_path / "repo")

    assert result.config is None


def test_invalid_env_override_reports_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.config import FractalConfigSchemaError, load_layered_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    write_global_config(tmp_path / "home")
    monkeypatch.setenv("FRACTAL_MAX_ITERATIONS", "lots")

    with pytest.raises(FractalConfigSchemaError, match="FRACTAL_MAX_ITERATIONS"):
        load_layered_config()


def test_project_config_rejects_raw_secrets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.config import FractalConfigSchemaError, load_layered_config

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    write_global_config(tmp_path / "home")
    workspace = tmp_path / "repo"
    write_project_config(
        workspace,
        """
[providers.openai-api]
api_key = "sk-leaked"
""",
    )

    with pytest.raises(FractalConfigSchemaError, match="api_key"):
        load_layered_config(workspace=workspace)


def test_resolve_runtime_lms_applies_project_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    install_fake_dspy_lm(monkeypatch)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    write_global_config(tmp_path / "home")
    workspace = tmp_path / "repo"
    write_project_config(workspace, 'active_model = "gpt-5.4-mini"')

    lm_config = cli.resolve_runtime_lms(
        SimpleNamespace(lm=None, sub_lm=None, workspace=workspace),
        stdin=StringIO(),
        stdout=StringIO(),
        stderr=StringIO(),
        auto_setup=False,
    )

    assert lm_config is not None
    assert lm_config.lm.kwargs == {
        "model": "openai/gpt-5.4-mini",
        "api_key": "sk-secret-value",
    }


def test_config_show_reports_layer_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal import cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "home"))
    write_global_config(tmp_path / "home")
    workspace = tmp_path / "repo"
    write_project_config(workspace, 'active_model = "gpt-5.4-mini"')
    monkeypatch.setenv("FRACTAL_VERBOSE", "1")
    args = cli.build_parser().parse_args(
        ["--workspace", str(workspace), "config", "show"]
    )
    stdout = StringIO()

    exit_code = cli.run_config_command(args, stdout=stdout, stderr=StringIO())

    assert exit_code == 0
    output = stdout.getvalue()
    assert "model: gpt-5.4-mini" in output
    assert f"project overrides: {workspace / '.fractal' / 'config.toml'}" in output
    assert "env overrides: FRACTAL_VERBOSE" in output
