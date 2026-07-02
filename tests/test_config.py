from __future__ import annotations

import os
import stat
import tomllib
from pathlib import Path

import pytest


def valid_config_text() -> str:
    return """
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.1"

[providers.openai-api]
auth_source = "env"
api_key_env = "OPENAI_API_KEY"

[providers.openai-codex]
auth_source = "codex-cli"
""".strip()


def test_default_config_path_uses_xdg_config_home(tmp_path: Path) -> None:
    from fractal.config import default_config_path

    path = default_config_path(env={"XDG_CONFIG_HOME": str(tmp_path)})

    assert path == tmp_path / "fractal" / "config.toml"


def test_default_config_path_uses_home_when_xdg_is_unset(tmp_path: Path) -> None:
    from fractal.config import default_config_path

    path = default_config_path(env={}, home=tmp_path)

    assert path == tmp_path / ".config" / "fractal" / "config.toml"


def test_load_config_distinguishes_missing_config(tmp_path: Path) -> None:
    from fractal.config import load_config

    result = load_config(tmp_path / "missing.toml")

    assert result.missing is True
    assert result.loaded is False
    assert result.config is None


def test_load_config_rejects_invalid_toml(tmp_path: Path) -> None:
    from fractal.config import FractalConfigParseError, load_config

    path = tmp_path / "config.toml"
    path.write_text("active_provider = [", encoding="utf-8")

    with pytest.raises(FractalConfigParseError) as exc_info:
        load_config(path)

    assert str(path) in str(exc_info.value)


def test_load_config_rejects_schema_errors(tmp_path: Path) -> None:
    from fractal.config import FractalConfigSchemaError, load_config

    path = tmp_path / "config.toml"
    path.write_text(
        """
schema_version = 2
active_provider = "openai-api"
active_model = "gpt-5.1"

[providers.openai-api]
auth_source = "env"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(FractalConfigSchemaError) as exc_info:
        load_config(path)

    assert "schema_version" in str(exc_info.value)


def test_load_config_requires_active_provider_entry(tmp_path: Path) -> None:
    from fractal.config import FractalConfigSchemaError, load_config

    path = tmp_path / "config.toml"
    path.write_text(
        """
schema_version = 1
active_provider = "anthropic"
active_model = "claude-opus-4.1"

[providers.openai-api]
auth_source = "env"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(FractalConfigSchemaError, match="active_provider"):
        load_config(path)


def test_valid_config_loads_into_typed_objects(tmp_path: Path) -> None:
    from fractal.config import load_config, resolve_effective_config

    path = tmp_path / "config.toml"
    path.write_text(valid_config_text(), encoding="utf-8")

    result = load_config(path)

    assert result.loaded is True
    assert result.config is not None
    assert result.config.active_provider == "openai-api"
    assert result.config.providers["openai-api"].api_key_env == "OPENAI_API_KEY"
    effective = resolve_effective_config(result.config, path=path)
    assert effective.provider == "openai-api"
    assert effective.model == "gpt-5.1"
    assert effective.config_path == path


def test_write_config_round_trips_toml(tmp_path: Path) -> None:
    from fractal.config import FractalConfig, ProviderConfig, load_config, write_config

    config = FractalConfig(
        active_provider="custom-openai-compatible",
        active_model="provider/model",
        providers={
            "custom-openai-compatible": ProviderConfig(
                auth_source="env",
                api_key_env="CUSTOM_API_KEY",
                base_url="https://example.test/v1",
            )
        },
    )
    path = tmp_path / "fractal" / "config.toml"

    written_path = write_config(config, path)
    loaded = load_config(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    assert written_path == path
    assert loaded.config == config
    assert raw["schema_version"] == 1
    assert raw["providers"]["custom-openai-compatible"]["api_key_env"] == (
        "CUSTOM_API_KEY"
    )


def test_write_config_uses_restrictive_permissions_where_supported(
    tmp_path: Path,
) -> None:
    from fractal.config import FractalConfig, ProviderConfig, write_config

    config = FractalConfig(
        active_provider="openai-api",
        active_model="gpt-5.1",
        providers={
            "openai-api": ProviderConfig(
                auth_source="env",
                api_key_env="OPENAI_API_KEY",
            )
        },
    )
    path = tmp_path / "config-dir" / "config.toml"

    write_config(config, path)

    if os.name != "posix":
        return
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_render_config_redacts_credential_references_and_env_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fractal.config import load_config, render_config

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    path = tmp_path / "config.toml"
    path.write_text(valid_config_text(), encoding="utf-8")
    result = load_config(path)
    assert result.config is not None

    rendered = render_config(result.config, path=path)

    assert "provider: openai-api" in rendered
    assert "model: gpt-5.1" in rendered
    assert "api_key_env: <redacted>" in rendered
    assert "OPENAI_API_KEY" not in rendered
    assert "sk-secret-value" not in rendered


def test_config_rejects_raw_secret_fields(tmp_path: Path) -> None:
    from fractal.config import FractalConfigSchemaError, load_config

    path = tmp_path / "config.toml"
    path.write_text(
        """
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.1"

[providers.openai-api]
auth_source = "env"
api_key = "sk-secret-value"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(FractalConfigSchemaError, match="not allowed"):
        load_config(path)


def test_api_key_env_must_be_env_var_name(tmp_path: Path) -> None:
    from fractal.config import FractalConfigSchemaError, load_config

    path = tmp_path / "config.toml"
    path.write_text(
        """
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.1"

[providers.openai-api]
auth_source = "env"
api_key_env = "sk-secret-value"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(FractalConfigSchemaError, match="api_key_env"):
        load_config(path)


def test_config_round_trips_sub_model_and_defaults(tmp_path: Path) -> None:
    from fractal.config import FractalConfig, ProviderConfig, load_config, write_config

    path = tmp_path / "config.toml"
    config = FractalConfig(
        active_provider="openai-api",
        active_model="gpt-5.5",
        active_sub_model="gpt-5.4-mini",
        providers={
            "openai-api": ProviderConfig(
                auth_source="env",
                api_key_env="OPENAI_API_KEY",
            )
        },
        defaults={"max_iterations": 12, "verbose": True},
    )

    write_config(config, path)
    loaded = load_config(path).config

    assert loaded is not None
    assert loaded.active_sub_model == "gpt-5.4-mini"
    assert loaded.defaults.max_iterations == 12
    assert loaded.defaults.verbose is True


def test_config_omits_empty_defaults_table(tmp_path: Path) -> None:
    from fractal.config import FractalConfig, ProviderConfig, write_config

    path = tmp_path / "config.toml"
    config = FractalConfig(
        active_provider="openai-api",
        active_model="gpt-5.5",
        providers={
            "openai-api": ProviderConfig(
                auth_source="env",
                api_key_env="OPENAI_API_KEY",
            )
        },
    )

    write_config(config, path)

    assert "defaults" not in path.read_text(encoding="utf-8")


def test_config_rejects_non_positive_max_iterations(tmp_path: Path) -> None:
    from fractal.config import FractalConfigSchemaError, load_config

    path = tmp_path / "config.toml"
    path.write_text(
        """
schema_version = 1
active_provider = "openai-api"
active_model = "gpt-5.5"

[providers.openai-api]
auth_source = "env"
api_key_env = "OPENAI_API_KEY"

[defaults]
max_iterations = 0
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(FractalConfigSchemaError, match="max_iterations"):
        load_config(path)



def test_render_config_shows_effective_num_retries_default(tmp_path: Path) -> None:
    from fractal.config import load_config, render_config

    path = tmp_path / "config.toml"
    path.write_text(valid_config_text(), encoding="utf-8")
    result = load_config(path)
    assert result.config is not None

    rendered = render_config(result.config, path=path)

    assert "defaults.num_retries: 3 (default)" in rendered


def test_config_accepts_num_retries_default(tmp_path: Path) -> None:
    from fractal.config import load_config

    path = tmp_path / "config.toml"
    path.write_text(
        valid_config_text()
        + """

[defaults]
num_retries = 5
""",
        encoding="utf-8",
    )

    result = load_config(path)

    assert result.config is not None
    assert result.config.defaults.num_retries == 5
