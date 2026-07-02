from __future__ import annotations

import os
import re
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

CONFIG_SCHEMA_VERSION = 1
CONFIG_DIR_NAME = "fractal"
CONFIG_FILE_NAME = "config.toml"
PROJECT_CONFIG_DIR_NAME = ".fractal"
ENV_PROVIDER = "FRACTAL_PROVIDER"
ENV_MODEL = "FRACTAL_MODEL"
ENV_SUB_PROVIDER = "FRACTAL_SUB_PROVIDER"
ENV_SUB_MODEL = "FRACTAL_SUB_MODEL"
ENV_MAX_ITERATIONS = "FRACTAL_MAX_ITERATIONS"
ENV_VERBOSE = "FRACTAL_VERBOSE"
ENV_LM_NUM_RETRIES = "FRACTAL_LM_NUM_RETRIES"
DEFAULT_LM_NUM_RETRIES = 3
REDACTED = "<redacted>"
_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RAW_SECRET_FIELD_NAMES = frozenset({
    "api_key",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "credential",
    "credentials",
})


class FractalConfigError(Exception):
    """Base class for global Fractal config errors."""

    def __init__(self, path: str | Path, message: str) -> None:
        self.path = Path(path)
        self.message = message
        super().__init__(f"{self.path}: {message}")


class FractalConfigParseError(FractalConfigError):
    """Raised when the config file is not valid TOML."""


class FractalConfigSchemaError(FractalConfigError):
    """Raised when valid TOML does not match Fractal's config schema."""


class ProviderConfig(BaseModel):
    """Non-secret provider settings stored in global Fractal config."""

    model_config = ConfigDict(extra="forbid")

    auth_source: Literal["env", "stored", "codex-cli", "local"] | None = None
    api_key_env: str | None = None
    base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_raw_secret_fields(cls, value: object) -> object:
        _reject_raw_secret_fields(value)
        return value

    @model_validator(mode="after")
    def validate_non_secret_fields(self) -> "ProviderConfig":
        if self.api_key_env is not None and not _ENV_VAR_NAME.fullmatch(
            self.api_key_env
        ):
            raise ValueError("api_key_env must be an environment variable name.")
        if self.base_url is not None and not self.base_url.strip():
            raise ValueError("base_url must not be blank when provided.")
        return self


class DefaultsConfig(BaseModel):
    """Optional run defaults applied when CLI flags are not passed."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int | None = Field(default=None, ge=1)
    verbose: bool | None = None
    num_retries: int | None = Field(default=None, ge=0)


class FractalConfig(BaseModel):
    """Versioned global Fractal config loaded from TOML."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = CONFIG_SCHEMA_VERSION
    active_provider: str = Field(min_length=1)
    active_model: str = Field(min_length=1)
    # Sub-LM provider; None means RLM sub-calls use the active provider.
    active_sub_provider: str | None = Field(default=None, min_length=1)
    active_sub_model: str | None = Field(default=None, min_length=1)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_raw_secret_fields(cls, value: object) -> object:
        _reject_raw_secret_fields(value)
        return value

    @model_validator(mode="after")
    def validate_active_provider(self) -> "FractalConfig":
        invalid_provider_ids = [
            provider_id
            for provider_id in self.providers
            if not _PROVIDER_ID.fullmatch(provider_id)
        ]
        if invalid_provider_ids:
            raise ValueError(
                "provider ids must contain only letters, numbers, dots, "
                "underscores, or hyphens."
            )
        if self.active_provider not in self.providers:
            raise ValueError("active_provider must reference a configured provider.")
        if (
            self.active_sub_provider is not None
            and self.active_sub_provider not in self.providers
        ):
            raise ValueError(
                "active_sub_provider must reference a configured provider."
            )
        return self


class ProjectFractalConfig(BaseModel):
    """Optional per-workspace overrides loaded from .fractal/config.toml."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = CONFIG_SCHEMA_VERSION
    active_provider: str | None = Field(default=None, min_length=1)
    active_model: str | None = Field(default=None, min_length=1)
    active_sub_provider: str | None = Field(default=None, min_length=1)
    active_sub_model: str | None = Field(default=None, min_length=1)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)

    @model_validator(mode="before")
    @classmethod
    def reject_raw_secret_fields(cls, value: object) -> object:
        _reject_raw_secret_fields(value)
        return value


class EffectiveFractalConfig(BaseModel):
    """Resolved config object consumed by future runtime and onboarding code."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = CONFIG_SCHEMA_VERSION
    provider: str
    model: str
    sub_provider: str | None = None
    sub_model: str | None = None
    provider_config: ProviderConfig
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    config_path: Path | None = None


@dataclass(frozen=True)
class ConfigLoadResult:
    path: Path
    config: FractalConfig | None

    @property
    def missing(self) -> bool:
        return self.config is None

    @property
    def loaded(self) -> bool:
        return self.config is not None


def default_config_path(
    *,
    env: dict[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    environment = os.environ if env is None else env
    xdg_config_home = environment.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home).expanduser() / CONFIG_DIR_NAME / CONFIG_FILE_NAME
    home_path = Path.home() if home is None else Path(home).expanduser()
    return home_path / ".config" / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def load_config(path: str | Path | None = None) -> ConfigLoadResult:
    config_path = Path(path) if path is not None else default_config_path()
    if not config_path.exists():
        return ConfigLoadResult(path=config_path, config=None)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FractalConfigError(config_path, f"could not read config: {exc}") from exc
    try:
        data = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise FractalConfigParseError(config_path, str(exc)) from exc
    try:
        config = FractalConfig.model_validate(data)
    except (ValidationError, ValueError) as exc:
        raise FractalConfigSchemaError(config_path, str(exc)) from exc
    return ConfigLoadResult(path=config_path, config=config)


@dataclass(frozen=True)
class LayeredConfigResult:
    """Global config merged with project-file and environment overrides."""

    path: Path
    config: FractalConfig | None
    project_path: Path | None = None
    env_overrides: tuple[str, ...] = ()

    @property
    def missing(self) -> bool:
        return self.config is None

    @property
    def loaded(self) -> bool:
        return self.config is not None


def project_config_path(workspace: str | Path) -> Path:
    return Path(workspace) / PROJECT_CONFIG_DIR_NAME / CONFIG_FILE_NAME


def load_project_config(workspace: str | Path) -> ProjectFractalConfig | None:
    config_path = project_config_path(workspace)
    if not config_path.exists():
        return None
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FractalConfigError(config_path, f"could not read config: {exc}") from exc
    try:
        data = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise FractalConfigParseError(config_path, str(exc)) from exc
    try:
        return ProjectFractalConfig.model_validate(data)
    except (ValidationError, ValueError) as exc:
        raise FractalConfigSchemaError(config_path, str(exc)) from exc


def load_layered_config(
    *,
    path: str | Path | None = None,
    workspace: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> LayeredConfigResult:
    """Resolve config precedence: global file < project file < FRACTAL_* env.

    CLI flags sit above all of these but are applied by the CLI itself.
    Environment overrides only apply when some config file exists; with no
    files at all the result is "not configured" so onboarding still runs.
    """
    environment = os.environ if env is None else env
    global_result = load_config(path)
    project_config = load_project_config(workspace) if workspace is not None else None

    if global_result.config is None and project_config is None:
        return LayeredConfigResult(path=global_result.path, config=None)

    if global_result.config is not None:
        data = global_result.config.model_dump(mode="python", exclude_none=True)
    else:
        data = {"schema_version": CONFIG_SCHEMA_VERSION, "providers": {}}

    if project_config is not None:
        overlay = project_config.model_dump(mode="python", exclude_none=True)
        for key in (
            "active_provider",
            "active_model",
            "active_sub_provider",
            "active_sub_model",
        ):
            if key in overlay:
                data[key] = overlay[key]
        for provider_id, provider_data in overlay.get("providers", {}).items():
            data.setdefault("providers", {})[provider_id] = provider_data
        for key, value in overlay.get("defaults", {}).items():
            data.setdefault("defaults", {})[key] = value

    env_overrides: list[str] = []
    error_path = (
        project_config_path(workspace)
        if project_config is not None and workspace is not None
        else global_result.path
    )
    for env_name, key in (
        (ENV_PROVIDER, "active_provider"),
        (ENV_MODEL, "active_model"),
        (ENV_SUB_PROVIDER, "active_sub_provider"),
        (ENV_SUB_MODEL, "active_sub_model"),
    ):
        value = environment.get(env_name)
        if value:
            data[key] = value
            env_overrides.append(env_name)
    max_iterations = environment.get(ENV_MAX_ITERATIONS)
    if max_iterations:
        try:
            data.setdefault("defaults", {})["max_iterations"] = int(max_iterations)
        except ValueError as exc:
            raise FractalConfigSchemaError(
                error_path, f"{ENV_MAX_ITERATIONS} must be an integer"
            ) from exc
        env_overrides.append(ENV_MAX_ITERATIONS)
    verbose = environment.get(ENV_VERBOSE)
    if verbose:
        data.setdefault("defaults", {})["verbose"] = _parse_env_bool(
            verbose, name=ENV_VERBOSE, path=error_path
        )
        env_overrides.append(ENV_VERBOSE)
    num_retries = environment.get(ENV_LM_NUM_RETRIES)
    if num_retries:
        try:
            data.setdefault("defaults", {})["num_retries"] = int(num_retries)
        except ValueError as exc:
            raise FractalConfigSchemaError(
                error_path, f"{ENV_LM_NUM_RETRIES} must be an integer"
            ) from exc
        env_overrides.append(ENV_LM_NUM_RETRIES)

    try:
        merged = FractalConfig.model_validate(data)
    except (ValidationError, ValueError) as exc:
        raise FractalConfigSchemaError(
            error_path,
            f"invalid config after merging global/project/env layers: {exc}",
        ) from exc
    return LayeredConfigResult(
        path=global_result.path,
        config=merged,
        project_path=(
            project_config_path(workspace)
            if project_config is not None and workspace is not None
            else None
        ),
        env_overrides=tuple(env_overrides),
    )


def _parse_env_bool(value: str, *, name: str, path: Path) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise FractalConfigSchemaError(path, f"{name} must be a boolean (true/false)")


def effective_lm_num_retries(defaults: DefaultsConfig | None = None) -> int:
    if defaults is not None and defaults.num_retries is not None:
        return defaults.num_retries
    return DEFAULT_LM_NUM_RETRIES


def resolve_effective_config(
    config: FractalConfig,
    *,
    path: str | Path | None = None,
) -> EffectiveFractalConfig:
    return EffectiveFractalConfig(
        provider=config.active_provider,
        model=config.active_model,
        sub_provider=config.active_sub_provider,
        sub_model=config.active_sub_model,
        provider_config=config.providers[config.active_provider],
        defaults=config.defaults,
        config_path=Path(path) if path is not None else None,
    )


def write_config(config: FractalConfig, path: str | Path | None = None) -> Path:
    config_path = Path(path) if path is not None else default_config_path()
    validated = FractalConfig.model_validate(config.model_dump(mode="python"))
    payload = validated.model_dump(mode="python", exclude_none=True)
    if not payload.get("defaults"):
        payload.pop("defaults", None)
    return _write_toml_file(config_path, payload)


def write_project_config(
    config: ProjectFractalConfig,
    workspace: str | Path,
) -> Path:
    config_path = project_config_path(workspace)
    validated = ProjectFractalConfig.model_validate(config.model_dump(mode="python"))
    payload = validated.model_dump(mode="python", exclude_none=True)
    for table in ("providers", "defaults"):
        if not payload.get(table):
            payload.pop(table, None)
    return _write_toml_file(config_path, payload)


def _write_toml_file(config_path: Path, payload: dict[str, object]) -> Path:
    _ensure_config_dir(config_path.parent)
    try:
        import tomli_w

        toml_text = tomli_w.dumps(payload)
        tmp_path = _write_temp_config(config_path.parent, config_path.name, toml_text)
        try:
            os.replace(tmp_path, config_path)
            _chmod_posix(config_path, 0o600)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    except OSError as exc:
        raise FractalConfigError(config_path, f"could not write config: {exc}") from exc
    return config_path


def render_effective_config(config: EffectiveFractalConfig) -> str:
    provider_config = config.provider_config
    lines = ["Fractal config"]
    if config.config_path is not None:
        lines.append(f"path: {config.config_path}")
    lines.extend([
        f"schema_version: {config.schema_version}",
        f"provider: {config.provider}",
        f"model: {config.model}",
    ])
    if config.sub_provider is not None:
        lines.append(f"sub_provider: {config.sub_provider}")
    if config.sub_model is not None:
        lines.append(f"sub_model: {config.sub_model}")
    if provider_config.auth_source is not None:
        lines.append(f"auth_source: {provider_config.auth_source}")
    if provider_config.auth_source == "stored":
        lines.append("api_key: stored in local credentials file")
    if provider_config.api_key_env is not None:
        lines.append(f"api_key_env: {REDACTED}")
    if provider_config.base_url is not None:
        lines.append(f"base_url: {provider_config.base_url}")
    if config.defaults.max_iterations is not None:
        lines.append(f"defaults.max_iterations: {config.defaults.max_iterations}")
    if config.defaults.verbose is not None:
        lines.append(f"defaults.verbose: {config.defaults.verbose}")
    retry_value = effective_lm_num_retries(config.defaults)
    if config.defaults.num_retries is None:
        lines.append(f"defaults.num_retries: {retry_value} (default)")
    else:
        lines.append(f"defaults.num_retries: {retry_value}")
    return "\n".join(lines)


def render_config(config: FractalConfig, *, path: str | Path | None = None) -> str:
    return render_effective_config(resolve_effective_config(config, path=path))


def _ensure_config_dir(path: Path) -> None:
    created = not path.exists()
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if created:
        _chmod_posix(path, 0o700)


def _write_temp_config(directory: Path, filename: str, toml_text: str) -> Path:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{filename}.",
        suffix=".tmp",
        dir=directory,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(toml_text)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        _chmod_posix(tmp_path, 0o600)
    except BaseException:
        try:
            tmp_path.unlink()
        finally:
            raise
    return tmp_path


def _chmod_posix(path: Path, mode: int) -> None:
    if os.name == "posix":
        path.chmod(mode)


def _reject_raw_secret_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str) and key in _RAW_SECRET_FIELD_NAMES:
                raise ValueError(
                    f"{key!r} is not allowed in Fractal config; store only "
                    "non-secret credential references."
                )
            _reject_raw_secret_fields(nested)
    elif isinstance(value, list):
        for item in value:
            _reject_raw_secret_fields(item)
