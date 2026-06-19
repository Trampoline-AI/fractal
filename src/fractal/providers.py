from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol
from urllib.parse import urlparse

from .lm_types import RuntimeLM

OPENAI_CODEX = "openai-codex"
OPENAI_API = "openai-api"
ANTHROPIC = "anthropic"
GEMINI = "gemini"
XAI = "xai"
ZAI = "zai"
DEEPSEEK = "deepseek"
MISTRAL = "mistral"
GROQ = "groq"
OPENROUTER = "openrouter"
OLLAMA = "ollama"
CUSTOM_OPENAI_COMPATIBLE = "custom-openai-compatible"
ProviderAuthType = Literal["api_key_env", "codex_cli", "none"]
ProviderAuthSource = Literal["env", "stored", "codex-cli", "local"]
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class ProviderError(ValueError):
    """Base class for provider registry and LM factory errors."""


class UnknownProviderError(ProviderError):
    """Raised when config references a provider Fractal does not know."""


class ProviderConfigError(ProviderError):
    """Raised when a provider selection is incomplete or inconsistent."""


class UnsupportedProviderModelError(ProviderError):
    """Raised when a provider cannot use the requested model."""


class MissingProviderCredentialError(ProviderConfigError):
    """Raised when a provider's configured credential source is unavailable."""


class MissingCodexCliError(ProviderConfigError):
    """Raised when the official Codex CLI is required but unavailable."""


class MissingCodexAuthError(MissingProviderCredentialError):
    """Raised when official Codex CLI auth is missing or unusable."""


@dataclass(frozen=True)
class ProviderSelection:
    provider: str
    model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    auth_source: ProviderAuthSource | None = None


class ProviderBehavior(Protocol):
    def validate_shape(
        self,
        selection: ProviderSelection,
        definition: "ProviderDefinition",
    ) -> None: ...

    def check_readiness(
        self,
        selection: ProviderSelection,
        definition: "ProviderDefinition",
        *,
        env: Mapping[str, str] | None,
    ) -> None: ...

    def build_lm(
        self,
        selection: ProviderSelection,
        definition: "ProviderDefinition",
        *,
        env: Mapping[str, str] | None,
    ) -> RuntimeLM: ...


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    display_name: str
    auth_type: ProviderAuthType
    auth_source: ProviderAuthSource
    default_model: str
    behavior: ProviderBehavior
    model_options: tuple[str, ...] = ()
    default_api_key_env: str | None = None
    model_prefix: str | None = None
    supports_base_url: bool = False
    base_url_label: str | None = None
    default_base_url: str | None = None
    setup_messages: tuple[str, ...] = ()
    restricted_models: tuple[str, ...] = ()

    @property
    def allows_custom_model(self) -> bool:
        # model_options are suggestions, not a contract; only providers with
        # an explicit restricted set reject arbitrary model ids.
        return not self.restricted_models

    def validate_shape(
        self,
        selection: ProviderSelection,
    ) -> None:
        self.behavior.validate_shape(selection, self)

    def check_readiness(
        self,
        selection: ProviderSelection,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.behavior.check_readiness(selection, self, env=env)

    def build_lm(
        self,
        selection: ProviderSelection,
        *,
        env: Mapping[str, str] | None = None,
    ) -> RuntimeLM:
        return self.behavior.build_lm(selection, self, env=env)


@dataclass(frozen=True)
class ApiKeyStringLMRuntime:
    model: str
    # None means litellm resolves the key from the env var at call time; a
    # value means the key came from Fractal's credential store. repr is
    # suppressed so the secret never lands in logs or tracebacks.
    api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class CodexLMRuntime:
    model: str
    auth_path: Path


@dataclass(frozen=True)
class CustomOpenAIRuntime:
    model: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class OllamaRuntime:
    model: str
    base_url: str


class ApiKeyStringLMBehavior:
    def validate_shape(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
    ) -> None:
        _validate_auth_source(selection, definition)
        _selection_model(selection, definition)
        if selection.auth_source != "stored":
            _api_key_env_name(selection, definition)

    def check_readiness(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> None:
        self._runtime(selection, definition, env=env)

    def build_lm(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> RuntimeLM:
        runtime = self._runtime(selection, definition, env=env)
        if runtime.api_key is None:
            return runtime.model

        try:
            import dspy
        except ImportError as exc:
            raise ProviderConfigError(
                f"provider {definition.id!r} requires DSPy for stored API keys"
            ) from exc

        return dspy.LM(model=runtime.model, api_key=runtime.api_key)

    def _runtime(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> ApiKeyStringLMRuntime:
        self.validate_shape(selection, definition)
        api_key: str | None = None
        if selection.auth_source == "stored":
            api_key = _require_stored_api_key(selection, definition)
        else:
            _require_api_key_env(selection, definition, env=env)
        return ApiKeyStringLMRuntime(
            model=_normalize_model(_selection_model(selection, definition), definition),
            api_key=api_key,
        )


class CodexCliLMBehavior:
    def validate_shape(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
    ) -> None:
        _validate_auth_source(selection, definition)
        _selection_model(selection, definition)

    def check_readiness(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> None:
        self._runtime(selection, definition)

    def build_lm(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> RuntimeLM:
        runtime = self._runtime(selection, definition)
        try:
            from dspy_codex_lm import CodexLM
        except ImportError as exc:
            raise ProviderConfigError(
                "openai-codex requires PredictRLM's dspy_codex_lm integration"
            ) from exc

        return CodexLM(model=runtime.model, auth_path=runtime.auth_path)

    def _runtime(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
    ) -> CodexLMRuntime:
        self.validate_shape(selection, definition)
        return CodexLMRuntime(
            model=_resolve_codex_model(selection, definition),
            auth_path=_codex_cli_auth_path(selection),
        )


class CustomOpenAICompatibleBehavior:
    def validate_shape(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
    ) -> None:
        _validate_auth_source(selection, definition)
        _custom_openai_base_url(selection)
        if selection.auth_source != "stored" and not selection.api_key_env:
            raise ProviderConfigError(
                "custom OpenAI-compatible provider requires api_key_env"
            )

    def check_readiness(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> None:
        self._runtime(selection, definition, env=env)

    def build_lm(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> RuntimeLM:
        runtime = self._runtime(selection, definition, env=env)

        try:
            import dspy
        except ImportError as exc:
            raise ProviderConfigError(
                "custom OpenAI-compatible provider requires DSPy"
            ) from exc

        return dspy.LM(
            model=runtime.model,
            api_base=runtime.base_url,
            api_key=runtime.api_key,
        )

    def _runtime(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> CustomOpenAIRuntime:
        self.validate_shape(selection, definition)
        base_url = _custom_openai_base_url(selection)
        if selection.auth_source == "stored":
            api_key = _require_stored_api_key(selection, definition)
        else:
            env_name = _require_api_key_env(selection, definition, env=env)
            values = os.environ if env is None else env
            api_key = values[env_name]
        return CustomOpenAIRuntime(
            model=_normalize_model(_selection_model(selection, definition), definition),
            base_url=base_url,
            api_key=api_key,
        )


class OllamaLMBehavior:
    """Local Ollama server: no credential, optional non-default base URL."""

    def validate_shape(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
    ) -> None:
        _validate_auth_source(selection, definition)
        _selection_model(selection, definition)
        _optional_base_url(selection, definition)

    def check_readiness(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> None:
        self._runtime(selection, definition)

    def build_lm(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
        *,
        env: Mapping[str, str] | None,
    ) -> RuntimeLM:
        runtime = self._runtime(selection, definition)

        try:
            import dspy
        except ImportError as exc:
            raise ProviderConfigError("ollama provider requires DSPy") from exc

        # litellm's ollama route ignores credentials; an empty key keeps it
        # from picking up unrelated OPENAI_API_KEY-style env vars.
        return dspy.LM(
            model=runtime.model,
            api_base=runtime.base_url,
            api_key="",
        )

    def _runtime(
        self,
        selection: ProviderSelection,
        definition: ProviderDefinition,
    ) -> OllamaRuntime:
        self.validate_shape(selection, definition)
        return OllamaRuntime(
            model=_normalize_model(_selection_model(selection, definition), definition),
            base_url=_optional_base_url(selection, definition),
        )


_API_KEY_STRING_LM = ApiKeyStringLMBehavior()
_CODEX_CLI_LM = CodexCliLMBehavior()
_CUSTOM_OPENAI_COMPATIBLE_LM = CustomOpenAICompatibleBehavior()
_OLLAMA_LM = OllamaLMBehavior()


_PROVIDERS: dict[str, ProviderDefinition] = {
    OPENAI_CODEX: ProviderDefinition(
        id=OPENAI_CODEX,
        display_name="OpenAI Codex",
        auth_type="codex_cli",
        auth_source="codex-cli",
        default_model="gpt-5.5",
        behavior=_CODEX_CLI_LM,
        setup_messages=(
            "OpenAI Codex uses the official Codex CLI auth store.",
            "If needed, run `codex login --device-auth` before continuing.",
        ),
        restricted_models=("gpt-5.5",),
    ),
    OPENAI_API: ProviderDefinition(
        id=OPENAI_API,
        display_name="OpenAI API",
        auth_type="api_key_env",
        auth_source="env",
        default_model="gpt-5.5",
        behavior=_API_KEY_STRING_LM,
        model_options=("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"),
        default_api_key_env="OPENAI_API_KEY",
        model_prefix="openai",
    ),
    ANTHROPIC: ProviderDefinition(
        id=ANTHROPIC,
        display_name="Anthropic",
        auth_type="api_key_env",
        auth_source="env",
        default_model="claude-fable-5",
        behavior=_API_KEY_STRING_LM,
        model_options=(
            "claude-opus-4-8",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ),
        default_api_key_env="ANTHROPIC_API_KEY",
        model_prefix="anthropic",
    ),
    GEMINI: ProviderDefinition(
        id=GEMINI,
        display_name="Google Gemini",
        auth_type="api_key_env",
        auth_source="env",
        default_model="gemini-3.5-flash",
        behavior=_API_KEY_STRING_LM,
        model_options=("gemini-3.1-pro-preview", "gemini-3.1-flash-lite"),
        default_api_key_env="GEMINI_API_KEY",
        model_prefix="gemini",
    ),
    XAI: ProviderDefinition(
        id=XAI,
        display_name="xAI",
        auth_type="api_key_env",
        auth_source="env",
        default_model="grok-4.3",
        behavior=_API_KEY_STRING_LM,
        model_options=("grok-code-fast-1", "grok-4-1-fast-reasoning"),
        default_api_key_env="XAI_API_KEY",
        model_prefix="xai",
    ),
    ZAI: ProviderDefinition(
        id=ZAI,
        display_name="Z.AI",
        auth_type="api_key_env",
        auth_source="env",
        default_model="glm-5.2",
        behavior=_API_KEY_STRING_LM,
        model_options=(
            "glm-5.1",
            "glm-4.7",
            "glm-4.6",
            "glm-4.5",
            "glm-4.5-flash",
        ),
        default_api_key_env="ZAI_API_KEY",
        model_prefix="zai",
    ),
    DEEPSEEK: ProviderDefinition(
        id=DEEPSEEK,
        display_name="DeepSeek",
        auth_type="api_key_env",
        auth_source="env",
        default_model="deepseek-v4-pro",
        behavior=_API_KEY_STRING_LM,
        model_options=("deepseek-v4-flash",),
        default_api_key_env="DEEPSEEK_API_KEY",
        model_prefix="deepseek",
    ),
    MISTRAL: ProviderDefinition(
        id=MISTRAL,
        display_name="Mistral",
        auth_type="api_key_env",
        auth_source="env",
        default_model="devstral-2-latest",
        behavior=_API_KEY_STRING_LM,
        model_options=(
            "mistral-large-latest",
            "mistral-medium-latest",
            "codestral-latest",
        ),
        default_api_key_env="MISTRAL_API_KEY",
        model_prefix="mistral",
    ),
    GROQ: ProviderDefinition(
        id=GROQ,
        display_name="Groq",
        auth_type="api_key_env",
        auth_source="env",
        default_model="openai/gpt-oss-120b",
        behavior=_API_KEY_STRING_LM,
        model_options=(
            "moonshotai/kimi-k2-instruct-0905",
            "qwen/qwen3-32b",
            "llama-3.3-70b-versatile",
        ),
        default_api_key_env="GROQ_API_KEY",
        model_prefix="groq",
    ),
    OPENROUTER: ProviderDefinition(
        id=OPENROUTER,
        display_name="OpenRouter",
        auth_type="api_key_env",
        auth_source="env",
        default_model="openai/gpt-5.5",
        behavior=_API_KEY_STRING_LM,
        model_options=(
            "openai/gpt-5.4",
            "openai/gpt-5.4-mini",
            "anthropic/claude-fable-5",
            "anthropic/claude-opus-4.8",
            "anthropic/claude-sonnet-4.6",
            "anthropic/claude-haiku-4.5",
            "deepseek/deepseek-v4-pro",
            "deepseek/deepseek-v4-flash",
            "qwen/qwen3-coder",
            "poolside/laguna-m.1",
            "openrouter/pareto-code",
            "openai/gpt-oss-120b",
        ),
        default_api_key_env="OPENROUTER_API_KEY",
        model_prefix="openrouter",
    ),
    OLLAMA: ProviderDefinition(
        id=OLLAMA,
        display_name="Ollama (local)",
        auth_type="none",
        auth_source="local",
        default_model="qwen3-coder",
        behavior=_OLLAMA_LM,
        model_options=("gpt-oss:20b", "gpt-oss:120b", "devstral"),
        model_prefix="ollama_chat",
        supports_base_url=True,
        base_url_label="Ollama server URL",
        default_base_url=DEFAULT_OLLAMA_BASE_URL,
        setup_messages=(
            "Ollama runs models locally; no API key is needed.",
            "Make sure the model is pulled, e.g. `ollama pull qwen3-coder`.",
        ),
    ),
    CUSTOM_OPENAI_COMPATIBLE: ProviderDefinition(
        id=CUSTOM_OPENAI_COMPATIBLE,
        display_name="Custom OpenAI-compatible",
        auth_type="api_key_env",
        auth_source="env",
        default_model="gpt-oss-120b",
        behavior=_CUSTOM_OPENAI_COMPATIBLE_LM,
        model_options=("qwen3-coder",),
        default_api_key_env="CUSTOM_OPENAI_API_KEY",
        supports_base_url=True,
        base_url_label="OpenAI-compatible base URL",
        model_prefix="openai",
    ),
}


def provider_registry() -> Mapping[str, ProviderDefinition]:
    return MappingProxyType(_PROVIDERS)


def list_providers() -> list[ProviderDefinition]:
    return list(_PROVIDERS.values())


def model_choices(definition: ProviderDefinition) -> tuple[str, ...]:
    choices: list[str] = []
    for model in (definition.default_model, *definition.model_options):
        if model and model not in choices:
            choices.append(model)
    return tuple(choices)


def get_provider(provider_id: str) -> ProviderDefinition:
    try:
        return _PROVIDERS[provider_id]
    except KeyError as exc:
        raise UnknownProviderError(f"unknown provider {provider_id!r}") from exc


def resolve_lm(
    explicit_lm: RuntimeLM | None,
    selection: ProviderSelection | None,
    *,
    env: Mapping[str, str] | None = None,
) -> RuntimeLM | None:
    if explicit_lm is not None:
        return explicit_lm
    if selection is None:
        return None
    return build_lm(selection, env=env)


def build_lm(
    selection: ProviderSelection,
    *,
    env: Mapping[str, str] | None = None,
) -> RuntimeLM:
    return get_provider(selection.provider).build_lm(selection, env=env)


def resolve_api_key(
    selection: ProviderSelection,
    *,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return the provider's API key, or None for providers without one."""
    definition = get_provider(selection.provider)
    if definition.auth_type != "api_key_env":
        return None
    if selection.auth_source == "stored":
        return _require_stored_api_key(selection, definition)
    env_name = _require_api_key_env(selection, definition, env=env)
    values = os.environ if env is None else env
    return values[env_name]


def validate_provider_selection(
    selection: ProviderSelection,
) -> None:
    get_provider(selection.provider).validate_shape(selection)


def check_provider_readiness(
    selection: ProviderSelection,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    get_provider(selection.provider).check_readiness(selection, env=env)


def _selection_model(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> str:
    model = selection.model or definition.default_model
    if not model:
        raise ProviderConfigError(f"provider {definition.id!r} requires a model")
    if definition.restricted_models:
        slug = _strip_provider_prefix(model)
        if slug not in definition.restricted_models:
            supported = ", ".join(definition.restricted_models)
            raise UnsupportedProviderModelError(
                f"provider {definition.id!r} only supports these models: {supported}"
            )
    return model


def _validate_auth_source(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> None:
    allowed: set[ProviderAuthSource | None] = {None, definition.auth_source}
    if definition.auth_type == "api_key_env":
        allowed.add("stored")
    if selection.auth_source in allowed:
        return
    raise ProviderConfigError(
        f"provider {definition.id!r} requires auth_source={definition.auth_source!r}"
    )


def _normalize_model(model: str, definition: ProviderDefinition) -> str:
    prefix = definition.model_prefix
    if prefix is None or model.startswith(f"{prefix}/"):
        return model
    return f"{prefix}/{model}"


def _strip_provider_prefix(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _api_key_env_name(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> str:
    env_name = selection.api_key_env or definition.default_api_key_env
    if not env_name:
        raise ProviderConfigError(
            f"provider {definition.id!r} requires an API key env var name"
        )
    return env_name


def _require_stored_api_key(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> str:
    from .credentials import get_stored_credential

    api_key = get_stored_credential(selection.provider)
    if not api_key:
        raise MissingProviderCredentialError(
            f"provider {definition.id!r} has no stored API key. "
            "Run `fractal config setup` to store one."
        )
    return api_key


def _require_api_key_env(
    selection: ProviderSelection,
    definition: ProviderDefinition,
    *,
    env: Mapping[str, str] | None,
) -> str:
    env_name = _api_key_env_name(selection, definition)
    values = os.environ if env is None else env
    if not values.get(env_name):
        raise MissingProviderCredentialError(
            f"provider {definition.id!r} requires environment variable "
            f"{env_name} to be set"
        )
    return env_name


def _resolve_codex_model(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> str:
    try:
        from dspy_codex_lm.cli import CodexLMUnsupportedModelError, resolve_codex_model
    except ImportError as exc:
        raise ProviderConfigError(
            "openai-codex requires PredictRLM's dspy_codex_lm integration"
        ) from exc

    model = _selection_model(selection, definition)
    try:
        return resolve_codex_model(model)
    except CodexLMUnsupportedModelError as exc:
        raise UnsupportedProviderModelError(str(exc)) from exc


def _codex_cli_auth_path(selection: ProviderSelection) -> Path:
    if shutil.which("codex") is None:
        raise MissingCodexCliError(
            "openai-codex requires the official `codex` CLI. "
            "Install it, then run `codex login --device-auth`."
        )
    try:
        from dspy_codex_lm.auth import codex_auth_path, load_codex_auth
    except ImportError as exc:
        raise ProviderConfigError(
            "openai-codex requires PredictRLM's dspy_codex_lm integration"
        ) from exc

    auth_path = codex_auth_path()
    try:
        load_codex_auth(auth_path)
    except FileNotFoundError as exc:
        raise MissingCodexAuthError(
            f"openai-codex could not find Codex CLI auth at {auth_path}. "
            "Run `codex login --device-auth`."
        ) from exc
    except Exception as exc:
        raise MissingCodexAuthError(
            f"openai-codex found unusable Codex CLI auth at {auth_path}. "
            "Run `codex login --device-auth` again."
        ) from exc
    return auth_path


def _optional_base_url(
    selection: ProviderSelection,
    definition: ProviderDefinition,
) -> str:
    base_url = (selection.base_url or definition.default_base_url or "").strip()
    if not base_url:
        raise ProviderConfigError(f"provider {definition.id!r} requires base_url")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProviderConfigError(
            f"provider {definition.id!r} requires base_url to be an HTTP(S) URL"
        )
    return base_url


def _custom_openai_base_url(
    selection: ProviderSelection,
) -> str:
    if not selection.model:
        raise ProviderConfigError("custom OpenAI-compatible provider requires model")
    if not selection.base_url:
        raise ProviderConfigError("custom OpenAI-compatible provider requires base_url")
    base_url = selection.base_url.strip()
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProviderConfigError(
            "custom OpenAI-compatible provider requires base_url to be an "
            "HTTP(S) URL"
        )
    return base_url
