from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


def install_fake_codex_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    auth_path: object = "/tmp/codex-auth.json",
    load_auth_error: Exception | None = None,
) -> dict[str, object]:
    calls: dict[str, object] = {}

    class FakeCodexLM:
        def __init__(self, *, model: str, auth_path: object | None = None) -> None:
            calls["model"] = model
            calls["auth_path"] = auth_path

    class FakeUnsupportedModelError(RuntimeError):
        pass

    def resolve_codex_model(model: str) -> str:
        calls["requested"] = model
        if model == "gpt-4o":
            raise FakeUnsupportedModelError(f"cannot route {model!r}")
        return "resolved-codex-model"

    def load_codex_auth(path: object) -> tuple[str, str]:
        calls["load_auth_path"] = path
        if load_auth_error is not None:
            raise load_auth_error
        return ("secret-token", "acct-123")

    codex_module = ModuleType("dspy_codex_lm")
    codex_module.CodexLM = FakeCodexLM

    cli_module = ModuleType("dspy_codex_lm.cli")
    cli_module.CodexLMUnsupportedModelError = FakeUnsupportedModelError
    cli_module.resolve_codex_model = resolve_codex_model

    auth_module = ModuleType("dspy_codex_lm.auth")
    auth_module.codex_auth_path = lambda: auth_path
    auth_module.load_codex_auth = load_codex_auth

    monkeypatch.setitem(sys.modules, "dspy_codex_lm", codex_module)
    monkeypatch.setitem(sys.modules, "dspy_codex_lm.cli", cli_module)
    monkeypatch.setitem(sys.modules, "dspy_codex_lm.auth", auth_module)
    monkeypatch.setattr("fractal.providers.shutil.which", lambda name: "/bin/codex")
    return calls


def test_registry_contains_initial_provider_set() -> None:
    from fractal.providers import (
        ANTHROPIC,
        CUSTOM_OPENAI_COMPATIBLE,
        DEEPSEEK,
        GEMINI,
        GROQ,
        MISTRAL,
        OLLAMA,
        OPENAI_API,
        OPENAI_CODEX,
        OPENROUTER,
        XAI,
        ZAI,
        get_provider,
        model_choices,
        provider_registry,
    )

    assert set(provider_registry()) == {
        OPENAI_CODEX,
        OPENAI_API,
        ANTHROPIC,
        GEMINI,
        XAI,
        ZAI,
        DEEPSEEK,
        MISTRAL,
        GROQ,
        OPENROUTER,
        OLLAMA,
        CUSTOM_OPENAI_COMPATIBLE,
    }
    assert get_provider(OPENAI_CODEX).auth_type == "codex_cli"
    assert get_provider(OPENAI_CODEX).default_model == "gpt-5.5"
    assert model_choices(get_provider(OPENAI_CODEX)) == ("gpt-5.5",)
    assert model_choices(get_provider(OPENAI_API)) == (
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    )
    assert model_choices(get_provider(ANTHROPIC)) == (
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    )
    assert model_choices(get_provider(OPENROUTER)) == (
        "openai/gpt-5.5",
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
    )
    assert model_choices(get_provider(CUSTOM_OPENAI_COMPATIBLE)) == (
        "gpt-oss-120b",
        "qwen3-coder",
    )
    assert get_provider(OPENAI_API).default_api_key_env == "OPENAI_API_KEY"
    assert get_provider(ANTHROPIC).default_api_key_env == "ANTHROPIC_API_KEY"
    assert get_provider(GEMINI).default_api_key_env == "GEMINI_API_KEY"
    assert get_provider(XAI).default_api_key_env == "XAI_API_KEY"
    assert model_choices(get_provider(ZAI)) == (
        "glm-5.2",
        "glm-5.1",
        "glm-4.7",
        "glm-4.6",
        "glm-4.5",
        "glm-4.5-flash",
    )
    assert get_provider(ZAI).default_api_key_env == "ZAI_API_KEY"
    assert get_provider(DEEPSEEK).default_api_key_env == "DEEPSEEK_API_KEY"
    assert get_provider(MISTRAL).default_api_key_env == "MISTRAL_API_KEY"
    assert get_provider(GROQ).default_api_key_env == "GROQ_API_KEY"
    assert get_provider(OPENROUTER).default_api_key_env == "OPENROUTER_API_KEY"
    assert get_provider(CUSTOM_OPENAI_COMPATIBLE).supports_base_url is True
    assert get_provider(OLLAMA).auth_type == "none"
    assert get_provider(OLLAMA).default_base_url == "http://localhost:11434"


def test_unknown_provider_raises_clear_error() -> None:
    from fractal.providers import UnknownProviderError, get_provider

    with pytest.raises(UnknownProviderError, match="unknown provider 'missing'"):
        get_provider("missing")


def test_resolve_lm_prefers_explicit_lm() -> None:
    from fractal.providers import OPENAI_API, ProviderSelection, resolve_lm

    explicit = "explicit-lm"

    assert resolve_lm(explicit, ProviderSelection(OPENAI_API, model="gpt-5.5")) is explicit


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai-api", "gpt-5.5", "openai/gpt-5.5"),
        ("openai-api", "openai/gpt-5.5", "openai/gpt-5.5"),
        ("anthropic", "claude-sonnet-4-6", "anthropic/claude-sonnet-4-6"),
        ("gemini", "gemini-3.5-flash", "gemini/gemini-3.5-flash"),
        ("xai", "grok-4.3", "xai/grok-4.3"),
        ("zai", "glm-5.2", "zai/glm-5.2"),
        ("zai", "zai/glm-4.5-flash", "zai/glm-4.5-flash"),
        ("deepseek", "deepseek-v4-pro", "deepseek/deepseek-v4-pro"),
        ("mistral", "devstral-2-latest", "mistral/devstral-2-latest"),
        ("groq", "openai/gpt-oss-120b", "groq/openai/gpt-oss-120b"),
        (
            "openrouter",
            "openai/gpt-5.5",
            "openrouter/openai/gpt-5.5",
        ),
        (
            "openrouter",
            "openrouter/openai/gpt-5.5",
            "openrouter/openai/gpt-5.5",
        ),
    ],
)
def test_api_backed_providers_normalize_model_strings(
    provider: str,
    model: str,
    expected: str,
) -> None:
    from fractal.providers import ProviderSelection, build_lm, get_provider

    env_name = get_provider(provider).default_api_key_env
    assert env_name is not None

    assert (
        build_lm(
            ProviderSelection(provider, model=model),
            env={env_name: "secret-value"},
        )
        == expected
    )


@pytest.mark.parametrize(
    ("provider", "env_name"),
    [
        ("openai-api", "OPENAI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
        ("xai", "XAI_API_KEY"),
        ("zai", "ZAI_API_KEY"),
        ("deepseek", "DEEPSEEK_API_KEY"),
        ("mistral", "MISTRAL_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_api_backed_providers_require_configured_env_vars(
    provider: str,
    env_name: str,
) -> None:
    from fractal.providers import (
        MissingProviderCredentialError,
        ProviderSelection,
        build_lm,
        check_provider_readiness,
        validate_provider_selection,
    )

    selection = ProviderSelection(provider, model="model-name")
    validate_provider_selection(selection)

    with pytest.raises(MissingProviderCredentialError) as excinfo:
        check_provider_readiness(selection, env={})

    message = str(excinfo.value)
    assert provider in message
    assert env_name in message
    assert "secret-value" not in message

    with pytest.raises(MissingProviderCredentialError, match=env_name):
        build_lm(selection, env={env_name: ""})

    check_provider_readiness(selection, env={env_name: "secret-value"})


def test_api_backed_providers_reject_codex_cli_auth_source() -> None:
    from fractal.providers import (
        OPENAI_API,
        ProviderConfigError,
        ProviderSelection,
        validate_provider_selection,
    )

    with pytest.raises(ProviderConfigError, match="auth_source='env'"):
        validate_provider_selection(
            ProviderSelection(
                OPENAI_API,
                model="gpt-5.5",
                api_key_env="OPENAI_API_KEY",
                auth_source="codex-cli",
            )
        )


def test_codex_factory_uses_codex_model_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal.providers import OPENAI_CODEX, ProviderSelection, build_lm

    auth_path = object()
    calls = install_fake_codex_modules(monkeypatch, auth_path=auth_path)

    lm = build_lm(ProviderSelection(OPENAI_CODEX, model="openai/gpt-5.5"))

    assert type(lm).__name__ == "FakeCodexLM"
    assert calls == {
        "requested": "openai/gpt-5.5",
        "load_auth_path": auth_path,
        "model": "resolved-codex-model",
        "auth_path": auth_path,
    }


def test_codex_factory_reports_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal.providers import (
        OPENAI_CODEX,
        ProviderSelection,
        UnsupportedProviderModelError,
        build_lm,
    )

    calls = install_fake_codex_modules(monkeypatch)

    with pytest.raises(UnsupportedProviderModelError, match="only supports"):
        build_lm(ProviderSelection(OPENAI_CODEX, model="gpt-4o"))

    assert "requested" not in calls
    assert "load_auth_path" not in calls


def test_codex_validation_requires_official_codex_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal.providers import (
        OPENAI_CODEX,
        MissingCodexCliError,
        ProviderSelection,
        check_provider_readiness,
    )

    install_fake_codex_modules(monkeypatch)
    monkeypatch.setattr("fractal.providers.shutil.which", lambda name: None)

    with pytest.raises(MissingCodexCliError) as excinfo:
        check_provider_readiness(ProviderSelection(OPENAI_CODEX))

    message = str(excinfo.value)
    assert "codex" in message
    assert "codex login --device-auth" in message
    assert "secret-token" not in message


def test_codex_validation_requires_usable_codex_cli_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal.providers import (
        OPENAI_CODEX,
        MissingCodexAuthError,
        ProviderSelection,
        build_lm,
    )

    auth_path = "/tmp/fake-codex-auth.json"
    install_fake_codex_modules(
        monkeypatch,
        auth_path=auth_path,
        load_auth_error=FileNotFoundError(auth_path),
    )

    with pytest.raises(MissingCodexAuthError) as excinfo:
        build_lm(ProviderSelection(OPENAI_CODEX))

    message = str(excinfo.value)
    assert auth_path in message
    assert "codex login --device-auth" in message
    assert "secret-token" not in message


def test_codex_validation_rejects_non_cli_auth_source(
) -> None:
    from fractal.providers import (
        OPENAI_CODEX,
        ProviderConfigError,
        ProviderSelection,
        validate_provider_selection,
    )

    with pytest.raises(ProviderConfigError, match="auth_source='codex-cli'"):
        validate_provider_selection(
            ProviderSelection(OPENAI_CODEX, auth_source="codex-lm-profile")
        )


def test_custom_openai_compatible_requires_complete_config() -> None:
    from fractal.providers import (
        CUSTOM_OPENAI_COMPATIBLE,
        ProviderConfigError,
        ProviderSelection,
        build_lm,
    )

    with pytest.raises(ProviderConfigError, match="requires base_url"):
        build_lm(
            ProviderSelection(
                CUSTOM_OPENAI_COMPATIBLE,
                model="custom-model",
                api_key_env="CUSTOM_KEY",
            ),
            env={"CUSTOM_KEY": "secret"},
        )

    with pytest.raises(ProviderConfigError, match="requires model"):
        build_lm(
            ProviderSelection(
                CUSTOM_OPENAI_COMPATIBLE,
                api_key_env="CUSTOM_KEY",
                base_url="https://llm.example.test/v1",
            ),
            env={"CUSTOM_KEY": "secret"},
        )

    with pytest.raises(ProviderConfigError, match="requires api_key_env"):
        build_lm(
            ProviderSelection(
                CUSTOM_OPENAI_COMPATIBLE,
                model="custom-model",
                base_url="https://llm.example.test/v1",
            ),
            env={"CUSTOM_KEY": "secret"},
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "ftp://llm.example.test/v1",
        "https://",
    ],
)
def test_custom_openai_compatible_validates_base_url_shape(base_url: str) -> None:
    from fractal.providers import (
        CUSTOM_OPENAI_COMPATIBLE,
        ProviderConfigError,
        ProviderSelection,
        build_lm,
    )

    with pytest.raises(ProviderConfigError, match="HTTP\\(S\\) URL"):
        build_lm(
            ProviderSelection(
                CUSTOM_OPENAI_COMPATIBLE,
                model="custom-model",
                api_key_env="CUSTOM_KEY",
                base_url=base_url,
            ),
            env={"CUSTOM_KEY": "secret"},
        )


def test_custom_openai_compatible_reads_key_from_env_without_leaking_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal.providers import (
        CUSTOM_OPENAI_COMPATIBLE,
        ProviderConfigError,
        ProviderSelection,
        build_lm,
    )

    created: dict[str, object] = {}

    def fake_lm(**kwargs: object) -> object:
        created.update(kwargs)
        return SimpleNamespace(kind="lm")

    dspy_module = ModuleType("dspy")
    dspy_module.LM = fake_lm
    monkeypatch.setitem(sys.modules, "dspy", dspy_module)

    selection = ProviderSelection(
        CUSTOM_OPENAI_COMPATIBLE,
        model="custom-model",
        api_key_env="CUSTOM_KEY",
        base_url="https://llm.example.test/v1",
    )

    with pytest.raises(ProviderConfigError) as excinfo:
        build_lm(selection, env={"CUSTOM_KEY": ""})

    assert "secret-value" not in str(excinfo.value)
    assert "CUSTOM_KEY" in str(excinfo.value)

    lm = build_lm(selection, env={"CUSTOM_KEY": "secret-value"})

    assert lm.kind == "lm"
    assert created == {
        "model": "openai/custom-model",
        "api_base": "https://llm.example.test/v1",
        "api_key": "secret-value",
    }


def test_ollama_builds_local_lm_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal.providers import (
        OLLAMA,
        ProviderSelection,
        build_lm,
        check_provider_readiness,
        validate_provider_selection,
    )

    created: dict[str, object] = {}

    def fake_lm(**kwargs: object) -> object:
        created.update(kwargs)
        return SimpleNamespace(kind="lm")

    dspy_module = ModuleType("dspy")
    dspy_module.LM = fake_lm
    monkeypatch.setitem(sys.modules, "dspy", dspy_module)

    selection = ProviderSelection(OLLAMA, model="qwen3-coder", auth_source="local")
    validate_provider_selection(selection)
    check_provider_readiness(selection, env={})

    lm = build_lm(selection, env={})

    assert lm.kind == "lm"
    assert created == {
        "model": "ollama_chat/qwen3-coder",
        "api_base": "http://localhost:11434",
        "api_key": "",
    }


def test_ollama_accepts_custom_base_url_and_rejects_bad_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fractal.providers import (
        OLLAMA,
        ProviderConfigError,
        ProviderSelection,
        build_lm,
        validate_provider_selection,
    )

    created: dict[str, object] = {}

    def fake_lm(**kwargs: object) -> object:
        created.update(kwargs)
        return SimpleNamespace(kind="lm")

    dspy_module = ModuleType("dspy")
    dspy_module.LM = fake_lm
    monkeypatch.setitem(sys.modules, "dspy", dspy_module)

    build_lm(
        ProviderSelection(
            OLLAMA,
            model="devstral",
            base_url="http://ollama.lan:11434",
        ),
        env={},
    )
    assert created["api_base"] == "http://ollama.lan:11434"

    with pytest.raises(ProviderConfigError, match="HTTP\\(S\\) URL"):
        validate_provider_selection(
            ProviderSelection(OLLAMA, model="devstral", base_url="not-a-url")
        )


def test_ollama_rejects_other_auth_sources() -> None:
    from fractal.providers import (
        OLLAMA,
        ProviderConfigError,
        ProviderSelection,
        validate_provider_selection,
    )

    with pytest.raises(ProviderConfigError, match="auth_source='local'"):
        validate_provider_selection(
            ProviderSelection(OLLAMA, model="devstral", auth_source="env")
        )
