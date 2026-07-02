from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, TextIO

import dspy

from .config import DefaultsConfig, FractalConfig, effective_lm_num_retries
from .providers import ProviderSelection


@dataclass(frozen=True)
class RuntimeLMConfig:
    lm: dspy.LM
    sub_lm: dspy.LM
    provider_selection: ProviderSelection | None = None
    sub_lm_follows_main: bool = True
    defaults: DefaultsConfig | None = None
    lm_num_retries: int = 3
    sub_model: str | None = None


class RuntimeLMArgs(Protocol):
    lm: dspy.LM | str | None
    sub_lm: dspy.LM | str | None
    workspace: Path | None


def coerce_lm(lm: dspy.LM | str) -> dspy.LM:
    """Normalize Fractal's accepted LM inputs into a concrete DSPy LM."""
    if isinstance(lm, dspy.LM):
        return lm
    return dspy.LM(model=lm)


def resolve_runtime_lms(
    args: RuntimeLMArgs,
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    auto_setup: bool,
    on_setup_start: Callable[[], None] | None = None,
) -> RuntimeLMConfig | None:
    from .config import FractalConfigError, load_layered_config
    from .providers import ProviderError, build_lm

    if args.lm is not None:
        num_retries = effective_lm_num_retries()
        lm = coerce_lm(args.lm)
        lm.num_retries = num_retries
        sub_lm = lm if args.sub_lm is None else coerce_lm(args.sub_lm)
        sub_lm.num_retries = num_retries
        return RuntimeLMConfig(
            lm=lm,
            sub_lm=sub_lm,
            provider_selection=None,
            sub_lm_follows_main=args.sub_lm is None,
            lm_num_retries=num_retries,
        )

    workspace = getattr(args, "workspace", None)
    try:
        result = load_layered_config(workspace=workspace)
    except FractalConfigError as exc:
        print(f"fractal config: {exc}", file=stderr)
        print("Run `fractal config setup` after fixing the config.", file=stderr)
        return None

    if result.config is None:
        if not auto_setup:
            print(f"fractal: no global config found at {result.path}", file=stderr)
            print("Run `fractal config setup` or pass `--lm` explicitly.", file=stderr)
            return None
        if on_setup_start is not None:
            on_setup_start()
        print("fractal: no global config found; starting setup.", file=stderr)
        from .config_commands import config_setup

        if config_setup(stdin=stdin, stdout=stdout, stderr=stderr) != 0:
            return None
        try:
            result = load_layered_config(workspace=workspace)
        except FractalConfigError as exc:
            print(f"fractal config: {exc}", file=stderr)
            return None
        if result.config is None:
            print("fractal: setup completed but config could not be loaded", file=stderr)
            return None

    num_retries = effective_lm_num_retries(result.config.defaults)
    selection = selection_from_config(result.config, path=result.path)
    try:
        lm = build_lm(selection)
        lm.num_retries = num_retries
    except ProviderError as exc:
        print(f"fractal config: {exc}", file=stderr)
        print(
            "Run `fractal config status` for details or "
            "`fractal config setup` to repair setup.",
            file=stderr,
        )
        return None
    sub_lm = None if args.sub_lm is None else coerce_lm(args.sub_lm)
    if sub_lm is not None:
        sub_lm.num_retries = num_retries
    sub_lm_follows_main = args.sub_lm is None
    sub_model = result.config.active_sub_model
    if sub_lm is None:
        try:
            sub_selection = sub_selection_from_config(result.config, path=result.path)
            if sub_selection is not None:
                sub_lm = build_lm(sub_selection)
                sub_lm.num_retries = num_retries
                sub_model = sub_selection.model
                sub_lm_follows_main = False
        except ProviderError as exc:
            print(f"fractal config: sub model: {exc}", file=stderr)
            print(
                "Fix `active_sub_provider`/`active_sub_model` in the config or remove them.",
                file=stderr,
            )
            return None
    if sub_lm is None:
        sub_lm = lm
    return RuntimeLMConfig(
        lm=lm,
        sub_lm=sub_lm,
        provider_selection=selection,
        sub_lm_follows_main=sub_lm_follows_main,
        defaults=result.config.defaults,
        lm_num_retries=num_retries,
        sub_model=sub_model,
    )


def selection_from_config(
    config: FractalConfig,
    *,
    path: str | Path | None = None,
) -> ProviderSelection:
    from .config import resolve_effective_config

    effective = resolve_effective_config(config, path=path)
    provider_config = effective.provider_config
    return ProviderSelection(
        provider=effective.provider,
        model=effective.model,
        api_key_env=provider_config.api_key_env,
        base_url=provider_config.base_url,
        auth_source=provider_config.auth_source,
    )


def sub_selection_from_config(
    config: FractalConfig,
    *,
    path: str | Path | None = None,
) -> ProviderSelection | None:
    """Provider selection for RLM sub-calls, or None to follow the main LM.

    The sub-LM may live on a different provider (``active_sub_provider``);
    when only ``active_sub_model`` is set it shares the main provider.
    """
    if config.active_sub_provider is None and config.active_sub_model is None:
        return None
    provider_id = config.active_sub_provider or config.active_provider
    provider_config = config.providers[provider_id]
    model = config.active_sub_model
    if model is None:
        from .providers import get_provider

        model = get_provider(provider_id).default_model
    return ProviderSelection(
        provider=provider_id,
        model=model,
        api_key_env=provider_config.api_key_env,
        base_url=provider_config.base_url,
        auth_source=provider_config.auth_source,
    )
