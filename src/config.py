from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

# provider -> (api key env var, base url env var)
PROVIDER_ENV = {
    "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    "custom": ("CUSTOM_API_KEY", "CUSTOM_BASE_URL"),
    "gemini": ("GEMINI_API_KEY", None),
    "anthropic": ("ANTHROPIC_API_KEY", None),
    "ollama": (None, "OLLAMA_BASE_URL"),
    "openrouter": ("OPENROUTER_API_KEY", None),
}


@dataclass
class LabConfig:
    """Shared configuration: paths, compact-memory knobs, main model and judge model."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    profile_confidence_threshold: float = 0.6


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    raw_provider = os.getenv(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai")
    provider = normalize_provider(raw_provider)
    key_env, url_env = PROVIDER_ENV[provider]
    model_name = os.getenv(f"{prefix}_MODEL") or (
        fallback.model_name if fallback and fallback.provider == provider else DEFAULT_MODELS[provider]
    )
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    else:
        api_key = os.getenv(key_env) if key_env else None
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=api_key,
        base_url=os.getenv(url_env) if url_env else None,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` (if present) and return a populated LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    state_dir = Path(os.getenv("LAB_STATE_DIR", root / "state")).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM")
    judge_model = _provider_config("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "4")),
        model=model,
        judge_model=judge_model,
        profile_confidence_threshold=float(os.getenv("PROFILE_CONFIDENCE_THRESHOLD", "0.6")),
    )
