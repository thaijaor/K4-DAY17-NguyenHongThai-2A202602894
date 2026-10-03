from __future__ import annotations

import os
from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents (main model and judge model)."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    @property
    def is_configured(self) -> bool:
        """True when the provider has what it needs to make a live call."""

        if self.provider == "ollama":
            return bool(self.model_name)
        if self.provider == "custom":
            return bool(self.model_name and self.base_url)
        return bool(self.model_name and self.api_key)


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`; reject unknown providers."""

    key = (value or "").strip().lower()
    if key not in _ALIASES:
        raise ValueError(f"Unsupported provider '{value}'. Choose one of: {', '.join(SUPPORTED_PROVIDERS)}")
    return _ALIASES[key]


_RATE_LIMITER = None


def shared_rate_limiter():
    """One limiter shared by every model (agents + judge), set via LLM_RPM for free-tier quotas."""

    global _RATE_LIMITER
    rpm = float(os.getenv("LLM_RPM", "0") or 0)
    if rpm <= 0:
        return None
    if _RATE_LIMITER is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        _RATE_LIMITER = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.2, max_bucket_size=1)
    return _RATE_LIMITER


def build_chat_model(config: ProviderConfig):
    """Instantiate the LangChain chat model for the selected provider."""

    model = _build_chat_model(config)
    limiter = shared_rate_limiter()
    if limiter is not None:
        model.rate_limiter = limiter
    return model


def _build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model_name, temperature=config.temperature, api_key=config.api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(model=config.model_name, temperature=config.temperature, api_key=config.api_key)

    raise ValueError(f"Unsupported provider: {provider}")
