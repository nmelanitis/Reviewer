"""
Small provider adapter for optional cloud LLM calls.

This module is intentionally isolated from the rest of the pipeline. Existing
offline/manual paths do not import provider SDKs. The selected provider is
imported only when a cloud LLM script is run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROVIDERS = {"openai", "anthropic", "gemini"}
DEFAULT_ENV_FILE = ".env"
ENV_LOADED = False


@dataclass
class LLMConfig:
    provider: str
    model: str
    max_output_tokens: int = 4096


def load_env_file(path: str = DEFAULT_ENV_FILE) -> None:
    """
    Load API keys from a local .env file without adding a dependency.

    Exported shell variables take priority. The .env file is for local secrets
    only and should stay ignored by git.
    """
    global ENV_LOADED
    if ENV_LOADED:
        return
    ENV_LOADED = True

    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def require_api_key(provider: str) -> str:
    load_env_file()
    env_names = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_name = env_names[provider]
    api_key = os.getenv(env_name)
    if not api_key:
        raise RuntimeError(f"{env_name} is required for provider '{provider}'.")
    return api_key


def generate_text(config: LLMConfig, prompt: str) -> str:
    if config.provider not in PROVIDERS:
        raise RuntimeError(f"Unsupported provider: {config.provider}")

    if config.provider == "openai":
        return generate_openai(config, prompt)
    if config.provider == "anthropic":
        return generate_anthropic(config, prompt)
    return generate_gemini(config, prompt)


def generate_openai(config: LLMConfig, prompt: str) -> str:
    api_key = require_api_key("openai")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install OpenAI support with: pip install openai") from exc

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=config.model,
        input=prompt,
        max_output_tokens=config.max_output_tokens,
    )
    text = getattr(response, "output_text", None)
    if text:
        return text
    return str(response)


def generate_anthropic(config: LLMConfig, prompt: str) -> str:
    api_key = require_api_key("anthropic")
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("Install Anthropic support with: pip install anthropic") from exc

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=config.model,
        max_tokens=config.max_output_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip() or str(response)


def generate_gemini(config: LLMConfig, prompt: str) -> str:
    api_key = require_api_key("gemini")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Install Gemini support with: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=config.model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=config.max_output_tokens),
    )
    text = getattr(response, "text", None)
    if text:
        return text
    return str(response)
