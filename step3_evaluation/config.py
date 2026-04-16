"""
Centralized configuration for the evaluation system.
Example keys.json:

{
  "openai": "sk-...",
  "anthropic": "sk-ant-...",
  "deepseek": "sk-...",
  "gemini": "AIza...",
  "openrouter": "sk-or-...",
  "xiaomi": "sk-...",
  "vertex_project_id": "your-gcp-project-id"
}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

# ============================================================================
# Keys JSON location
# ============================================================================
# Priority:
# 1) EVAL_KEYS_JSON env var (optional, for advanced use / deployment)
# 2) configs/keys.json relative to repo root (best for UI)
#
# If you truly want "no env at all", just don't set EVAL_KEYS_JSON.
DEFAULT_KEYS_JSON = "step3_evaluation/configs/keys.json"
KEYS_JSON_PATH = os.getenv("EVAL_KEYS_JSON", DEFAULT_KEYS_JSON)


# ============================================================================
# Internal: load keys
# ============================================================================
_KEYS_CACHE: Optional[Dict[str, Any]] = None


def _repo_root() -> Path:
    """
    Try to infer repo root robustly.
    This file is under step3_evaluation/, so repo root is likely parents[1] or parents[2].
    We fall back to cwd if unsure.
    """
    here = Path(__file__).resolve()
    # step3_evaluation/config.py -> repo root is usually two levels up
    # adjust if your layout differs
    for up in (2, 3, 1):
        try:
            cand = here.parents[up]
            if (cand / "problems").exists() and (cand / "pipeline").exists():
                return cand
        except Exception:
            pass
    return Path.cwd()


def _resolve_keys_json_path() -> Path:
    p = Path(KEYS_JSON_PATH)
    if p.is_absolute():
        return p
    return _repo_root() / p


def _load_keys_json() -> Dict[str, Any]:
    """
    Load keys from JSON once and cache it.
    """
    global _KEYS_CACHE
    if _KEYS_CACHE is not None:
        return _KEYS_CACHE

    p = _resolve_keys_json_path()
    if not p.exists():
        raise RuntimeError(
            f"Keys JSON not found: {p}\n"
            f"Create it (recommended) at: {DEFAULT_KEYS_JSON}\n\n"
            "Example:\n"
            '{\n'
            '  "openai": "sk-...",\n'
            '  "anthropic": "sk-ant-...",\n'
            '  "deepseek": "sk-...",\n'
            '  "gemini": "AIza...",\n'
            '  "openrouter": "sk-or-...",\n'
            '  "xiaomi": "sk-...",\n'
            '  "vertex_project_id": "your-gcp-project-id"\n'
            '}\n'
        )

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse keys JSON: {p}\nError: {e}")

    if not isinstance(obj, dict):
        raise RuntimeError(f"Keys JSON must be a JSON object/dict: {p}")

    _KEYS_CACHE = obj
    return obj


def reload_keys() -> None:
    """
    Force reload keys JSON (useful in UI after editing keys.json).
    """
    global _KEYS_CACHE
    _KEYS_CACHE = None


def _get_key(name: str) -> str:
    keys = _load_keys_json()
    val = keys.get(name)
    if not val or not isinstance(val, str):
        raise RuntimeError(
            f"Missing API key '{name}' in keys JSON: {_resolve_keys_json_path()}\n"
            f"Please add: \"{name}\": \"...\""
        )
    return val


def _get_optional_value(name: str, default: Any = None) -> Any:
    keys = _load_keys_json()
    return keys.get(name, default)


# ============================================================================
# API Keys (from keys.json)
# ============================================================================
def get_openai_api_key() -> str:
    return _get_key("openai")


def get_anthropic_api_key() -> str:
    return _get_key("anthropic")


def get_deepseek_api_key() -> str:
    return _get_key("deepseek")


def get_gemini_api_key() -> str:
    # we store as "gemini" in keys.json
    return _get_key("gemini")


def get_openrouter_api_key() -> str:
    return _get_key("openrouter")


def get_xiaomi_api_key() -> str:
    return _get_key("xiaomi")


# ============================================================================
# Vertex AI Configuration
# ============================================================================
# Load lazily from keys.json when needed (do not hard-fail at import time).
DEFAULT_VERTEX_PROJECT_ID = "booming-entity-322713"
VERTEX_LOCATION_GEMINI = "global"
VERTEX_LOCATION_LLAMA = "us-east5"
VERTEX_LOCATION_QWEN = "us-south1"


# ============================================================================
# Model-specific defaults
# ============================================================================
DEFAULT_REASONING_EFFORT = {
    "openai": {
        "o4-mini": "high",
        "gpt-5.1": "medium",
    }
}


# ============================================================================
# OpenRouter Model Aliases
# ============================================================================
OPENROUTER_MODEL_ALIASES: Dict[str, str] = {
    # Qwen
    "qwen3-14b": "qwen/qwen3-14b",
    "qwq-32b": "qwen/qwq-32b",

    # Mistral
    "ministral-14b-2512": "mistralai/ministral-14b-2512",

    # NVIDIA
    "nemotron-3-nano-30b-a3b": "nvidia/nemotron-3-nano-30b-a3b:free",

    # Xiaomi
    "mimo-v2-flash": "xiaomi/mimo-v2-flash:free",

    # xAI
    "grok-4.1-fast": "x-ai/grok-4.1-fast",

    # Moonshot
    "kimi-k2-thinking": "moonshotai/kimi-k2-thinking",

    # Gemini on OpenRouter (optional)
    "gemini-3-flash-preview": "google/gemini-3-flash-preview",
}


# ============================================================================
# Provider-specific settings
# ============================================================================
PROVIDER_SETTINGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
    },
    "gemini": {},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
    },
    "vertex_maas": {},
    "xiaomi": {},
}


# ============================================================================
# Evaluation settings
# ============================================================================
DEFAULT_CONCURRENCY = 50
DEFAULT_SIZES = ["S", "M", "L"]
ALL_PROBLEMS = [
    "MIS",
    "CVRP",
    "MAXCUT",
    "TSP",
    "TSPTW",
    "OP",
    "PCTSP",
    "MLP",
    "MCP",
    "MDS",
    "GCP",
    "MVC",
    "PDP",
    "QSPP",
    "2SP",
    "BPP",
    "AP3",
    "CFLP",
    "CMP",
    "CSP",
    "FSP",
    "GAP",
    "HSP",
    "JSP",
    "KMST",
    "KP",
    "LOP",
    "MDP",
    "MkC",
    "PCENTER",
    "QKP",
    "SCP",
    "SFP",
    "SP",
    "SPP",
    "STP",
    "UFLP",
    "PMED",
    "OSP",
    "PMS",
    "QAP",
    "RCPSP",
    "SMTWT",
]


# ============================================================================
# Utility functions
# ============================================================================
def get_vertex_project_id() -> str:
    return str(_get_optional_value("vertex_project_id", DEFAULT_VERTEX_PROJECT_ID))


def get_provider_setting(provider: str, setting: str, default: Any = None) -> Any:
    return PROVIDER_SETTINGS.get(provider, {}).get(setting, default)


def normalize_openrouter_model(model: str) -> str:
    if "/" in model:
        return model
    return OPENROUTER_MODEL_ALIASES.get(model, model)
