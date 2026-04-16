import asyncio
import json
import re
from typing import Any, Dict, Tuple, Optional, Literal, List

import google
from google import genai
from google.genai import types as genai_types
import pandas as pd
import requests
from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic, Anthropic
from google.auth import default
from google.auth.transport.requests import Request

# Import configuration
from step3_evaluation.config import (
    get_openai_api_key,
    get_anthropic_api_key,
    get_deepseek_api_key,
    get_gemini_api_key,
    get_openrouter_api_key,
    get_xiaomi_api_key,
    get_vertex_project_id,
    normalize_openrouter_model,
    VERTEX_LOCATION_GEMINI,
    VERTEX_LOCATION_LLAMA,
    VERTEX_LOCATION_QWEN,
)
from step3_evaluation.utils.eval_parsing import safe_json_loads

Provider = Literal[
    "openai",
    "anthropic",
    "deepseek",
    "gemini",
    "vertex_maas",
    "openrouter",
    "xiaomi"
]

# Client instances (initialized lazily)
_openai_async_client: Optional[AsyncOpenAI] = None
_openai_client: Optional[OpenAI] = None

_anthropic_async_client: Optional[AsyncAnthropic] = None
_anthropic_client: Optional[Anthropic] = None
_deepseek_async_client: Optional[AsyncOpenAI] = None
_gemini_client: Optional[genai.Client] = None
_openrouter_async_client: Optional[AsyncOpenAI] = None
_openrouter_client: Optional[OpenAI] = None
# Xiaomi client instances (initialized lazily)
_xiaomi_async_client: Optional[AsyncOpenAI] = None
_xiaomi_client: Optional[OpenAI] = None

def _get_xiaomi_async_client() -> AsyncOpenAI:
    global _xiaomi_async_client
    if _xiaomi_async_client is None:
        _xiaomi_async_client = AsyncOpenAI(
            api_key=get_xiaomi_api_key(),
            base_url="https://api.xiaomimimo.com/v1",
        )
    return _xiaomi_async_client


def _get_xiaomi_client() -> OpenAI:
    global _xiaomi_client
    if _xiaomi_client is None:
        _xiaomi_client = OpenAI(
            api_key=get_xiaomi_api_key(),
            base_url="https://api.xiaomimimo.com/v1",
        )
    return _xiaomi_client


def _get_openrouter_async_client() -> AsyncOpenAI:
    global _openrouter_async_client
    if _openrouter_async_client is None:
        api_key = get_openrouter_api_key()
        _openrouter_async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_async_client


def _get_openrouter_client() -> OpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        api_key = get_openrouter_api_key()
        _openrouter_client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


def _get_adc_token() -> str:
    creds, _ = default()
    creds.refresh(Request())
    return creds.token


def _vertex_maas_base_url(project_id: str, location: str, api_version: str = "v1beta1") -> str:
    return (
        f"https://{location}-aiplatform.googleapis.com/"
        f"{api_version}/projects/{project_id}/locations/{location}/endpoints/openapi"
    )


def _make_vertex_maas_async_client(project_id: str, location: str, api_version: str = "v1beta1") -> AsyncOpenAI:
    token = _get_adc_token()
    return AsyncOpenAI(
        api_key=token,
        base_url=_vertex_maas_base_url(project_id, location, api_version),
    )

def _make_vertex_maas_client(project_id: str, location: str, api_version: str = "v1beta1") -> OpenAI:
    token = _get_adc_token()
    return OpenAI(
        api_key=token,
        base_url=_vertex_maas_base_url(project_id, location, api_version),
    )



async def _vertex_gemini_generate_content(
    *,
    project_id: str,
    location: str,  # usually "global" for publisher models
    model_id: str,  # e.g. "gemini-3-pro-preview"
    contents: List[Dict[str, Any]],
    generation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call Vertex AI Gemini publisher model via generateContent REST API.
    Returns parsed JSON response dict.
    """
    url = (
        f"https://aiplatform.googleapis.com/v1/"
        f"projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"
    )

    payload: Dict[str, Any] = {"contents": contents}
    if generation_config is not None:
        payload["generationConfig"] = generation_config

    token = _get_adc_token()

    def _sync_post() -> Dict[str, Any]:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
        )
        resp.raise_for_status()
        return resp.json()

    # run blocking requests in a thread
    import asyncio
    return await asyncio.to_thread(_sync_post)


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = get_gemini_api_key()
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

def _get_deepseek_async_client() -> AsyncOpenAI:
    global _deepseek_async_client
    if _deepseek_async_client is None:
        api_key = get_deepseek_api_key()
        _deepseek_async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _deepseek_async_client

def _get_openai_async_client() -> AsyncOpenAI:
    global _openai_async_client
    if _openai_async_client is None:
        api_key = get_openai_api_key()
        print(api_key)
        _openai_async_client = AsyncOpenAI(api_key=api_key)
    return _openai_async_client


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = get_openai_api_key()
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _get_anthropic_async_client() -> AsyncAnthropic:
    global _anthropic_async_client
    if _anthropic_async_client is None:
        api_key = get_anthropic_api_key()
        _anthropic_async_client = AsyncAnthropic(api_key=api_key)
    return _anthropic_async_client


def _get_anthropic_client() -> Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = get_anthropic_api_key()
        _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client

async def call_llm(
    instruction: str,
    *,
    model: str,
    provider: Optional[Provider] = None,
    reasoning: bool = False,
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    max_output_tokens: Optional[int] = None,  # Anthropic / others
    budget: Optional[int] = None,             # Anthropic thinking.budget_tokens
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    Call LLM and return raw text only.
    Returns: (raw_response, text, meta)

    Semantics:
    - reasoning controls whether "thinking / reasoning mode" is enabled.
    - reasoning_effort is only meaningful when reasoning=True (and only for some providers/models).
    - budget is only meaningful when reasoning=True and provider==anthropic.
    """

    # ---- strict local validation (avoid ambiguity) ----
    if not reasoning:
        if reasoning_effort is not None:
            raise ValueError("reasoning_effort is only valid when reasoning=True")
        if budget is not None:
            raise ValueError("budget is only valid when reasoning=True")
    else:
        # If you want to be strict: require Anthropic budget when reasoning enabled.
        if provider == "anthropic" and budget is None:
            raise ValueError("Anthropic requires budget when reasoning=True (thinking.budget_tokens)")

    # ---------- OpenAI ----------
    if provider == "openai":
        client = _get_openai_async_client()
        kwargs: Dict[str, Any] = {
            "model": model,
            "input": instruction,
            "service_tier": "flex",
        }

        # reasoning enabled => set reasoning object
        if reasoning:
            # allow effort optional; if None, let API default behavior
            if reasoning_effort is not None:
                kwargs["reasoning"] = {"effort": reasoning_effort}
        else:
            if model == "gpt-5.1":
                kwargs["temperature"] = 0

        temperature = kwargs.get("temperature", "None")
        eff = reasoning_effort if reasoning else None
        print(
            "[LLM CALL][openai]",
            f"model={model},",
            f"reasoning={'enabled' if reasoning else 'disabled'},",
            f"reasoning_effort={eff},",
            f"temperature={temperature}"
        )

        resp = await client.responses.create(**kwargs)

        text: Optional[str] = getattr(resp, "output_text", None)
        if text is None and getattr(resp, "output", None):
            try:
                text = resp.output[0].content[0].text  # type: ignore[attr-defined]
            except Exception:
                text = None
        if not text:
            raise ValueError("OpenAI response has no text content.")

        meta = {
            "provider": provider,
            "model": model,
            "reasoning": reasoning,
            "reasoning_effort": (reasoning_effort if reasoning else None),
            "temperature": temperature,
        }
        return resp, text, meta

    # ---------- Xiaomi (MiMo, OpenAI-compatible Chat Completions) ----------
    elif provider == "xiaomi":
        client = _get_xiaomi_async_client()
        messages = [{"role": "user", "content": instruction}]

        body: Dict[str, Any] = {"model": model, "messages": messages}

        if max_output_tokens is not None:
            body["max_completion_tokens"] = int(max_output_tokens)

        if reasoning:
            # Xiaomi: you were using thinking enabled/disabled in extra_body
            body["extra_body"] = {"thinking": {"type": "enabled"}}
            print("[LLM CALL][xiaomi]", f"model={model}, thinking=enabled")
        else:
            body["temperature"] = 0.0
            body["extra_body"] = {"thinking": {"type": "disabled"}}
            print(
                "[LLM CALL][xiaomi]",
                f"model={model}, thinking=disabled, temperature=0, max_tokens={body.get('max_completion_tokens')}",
            )

        resp = await client.chat.completions.create(**body)
        text = resp.choices[0].message.content or ""
        if not text:
            raise ValueError("Xiaomi response has no text content.")

        meta = {
            "provider": provider,
            "model": model,
            "reasoning": reasoning,
            "max_tokens": body.get("max_completion_tokens"),
            "temperature": body.get("temperature"),
            "thinking": body.get("extra_body", {}).get("thinking", {}).get("type"),
        }
        return resp, text, meta

    # ---------- OpenRouter ----------
    elif provider == "openrouter":
        client = _get_openrouter_async_client()
        model_norm = normalize_openrouter_model(model)

        if model_norm == "qwen/qwen3-14b" and (not reasoning):
            instruction = f"{instruction}\n/no_thinking"

        messages = [{"role": "user", "content": instruction}]
        body: Dict[str, Any] = {"model": model_norm, "messages": messages}

        if max_output_tokens is not None:
            body["max_tokens"] = int(max_output_tokens)

        if reasoning:
            body["extra_body"] = {"reasoning": {"enabled": True}}
            print(
                "[LLM CALL][openrouter]",
                f"model={model_norm}, reasoning=enabled, max_tokens={max_output_tokens}, effort={reasoning_effort}",
            )
        else:
            body["temperature"] = 0
            body["extra_body"] = {"reasoning": {"enabled": False}}
            print(
                "[LLM CALL][openrouter]",
                f"model={model_norm}, reasoning=disabled, temperature=0, max_tokens={max_output_tokens}",
            )

        resp = await client.chat.completions.create(**body)
        text = resp.choices[0].message.content or ""
        if not text:
            raise ValueError("OpenRouter response has no text content.")

        meta = {
            "provider": provider,
            "model": model_norm,
            "reasoning": reasoning,
            "reasoning_effort": (reasoning_effort if reasoning else None),
            "max_tokens": max_output_tokens,
        }
        return resp, text, meta

    # ---------- Vertex MaaS ----------
    elif provider == "vertex_maas":
        vertex_project_id = get_vertex_project_id()

        # --- Vertex Gemini publisher via REST ---
        if model.startswith("gemini-"):
            location = VERTEX_LOCATION_GEMINI
            model_id = model
            generation_config: Dict[str, Any] = {}

            if "gemini-3" in model_id:
                # Gemini-3: thinkingConfig only if reasoning enabled
                if reasoning:
                    level: Optional[str] = None
                    if reasoning_effort == "low":
                        level = "LOW"
                    elif reasoning_effort in ("medium", "high"):
                        level = "HIGH"
                    generation_config["thinkingConfig"] = {"thinkingLevel": level}
                else:
                    generation_config["temperature"] = 0.0


            think_level = generation_config.get("thinkingConfig", {}).get("thinkingLevel", "disabled")
            temp_val = generation_config.get("temperature", "None")
            print(
                "[LLM CALL][vertex_maas][gemini]",
                f"model={model_id}, location={location},",
                f"reasoning={'enabled' if reasoning else 'disabled'},",
                f"thinkingLevel={think_level},",
                f"temperature={temp_val},",
            )

            contents = [{"role": "user", "parts": [{"text": instruction}]}]
            resp = await _vertex_gemini_generate_content(
                project_id=vertex_project_id,
                location=location,
                model_id=model_id,
                contents=contents,
                generation_config=generation_config,
            )

            text_parts: List[str] = []
            try:
                c0 = (resp.get("candidates") or [])[0]
                parts = ((c0.get("content") or {}).get("parts") or [])
                for p in parts:
                    t = p.get("text")
                    if isinstance(t, str) and t:
                        text_parts.append(t)
            except Exception:
                pass

            text = "\n".join(text_parts).strip()
            if not text:
                raise ValueError(f"Vertex Gemini response has no text content. raw={resp}")

            meta = {
                "provider": provider,
                "model": model,
                "reasoning": reasoning,
                "reasoning_effort": (reasoning_effort if reasoning else None),
                "vertex_location": location,
                "think_level": think_level,
                "temperature": temp_val,
            }
            return resp, text, meta

        # --- Vertex MaaS OpenAI-compatible endpoints (Llama/Qwen etc.) ---
        else:
            if model.startswith("llama-4-"):
                location = VERTEX_LOCATION_LLAMA
                model_norm = "meta/" + model
                max_tokens = max_output_tokens
            elif model.startswith("qwen"):
                location = VERTEX_LOCATION_QWEN
                model_norm = "qwen/" + model
                max_tokens = max_output_tokens
            else:
                location = VERTEX_LOCATION_LLAMA
                model_norm = model
                max_tokens = max_output_tokens

            client = _make_vertex_maas_async_client(
                project_id=vertex_project_id,
                location=location,
                api_version="v1beta1",
            )

            resp = await client.chat.completions.create(
                model=model_norm,
                messages=[{"role": "user", "content": instruction}],
                max_tokens=max_tokens,
                temperature=0.0,
                stream=False,
            )

            print("[LLM CALL][vertex_maas]", f"model={model_norm}, max_tokens={max_tokens}, temperature=0")

            text = resp.choices[0].message.content or ""
            if not text:
                raise ValueError("Vertex MaaS response has no text content.")

            meta = {
                "provider": provider,
                "model": model,
                "reasoning": reasoning,
                "reasoning_effort": (reasoning_effort if reasoning else None),
                "vertex_location": location,
                "max_tokens": max_tokens,
                "temperature": 0,
            }
            return resp, text, meta


    # ---------- Anthropic ----------
    elif provider == "anthropic":
        client = _get_anthropic_async_client()
        msg_kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": instruction}],
        }

        if reasoning:
            # budget validated above
            msg_kwargs["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}  # type: ignore[arg-type]
        else:
            msg_kwargs["temperature"] = 0.0

        temperature = msg_kwargs.get("temperature", "None")
        print(
            "[LLM CALL][anthropic]",
            f"model={model},",
            f"reasoning={'enabled' if reasoning else 'disabled'},",
            f"thinking_budget={budget if reasoning else None},",
            f"temperature={temperature},",
            f"max_tokens={max_output_tokens}"
        )

        resp = await client.messages.create(**msg_kwargs)

        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "".join(text_parts).strip()
        if not text:
            raise ValueError("Anthropic response has no text content.")

        meta = {
            "provider": provider,
            "model": model,
            "reasoning": reasoning,
            "thinking_budget": (budget if reasoning else None),
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        return resp, text, meta

    # ---------- DeepSeek ----------
    elif provider == "deepseek":
        client = _get_deepseek_async_client()
        messages = [{"role": "user", "content": instruction}]

        if reasoning and model == "deepseek-reasoner":
            extra_body: Dict[str, Any] = {"thinking": {"type": "enabled"}}
            print("[LLM CALL][deepseek]", f"model={model}, thinking=enabled, max_tokens={max_output_tokens}")
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                extra_body=extra_body,
                max_tokens=max_output_tokens,
            )
            temperature = None
        else:
            # reasoning disabled OR non-reasoner model
            print("[LLM CALL][deepseek]", f"model={model}, thinking=disabled, temperature=0, max_tokens={max_output_tokens}")
            temperature = 0
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_output_tokens,
                temperature=temperature,
            )

        text = resp.choices[0].message.content or ""
        if not text:
            raise ValueError("DeepSeek response has no content text.")

        meta = {
            "provider": provider,
            "model": model,
            "reasoning": reasoning,
            "reasoning_effort": (reasoning_effort if reasoning else None),
            "max_tokens": max_output_tokens,
            "temperature": temperature,
        }
        return resp, text, meta

    # ---------- Gemini ----------
    elif provider == "gemini":
        client = _get_gemini_client()
        is_g3 = "gemini-3" in model

        thinking_cfg_kwargs: Dict[str, Any] = {}
        if is_g3 and reasoning:
            # g3 supports low/high (you already did that)
            if reasoning_effort == "low":
                thinking_cfg_kwargs["thinking_level"] = "low"
            elif reasoning_effort in ("medium", "high"):
                thinking_cfg_kwargs["thinking_level"] = "high"
            else:
                # enabled but no effort => default high
                thinking_cfg_kwargs["thinking_level"] = "high"

        gen_cfg_kwargs: Dict[str, Any] = {}
        if thinking_cfg_kwargs:
            gen_cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(**thinking_cfg_kwargs)
        else:
            gen_cfg_kwargs["temperature"] = 0.0

        print(
            "[LLM CALL][gemini]",
            f"model={model}, reasoning={'enabled' if reasoning else 'disabled'}, timeout=1200s"
        )

        def _sync_call() -> Any:
            return client.models.generate_content(
                model=model,
                contents=instruction,
                config=genai_types.GenerateContentConfig(**gen_cfg_kwargs),
            )

        try:
            resp = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=1200)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Gemini call timed out after 1200 seconds (model={model})")

        text = getattr(resp, "text", None) or ""
        if not text:
            raise ValueError("Gemini response has no text content.")

        meta = {
            "provider": provider,
            "model": model,
            "reasoning": reasoning,
            "reasoning_effort": (reasoning_effort if reasoning else None),
            "timeout": 1200,
        }
        return resp, text, meta

    else:
        raise ValueError(f"Unknown provider: {provider}")
