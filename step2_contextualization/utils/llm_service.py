
import os
import re
import json
from pathlib import Path
from typing import Optional, List, Any

try:
    from openai import OpenAI, AsyncOpenAI
except ModuleNotFoundError:
    OpenAI = None
    AsyncOpenAI = None


class OpenAILlmService:
    @staticmethod
    def _load_api_key_from_repo_config() -> Optional[str]:
        repo_root = Path(__file__).resolve().parents[2]
        key_path = repo_root / "step3_evaluation" / "configs" / "keys.json"
        if not key_path.exists():
            return None

        try:
            with open(key_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        key = data.get("openai")
        if isinstance(key, str):
            key = key.strip()
            return key or None
        return None

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or self._load_api_key_from_repo_config()
        )
        self.client = None
        self.async_client = None

        if OpenAI is not None and AsyncOpenAI is not None:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)

    def _ensure_clients(self):
        if self.client is None or self.async_client is None:
            raise RuntimeError(
                "OpenAI SDK is not installed or OPENAI_API_KEY is missing. "
                "Install `openai` and set OPENAI_API_KEY, or use cached contexts."
            )

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self._ensure_clients()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop,
            #temperature=temperature if temperature is not None else self.temperature,
        )
        return resp.choices[0].message.content.strip()

    async def acomplete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
    ) -> str:
        self._ensure_clients()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=stop,
            #temperature=temperature if temperature is not None else self.temperature,
        )
        return resp.choices[0].message.content.strip()



def call_llm(llm: OpenAILlmService, prompt: str, **kwargs) -> str:
    return llm.complete(prompt, **kwargs)


async def acall_llm(llm: OpenAILlmService, prompt: str, **kwargs) -> str:
    return await llm.acomplete(prompt, **kwargs)




