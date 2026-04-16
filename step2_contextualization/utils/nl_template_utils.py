import json
from typing import Dict, Any

from step2_contextualization.utils.llm_service import OpenAILlmService, call_llm
from step2_contextualization.utils.prompt_loader import PromptLoader
import json
import re

import json
import re

def safe_json_loads(raw: str):
    if raw is None:
        raise ValueError("safe_json_loads: raw is None")

    text = raw.strip()

    # ---- 1. Remove code fences ----
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            inner = parts[1].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            text = inner

    # ---- 2. Pre-cleaning: fix common LLM JSON errors ----

    # 2a. Fix bare newlines inside JSON strings:
    # Replace `" ... \n ... "` with `" ...  ... "`
    def fix_string_newlines(s):
        pattern = r'"([^"\\]*(?:\\.[^"\\]*)*)"'  # match JSON strings
        def repl(m):
            content = m.group(1)
            fixed = content.replace("\n", " ").replace("\r", " ")
            return f"\"{fixed}\""
        return re.sub(pattern, repl, s)

    text = fix_string_newlines(text)

    # 2b. Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)

    # ---- 3. Try direct load ----
    try:
        return json.loads(text)
    except Exception:
        pass

    # ---- 4. Extract first JSON object or array ----
    match = re.search(r'(\{.*\}|\[.*\])', text, flags=re.DOTALL)
    if match:
        json_candidate = match.group(1)

        # Also fix newlines in that candidate
        json_candidate = fix_string_newlines(json_candidate)
        json_candidate = re.sub(r',\s*([}\]])', r'\1', json_candidate)

        try:
            return json.loads(json_candidate)
        except Exception:
            pass

    # ---- 5. If still fails, print debug ----
    print("[ERROR] safe_json_loads: Could not parse JSON. Raw text:")
    print(raw)
    raise ValueError("safe_json_loads: Invalid JSON structure")


import asyncio
from typing import Dict, Any
async def build_nl_styles_for_context(
    task_name: str,
    precomputed_templates: Dict[int, str],
    scenario_hints: Dict[int, Dict[str, Any]],
    llm: OpenAILlmService,
    prompter: PromptLoader,
) -> Dict[tuple, Dict[str, Any]]:


    nl_styles: Dict[tuple, Dict[str, Any]] = {}

    async def build_one(i: int, tmpl: str):
        data_shape_hint = scenario_hints[i]

        prompt = prompter.render(
            "input_nl_template.md",
            problem_type=task_name,
            instruction_template=tmpl,
            data_shape_hint=data_shape_hint,
        )

        raw = await llm.acomplete(prompt)
        print(f"[DEBUG] Raw NL style output for {i}:\n{raw[:400]}...\n")

        parsed = safe_json_loads(raw)
        return (task_name.upper(), i), parsed

    tasks = [
        build_one(i, tmpl)
        for i, tmpl in precomputed_templates.items()
    ]

    results = await asyncio.gather(*tasks)

    for key, val in results:
        nl_styles[key] = val

    return nl_styles


def _format_safe(template, **kwargs):
    from string import Formatter
    if not template:
        return ""
    fields = {fname for _, fname, _, _ in Formatter().parse(template) if fname}
    safe_kwargs = {k: v for k, v in kwargs.items() if k in fields}
    for f in fields:
        safe_kwargs.setdefault(f, "")
    return template.format(**safe_kwargs)
import string

def make_labels(n, index_base):
    if index_base == 0:
        return list(range(n))
    if index_base == 1:
        return [i + 1 for i in range(n)]

    # alphabetical
    labels = []
    chars = string.ascii_uppercase
    base = len(chars)
    for i in range(n):
        s, x = "", i
        while True:
            s = chars[x % base] + s
            x = x // base - 1
            if x < 0:
                break
        labels.append(s)
    return labels

