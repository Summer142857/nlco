import json
import os
import re

import numpy as np
from typing import List, Dict, Any, Set

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError:
    SentenceTransformer = None

from step2_contextualization.utils.nl_template_utils import  safe_json_loads, \
    build_nl_styles_for_context
from step2_contextualization.utils.llm_service import OpenAILlmService, call_llm
from step2_contextualization.utils.prompt_loader import PromptLoader
MAX_GEN_ITER = 10
from step2_contextualization.utils.llm_service import acall_llm

async def build_scenario_hint_for_context(
    task_name: str,
    task_description: str,
    contexts_with_baseline: List[Dict[str, Any]],
    base_hint: Dict[str, Any],
    llm: OpenAILlmService,
    prompter: PromptLoader,
    part12_templates: Dict[int, str],
) -> Dict[int, Dict[str, Any]]:



    scenario_hints: Dict[int, Dict[str, Any]] = {0: base_hint}

    if len(contexts_with_baseline) == 1:
        return scenario_hints

    async def build_one(i: int, ctx: Dict[str, Any]):
        base_text = ctx.get("text") or "Solve the task as described."
        hint_json = json.dumps(base_hint, ensure_ascii=False, indent=2)

        part12 = part12_templates.get(i, "")

        prompt = prompter.render(
            "scenario_hint.md",
            task_name=task_name,
            task_description=task_description,
            base_text=base_text,
            part12_intro=part12,
            hint_json=hint_json,
        )

        raw = await llm.acomplete(prompt)
        raw = raw.strip()
        print(f"[DEBUG] Scenario hint raw response for ctx[{i}] (first 400 chars):\n"
              f"{raw[:400]}...\n")

        scenario_hint = safe_json_loads(raw)
        return i, scenario_hint

    tasks = [
        build_one(i, ctx)
        for i, ctx in enumerate(contexts_with_baseline[1:], start=1)
    ]

    results = await asyncio.gather(*tasks)

    for i, scen in results:
        scenario_hints[i] = scen

    return scenario_hints

async def generate_diverse_instructions(
    llm: OpenAILlmService,
    prompter: PromptLoader,
    problem_type: str,
    k: int,
    n: int,
    problem_description: str = "",
) -> (List[Dict[str, Any]], Dict[str, Any]):

    prompt_template = prompter.render(
        "instruction_list.md",
        problem_type=problem_type,
        k=k,
        task_description=problem_description or "N/A",
    )

    accepted_texts: List[str] = []
    rejected_texts: Set[str] = set()
    accepted_structured: List[Dict[str, Any]] = []

    per_round_audit: List[Dict[str, Any]] = []
    rounds_run = 0
    SIM_THRESHOLD = 0.7

    for it in range(MAX_GEN_ITER):
        if len(accepted_structured) >= n:
            break
        rounds_run = it + 1

        raw = await llm.acomplete(prompt_template)
        raw_candidates = parse_numbered_list(raw)

        new_raw = [
            s for s in raw_candidates
            if (s not in accepted_texts and s not in rejected_texts)
        ]

        accepted_before = list(accepted_texts)
        cluster_input = accepted_before + new_raw

        kept_to_verify, cluster_removed_details = union_cluster_prune_with_reasons(
            accepted_list=accepted_before,
            fresh_raw=new_raw,
            thr=SIM_THRESHOLD,
        )

        accepted_this_round: List[str] = []
        rejected_this_round: List[str] = []

        if kept_to_verify:
            tasks = [
                verify_problem_semantics(
                    llm=llm,
                    prompter=prompter,
                    text=s,
                    expected_problem=problem_type,
                    problem_description=problem_description,
                )
                for s in kept_to_verify
            ]

            # results: List[(bool, raw)]
            results = await asyncio.gather(*tasks)

            for s, (is_valid, raw_resp) in zip(kept_to_verify, results):
                if is_valid:
                    if s not in accepted_texts:
                        accepted_texts.append(s)
                        accepted_structured.append(
                            {"text": s, "problem_type": problem_type}
                        )
                        accepted_this_round.append(s)
                else:
                    if s not in rejected_texts:
                        rejected_texts.add(s)
                        rejected_this_round.append(s)

        per_round_audit.append({
            "round_index": it,
            "new_raw": new_raw,
            "cluster_input": cluster_input,
            "cluster_removed": cluster_removed_details,
            "kept_to_verify": kept_to_verify,
            "accepted": accepted_this_round,
            "rejected": rejected_this_round,
        })

        if len(accepted_structured) >= n:
            break

    audit = {
        "problem_type": problem_type,
        "rounds_run": rounds_run,
        "per_round": per_round_audit,
        "accepted_total": len(accepted_texts),
        "rejected_total": len(rejected_texts),
        "similarity_threshold": SIM_THRESHOLD,
    }
    return accepted_structured, audit



def _fallback_embed_norm(texts: List[str], dim: int = 512):
    """Lightweight hashed bag-of-words embedding when sentence-transformers is unavailable."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    mat = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        tokens = re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
        if not tokens:
            continue
        for tok in tokens:
            mat[i, hash(tok) % dim] += 1.0
        norm = np.linalg.norm(mat[i])
        if norm > 0:
            mat[i] /= norm
    return mat


def embed_norm(texts, model_name="all-MiniLM-L6-v2"):
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    if SentenceTransformer is None:
        print("[WARN] sentence_transformers not installed, using fallback text embeddings.")
        return _fallback_embed_norm(texts)
    return SentenceTransformer(model_name).encode(texts, normalize_embeddings=True)


def union_cluster_prune_with_reasons(
        accepted_list: List[str],
        fresh_raw: List[str],
        thr: float = 0.84,
):


    if not fresh_raw:
        return [], []

    union = accepted_list + fresh_raw
    n_acc = len(accepted_list)
    n_total = len(union)

    emb = embed_norm(union)
    S = emb @ emb.T
    np.fill_diagonal(S, 0.0)

    kept_fresh: List[str] = []
    removed_details: List[Dict[str, Any]] = []

    pool_idxs = list(range(n_acc))

    for r in range(n_acc, n_total):
        if pool_idxs:
            sims = S[r, pool_idxs]
            best_pos = int(np.argmax(sims))
            best_idx = pool_idxs[best_pos]
            best_sim = float(sims[best_pos])
        else:
            best_idx = None
            best_sim = 0.0

        if best_idx is not None and best_sim >= thr:
            removed_details.append({
                "text": union[r],
                "anchor": union[best_idx],
                "anchor_type": "accepted" if best_idx < n_acc else "new_kept",
                "similarity": best_sim,
            })
        else:
            kept_fresh.append(union[r])
            pool_idxs.append(r)

    return kept_fresh, removed_details



def parse_numbered_list(text: str) -> List[str]:
    out = []
    for line in text.strip().splitlines():
        if "." in line:
            try:
                out.append(line.split(".", 1)[1].strip())
            except Exception:
                pass
    return [s for s in out if s]


async def verify_problem_semantics(
    llm: OpenAILlmService,
    prompter: PromptLoader,
    text: str,
    expected_problem: str,
    problem_description: str,
) -> (bool, str):
    prompt = prompter.render(
        "verifier.md",
        text=text,
        expected_problem=expected_problem.upper(),
        problem_description=problem_description or "N/A",
    )
    raw = await llm.acomplete(prompt)   # ⭐ 用异步接口
    raw = raw.strip()
    if os.getenv("NLCO_DEBUG_VERIFICATION", "").lower() in {"1", "true", "yes", "on"}:
        print(f"[DEBUG] Verification response:\n{raw}\n")

    is_yes = any(
        line.strip().lower().endswith("yes")
        for line in raw.splitlines()
        if line.lower().startswith("answer:")
    )
    return bool(is_yes), raw


import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple

PLACEHOLDER = "{{INSTANCE_INPUT}}"

import re

def _select_one_intro_variant(raw: str) -> str:

    text = raw.strip()

    pattern_paren = re.compile(
        r'\(([A-Z])\)\s*(.*?)(?=(\([A-Z]\))|\Z)', re.S
    )
    matches = [m.group(2).strip() for m in pattern_paren.finditer(text)]
    if matches:
        return random.choice(matches)

    pattern_suffix = re.compile(
        r'([A-Z])\)\s*(.*?)(?=([A-Z]\))|\Z)', re.S
    )
    matches = [m.group(2).strip() for m in pattern_suffix.finditer(text)]
    if matches:
        return random.choice(matches)

    pattern_fullwidth = re.compile(
        r'([A-Z])）\s*(.*?)(?=([A-Z]）)|\Z)', re.S
    )
    matches = [m.group(2).strip() for m in pattern_fullwidth.finditer(text)]
    if matches:
        return random.choice(matches)

    text = re.sub(r'^[A-Z][\)\）]\s*', '', text)
    text = re.sub(r'^\([A-Z]\)\s*', '', text)

    return text.strip()

async def generate_instruction_template(
    contexts_with_baseline: List[Dict[str, Any]],
    baseline_template: str,
    task_name: str,
    output_format: str,
    task_description: str,
    llm,
    prompter,
) -> Dict[str, Dict[int, str]]:

    # idx -> full template
    full_templates: Dict[int, str] = {0: baseline_template}
    # idx -> part 1&2 (casual intro + reference to details, no placeholder)
    part12_templates: Dict[int, str] = {}
    # idx -> part 3&4 (JSON + explanation + closing)
    part34_templates: Dict[int, str] = {}

    if len(contexts_with_baseline) == 1:
        return {
            "full": full_templates,
            "part12": part12_templates,
            "part34": part34_templates,
        }

    async def build_one(i: int, ctx: Dict[str, Any]) -> Tuple[int, str, str, str]:
        base_text = ctx.get("text") or "Solve the task as described."

        prompt_part1 = prompter.render(
            "instruction_template_part1.md",
            base_text=base_text,
            problem_type=task_name.upper(),
            task_description=task_description,
        )
        part12_raw = await llm.acomplete(prompt_part1)
        part12_raw = part12_raw.strip()
        print(f"[DEBUG] template[{i}] part12_raw first 200 chars:\n{part12_raw[:200]}\n")

        part12 = _select_one_intro_variant(part12_raw)
        print(f"[DEBUG] template[{i}] part12_selected first 200 chars:\n{part12[:200]}\n")

        prompt_part2 = prompter.render(
            "instruction_template_part2.md",
            partial_intro=part12,
            problem_type=task_name.upper(),
            output_format=output_format,
            task_description=task_description,
        )
        part34_raw = await llm.acomplete(prompt_part2)
        part34 = part34_raw.strip()
        print(f"[DEBUG] template[{i}] part34 first 200 chars:\n{part34[:200]}\n")

        full = (
            part12.rstrip()
            + "\n\n"
            + PLACEHOLDER
            + "\n\n"
            + part34.lstrip()
        )

        return i, part12, part34, full

    tasks = [
        build_one(i, ctx)
        for i, ctx in enumerate(contexts_with_baseline[1:], start=1)
    ]

    results = await asyncio.gather(*tasks)

    for i, p12, p34, full in results:
        part12_templates[i] = p12
        part34_templates[i] = p34
        full_templates[i] = full

    return {
        "full": full_templates,
        "part12": part12_templates,
        "part34": part34_templates,
    }





async def generate_context(
    task_name: str,
    task_description: str,
    output_root_dir: str,
    llm: OpenAILlmService,
    prompter: PromptLoader,
    k_per_call: int,
    n_target: int,
    hint,
    baseline_template: str,
    output_format,
):
    print(f"[STAGE] Generating contexts for task: {task_name}")

    # STEP 1: Generate diverse instructions
    structured_contexts, contexts_audit_base = await generate_diverse_instructions(
        llm=llm,
        prompter=prompter,
        problem_type=task_name,
        k=k_per_call,
        n=n_target,
        problem_description=task_description,
    )

    if not structured_contexts:
        print(f"[WARN] No valid instructions generated for {task_name}, aborting.")
        return None

    baseline_ctx = {
        "text": "[BASELINE] Canonical task wording.",
        "problem_type": task_name,
    }
    contexts_with_baseline = [baseline_ctx] + structured_contexts

    os.makedirs(output_root_dir, exist_ok=True)

    # STEP 2: Build templates (baseline + 1 per context)
    # {
    #   "full":  {idx: full_template_str},
    #   "part12": {idx: part12_str},
    #   "part34": {idx: part34_str},
    # }
    template_bundle = await generate_instruction_template(
        contexts_with_baseline=contexts_with_baseline,
        baseline_template=baseline_template,
        task_name=task_name,
        output_format=output_format,
        task_description=task_description,
        llm=llm,
        prompter=prompter,
    )

    full_templates: Dict[int, str] = template_bundle["full"]
    part12_templates: Dict[int, str] = template_bundle["part12"]
    part34_templates: Dict[int, str] = template_bundle["part34"]

    # STEP 3: build scenario hints
    scenario_hints = await build_scenario_hint_for_context(
        task_name=task_name,
        task_description=task_description,
        contexts_with_baseline=contexts_with_baseline,
        base_hint=hint,
        llm=llm,
        prompter=prompter,
        part12_templates=part12_templates,
    )

    # STEP 4: build nl styles
    nl_styles = await build_nl_styles_for_context(
        task_name=task_name,
        precomputed_templates=part12_templates,
        scenario_hints=scenario_hints,
        llm=llm,
        prompter=prompter,
    )

    contexts_audit = dict(contexts_audit_base)

    contexts_audit["precomputed_templates"] = {
        "full": {str(i): t for i, t in full_templates.items()},
        "part12": {str(i): t for i, t in part12_templates.items()},
        "part34": {str(i): t for i, t in part34_templates.items()},
    }

    contexts_audit["nl_styles"] = {
        str(i): style for (task, i), style in nl_styles.items()
    }
    contexts_audit["contexts"] = contexts_with_baseline

    contexts_audit["scenario_hints"] = {
        str(i): sh for i, sh in scenario_hints.items()
    }

    audit_path = os.path.join(
        output_root_dir,
        f"{task_name}_contexts.json",
    )
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(contexts_audit, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote contexts audit: {audit_path}")

    return {
        "contexts": contexts_with_baseline,
        "precomputed_templates": template_bundle,
        "nl_styles": nl_styles,
        "scenario_hints": scenario_hints,
        "audit": contexts_audit,
        "audit_path": audit_path,
    }


