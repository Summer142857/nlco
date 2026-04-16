
import json
import asyncio
import re
import time
from typing import Any, Dict, Optional, Callable, Literal

import os

import numpy as np
import pandas as pd
from pandas.io.clipboard import paste

from step3_evaluation.utils.eval_parsing import _normalize_decision
from step3_evaluation.utils.llm import (
    Provider, safe_json_loads, call_llm,
)
from step3_evaluation.utils.objective_map import ObjectiveSense, OBJECTIVE_SENSE_MAP


def normalize_eval_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add compatibility aliases so evaluators written against the older Step 2 schema
    can operate on the normalized NLCO release schema.
    """
    df = df.copy()

    alias_map = {
        "instruction": "prompt",
        "context_index": "example_index",
        "problem_type": "task_id",
        "task_name": "task_id",
        "obj": "reference_objective_value",
        "instance": "instance_canonical_json",
        "solution": "reference_solution_canonical_json",
        "instance_variant": "instance_surface_json",
        "solution_variant": "reference_solution_surface_json",
        "input_format": "surface_format",
        "input_index_base": "indexing_scheme",
    }

    for target, source in alias_map.items():
        if target not in df.columns and source in df.columns:
            df[target] = df[source]

    if "context_index" in df.columns:
        try:
            df["context_index"] = df["context_index"].astype(int)
        except Exception:
            pass

    return df


def reevaluate_from_existing_csv(
    existing_csv: str,
    output_csv: str,
    *,
    decision_field: str,
    parse_instance_fn: Callable[[pd.Series], Any],
    feasibility_fn: Callable[[Any, Any], Any],
    objective_model_fn: Callable[[Any, Any], float],
    model_name: str,
) -> None:

    df = normalize_eval_dataframe(pd.read_csv(existing_csv))
    new_rows: list[Dict[str, Any]] = []

    # Helper: safe column access
    def _get_col(row: pd.Series, col: str) -> Any:
        return row[col] if col in df.columns else None

    for _, row in df.iterrows():
        instance = parse_instance_fn(row)

        raw_text = row.get("model_raw", "") or ""

        # Read token stats if present
        input_tokens = _get_col(row, "input_tokens")
        output_tokens = _get_col(row, "output_tokens")
        reasoning_tokens = _get_col(row, "reasoning_tokens")

        # Load llm_meta from CSV if present (usually stored as JSON string)
        llm_meta = _get_col(row, "llm_meta")


        # Parse JSON from saved model_raw
        if isinstance(raw_text, str) and raw_text.strip():
            try:
                parsed_json = safe_json_loads(raw_text)
                print("parsed_json: ", parsed_json)
            except Exception as e:
                parsed_json = {
                    "_error": "json_parse_failed",
                    "_detail": f"{type(e).__name__}: {e!s}",
                }
        else:
            parsed_json = {
                "_error": "llm_call_failed",
                "_detail": "model_raw is empty in CSV",
            }

        parsed_decision = (
            parsed_json.get(decision_field, None)
        )

        row_result = _evaluate_from_parsed_json(
            row=row,
            instance=instance,
            parsed_json=parsed_json,
            decision_field=decision_field,
            feasibility_fn=feasibility_fn,
            objective_model_fn=objective_model_fn,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            llm_meta=llm_meta,
        )

        new_rows.append(
            {
                "feasible": row_result.get("feasible"),
                "feasibility_reason": row_result.get("feasibility_reason"),
                "pattern_format": row_result.get("pattern_format"),
                "obj_model": row_result.get("obj_model"),
                "obj_opt": row_result.get("obj_opt"),
                "gap_ratio": row_result.get("gap_ratio"),
                "log_gap": row_result.get("log_gap"),
                "model_decision": parsed_decision,
            }
        )

    new_eval_df = pd.DataFrame(new_rows, index=df.index)

    # Overwrite / create columns
    for col in [
        "feasible",
        "feasibility_reason",
        "pattern_format",
        "obj_model",
        "obj_opt",
        "log_gap",
        "gap_ratio",
        "model_decision",
    ]:
        df[col] = new_eval_df[col]

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"[RE-EVAL] Re-evaluated from existing CSV (no LLM calls): {output_csv}")
    print(f"[INFO] Number of rows: {len(df)}")





def _evaluate_from_parsed_json(
    row: pd.Series,
    instance: Any,
    parsed_json: Dict[str, Any],
    *,
    decision_field: str,
    feasibility_fn: Callable[[Any, Any], Any],
    objective_model_fn: Callable[[Any, Any], float],
    model_name: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    llm_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    instruction = row["instruction"]
    chars_in = len(instruction)

    # ---- normalize llm_meta: string -> dict ----
    if llm_meta is None:
        llm_meta = {}

    elif isinstance(llm_meta, dict):
        # already correct
        pass

    elif isinstance(llm_meta, str):
        s = llm_meta.strip()

        if not s:
            llm_meta = {}
        else:
            # 1) try JSON (recommended format)
            try:
                obj = safe_json_loads(s)
                llm_meta = obj if isinstance(obj, dict) else {}
            except Exception:
                # 2) fallback: Python-literal dict string (single quotes)
                try:
                    obj = eval(s, {"__builtins__": {}})
                    llm_meta = obj if isinstance(obj, dict) else {}
                except Exception:
                    llm_meta = {}

    else:
        # last-resort conversion
        try:
            llm_meta = dict(llm_meta)
        except Exception:
            llm_meta = {}
    # ---------- 1) Handle JSON parse / call errors ----------
    if "_error" in parsed_json:
        err = parsed_json.get("_error")       # "json_parse_failed" or "llm_call_failed"
        detail = parsed_json.get("_detail")   # exception message or call_error string

        # Extract limits from meta:
        # - max_tokens: output_tokens limit
        # - budget: reasoning_tokens limit (if present)
        output_limit = None
        reasoning_limit = None
        llm_meta = dict(llm_meta)
        if isinstance(llm_meta, dict):
            # Your contract: meta only contains these two knobs (global per run / provider config)
            output_limit = llm_meta.get("max_tokens")
            print(output_limit)
            reasoning_limit = llm_meta.get("budget")

            # Best-effort int cast
            try:
                if output_limit is not None:
                    output_limit = int(output_limit)
            except Exception:
                output_limit = None
            try:
                if reasoning_limit is not None:
                    reasoning_limit = int(reasoning_limit)
            except Exception:
                reasoning_limit = None

        pattern_format = err or "unknown_error"

        if err == "json_parse_failed":
            # Determine whether this parse failure is likely due to truncation.
            # Rule:
            # - output_tokens near output_limit => truncated
            # - reasoning_tokens near reasoning_limit (if budget exists) => truncated
            # Otherwise => non_json (model did not follow JSON format)
            near_output_limit = False
            near_reasoning_limit = False

            if output_limit is not None and output_limit > 0 and output_tokens is not None:
                near_output_limit = (output_tokens >= 0.95 * output_limit)

            if reasoning_limit is not None and reasoning_limit > 0 and reasoning_tokens is not None:
                near_reasoning_limit = (reasoning_tokens >= 0.95 * reasoning_limit)

            if near_output_limit or near_reasoning_limit:
                pattern_format = "OutputTruncated"
                feasibility_reason = (
                    "No valid JSON output (likely truncated: tokens near limit)."
                )
                # Add evidence for debugging/analysis
                evidence = []
                if near_output_limit:
                    evidence.append(f"output_tokens={output_tokens}~limit={output_limit}")
                if near_reasoning_limit:
                    evidence.append(f"reasoning_tokens={reasoning_tokens}~budget={reasoning_limit}")
                if evidence:
                    feasibility_reason += " evidence=" + ",".join(evidence)
            else:
                pattern_format = "FormatError"
                feasibility_reason = (
                    "No valid JSON output (model output is not valid JSON)."
                )
                # If limits are missing, be explicit so you know why truncation was not attributed
                if output_limit is None and reasoning_limit is None:
                    feasibility_reason += " note=no_limits_in_meta"

            if detail:
                feasibility_reason = f"{feasibility_reason} detail={detail}"

        elif err == "llm_call_failed":
            pattern_format = "llm_call_failed"
            feasibility_reason = "LLM call failed."
            if detail:
                feasibility_reason = f"{feasibility_reason} detail={detail}"

        else:
            feasibility_reason = "Invalid model output."
            if detail:
                feasibility_reason = f"{feasibility_reason} detail={detail}"

        return {
            "model_name": model_name,
            "model_decision": None,
            "feasible": False,
            "feasibility_reason": feasibility_reason,
            "pattern_format": pattern_format,
            "obj_model": None,
            "obj_opt": None,
            "gap_ratio": None,
            "input_tokens": input_tokens,
            "chars_in": chars_in,
        }
    # ---------- 2) Extract decision ----------
    decision = parsed_json.get(decision_field, None)
    #print(parsed_json)
    #print(decision)
    decision = _normalize_decision(decision)

    if decision is None:
        return {
            "model_name": model_name,
            "model_decision": None,
            "feasible": False,
            "feasibility_reason": f"Missing decision field '{decision_field}' in JSON output",
            "pattern_format": "FormatError",
            "obj_model": None,
            "obj_opt": None,
            "gap_ratio": None,
            "input_tokens": input_tokens,
            "chars_in": chars_in,
        }

    # ---------- 3) Feasibility check ----------
    try:
        is_feasible, pattern_format, reason = feasibility_fn(decision, instance)
    except Exception as e:
        return {
            "model_name": model_name,
            "model_decision": decision,
            "feasible": False,
            "feasibility_reason": f"feasibility_fn_error: {e}",
            "pattern_format": "feasibility_fn_error",
            "obj_model": None,
            "obj_opt": None,
            "gap_ratio": None,
            "input_tokens": input_tokens,
            "chars_in": chars_in,
        }

    result: Dict[str, Any] = {
        "model_name": model_name,
        "model_decision": decision,
        "feasible": is_feasible,
        "feasibility_reason": reason,
        "pattern_format": pattern_format,
        "obj_model": None,
        "obj_opt": None,
        "gap_ratio": None,
        "input_tokens": input_tokens,
        "chars_in": chars_in,
    }

    if not is_feasible:
        return result

    # ---------- 4) Objective ----------
    try:
        obj_model = float(objective_model_fn(decision, instance))
    except Exception as e:
        result["feasible"] = False
        result["feasibility_reason"] = f"objective_model_fn_error: {e}"
        result["pattern_format"] = "objective_model_fn_error"
        return result

    obj_opt = row.get("obj", None)
    result["obj_model"] = obj_model
    result["obj_opt"] = obj_opt

    # ---------- 5) gap_ratio (sense-aware) ----------
    problem_type = row.get("problem_type", None)
    if problem_type is None:
        result["feasible"] = False
        result["feasibility_reason"] = "Missing 'problem_type' in row; cannot infer objective sense."
        result["pattern_format"] = "missing_problem_type"
        return result

    if problem_type not in OBJECTIVE_SENSE_MAP:
        result["feasible"] = False
        result["feasibility_reason"] = f"Unknown problem_type {problem_type!r}; please add it to OBJECTIVE_SENSE_MAP."
        result["pattern_format"] = "unknown_problem_type"
        return result

    objective_sense: ObjectiveSense = OBJECTIVE_SENSE_MAP[problem_type]
    eps = 1e-2
    gap_ratio = None
    if obj_opt is not None:
        obj_opt_f = float(obj_opt)
        if abs(obj_opt_f) < 1e-9:
            # Handle zero optimal value
            if abs(obj_model - obj_opt_f) < 1e-9:
                gap_ratio = 0.0
            else:
                den = max(abs(obj_opt_f), eps)
                # If optimal is 0 and model is not 0, gap is undefined/infinite
                if objective_sense == "max":
                    gap_ratio = (obj_opt_f - obj_model) / den
                else:  # "min"
                    gap_ratio = (obj_model - obj_opt_f) / den
        else:
            if objective_sense == "max":
                gap_ratio = (obj_opt_f - obj_model) / obj_opt_f
            else:  # "min"
                gap_ratio = (obj_model - obj_opt_f) / obj_opt_f
            
            if gap_ratio is not None and gap_ratio < 0:
                gap_ratio = 0.0

    log_gap = None
    try:
        if gap_ratio is not None and np.isfinite(gap_ratio) and gap_ratio >= 0:
            log_gap = float(np.log1p(gap_ratio))  # log(1+gap)
    except Exception:
        log_gap = None
    result["log_gap"] = log_gap
    result["gap_ratio"] = gap_ratio

    return result






async def evaluate_single_row_generic(
    row: pd.Series,
    *,
    decision_field: str,
    parse_instance_fn: Callable[[pd.Series], Any],
    feasibility_fn: Callable[[Any, Any], Any],
    objective_model_fn: Callable[[Any, Any], float],
    model_name: str,
    provider: Optional[Provider],
    reasoning: bool,
    reasoning_effort: Optional[Literal["low", "medium", "high"]],
    anthropic_budget: Optional[int],
    max_tokens: int,
) -> Dict[str, Any]:

    instruction = row["instruction"]
    instance = parse_instance_fn(row)

    resp = None
    raw_text: Optional[str] = None

    call_error: Optional[str] = None
    llm_meta: Dict[str, Any] = {}

    # 1) Call LLM (text only)
    try:
        if provider == "openai":
            resp, raw_text, llm_meta = await call_llm(
                instruction,
                model=model_name,
                provider="openai",
                reasoning_effort=reasoning_effort,
                reasoning=reasoning,
                max_output_tokens=None,
                budget=None,
            )

        elif provider == "deepseek":
            resp, raw_text, llm_meta = await call_llm(
                instruction=instruction,
                model=model_name,
                max_output_tokens=max_tokens,
                reasoning = reasoning,

                provider="deepseek",
            )

        elif provider == "gemini":
            resp, raw_text, llm_meta = await call_llm(
                instruction=instruction,
                model=model_name,
                reasoning_effort=reasoning_effort,
                reasoning=reasoning,
                provider="gemini",
            )

        elif provider == "vertex_maas":
            resp, raw_text, llm_meta = await call_llm(
                instruction=instruction,
                model=model_name,
                max_output_tokens=max_tokens,
                reasoning = reasoning,

                reasoning_effort=reasoning_effort,
                provider="vertex_maas",
            )

        elif provider == "openrouter":
            resp, raw_text, llm_meta = await call_llm(
                instruction=instruction,
                model=model_name,
                reasoning=reasoning,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_tokens,

                provider="openrouter",
            )
        elif provider == "xiaomi":
            resp, raw_text, llm_meta = await call_llm(
                instruction=instruction,
                model=model_name,
                reasoning=reasoning,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_tokens,

                provider="xiaomi",
            )

        else:  # anthropic
            resp, raw_text, llm_meta = await call_llm(
                instruction,
                model=model_name,
                reasoning=reasoning,
                provider="anthropic",
                max_output_tokens=max_tokens,
                budget=anthropic_budget,
            )

    except Exception as e:
        call_error = f"{type(e).__name__}: {e!s}"

    # 2) Parse JSON
    if raw_text is not None:
        try:
            parsed_json = safe_json_loads(raw_text)
        except Exception as e:
            parse_error = f"{type(e).__name__}: {e!s}"
            head = raw_text[:500]
            tail = raw_text[-500:]
            lines = raw_text.splitlines()
            line_855 = lines[854][:300] if len(lines) >= 855 else None

            parsed_json = {
                "_error": "json_parse_failed",
                "_detail": parse_error,
                "_raw_head": head,
                "_raw_tail": tail,
                "_num_lines": len(lines),
                "_line_855": line_855,
            }
    else:
        parsed_json = {
            "_error": "llm_call_failed",
            "_detail": call_error,
        }

    # 3) Token usage extraction
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None

    if resp is not None:
        if provider == "openai":
            usage = getattr(resp, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

                otd = getattr(usage, "output_tokens_details", None)
                if otd is not None:
                    reasoning_tokens = getattr(otd, "reasoning_tokens", None)
                    if reasoning_tokens is None and isinstance(otd, dict):
                        reasoning_tokens = otd.get("reasoning_tokens")

        elif provider == "anthropic":
            usage = getattr(resp, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                if input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens

        elif provider == "openrouter":
            usage = getattr(resp, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                print("openrouter", input_tokens)
                ctd = getattr(usage, "completion_tokens_details", None)
                if ctd is not None:
                    reasoning_tokens = getattr(ctd, "reasoning_tokens", None)
                if reasoning_tokens is None and isinstance(ctd, dict):
                    reasoning_tokens = ctd.get("reasoning_tokens")
        elif provider == "xiaomi":
            usage = getattr(resp, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                ctd = getattr(usage, "completion_tokens_details", None)
                if ctd is not None:
                    reasoning_tokens = getattr(ctd, "reasoning_tokens", None)
                if reasoning_tokens is None and isinstance(ctd, dict):
                    reasoning_tokens = ctd.get("reasoning_tokens")

        elif provider == "deepseek":
            usage = getattr(resp, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                if total_tokens is None and input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens

                ctd = getattr(usage, "completion_tokens_details", None)
                if ctd is not None:
                    reasoning_tokens = getattr(ctd, "reasoning_tokens", None)
                if reasoning_tokens is None and isinstance(ctd, dict):
                    reasoning_tokens = ctd.get("reasoning_tokens")


        elif provider == "vertex_maas":
            usage = getattr(resp, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)
                if total_tokens is None and input_tokens is not None and output_tokens is not None:
                    total_tokens = input_tokens + output_tokens
                ctd = getattr(usage, "completion_tokens_details", None)
                if ctd is not None:
                    reasoning_tokens = getattr(ctd, "reasoning_tokens", None)
                if reasoning_tokens is None and isinstance(ctd, dict):
                    reasoning_tokens = ctd.get("reasoning_tokens")

            elif isinstance(resp, dict):
                usage_md = resp.get("usageMetadata") or {}
                input_tokens = usage_md.get("promptTokenCount")
                reasoning_tokens = usage_md.get("thoughtsTokenCount")
                candidates_tokens = usage_md.get("candidatesTokenCount")
                total_tokens = usage_md.get("totalTokenCount")

                if total_tokens is not None and input_tokens is not None:
                    output_tokens = total_tokens - input_tokens
                else:
                    tt = reasoning_tokens or 0
                    ct = candidates_tokens or 0
                    est = tt + ct
                    output_tokens = est if est > 0 else None

        elif provider == "gemini":
            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                input_tokens = getattr(usage, "prompt_token_count", None)
                reasoning_tokens = getattr(usage, "thoughts_token_count", None)
                candidates_tokens = getattr(usage, "candidates_token_count", None)
                total_tokens = getattr(usage, "total_token_count", None)
                if total_tokens is not None and input_tokens is not None:
                    output_tokens = total_tokens - input_tokens
                else:
                    tt = reasoning_tokens or 0
                    ct = candidates_tokens or 0
                    est = tt + ct
                    output_tokens = est if est > 0 else None

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    base = _evaluate_from_parsed_json(
        row=row,
        instance=instance,
        parsed_json=parsed_json,
        decision_field=decision_field,
        feasibility_fn=feasibility_fn,
        objective_model_fn=objective_model_fn,
        model_name=model_name,
        input_tokens=input_tokens,
        reasoning_tokens=reasoning_tokens,
        output_tokens=output_tokens,
        llm_meta=llm_meta,
    )

    base["output_tokens"] = output_tokens
    base["total_tokens"] = total_tokens

    if provider != "anthropic":
        base["reasoning_tokens"] = reasoning_tokens


    # 5) Always store raw + meta + instance
    base["model_raw"] = raw_text
    base["llm_meta"] = llm_meta
    base["instance_parsed"] = instance

    return base


def _augment_instruction_for_reasoning(
    instr: str,
    *,
    model_name: str,
    provider: Provider,
    reasoning_effort: Optional[Literal["low", "medium", "high"]],
    anthropic_budget: Optional[int],
) -> str:
    """
    Append a small instruction suffix.
    - Anthropic: reasoning is controlled by anthropic_budget (thinking budget).
    - Non-Anthropic: reasoning is controlled by reasoning_effort.
    """
    if not isinstance(instr, str):
        instr = "" if instr is None else str(instr)

    if provider == "anthropic":
        if anthropic_budget is not None:
            # Anthropic reasoning ON
            return instr + "\n\nPlease output only the JSON in the required format and nothing else."
        else:
            # Anthropic reasoning OFF
            return (
                    instr
                    + "\n\nPlease think step by step, output your reasoning trace, and then output the JSON in the required format."
            )
    elif provider =="deepseek":
        if model_name =="deepseek-reasoner":
            return instr + "\n\nPlease output only the JSON in the required format and nothing else."
        else:
            return (
                    instr
                    + "\n\nPlease think step by step, output your reasoning trace, and then output the JSON in the required format."
            )

    else:
        # Non-Anthropic providers: reasoning controlled by reasoning_effort
        if reasoning_effort is not None:
            return instr + "\n\nPlease output only the JSON in the required format and nothing else."
        else:
            # Anthropic reasoning OFF
            return (
                    instr
                    + "\n\nPlease think step by step, output your reasoning trace, and then output the JSON in the required format."
            )



def _build_reasoning_columns(
    *,
    model_name: str,
    provider: Provider,
    reasoning_effort: Optional[Literal["low", "medium", "high"]],
    anthropic_budget: Optional[int],
) -> Dict[str, Any]:

    if provider == "anthropic":
        # Anthropic: reasoning iff thinking.budget_tokens is set
        reasoning_enabled = anthropic_budget is not None

    elif provider == "deepseek":
        # DeepSeek: only deepseek-reasoner actually reasons
        reasoning_enabled = (model_name == "deepseek-reasoner")

    else:
        # OpenAI / Gemini / others:
        # reasoning explicitly controlled by reasoning_effort
        reasoning_enabled = reasoning_effort is not None

    return {
        "reasoning_enabled": reasoning_enabled,
    }


async def evaluate_csv_generic_async(
    input_csv: str,
    output_csv: str,
    *,
    task_name: Optional[str],
    decision_field: str,
    parse_instance_fn: Callable[[pd.Series], Any],
    feasibility_fn: Callable[[Any, Any], Any],
    objective_model_fn: Callable[[Any, Any], float],
    model_name: str,
    max_concurrency: int,
    provider_override: Optional[Provider] = None,
    reasoning: bool,
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    anthropic_budget: Optional[int] = None,
    max_tokens: int = 20000,
    num_test: Optional[int] = None,
):

    base = os.path.basename(input_csv)
    size_match = re.search(r"_(S|M|L)(?:_context)?\.csv$", base, flags=re.IGNORECASE)
    if size_match:
        problem_size = size_match.group(1).upper()
    else:
        problem_size = None

    df = normalize_eval_dataframe(pd.read_csv(input_csv))

    # Optional filter by task_name
    if task_name is not None and "task_name" in df.columns:
        df = df[df["task_name"] == task_name].copy()
        print(df)
    
    # Optional limit to first num_test instances (for quick testing)
    if num_test is not None and num_test > 0:
        original_count = len(df)
        df = df.head(num_test).copy()
        print(f"[INFO] Limited to first {num_test} instances (out of {original_count} total)")

    # Keep original row index to map back
    df = df.reset_index(drop=False).rename(columns={"index": "orig_index"})
    prov: Provider = provider_override

    # Persist modified instruction into df so it will be written into the output CSV
    if "instruction" in df.columns:
        df["instruction"] = df["instruction"].apply(
            lambda s: _augment_instruction_for_reasoning(
                s,
                model_name=model_name,
                provider=prov,
                reasoning_effort=reasoning_effort,
                anthropic_budget=anthropic_budget,
            )
        )
    if "problem_size" not in df.columns:
        if problem_size is not None:
            df["problem_size"] = problem_size
        else:
            df["problem_size"] = "UNKNOWN"

    semaphore = asyncio.Semaphore(max_concurrency)

    async def job(row: pd.Series):
        async with semaphore:
            return await evaluate_single_row_generic(
                row,
                decision_field=decision_field,
                parse_instance_fn=parse_instance_fn,
                feasibility_fn=feasibility_fn,
                objective_model_fn=objective_model_fn,
                model_name=model_name,
                provider=prov,
                reasoning=reasoning,
                reasoning_effort=reasoning_effort,
                anthropic_budget=anthropic_budget,
                max_tokens=max_tokens,
            )

    tasks = [job(row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    reasoning_col = _build_reasoning_columns(
        provider=prov,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        anthropic_budget=anthropic_budget,
    )
    out_df["reasoning_enabled"] = reasoning_col["reasoning_enabled"]

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"[OK] Evaluation written to: {output_csv}")
    print(f"[INFO] Number of evaluated rows: {len(out_df)}")
    print(f"[INFO] Model used: {model_name}")
