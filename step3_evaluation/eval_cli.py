# step3_evaluation/eval_cli.py

import argparse
import asyncio
import os
import re
from typing import Dict, Callable, List, Optional, Literal

import pandas as pd
from step3_evaluation.utils.problems_registry import PROBLEM_REGISTRY
from step3_evaluation.utils.eval_generic import (
    evaluate_csv_generic_async, reevaluate_from_existing_csv, normalize_eval_dataframe,
)
from step3_evaluation.utils.llm import Provider
from step3_evaluation.config import ALL_PROBLEMS, DEFAULT_CONCURRENCY


def parse_sizes_arg(sizes_arg: str) -> List[str]:
    """
    Parse sizes like:
      "S"          -> ["S"]
      "S,M,L"      -> ["S", "M", "L"]
      "all"        -> ["S", "M", "L"]
    """
    sizes_arg = sizes_arg.strip().lower()
    if sizes_arg == "all":
        return ["S", "M", "L"]
    parts = [p for p in re.split(r"[,\s]+", sizes_arg) if p]
    return [p.upper() for p in parts]


def parse_problems_arg(problems_arg: str) -> List[str]:
    """
    Parse problems like:
      "TSP"           -> ["TSP"]
      "TSP,CVRP,MIS"  -> ["TSP", "CVRP", "MIS"]
      "all"           -> all problems from ALL_PROBLEMS
    """
    problems_arg = problems_arg.strip()
    if problems_arg == "all":
        return ALL_PROBLEMS
    parts = [p for p in re.split(r"[,\s]+", problems_arg) if p]
    return [p for p in parts]


async def run_for_problem(
    problem: str,
    sizes: List[str],
    dataset_root: str,
    output_root: str,
    model: str,
    max_concurrency: int,
    provider: Optional[Provider] = None,
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None,
    anthropic_budget: Optional[int] = None,
    max_tokens: int = 2048,
    decision_field: str = "solution",
    reasoning: bool = False,
    without_reasoning: bool = False,
    num_test: Optional[int] = None,
):
    if problem not in PROBLEM_REGISTRY:
        raise ValueError(
            f"Unknown problem type: {problem}. "
            f"Available: {list(PROBLEM_REGISTRY.keys())}"
        )

    cfg = PROBLEM_REGISTRY[problem]
    print(cfg)
    parse_instance_fn = cfg["parse_instance_fn"]
    feasibility_fn = cfg["feasibility_fn"]
    objective_model_fn = cfg["objective_model_fn"]

    per_size_outputs: List[str] = []
    model_dir_name = model
    if without_reasoning:
        model_dir_name = f"{model}__no_reasoning"
    safe_model = model_dir_name.replace(":", "_").replace("/", "_")

    for size in sizes:
        size = size.upper()

        input_csv = os.path.join(
            dataset_root,
            problem,
            f"{problem}_{size}_context.csv",
        )

        if not os.path.exists(input_csv):
            print(f"[WARN] Input CSV not found, skip: {input_csv}")
            continue

        output_dir = os.path.join(output_root, problem, model_dir_name)
        os.makedirs(output_dir, exist_ok=True)

        output_csv = os.path.join(output_dir, f"{problem}_{size}_eval.csv")

        # If output exists, check whether it already has enough rows (default 50 or num_test)
        # If output exists, resume / re-run failed rows
        if os.path.exists(output_csv):
            target_n = int(num_test) if num_test is not None else 50

            try:
                existing_df = normalize_eval_dataframe(pd.read_csv(output_csv))
            except Exception as e:
                print(f"[WARN] Failed to read existing CSV {output_csv}: {e}. Will re-run from scratch.")
                existing_df = pd.DataFrame()

            # ---- drop llm_call_failed rows and re-run them ----
            failed_ids = set()
            clean_existing_df = existing_df

            failed_col = "pattern_format"
            if (not existing_df.empty) and (failed_col in existing_df.columns) and ("context_index" in existing_df.columns):
                failed_mask = existing_df[failed_col].astype(str).str.strip().str.lower() == "llm_call_failed"
                if failed_mask.any():
                    try:
                        failed_ids = set(existing_df.loc[failed_mask, "context_index"].dropna().astype(int).tolist())
                    except Exception:
                        failed_ids = set()
                    clean_existing_df = existing_df.loc[~failed_mask].copy()
                    print(f"[CLEAN] Found {len(failed_ids)} rows with {failed_col}=llm_call_failed. Will delete & re-run them.")

            existing_n = len(clean_existing_df)

            # If already enough AND no failed rows -> just re-eval locally
            if existing_n >= target_n and len(failed_ids) == 0:
                print(
                    f"[RE-EVAL] Found existing eval CSV with {existing_n} rows (>= {target_n}). "
                    f"Recomputing metrics locally (no new LLM calls): {output_csv}"
                )
                reevaluate_from_existing_csv(
                    existing_csv=output_csv,
                    output_csv=output_csv,
                    decision_field=decision_field,
                    parse_instance_fn=parse_instance_fn,
                    feasibility_fn=feasibility_fn,
                    objective_model_fn=objective_model_fn,
                    model_name=model,
                )
                per_size_outputs.append(output_csv)
                continue

            # Load input
            in_df = normalize_eval_dataframe(pd.read_csv(input_csv))

            # Determine which IDs already evaluated (AFTER removing failed rows)
            evaluated_ids = set()
            if not clean_existing_df.empty and "context_index" in clean_existing_df.columns:
                try:
                    evaluated_ids = set(clean_existing_df["context_index"].dropna().astype(int).tolist())
                except Exception:
                    evaluated_ids = set()

            # Build remain_df: prioritize failed IDs first, then other missing
            remain_parts = []

            if len(failed_ids) > 0:
                failed_part = in_df[in_df["context_index"].astype(int).isin(failed_ids)].copy()
                if not failed_part.empty:
                    remain_parts.append(failed_part)

            other_missing_part = in_df[~in_df["context_index"].astype(int).isin(evaluated_ids)].copy()
            if not other_missing_part.empty:
                remain_parts.append(other_missing_part)

            if not remain_parts:
                print(f"[WARN] No remaining rows found to evaluate. Keeping as-is: {output_csv}")
                per_size_outputs.append(output_csv)
                continue

            remain_df = pd.concat(remain_parts, ignore_index=True)

            # de-duplicate by context_index (keep first => failed rows take precedence)
            try:
                remain_df["context_index"] = remain_df["context_index"].astype(int)
                remain_df = remain_df.drop_duplicates(subset=["context_index"], keep="first")
            except Exception:
                pass

            need_topup = max(target_n - existing_n, 0)
            run_k = max(need_topup, len(failed_ids))

            remain_df = remain_df.head(run_k)

            print(
                f"[RESUME] Existing(clean)={existing_n}/{target_n}. "
                f"Failed_to_rerun={len(failed_ids)}. Will run {len(remain_df)} rows and append: {output_csv}"
            )

            # Write a temp input CSV for remain rows
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tf_in:
                tmp_input_csv = tf_in.name
                remain_df.to_csv(tmp_input_csv, index=False, encoding="utf-8")

            # Write to a temp output CSV
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as tf_out:
                tmp_output_csv = tf_out.name

            try:
                await evaluate_csv_generic_async(
                    input_csv=tmp_input_csv,
                    output_csv=tmp_output_csv,
                    task_name=problem,
                    decision_field=decision_field,
                    parse_instance_fn=parse_instance_fn,
                    feasibility_fn=feasibility_fn,
                    objective_model_fn=objective_model_fn,
                    model_name=model,
                    max_concurrency=max_concurrency,
                    provider_override=provider,
                    reasoning = reasoning,
                    reasoning_effort=reasoning_effort,
                    anthropic_budget=anthropic_budget,
                    max_tokens=max_tokens,
                    num_test=None,  # IMPORTANT: don't re-limit; we already sliced remain_df
                )
            finally:
                try:
                    os.remove(tmp_input_csv)
                except Exception:
                    pass

            # Append new results to existing CSV (BUT use clean_existing_df: failed rows removed)
            try:
                new_df = pd.read_csv(tmp_output_csv)
                new_df["orig_index"] = new_df["context_index"] - 1
                new_df["problem_size"] = size
            except Exception as e:
                print(f"[ERROR] Failed to read temp eval CSV {tmp_output_csv}: {e}")
                new_df = pd.DataFrame()
            finally:
                try:
                    os.remove(tmp_output_csv)
                except Exception:
                    pass

            if new_df.empty:
                print(f"[WARN] No new rows were produced; keeping existing CSV: {output_csv}")
                if len(failed_ids) > 0 and (len(clean_existing_df) != len(existing_df)):
                    clean_existing_df.to_csv(output_csv, index=False, encoding="utf-8")
                    print(f"[CLEAN] Wrote cleaned CSV without failed rows: {output_csv}")
                per_size_outputs.append(output_csv)
                continue

            out_df = pd.concat([clean_existing_df, new_df], ignore_index=True)

            # De-duplicate by context_index
            if "context_index" in out_df.columns:
                try:
                    out_df["context_index"] = out_df["context_index"].astype(int)
                    out_df = out_df.drop_duplicates(subset=["context_index"], keep="first")
                except Exception:
                    pass

            out_df.to_csv(output_csv, index=False, encoding="utf-8")
            print(f"[RESUME] Now {len(out_df)} rows in: {output_csv}")

            # Re-eval locally to ensure metrics columns are consistent
            reevaluate_from_existing_csv(
                existing_csv=output_csv,
                output_csv=output_csv,
                decision_field=decision_field,
                parse_instance_fn=parse_instance_fn,
                feasibility_fn=feasibility_fn,
                objective_model_fn=objective_model_fn,
                model_name=model,
            )

            per_size_outputs.append(output_csv)
            continue

        print(f"[RUN] Evaluating {problem} {size} from {input_csv} -> {output_csv}")
        await evaluate_csv_generic_async(
            input_csv=input_csv,
            output_csv=output_csv,
            task_name=problem,
            decision_field=decision_field,
            parse_instance_fn=parse_instance_fn,
            feasibility_fn=feasibility_fn,
            objective_model_fn=objective_model_fn,
            model_name=model,
            max_concurrency=max_concurrency,
            provider_override=provider,
            reasoning = reasoning,
            reasoning_effort=reasoning_effort,
            anthropic_budget=anthropic_budget,
            max_tokens=max_tokens,
            num_test=num_test,
        )

        per_size_outputs.append(output_csv)

    if not per_size_outputs:
        print(f"[INFO] No outputs generated for {problem}, skip overall.")
        return

    overall_dir = os.path.join(output_root, problem, safe_model)
    os.makedirs(overall_dir, exist_ok=True)
    overall_csv = os.path.join(overall_dir, f"{problem}_{safe_model}_overall.csv")

    dfs = []
    for path in per_size_outputs:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))

    if not dfs:
        print(f"[INFO] No valid per-size outputs for {problem}, cannot build overall.")
        return

    overall_df = pd.concat(dfs, ignore_index=True)
    overall_df.to_csv(overall_csv, index=False, encoding="utf-8")
    print(f"[OK] Overall {problem} results written to: {overall_csv}")


async def amain():
    parser = argparse.ArgumentParser(
        description="Run evaluation for combinatorial problems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single problem (default: reasoning disabled)
  python -m step3_evaluation.eval_cli --problem TSP --sizes S --model o4-mini --provider openai --dataset_root step2_contextualization/dataset

  # Enable reasoning explicitly + optional effort
  python -m step3_evaluation.eval_cli --problem TSP --sizes S --model o4-mini --provider openai --dataset_root step2_contextualization/dataset --reasoning enabled --reasoning_effort high

  # Anthropic: enable reasoning + provide budget
  python -m step3_evaluation.eval_cli --problem TSP --sizes S --model claude-sonnet-4-5 --provider anthropic --dataset_root step2_contextualization/dataset --reasoning enabled --anthropic_budget 2048
        """
    )
    parser.add_argument(
        "--problem", type=str, required=True,
        help="Problem type(s) to evaluate. Use comma-separated list (e.g., 'TSP,CVRP,MIS') or 'all' for all problems."
    )
    parser.add_argument(
        "--sizes", type=str, default="S,M,L",
        help="Size tags to evaluate, e.g. 'S', 'S,M', or 'all'. (default: 'S,M,L')"
    )
    parser.add_argument(
        "--dataset_root", type=str, required=True,
        help="Root directory for input datasets."
    )
    parser.add_argument(
        "--output_root", type=str, default="eval_outputs",
        help="Root directory for evaluation outputs. (default: 'eval_outputs')"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name for evaluation."
    )
    parser.add_argument(
        "--max_concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent LLM calls (online mode). (default: {DEFAULT_CONCURRENCY})"
    )

    parser.add_argument(
        "--provider", type=str,
        choices=["openai", "anthropic", "deepseek", "gemini", "vertex_maas", "openrouter", "xiaomi"],
        default=None,
        help="Force provider (no auto reasoning behavior; provider only routes requests)."
    )

    parser.add_argument(
        "--reasoning",
        type=str,
        choices=["enabled", "disabled"],
        default="disabled",
        help="Explicitly enable/disable reasoning. (default: disabled)"
    )

    parser.add_argument(
        "--reasoning_effort", type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort (provider-specific). Only valid when --reasoning enabled."
    )
    parser.add_argument(
        "--anthropic_budget", type=int, default=None,
        help="Claude thinking.budget_tokens (Anthropic only). Only valid when --reasoning enabled."
    )

    parser.add_argument(
        "--max_tokens", type=int, default=2048,
        help="max_tokens per response. (default: 2048)"
    )
    parser.add_argument(
        "--decision_field", type=str, default="solution",
        help="Field name for decision/solution in CSV. (default: 'solution')"
    )
    parser.add_argument(
        "--num_test", type=int, default=None,
        help="Limit to first N instances for quick testing. (default: None, use all instances)"
    )

    args = parser.parse_args()

    problems = parse_problems_arg(args.problem)
    sizes = parse_sizes_arg(args.sizes)

    provider: Optional[Provider] = args.provider if args.provider is not None else None
    reasoning_enabled = (args.reasoning == "enabled")

    if not reasoning_enabled:
        if args.reasoning_effort is not None:
            raise ValueError("--reasoning_effort is only valid when --reasoning enabled.")
        if args.anthropic_budget is not None:
            raise ValueError("--anthropic_budget is only valid when --reasoning enabled.")
    else:
        # Provider-specific validation (keep strict to avoid ambiguity)
        if provider == "anthropic" and args.anthropic_budget is None:
            raise ValueError("For Anthropic, --reasoning enabled requires --anthropic_budget.")
        if provider == "gemini" and args.reasoning_effort == "medium":
            raise ValueError("Gemini provider currently only supports reasoning effort in (high,low).")

    # Print summary
    print("=" * 80)
    print(f"Starting evaluation for {len(problems)} problem(s): {', '.join(problems)}")
    print(f"Sizes: {', '.join(sizes)}")
    print(f"Model: {args.model}")
    print(f"Provider: {provider or 'unset'}")
    print(f"Reasoning: {'enabled' if reasoning_enabled else 'disabled'}")
    if reasoning_enabled:
        if args.reasoning_effort is not None:
            print(f"Reasoning effort: {args.reasoning_effort}")
        if args.anthropic_budget is not None:
            print(f"Anthropic budget: {args.anthropic_budget}")
    print("Mode: online")
    if args.num_test:
        print(f"Test mode: Using first {args.num_test} instances only")
    print("=" * 80)

    # Run evaluation for each problem
    for idx, problem in enumerate(problems, 1):
        print(f"\n[{idx}/{len(problems)}] Processing problem: {problem}")
        print("-" * 80)

        try:
            await run_for_problem(
                problem=problem,
                sizes=sizes,
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                model=args.model,
                max_concurrency=args.max_concurrency,
                provider=provider,
                reasoning_effort=(args.reasoning_effort if reasoning_enabled else None),
                anthropic_budget=(args.anthropic_budget if reasoning_enabled else None),
                max_tokens=args.max_tokens,
                decision_field=args.decision_field,
                reasoning = reasoning_enabled,
                without_reasoning=(not reasoning_enabled),
                num_test=args.num_test,
            )
            print(f"✓ Completed problem: {problem}")
        except Exception as e:
            print(f"✗ Error processing problem {problem}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Evaluation complete! Processed {len(problems)} problem(s).")
    print("=" * 80)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
