
import glob
import json
import os

from step2_contextualization.utils.generate_scenarios import generate_context
from step2_contextualization.utils.dataset_schema import (
    infer_task_and_tier_from_base,
    write_normalized_dataset,
)

import asyncio

def generate_dataset_for_task(
        task_name: str,
        task_description: str,
        instance_dir: str,
        output_root_dir: str,
        contextualize_fn,
        llm,
        prompter,
        hint,
        baseline_template: str,
        output_format,
        k_per_call: int = 5,
        n_target: int = 10,
        scale: str | None = None,

        load_cached_contexts: bool = True,
        contexts_audit_path: str | None = None,
):


    """
    Generic dataset generation pipeline for any task.
    If a precomputed context JSON exists, reuse it instead of regenerating.
    """

    # 1) Resolve instance json files
    if os.path.isfile(instance_dir):
        # instance_dir is actually a single json file path
        json_files = [instance_dir]
    else:
        # instance_dir is a directory
        json_files = sorted(
            p for p in glob.glob(os.path.join(instance_dir, "*.json"))
            if not os.path.basename(p).endswith("_stats.json")
        )

    # optional: filter by scale (S/M/L)
    if scale is not None:
        scale = str(scale).upper()
        json_files = [p for p in json_files if f"_{scale}" in os.path.basename(p)]

    if not json_files:
        print(f"[WARN] No JSON files found for instance_dir={instance_dir} scale={scale}")
        return

    os.makedirs(output_root_dir, exist_ok=True)

    print(f"[INFO] Generating / loading NL contexts for task={task_name}")

    # --------------------------------------------------
    # 2) Load cached contexts (optional) or regenerate
    # --------------------------------------------------
    default_audit_path = os.path.join(output_root_dir, f"{task_name}_contexts.json")

    audit_path = contexts_audit_path or default_audit_path

    contexts = None
    templates = None
    nl_styles = None
    scenario_hints = None

    if load_cached_contexts:
        if os.path.exists(audit_path):
            print(f"[INFO] Loading cached contexts from: {audit_path}")
            with open(audit_path, "r", encoding="utf-8") as f:
                contexts_audit = json.load(f)

            contexts = contexts_audit.get("contexts")
            if contexts is None:
                print("[WARN] 'contexts' missing in audit JSON; will regenerate.")
            else:
                # templates
                raw_templates = contexts_audit.get("precomputed_templates", {})
                if isinstance(raw_templates, dict) and "full" in raw_templates:
                    templates = {int(k): v for k, v in raw_templates["full"].items()}
                else:
                    print("[WARN] 'precomputed_templates.full' missing; will regenerate.")

                # nl_styles
                raw_styles = contexts_audit.get("nl_styles", {})
                if isinstance(raw_styles, dict) and raw_styles:
                    nl_styles = {(task_name.upper(), int(k)): v for k, v in raw_styles.items()}
                else:
                    print("[WARN] 'nl_styles' missing; will regenerate.")

                # scenario_hints
                raw_scenario_hints = contexts_audit.get("scenario_hints", None)
                if raw_scenario_hints is not None:
                    scenario_hints = {int(k): v for k, v in raw_scenario_hints.items()}
                else:
                    print("[WARN] 'scenario_hints' missing; will regenerate.")
        else:
            print(f"[INFO] Cached contexts not found at: {audit_path} (will regenerate)")
    else:
        print("[INFO] load_cached_contexts=False, will regenerate contexts (ignore cache).")

    if contexts is None or templates is None or nl_styles is None or scenario_hints is None:
        print("[INFO] Regenerating contexts with LLM...")
        ctx_result = asyncio.run(
            generate_context(
                task_name=task_name,
                task_description=task_description,
                output_root_dir=output_root_dir,
                llm=llm,
                prompter=prompter,
                k_per_call=k_per_call,
                n_target=n_target,
                hint=hint,
                baseline_template=baseline_template,
                output_format=output_format,
            )
        )
        if ctx_result is None:
            print("[ERROR] Context generation failed.")
            return

        contexts = ctx_result["contexts"]
        templates = ctx_result["precomputed_templates"]["full"]
        nl_styles = ctx_result["nl_styles"]
        scenario_hints = ctx_result["scenario_hints"]


    # 4) Iterate over all instance files
    for src_path in json_files:
        print(f"\n[STAGE] Processing instance file: {src_path}")
        base = os.path.splitext(os.path.basename(src_path))[0]
        _, difficulty_tier = infer_task_and_tier_from_base(base)

        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[WARN] {src_path} is not a list; skipping.")
            continue

        instances = data[:n_target]
        all_records = []

        for idx, inst in enumerate(instances):
            print(f"[STAGE] Contextualizing instance #{idx} in {os.path.basename(src_path)}")

            recs = contextualize_fn(
                inst=inst,
                task_name=task_name,
                contexts=contexts,
                precomputed_templates=templates,
                nl_styles=nl_styles,
                scenario_hints=scenario_hints,
                instance_idx=idx,
                difficulty_tier=difficulty_tier,
            )

            all_records.extend(recs)

        # 5) Save results
        json_path = os.path.join(output_root_dir, f"{base}_context.json")
        csv_path = os.path.join(output_root_dir, f"{base}_context.csv")

        write_normalized_dataset(all_records, json_path, csv_path)

        print(f"[OK] Saved JSON: {json_path}")
        print(f"[OK] Saved CSV: {csv_path}")
        print(f"[OK] Total records: {len(all_records)}")
