import json
import random
from typing import Any, Dict, List, Optional, Sequence, Union

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for SMTWT (Single-Machine Total Weighted Tardiness).

Instance format:
{
  "instance": {
    "n_jobs": n,
    "processing_times": [p0, p1, ...],
    "weights": [w0, w1, ...],
    "due_dates": [d0, d1, ...]
  },
  "solution": <permutation>,            # preferred: list of job indices/ids in processing order
  # OR legacy (optional): {"start_times": [...]}
  "obj": <number>,
  "problem_type": "SMTWT"
}

We ask the model to output ONLY a permutation (job order):
{
  "solution": [<job_id_first>, <job_id_second>, ..., <job_id_last>]
}

Interpretation (no idle time implied by permutation):
- Single machine => jobs are processed one-by-one in the given order.
- If the permutation is pi = [pi1, pi2, ..., pin], then completion times are:
  C(pi1) = p(pi1)
  C(pik) = C(pi(k-1)) + p(pik)
- Tardiness(j) = max(0, C(j) - d(j))
- Objective = sum_j w(j) * Tardiness(j) (minimize)
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _solution_to_permutation(solution: Any) -> Optional[List[int]]:
    """
    Convert different solution representations into a 0-based permutation of job indices.

    Supported:
    - permutation as a list of ints (already 0-based indices)
    - dict with key "permutation" (list of ints)
    - legacy dict with key "start_times" (derive permutation by sorting start times)
    """
    if solution is None:
        return None

    # Case 1: solution is already a list -> assume it is a permutation of 0-based job indices.
    if isinstance(solution, list) and all(isinstance(x, int) for x in solution):
        return list(solution)

    # Case 2: solution is a dict with explicit permutation
    if isinstance(solution, dict):
        perm = solution.get("permutation")
        if isinstance(perm, list) and all(isinstance(x, int) for x in perm):
            return list(perm)

        # Case 3: legacy dict with start_times -> sort to derive order
        start_times = solution.get("start_times")
        if isinstance(start_times, list) and all(isinstance(x, (int, float)) for x in start_times):
            indexed = list(enumerate(start_times))
            indexed.sort(key=lambda t: (t[1], t[0]))  # stable tie-break by job index
            return [i for i, _ in indexed]

    return None


def _remap_solution_smtwt(solution: Any, job_id_map: Optional[Dict[int, Any]] = None) -> Any:
    """
    Remap a 0-based permutation of job indices into the labeled job identifiers used in the input variant.

    - If job_id_map is provided, each job index i is replaced by job_id_map[i].
    - Returns a list representing the permutation in the same identifier space shown to the model.
    """
    perm0 = _solution_to_permutation(solution)
    if perm0 is None:
        return solution  # fall back unchanged if unexpected format

    if job_id_map:
        return [job_id_map.get(i, i) for i in perm0]
    return perm0


def _build_labeled_smtwt_instance(
    n_jobs: int,
    processing_times: List[Any],
    weights: List[Any],
    due_dates: List[Any],
    job_id_map: Optional[Dict[int, Any]],
) -> Dict[str, Any]:
    job_ids = [job_id_map.get(i, i) if job_id_map else i for i in range(n_jobs)]
    job_recs: List[Dict[str, Any]] = []
    for i in range(n_jobs):
        job_recs.append(
            {
                "job_id": job_ids[i],
                "processing_time": processing_times[i],
                "weight": weights[i],
                "due_date": due_dates[i],
            }
        )
    return {
        "problem_type": "SMTWT",
        "n_jobs": n_jobs,
        "jobs": job_ids,
        "job_records": job_recs,
    }


def smtwt_render_input(
    fmt: str,
    n_jobs: int,
    processing_times: List[Any],
    weights: List[Any],
    due_dates: List[Any],
    job_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    # defaults
    n_jobs_key = "n_jobs"
    jobs_key = "jobs"

    job_id_key = "job_id"
    p_key = "processing_time"
    w_key = "weight"
    d_key = "due_date"

    # optional renaming
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "n_jobs":
                n_jobs_key = new
            elif orig == "jobs":
                jobs_key = new

        for field in scenario_hint.get("job_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "job_id":
                job_id_key = new
            elif orig == "processing_time":
                p_key = new
            elif orig == "weight":
                w_key = new
            elif orig == "due_date":
                d_key = new

    job_ids = [job_id_map.get(i, i) if job_id_map else i for i in range(n_jobs)]
    jobs_str = ", ".join(str(x) for x in job_ids)

    job_recs: List[Dict[str, Any]] = []
    for i in range(n_jobs):
        job_recs.append(
            {
                job_id_key: job_ids[i],
                p_key: processing_times[i],
                w_key: weights[i],
                d_key: due_dates[i],
            }
        )

    if fmt == "json":
        obj = {
            n_jobs_key: n_jobs,
            jobs_key: job_ids,
            "data": job_recs,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {n_jobs_key}={n_jobs}")
        lines.append(f"# {jobs_key}={jobs_str}")
        lines.append(f"{job_id_key},{p_key},{w_key},{d_key}")
        for r in job_recs:
            lines.append(f"{r[job_id_key]},{r[p_key]},{r[w_key]},{r[d_key]}")
        return "\n".join(lines)

    if fmt == "markdown_table":
        lines: List[str] = []

        # optional header/footer only; NO line template
        if isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            if header_tmpl:
                lines.append(
                    _format_safe(
                        header_tmpl,
                        n_jobs=n_jobs,
                        jobs=jobs_str,
                        **{n_jobs_key: n_jobs, jobs_key: jobs_str},
                    )
                )
                lines.append("")

        lines.append(f"| {job_id_key} | {p_key} | {w_key} | {d_key} |")
        lines.append("|---|---|---|---|")
        for r in job_recs:
            lines.append(f"| {r[job_id_key]} | {r[p_key]} | {r[w_key]} | {r[d_key]} |")

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        n_jobs=n_jobs,
                        jobs=jobs_str,
                        **{n_jobs_key: n_jobs, jobs_key: jobs_str},
                    )
                )

        return "\n".join(lines)

    # fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        job_line_tmpl = nl_style.get("job_item_fields_line_template", "") or ""

        parts: List[str] = []
        base_vars = {
            "n_jobs": n_jobs,
            "jobs": jobs_str,
            n_jobs_key: n_jobs,
            jobs_key: jobs_str,
        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        if job_line_tmpl:
            for r in job_recs:
                vars_ = dict(base_vars)
                vars_.update(
                    {
                        # generic keys
                        "job_id": r.get(job_id_key, ""),
                        "processing_time": r.get(p_key, ""),
                        "weight": r.get(w_key, ""),
                        "due_date": r.get(d_key, ""),
                        # renamed keys
                        job_id_key: r.get(job_id_key, ""),
                        p_key: r.get(p_key, ""),
                        w_key: r.get(w_key, ""),
                        d_key: r.get(d_key, ""),
                    }
                )
                parts.append(_format_safe(job_line_tmpl, **vars_))
        else:
            for r in job_recs:
                parts.append(
                    f"Job {r[job_id_key]}: processing_time={r[p_key]}, weight={r[w_key]}, due_date={r[d_key]}"
                )

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)

    return ""


def contextualize_instance_smtwt(
    inst: Dict[str, Any],
    task_name: str,
    contexts: List[Dict[str, Any]],
    precomputed_templates: Dict[int, str],
    nl_styles: Dict[tuple, Dict[str, Any]],
    scenario_hints: Dict[int, Dict[str, Any]],
    instance_idx: int = 0,
    difficulty_tier: str | None = None,
) -> List[Dict[str, Any]]:
    rng = random.Random()
    results: List[Dict[str, Any]] = []

    core = inst.get("instance")
    solution = inst.get("solution")
    obj = inst.get("obj")
    problem_type = inst.get("problem_type", task_name)

    if not isinstance(core, dict):
        return results

    n_jobs = core.get("n_jobs")
    processing_times = core.get("processing_times")
    weights = core.get("weights")
    due_dates = core.get("due_dates")

    if not isinstance(n_jobs, int) or n_jobs <= 0:
        return results
    if not isinstance(processing_times, list) or len(processing_times) != n_jobs:
        return results
    if not isinstance(weights, list) or len(weights) != n_jobs:
        return results
    if not isinstance(due_dates, list) or len(due_dates) != n_jobs:
        return results

    # choose context
    K = len(contexts) if contexts else 1
    valid_context_indices = sorted(idx for idx in precomputed_templates if idx != 0)
    if not valid_context_indices:
        valid_context_indices = sorted(precomputed_templates)

    K = len(valid_context_indices)
    if K == 0:
        return results

    context_index = valid_context_indices[instance_idx % K]
    template = precomputed_templates[context_index]
    nl_style = nl_styles.get((task_name.upper(), context_index))
    scenario_hint = scenario_hints.get(context_index)

    fmt = rng.choice(FORMATS)
    index_base = rng.choice(INDEX_BASES)

    # Build labels for jobs (number-based or name/letter-based)
    job_labels = _build_labels(n_jobs, index_base)
    job_id_map = {i: job_labels[i] for i in range(n_jobs)}

    # Remap the stored solution into the same label space shown in the input variant
    solution_variant = _remap_solution_smtwt(solution, job_id_map=job_id_map)

    input_text = smtwt_render_input(
        fmt=fmt,
        n_jobs=n_jobs,
        processing_times=processing_times,
        weights=weights,
        due_dates=due_dates,
        job_id_map=job_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_smtwt_instance(
        n_jobs=n_jobs,
        processing_times=processing_times,
        weights=weights,
        due_dates=due_dates,
        job_id_map=job_id_map,
    )

    record = build_output_record(
        task_id=task_name,
        difficulty_tier=difficulty_tier or "UNKNOWN",
        example_index=context_index,
        prompt=full_instruction,
        surface_format=fmt,
        indexing_scheme=index_base,
        instance_canonical_json=core,
        reference_solution_canonical_json=solution,
        reference_objective_value=obj,
        instance_surface_json=instance_variant,
        reference_solution_surface_json=solution_variant,
    )
    results.append(record)
    return results


def _get_problem_specs(problem_type: str) -> Dict[str, str]:
    # SMTWT only
    desc = (
        "Single-Machine Total Weighted Tardiness (SMTWT): you have several jobs to run on one machine.\n\n"
        "Each job i has a processing time p_i, a due date d_i, and a weight w_i that reflects how painful it is to finish late.\n"
        "Jobs cannot overlap because there is only one machine.\n\n"
        "If a job finishes after its due date, it is late by (finish_time - due_date). If it finishes on or before the due date, "
        "its lateness is zero. The goal is to schedule all jobs to minimize the total weighted lateness across jobs."
    )

    baseline_template = (
        "You are given a Single-Machine Total Weighted Tardiness (SMTWT) instance.\n\n"
        "Problem description:\n"
        "- There are n jobs processed on a single machine.\n"
        "- Each job i has: processing time p_i, due date d_i, and weight w_i.\n"
        "- The machine can run at most one job at a time.\n"
        "- Each job must be scheduled exactly once.\n\n"
        "How the score is computed (using ONLY the permutation/order you return):\n"
        "- Let the permutation be pi = [pi1, pi2, ..., pin].\n"
        "- Completion time: C(pi1) = p(pi1), and C(pik) = C(pi(k-1)) + p(pik).\n"
        "- Tardiness(j) = max(0, C(j) - d(j)).\n"
        "- Total cost = sum_j w(j) * Tardiness(j).\n"
        "- Your goal is to MINIMIZE this total cost.\n\n"
        "You will receive the instance below:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your permutation using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [\n'
        '    <job_id_first>,\n'
        '    <job_id_second>,\n'
        '    ...,\n'
        '    <job_id_last>\n'
        "  ]\n"
        "}\n"
        "```\n"
        "Rules:\n"
        "- The list `solution` must have length n.\n"
        "- It must contain each job identifier exactly once.\n"
        "- Use identifiers exactly as they appear in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [\n'
        '    <job_id_first>,\n'
        '    <job_id_second>,\n'
        '    ...,\n'
        '    <job_id_last>\n'
        "  ]\n"
        "}\n"
        "```\n"
    )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    global_fields = [
        {"name": "n_jobs", "description": "number of jobs"},
        {"name": "jobs", "description": "job identifiers"},
    ]
    job_item_fields = [
        {"name": "job_id", "description": "job id"},
        {"name": "processing_time", "description": "processing time of the job"},
        {"name": "weight", "description": "weight for tardiness penalty"},
        {"name": "due_date", "description": "due date of the job"},
    ]
    allowed = ["n_jobs", "jobs", "job_id", "processing_time", "weight", "due_date"]
    return {
        "global_fields": global_fields,
        "job_item_fields": job_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize SMTWT instances into NL/JSON/CSV/Markdown.")
    parser.add_argument("--problem_type", type=str, default="SMTWT", choices=["SMTWT"])
    parser.add_argument("--instance_dir", type=str, default=None)
    parser.add_argument("--output_root_dir", type=str, default=None)
    parser.add_argument("--k_per_call", type=int, default=20)
    parser.add_argument("--n_target", type=int, default=50)

    args = parser.parse_args()
    problem_type = args.problem_type
    task_name = problem_type

    specs = _get_problem_specs(problem_type)
    desc = specs["task_description"]
    baseline_template = specs["baseline_template"]
    output_format = specs["output_format"]

    instance_dir = args.instance_dir or f"./step1_instance_creation/generated_data/{problem_type}"
    output_root_dir = args.output_root_dir or f"step2_contextualization/dataset/{problem_type}"

    llm = OpenAILlmService()
    prompter = PromptLoader()
    hint = _get_hint(problem_type)

    generate_dataset_for_task(
        task_name=task_name,
        task_description=desc,
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_instance_smtwt,
        llm=llm,
        prompter=prompter,
        hint=hint,
        baseline_template=baseline_template,
        output_format=output_format,
        k_per_call=args.k_per_call,
        n_target=args.n_target,
    )


if __name__ == "__main__":
    main()

"""
python -m step2_contextualization.contextualize_smtwt \
  --problem_type SMTWT \
  --instance_dir ./step1_instance_creation/generated_data/SMTWT \
  --output_root_dir step2_contextualization/dataset/SMTWT \
  --k_per_call 20 \
  --n_target 50
"""
