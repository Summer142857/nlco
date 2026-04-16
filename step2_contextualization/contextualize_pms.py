import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for PMS (Parallel Machines Scheduling).

Instance format:
{
  "instance": {
    "nr_jobs": J,
    "nr_machines": M,
    "jobs": [
      {"processing_time": p, "release_date": r, "deadline": d},
      ...
    ]
  },
  "solution": [
    {"job": 0, "machine": 1, "start": 0, "end": 15, "processing_time": 15, "release_date": 0, "deadline": 417},
    ...
  ],
  "obj": <number>,
  "problem_type": "PMS"
}

We render input as a list of job records:
(job_id, processing_time, release_date, deadline)
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _remap_solution_pms(
    solution: Any,
    job_id_map: Optional[Dict[int, Any]],
    machine_id_map: Optional[Dict[int, Any]],
) -> Any:
    if solution is None:
        return None
    if not isinstance(solution, list):
        return solution

    out = []
    for rec in solution:
        if not isinstance(rec, dict):
            out.append(rec)
            continue
        rr = dict(rec)

        if job_id_map is not None and "job" in rr:
            try:
                rr["job"] = job_id_map.get(int(rr["job"]), rr["job"])
            except Exception:
                pass

        if machine_id_map is not None and "machine" in rr:
            try:
                rr["machine"] = machine_id_map.get(int(rr["machine"]), rr["machine"])
            except Exception:
                pass

        out.append(rr)
    return out


def _build_labeled_pms_instance(
    nr_jobs: int,
    nr_machines: int,
    jobs: List[Dict[str, Any]],
    job_id_map: Optional[Dict[int, Any]],
    machine_id_map: Optional[Dict[int, Any]],
) -> Dict[str, Any]:
    job_ids = [job_id_map.get(j, j) if job_id_map else j for j in range(nr_jobs)]
    machine_ids = [machine_id_map.get(m, m) if machine_id_map else m for m in range(nr_machines)]

    job_records: List[Dict[str, Any]] = []
    for j in range(nr_jobs):
        jj = job_id_map.get(j, j) if job_id_map else j
        info = jobs[j]
        job_records.append(
            {
                "job_id": jj,
                "processing_time": info.get("processing_time"),
                "release_date": info.get("release_date"),
                "deadline": info.get("deadline"),
            }
        )

    return {
        "problem_type": "PMS",
        "nr_jobs": nr_jobs,
        "nr_machines": nr_machines,
        "jobs": job_ids,
        "machines": machine_ids,
        "job_records": job_records,
    }


def pms_render_input(
    fmt: str,
    nr_jobs: int,
    nr_machines: int,
    jobs: List[Dict[str, Any]],
    job_id_map: Optional[Dict[int, Any]],
    machine_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    # defaults
    nr_jobs_key = "nr_jobs"
    nr_machines_key = "nr_machines"
    jobs_key = "jobs"
    machines_key = "machines"

    job_id_key = "job_id"
    p_key = "processing_time"
    r_key = "release_date"
    d_key = "deadline"

    # optional renaming
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "nr_jobs":
                nr_jobs_key = new
            elif orig == "nr_machines":
                nr_machines_key = new
            elif orig == "jobs":
                jobs_key = new
            elif orig == "machines":
                machines_key = new

        for field in scenario_hint.get("job_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "job_id":
                job_id_key = new
            elif orig == "processing_time":
                p_key = new
            elif orig == "release_date":
                r_key = new
            elif orig == "deadline":
                d_key = new

    job_ids = [job_id_map.get(j, j) if job_id_map else j for j in range(nr_jobs)]
    machine_ids = [machine_id_map.get(m, m) if machine_id_map else m for m in range(nr_machines)]
    job_str = ", ".join(str(x) for x in job_ids)
    machine_str = ", ".join(str(x) for x in machine_ids)

    recs: List[Dict[str, Any]] = []
    for j in range(nr_jobs):
        jj = job_id_map.get(j, j) if job_id_map else j
        info = jobs[j]
        recs.append(
            {
                job_id_key: jj,
                p_key: info.get("processing_time"),
                r_key: info.get("release_date"),
                d_key: info.get("deadline"),
            }
        )

    if fmt == "json":
        obj = {
            nr_jobs_key: nr_jobs,
            nr_machines_key: nr_machines,
            jobs_key: job_ids,
            machines_key: machine_ids,
            "job_data": recs,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {nr_jobs_key}={nr_jobs}")
        lines.append(f"# {nr_machines_key}={nr_machines}")
        lines.append(f"# {jobs_key}={job_str}")
        lines.append(f"# {machines_key}={machine_str}")
        lines.append(f"{job_id_key},{p_key},{r_key},{d_key}")
        for r in recs:
            lines.append(f"{r[job_id_key]},{r[p_key]},{r[r_key]},{r[d_key]}")
        return "\n".join(lines)

    if fmt == "markdown_table":
        lines: List[str] = []

        if isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            if header_tmpl:
                print(header_tmpl)
                lines.append(
                    _format_safe(
                        header_tmpl,
                        nr_jobs=nr_jobs,
                        nr_machines=nr_machines,
                        jobs=job_str,
                        machines=machine_str,
                    )
                )
                lines.append("")

        lines.append(f"| {job_id_key} | {p_key} | {r_key} | {d_key} |")
        lines.append("|---|---|---|---|")
        for r in recs:
            lines.append(f"| {r[job_id_key]} | {r[p_key]} | {r[r_key]} | {r[d_key]} |")

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        nr_jobs=nr_jobs,
                        nr_machines=nr_machines,
                        jobs=job_str,
                        machines=machine_str,
                    )
                )

        return "\n".join(lines)

    # fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        line_tmpl = nl_style.get("job_item_fields_line_template", "") or ""

        parts: List[str] = []
        base_vars = {
            "nr_jobs": nr_jobs,
            "nr_machines": nr_machines,
            "jobs": job_str,
            "machines": machine_str,
            nr_jobs_key: nr_jobs,
            nr_machines_key: nr_machines,
            jobs_key: job_str,
            machines_key: machine_str,
        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        if line_tmpl:
            for r in recs:
                vars_ = dict(base_vars)
                vars_.update(
                    {
                        # generic names
                        "job_id": r.get(job_id_key, ""),
                        "processing_time": r.get(p_key, ""),
                        "release_date": r.get(r_key, ""),
                        "deadline": r.get(d_key, ""),
                        # renamed names
                        job_id_key: r.get(job_id_key, ""),
                        p_key: r.get(p_key, ""),
                        r_key: r.get(r_key, ""),
                        d_key: r.get(d_key, ""),
                    }
                )
                parts.append(_format_safe(line_tmpl, **vars_))
        else:
            for r in recs:
                parts.append(f"{r[job_id_key]}, {r[p_key]}, {r[r_key]}, {r[d_key]}")

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)

    return ""


def contextualize_instance_pms(
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

    nr_jobs = core.get("nr_jobs")
    nr_machines = core.get("nr_machines")
    jobs = core.get("jobs")

    if not isinstance(nr_jobs, int) or not isinstance(nr_machines, int):
        return results
    if not isinstance(jobs, list) or len(jobs) != nr_jobs:
        return results
    for j in range(nr_jobs):
        if not isinstance(jobs[j], dict):
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

    # labels
    job_labels = [f"J{j+1}" for j in range(nr_jobs)]
    machine_labels = _build_labels(nr_machines, index_base)
    job_id_map = {j: job_labels[j] for j in range(nr_jobs)}
    machine_id_map = {m: machine_labels[m] for m in range(nr_machines)}

    solution_variant = _remap_solution_pms(solution, job_id_map=job_id_map, machine_id_map=machine_id_map)

    input_text = pms_render_input(
        fmt=fmt,
        nr_jobs=nr_jobs,
        nr_machines=nr_machines,
        jobs=jobs,
        job_id_map=job_id_map,
        machine_id_map=machine_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_pms_instance(
        nr_jobs=nr_jobs,
        nr_machines=nr_machines,
        jobs=jobs,
        job_id_map=job_id_map,
        machine_id_map=machine_id_map,
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
    desc = (
        "Parallel Machines Scheduling (PMS): you have several identical machines and a set of jobs to run.\n\n"
        "Each job has a processing time, may not start before its release date, and has a deadline it should meet.\n"
        "You must assign every job to exactly one machine and choose a start time for it.\n"
        "A machine can process at most one job at a time.\n\n"
        "Your task is to produce a feasible schedule that avoids overlaps and respects release dates, while minimizing the "
        "maximum tardiness (the worst deadline overrun across all jobs)."
    )

    baseline_template = (
        "You are given a Parallel Machines Scheduling (PMS) instance.\n\n"
        "Problem description:\n"
        "- There are multiple identical machines that can process jobs.\n"
        "- Each job has:\n"
        "  (1) `processing_time`: how long it takes,\n"
        "  (2) `release_date`: it cannot start earlier than this,\n"
        "  (3) `deadline`: it is considered late if it finishes after this.\n"
        "- You must assign every job to exactly one machine and choose a start time.\n"
        "- Each machine can run at most one job at a time (no overlaps).\n\n"
        "Objective:\n"
        "- Minimize the **maximum tardiness** across all jobs.\n"
        "  For each job j: completion C_j = end_j, tardiness T_j = max(0, C_j - deadline_j).\n"
        "  Optimize: minimize T_max = max_j T_j.\n\n"
        "You will receive the instance below:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your schedule using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  \"solution\": [\n'
        "    {\n"
        '      \"job\": <job_id>,\n'
        '      \"machine\": <machine_id>,\n'
        '      \"start\": <start_time>,\n'
        '      \"end\": <end_time>\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "Rules:\n"
        "- The list must contain every job exactly once.\n"
        "- For each record: `end = start + processing_time`.\n"
        "- Each job must start at or after its release date.\n"
        "- Jobs assigned to the same machine must not overlap in time.\n"
        "- Use identifiers exactly as they appear in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [\n'
        "    {\n"
        '      "job": <job_id>,\n'
        '      "machine": <machine_id>,\n'
        '      "start": <start_time>,\n'
        '      "end": <end_time>\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
    )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    global_fields = [
        {"name": "nr_jobs", "description": "number of jobs"},
        {"name": "nr_machines", "description": "number of machines"},
        {"name": "jobs", "description": "job identifiers"},
        {"name": "machines", "description": "machine identifiers"},
    ]
    job_item_fields = [
        {"name": "job_id", "description": "job identifier"},
        {"name": "processing_time", "description": "job processing time"},
        {"name": "release_date", "description": "earliest start time"},
        {"name": "deadline", "description": "due time for being on time"},
    ]
    allowed = ["nr_jobs", "nr_machines", "jobs", "machines", "job_id", "processing_time", "release_date", "deadline"]
    return {
        "global_fields": global_fields,
        "job_item_fields": job_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize PMS instances into NL/JSON/CSV/Markdown.")
    parser.add_argument("--problem_type", type=str, default="PMS", choices=["PMS"])
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
        contextualize_fn=contextualize_instance_pms,
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

'''
python -m step2_contextualization.contextualize_pms \
  --problem_type PMS \
  --instance_dir ./step1_instance_creation/generated_data/PMS \
  --output_root_dir step2_contextualization/dataset/PMS \
  --k_per_call 20 \
  --n_target 50


'''
