import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for:
- JSP: Job-Shop Scheduling Problem
- FSP: Flow-Shop Scheduling Problem (we follow the task order as given in the instance)
- OSP: Open-Shop Scheduling Problem

Instance format (shared):
{
  "instance": {
    "nr_machines": M,
    "nr_jobs": J,
    "jobs": [
      [ [machine_id, duration], [machine_id, duration], ... ],  # job 0 tasks
      ...
    ]
  },
  "solution": [
    {
      "job": 0,
      "tasks": [
        {"task": 0, "start": 0, "machine": 1, "duration": 10},
        ...
      ]
    },
    ...
  ],
  "obj": <number>,
  "problem_type": "JSP" | "FSP" | "OSP"
}

We render input as a flat list of job-task records:
(job_id, task_id, machine_id, duration)
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _remap_solution_shop(
    solution: Any,
    job_id_map: Optional[Dict[int, Any]],
    machine_id_map: Optional[Dict[int, Any]],
) -> Any:
    """
    Remap:
    - solution[*]["job"]
    - solution[*]["tasks"][*]["machine"]
    Keep task indices and starts as-is.
    """
    if solution is None:
        return None
    if not isinstance(solution, list):
        return solution

    out = []
    for job_block in solution:
        if not isinstance(job_block, dict):
            out.append(job_block)
            continue

        jb = dict(job_block)
        if "job" in jb and job_id_map is not None:
            try:
                jb["job"] = job_id_map.get(int(jb["job"]), jb["job"])
            except Exception:
                pass

        tasks = jb.get("tasks")
        if isinstance(tasks, list) and machine_id_map is not None:
            new_tasks = []
            for t in tasks:
                if not isinstance(t, dict):
                    new_tasks.append(t)
                    continue
                tt = dict(t)
                if "machine" in tt:
                    try:
                        tt["machine"] = machine_id_map.get(int(tt["machine"]), tt["machine"])
                    except Exception:
                        pass
                new_tasks.append(tt)
            jb["tasks"] = new_tasks

        out.append(jb)
    return out


def _build_labeled_shop_instance(
    problem_type: str,
    nr_machines: int,
    nr_jobs: int,
    jobs: List[List[List[int]]],
    job_id_map: Optional[Dict[int, Any]],
    machine_id_map: Optional[Dict[int, Any]],
) -> Dict[str, Any]:
    job_ids = [job_id_map.get(j, j) if job_id_map else j for j in range(nr_jobs)]
    machine_ids = [machine_id_map.get(m, m) if machine_id_map else m for m in range(nr_machines)]

    records: List[Dict[str, Any]] = []
    for j in range(nr_jobs):
        jj = job_id_map.get(j, j) if job_id_map else j
        for t_idx, (m, d) in enumerate(jobs[j]):
            mm = machine_id_map.get(m, m) if machine_id_map else m
            records.append({"job_id": jj, "task_id": t_idx, "machine_id": mm, "duration": d})

    return {
        "problem_type": problem_type,
        "nr_jobs": nr_jobs,
        "nr_machines": nr_machines,
        "jobs": job_ids,
        "machines": machine_ids,
        "records": records,
    }


def shop_render_input(
    fmt: str,
    problem_type: str,
    nr_machines: int,
    nr_jobs: int,
    jobs: List[List[List[int]]],
    job_id_map: Optional[Dict[int, Any]],
    machine_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    # defaults
    nr_machines_key = "nr_machines"
    nr_jobs_key = "nr_jobs"
    jobs_key = "jobs"
    machines_key = "machines"

    job_id_key = "job_id"
    task_id_key = "task_id"
    machine_id_key = "machine_id"
    duration_key = "duration"

    # optional renaming
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "nr_machines":
                nr_machines_key = new
            elif orig == "nr_jobs":
                nr_jobs_key = new
            elif orig == "jobs":
                jobs_key = new
            elif orig == "machines":
                machines_key = new

        for field in scenario_hint.get("job_task_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "job_id":
                job_id_key = new
            elif orig == "task_id":
                task_id_key = new
            elif orig == "machine_id":
                machine_id_key = new
            elif orig == "duration":
                duration_key = new

    job_ids = [job_id_map.get(j, j) if job_id_map else j for j in range(nr_jobs)]
    machine_ids = [machine_id_map.get(m, m) if machine_id_map else m for m in range(nr_machines)]
    job_str = ", ".join(str(x) for x in job_ids)
    machine_str = ", ".join(str(x) for x in machine_ids)

    records: List[Dict[str, Any]] = []
    for j in range(nr_jobs):
        jj = job_id_map.get(j, j) if job_id_map else j
        for t_idx, (m, d) in enumerate(jobs[j]):
            mm = machine_id_map.get(m, m) if machine_id_map else m
            records.append({job_id_key: jj, task_id_key: t_idx, machine_id_key: mm, duration_key: d})

    if fmt == "json":
        obj = {
            nr_jobs_key: nr_jobs,
            nr_machines_key: nr_machines,
            jobs_key: job_ids,
            machines_key: machine_ids,
            "tasks": records,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {nr_jobs_key}={nr_jobs}")
        lines.append(f"# {nr_machines_key}={nr_machines}")
        lines.append(f"# {jobs_key}={job_str}")
        lines.append(f"# {machines_key}={machine_str}")
        lines.append(f"{job_id_key},{task_id_key},{machine_id_key},{duration_key}")
        for r in records:
            lines.append(f"{r[job_id_key]},{r[task_id_key]},{r[machine_id_key]},{r[duration_key]}")
        return "\n".join(lines)

    if fmt == "markdown_table":
        lines: List[str] = []

        if isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            if header_tmpl:
                lines.append(
                    _format_safe(
                        header_tmpl,
                        nr_jobs=nr_jobs,
                        nr_machines=nr_machines,
                        jobs=job_str,
                        machines=machine_str,
                        **{nr_jobs_key: nr_jobs, nr_machines_key: nr_machines, jobs_key: job_str, machines_key: machine_str},
                    )
                )
                lines.append("")

        lines.append(f"| {job_id_key} | {task_id_key} | {machine_id_key} | {duration_key} |")
        lines.append("|---|---|---|---|")
        for r in records:
            lines.append(f"| {r[job_id_key]} | {r[task_id_key]} | {r[machine_id_key]} | {r[duration_key]} |")

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
                        **{nr_jobs_key: nr_jobs, nr_machines_key: nr_machines, jobs_key: job_str, machines_key: machine_str},
                    )
                )

        return "\n".join(lines)

    # fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        line_tmpl = nl_style.get("job_task_item_fields_line_template", "") or ""

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
            for r in records:
                vars_ = dict(base_vars)
                vars_.update(
                    {
                        # generic placeholders
                        "job_id": r.get(job_id_key, ""),
                        "task_id": r.get(task_id_key, ""),
                        "machine_id": r.get(machine_id_key, ""),
                        "duration": r.get(duration_key, ""),
                        # renamed placeholders
                        job_id_key: r.get(job_id_key, ""),
                        task_id_key: r.get(task_id_key, ""),
                        machine_id_key: r.get(machine_id_key, ""),
                        duration_key: r.get(duration_key, ""),
                    }
                )
                parts.append(_format_safe(line_tmpl, **vars_))

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)

    return ""


def contextualize_instance_shop(
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

    nr_machines = core.get("nr_machines")
    nr_jobs = core.get("nr_jobs")
    jobs = core.get("jobs")

    if not isinstance(nr_machines, int) or not isinstance(nr_jobs, int):
        return results
    if not isinstance(jobs, list) or len(jobs) != nr_jobs:
        return results
    for j in range(nr_jobs):
        if not isinstance(jobs[j], list):
            return results
        for t in jobs[j]:
            if not (isinstance(t, list) and len(t) == 2):
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

    solution_variant = _remap_solution_shop(solution, job_id_map=job_id_map, machine_id_map=machine_id_map)

    input_text = shop_render_input(
        fmt=fmt,
        problem_type=problem_type,
        nr_machines=nr_machines,
        nr_jobs=nr_jobs,
        jobs=jobs,
        job_id_map=job_id_map,
        machine_id_map=machine_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_shop_instance(
        problem_type=problem_type,
        nr_machines=nr_machines,
        nr_jobs=nr_jobs,
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
    if problem_type == "JSP":
        desc = (
            "Job-Shop Scheduling Problem (JSP): multiple jobs must be processed on multiple machines.\n\n"
            "Each job consists of a fixed sequence of tasks. Every task must run on a specified machine for a given duration.\n"
            "A machine can handle at most one task at a time, and tasks within the same job must follow their given order.\n\n"
            "Your task is to schedule all tasks to minimize the time when all jobs are finished (the makespan)."
        )
    elif problem_type == "FSP":
        desc = (
            "Flow-Shop Scheduling Problem (FSP): multiple jobs must pass through processing steps on machines.\n\n"
            "Each job consists of a sequence of tasks with given durations, and all jobs share the same machine order.\n"
            "A machine processes at most one task at a time.\n\n"
            "Your task is to schedule all tasks to minimize the time when all jobs are finished (the makespan)."
        )


    else:  # OSP

        desc = (
            "Open-Shop Scheduling Problem (OSP): multiple jobs must be processed on multiple machines.\n\n"
            "Each job requires processing on several specified machines, exactly once per machine, "
            "but **there is NO predefined order between a job’s tasks**.\n"
            "The execution order of tasks within a job is a decision variable and can differ across jobs.\n\n"
            "A machine can process at most one task at a time, and a job cannot process two tasks simultaneously.\n\n"
            "Your task is to choose both the task ordering and start times so as to minimize "
            "the time when all jobs are finished (the makespan)."
        )

    baseline_template = (
        f"You are given a {problem_type} scheduling instance.\n\n"
        "Problem description:\n"
        "- There are multiple jobs and multiple machines.\n"
        "- Each job consists of tasks; each task requires a specific machine and has a processing duration.\n"
        "- A machine can process at most one task at a time.\n"
        "- You must assign a start time to every task so that all machine conflicts are avoided.\n"
    )
    if problem_type == "JSP":
        baseline_template += "- Tasks within the same job must follow the given task order (task 0 then task 1 then ...).\n\n"
    elif problem_type == "OSP":
        baseline_template += "- Tasks within the same job can be in any order, but a job cannot run two tasks at the same time.\n\n"
    else:
        baseline_template += "- Use the task order as provided in the instance.\n\n"

    baseline_template += (
        "Objective:\n"
        "- Minimize the makespan (the completion time of the last finishing task).\n\n"
        "You will receive the instance below:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your schedule using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [\n'
        "    {\n"
        '      "job": <job_id>,\n'
        '      "tasks": [\n'
        "        {\n"
        '          "task": <task_index>,\n'
        '          "start": <start_time>,\n'
        '          "machine": <machine_id>,\n'
        '          "duration": <duration>\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "Rules:\n"
        "- Include every job exactly once in the outer list.\n"
        "- For each job, include all of its tasks exactly once.\n"
        "- The `machine` and `duration` must match the instance for that job/task.\n"
        "- Start times must produce a valid schedule (no machine overlap).\n"
        "- Use identifiers exactly as they appear in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [\n'
        "    {\n"
        '      "job": <job_id>,\n'
        '      "tasks": [\n'
        "        {\n"
        '          "task": <task_index>,\n'
        '          "start": <start_time>,\n'
        '          "machine": <machine_id>,\n'
        '          "duration": <duration>\n'
        "        }\n"
        "      ]\n"
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
    job_task_item_fields = [
        {"name": "job_id", "description": "job identifier"},
        {"name": "task_id", "description": "task index within a job"},
        {"name": "machine_id", "description": "required machine for the task"},
        {"name": "duration", "description": "processing duration"},
    ]
    allowed = ["nr_jobs", "nr_machines", "jobs", "machines", "job_id", "task_id", "machine_id", "duration"]
    return {
        "global_fields": global_fields,
        "job_task_item_fields": job_task_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize JSP/FSP/OSP instances into NL/JSON/CSV/Markdown.")
    parser.add_argument("--problem_type", type=str, default="JSP", choices=["JSP", "FSP", "OSP"])
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
        contextualize_fn=contextualize_instance_shop,
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
python -m step2_contextualization.contextualize_shop_scheduling \
  --problem_type JSP \
  --instance_dir ./step1_instance_creation/generated_data/JSP \
  --output_root_dir step2_contextualization/dataset/JSP \
  --k_per_call 20 \
  --n_target 50
python -m step2_contextualization.contextualize_shop_scheduling \
  --problem_type FSP \
  --instance_dir ./step1_instance_creation/generated_data/FSP \
  --output_root_dir step2_contextualization/dataset/FSP \
  --k_per_call 20 \
  --n_target 50
python -m step2_contextualization.contextualize_shop_scheduling \
  --problem_type OSP \
  --instance_dir ./step1_instance_creation/generated_data/OSP \
  --output_root_dir step2_contextualization/dataset/OSP \
  --k_per_call 20 \
  --n_target 50

'''
