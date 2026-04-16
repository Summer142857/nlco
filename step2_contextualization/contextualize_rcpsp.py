import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for RCPSP (Resource-Constrained Project Scheduling Problem).

Instance format:
{
  "instance": {
    "nr_tasks": T,
    "nr_resources": R,
    "capacities": [cap_r...],
    "tasks": [
      {"duration": d, "demands": [dem_r...], "successors": [t2, t3, ...]},
      ...
    ]
  },
  "solution": [
    {"task": 0, "start": 0, "end": 1, "duration": 1, "demands": [...]},
    ...
  ],
  "obj": <number>,
  "problem_type": "RCPSP"
}

We render per-task records:
(task_id, duration, demands, successors)
Successors may be empty; we render it as "none" in NL/markdown for clarity.
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _succ_to_str(succ: Any) -> str:
    if not isinstance(succ, list) or len(succ) == 0:
        return "none"
    return ", ".join(str(x) for x in succ)


def _list_to_str(xs: Any) -> str:
    if not isinstance(xs, list):
        return str(xs)
    if len(xs) == 0:
        return "none"
    return ", ".join(str(x) for x in xs)


def _remap_solution_rcpsp(
    solution: Any,
    task_id_map: Optional[Dict[int, Any]],
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
        if task_id_map is not None and "task" in rr:
            try:
                rr["task"] = task_id_map.get(int(rr["task"]), rr["task"])
            except Exception:
                pass
        out.append(rr)
    return out


def _build_labeled_rcpsp_instance(
    nr_tasks: int,
    nr_resources: int,
    capacities: List[Any],
    tasks: List[Dict[str, Any]],
    task_id_map: Optional[Dict[int, Any]],
    resource_id_map: Optional[Dict[int, Any]],
) -> Dict[str, Any]:
    task_ids = [task_id_map.get(t, t) if task_id_map else t for t in range(nr_tasks)]
    resource_ids = [resource_id_map.get(r, r) if resource_id_map else r for r in range(nr_resources)]
    cap_recs = []
    for r in range(nr_resources):
        rid = resource_id_map.get(r, r) if resource_id_map else r
        cap_recs.append({"resource_id": rid, "capacity": capacities[r]})

    task_recs: List[Dict[str, Any]] = []
    for t in range(nr_tasks):
        tid = task_id_map.get(t, t) if task_id_map else t
        info = tasks[t]
        dem = info.get("demands", [])
        succ = info.get("successors", [])

        # remap successor ids if mapping exists
        if isinstance(succ, list) and task_id_map:
            succ = [task_id_map.get(int(x), x) if isinstance(x, (int, float)) else task_id_map.get(x, x) for x in succ]

        task_recs.append(
            {
                "task_id": tid,
                "duration": info.get("duration"),
                "demands": dem,
                "successors": succ,
            }
        )

    return {
        "problem_type": "RCPSP",
        "nr_tasks": nr_tasks,
        "nr_resources": nr_resources,
        "tasks": task_ids,
        "resources": resource_ids,
        "resource_capacities": cap_recs,
        "task_records": task_recs,
    }


def rcpsp_render_input(
    fmt: str,
    nr_tasks: int,
    nr_resources: int,
    capacities: List[Any],
    tasks: List[Dict[str, Any]],
    task_id_map: Optional[Dict[int, Any]],
    resource_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    # defaults
    nr_tasks_key = "nr_tasks"
    nr_resources_key = "nr_resources"
    capacities_key = "capacities"
    tasks_key = "tasks"

    task_id_key = "task_id"
    duration_key = "duration"
    demands_key = "demands"
    successors_key = "successors"

    # optional renaming
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "nr_tasks":
                nr_tasks_key = new
            elif orig == "nr_resources":
                nr_resources_key = new
            elif orig == "capacities":
                capacities_key = new
            elif orig == "tasks":
                tasks_key = new

        for field in scenario_hint.get("task_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "task_id":
                task_id_key = new
            elif orig == "duration":
                duration_key = new
            elif orig == "demands":
                demands_key = new
            elif orig == "successors":
                successors_key = new

    task_ids = [task_id_map.get(t, t) if task_id_map else t for t in range(nr_tasks)]
    res_ids = [resource_id_map.get(r, r) if resource_id_map else r for r in range(nr_resources)]
    task_str = ", ".join(str(x) for x in task_ids)
    res_str = ", ".join(str(x) for x in res_ids)

    # capacities with remapped resource ids
    cap_recs: List[Dict[str, Any]] = []
    for r in range(nr_resources):
        rid = resource_id_map.get(r, r) if resource_id_map else r
        cap_recs.append({"resource_id": rid, "capacity": capacities[r]})

    # task records
    task_recs: List[Dict[str, Any]] = []
    for t in range(nr_tasks):
        tid = task_id_map.get(t, t) if task_id_map else t
        info = tasks[t]
        dem = info.get("demands", [])
        succ = info.get("successors", [])

        # remap successor ids if mapping exists
        if isinstance(succ, list) and task_id_map:
            succ = [task_id_map.get(int(x), x) if isinstance(x, (int, float)) else task_id_map.get(x, x) for x in succ]

        task_recs.append(
            {
                task_id_key: tid,
                duration_key: info.get("duration"),
                demands_key: dem,
                successors_key: succ,
            }
        )

    if fmt == "json":
        obj = {
            nr_tasks_key: nr_tasks,
            nr_resources_key: nr_resources,
            "resources": res_ids,
            capacities_key: cap_recs,
            tasks_key: task_recs,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {nr_tasks_key}={nr_tasks}")
        lines.append(f"# {nr_resources_key}={nr_resources}")
        lines.append(f"# resources={res_str}")
        lines.append("resource_id,capacity")
        for cr in cap_recs:
            lines.append(f"{cr['resource_id']},{cr['capacity']}")
        lines.append(f"{task_id_key},{duration_key},{demands_key},{successors_key}")
        for tr in task_recs:
            lines.append(
                f"{tr[task_id_key]},{tr[duration_key]},"
                f"\"{_list_to_str(tr[demands_key])}\",\"{_succ_to_str(tr[successors_key])}\""
            )
        return "\n".join(lines)

    if fmt == "markdown_table":
        lines: List[str] = []

        if isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            if header_tmpl:
                lines.append(
                    _format_safe(
                        header_tmpl,
                        nr_tasks=nr_tasks,
                        nr_resources=nr_resources,
                        tasks=task_str,
                        resources=res_str,
                        **{nr_tasks_key: nr_tasks, nr_resources_key: nr_resources},
                    )
                )
                lines.append("")

        lines.append("| resource_id | capacity |")
        lines.append("|---|---|")
        for cr in cap_recs:
            lines.append(f"| {cr['resource_id']} | {cr['capacity']} |")

        lines.append("")
        lines.append(f"| {task_id_key} | {duration_key} | {demands_key} | {successors_key} |")
        lines.append("|---|---|---|---|")
        for tr in task_recs:
            dem_str = _list_to_str(tr[demands_key])
            succ_str = _succ_to_str(tr[successors_key])
            lines.append(f"| {tr[task_id_key]} | {tr[duration_key]} | {dem_str} | {succ_str} |")

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        nr_tasks=nr_tasks,
                        nr_resources=nr_resources,
                        tasks=task_str,
                        resources=res_str,
                        **{nr_tasks_key: nr_tasks, nr_resources_key: nr_resources},
                    )
                )

        return "\n".join(lines)

    # fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        cap_line_tmpl = nl_style.get("capacity_item_fields_line_template", "") or ""
        task_line_tmpl = nl_style.get("task_item_fields_line_template", "") or ""

        parts: List[str] = []
        base_vars = {
            "nr_tasks": nr_tasks,
            "nr_resources": nr_resources,
            "tasks": task_str,
            "resources": res_str,
            nr_tasks_key: nr_tasks,
            nr_resources_key: nr_resources,
        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        if cap_line_tmpl:
            for cr in cap_recs:
                vars_ = dict(base_vars)
                vars_.update(
                    {
                        "resource_id": cr.get("resource_id", ""),
                        "capacity": cr.get("capacity", ""),
                    }
                )
                parts.append(_format_safe(cap_line_tmpl, **vars_))

        if task_line_tmpl:
            for tr in task_recs:
                vars_ = dict(base_vars)
                vars_.update(
                    {
                        # generic keys
                        "task_id": tr.get(task_id_key, ""),
                        "duration": tr.get(duration_key, ""),
                        "demands": _list_to_str(tr.get(demands_key, [])),
                        "successors": _succ_to_str(tr.get(successors_key, [])),
                        # renamed keys
                        task_id_key: tr.get(task_id_key, ""),
                        duration_key: tr.get(duration_key, ""),
                        demands_key: _list_to_str(tr.get(demands_key, [])),
                        successors_key: _succ_to_str(tr.get(successors_key, [])),
                    }
                )
                parts.append(_format_safe(task_line_tmpl, **vars_))
        else:
            for tr in task_recs:
                parts.append(
                    f"{tr[task_id_key]}: dur={tr[duration_key]}, "
                    f"demands={_list_to_str(tr[demands_key])}, successors={_succ_to_str(tr[successors_key])}"
                )

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)

    return ""


def contextualize_instance_rcpsp(
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

    nr_tasks = core.get("nr_tasks")
    nr_resources = core.get("nr_resources")
    capacities = core.get("capacities")
    tasks = core.get("tasks")

    if not isinstance(nr_tasks, int) or not isinstance(nr_resources, int):
        return results
    if not isinstance(capacities, list) or len(capacities) != nr_resources:
        return results
    if not isinstance(tasks, list) or len(tasks) != nr_tasks:
        return results
    for t in range(nr_tasks):
        if not isinstance(tasks[t], dict):
            return results
        if "duration" not in tasks[t] or "demands" not in tasks[t] or "successors" not in tasks[t]:
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
    task_labels = [f"T{t+1}" for t in range(nr_tasks)]
    resource_labels = _build_labels(nr_resources, index_base)

    task_id_map = {t: task_labels[t] for t in range(nr_tasks)}
    resource_id_map = {r: resource_labels[r] for r in range(nr_resources)}

    solution_variant = _remap_solution_rcpsp(solution, task_id_map=task_id_map)

    input_text = rcpsp_render_input(
        fmt=fmt,
        nr_tasks=nr_tasks,
        nr_resources=nr_resources,
        capacities=capacities,
        tasks=tasks,
        task_id_map=task_id_map,
        resource_id_map=resource_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_rcpsp_instance(
        nr_tasks=nr_tasks,
        nr_resources=nr_resources,
        capacities=capacities,
        tasks=tasks,
        task_id_map=task_id_map,
        resource_id_map=resource_id_map,
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
    # RCPSP only
    desc = (
        "Resource-Constrained Project Scheduling (RCPSP): you have a set of tasks that form a project.\n\n"
        "Each task has a fixed duration, requires certain amounts of one or more limited resources while it runs, "
        "and may depend on other tasks (some tasks must finish before others can start).\n\n"
        "You must choose a start time for every task so that all precedence relations are respected and total resource "
        "usage never exceeds the available capacities.\n\n"
        "Your task is to produce a feasible schedule, typically aiming to finish the whole project as early as possible."
    )

    baseline_template = (
        "You are given a Resource-Constrained Project Scheduling (RCPSP) instance.\n\n"
        "Problem description:\n"
        "- There are multiple tasks that must be scheduled over time.\n"
        "- Each task has a `duration`, a resource `demands` vector, and a list of `successors`.\n"
        "- If task A lists B as a successor, then A must finish before B can start.\n"
        "- While a task is running, it consumes its demanded amount of each resource.\n"
        "- At any time, total consumption of each resource must not exceed its capacity.\n\n"
        "You will receive the instance below:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your schedule using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [\n'
        "    {\n"
        '      "task": <task_id>,\n'
        '      "start": <start_time>,\n'
        '      "end": <end_time>\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "Rules:\n"
        "- The list must contain every task exactly once.\n"
        "- For each record: `end = start + duration` (use the task's given duration).\n"
        "- All precedence relations must be respected (a task must finish before any successor starts).\n"
        "- At any time, total resource usage must not exceed the given capacities.\n"
        "- Use identifiers exactly as they appear in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [\n'
        "    {\n"
        '      "task": <task_id>,\n'
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
        {"name": "nr_tasks", "description": "number of tasks"},
        {"name": "nr_resources", "description": "number of resources"},
        {"name": "capacities", "description": "resource capacity list"},
    ]
    task_item_fields = [
        {"name": "task_id", "description": "task identifier"},
        {"name": "duration", "description": "task duration"},
        {"name": "demands", "description": "resource demand vector while running"},
        {"name": "successors", "description": "tasks that must start after this task finishes (may be empty)"},
    ]
    allowed = ["nr_tasks", "nr_resources", "capacities", "task_id", "duration", "demands", "successors"]
    return {
        "global_fields": global_fields,
        "task_item_fields": task_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize RCPSP instances into NL/JSON/CSV/Markdown.")
    parser.add_argument("--problem_type", type=str, default="RCPSP", choices=["RCPSP"])
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
        contextualize_fn=contextualize_instance_rcpsp,
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

python -m step2_contextualization.contextualize_rcpsp \
  --problem_type RCPSP \
  --instance_dir ./step1_instance_creation/generated_data/RCPSP \
  --output_root_dir step2_contextualization/dataset/RCPSP \
  --k_per_call 20 \
  --n_target 50

'''
