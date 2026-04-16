import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for GAP (Generalized Assignment Problem).

Instance format example:
{
  "instance": {
    "resource_consumption": [[...], ...],  # shape: num_agents x num_tasks
    "assignment_costs": [[...], ...],      # shape: num_agents x num_tasks
    "capacities": [...],                   # length: num_agents
    "objective": <number>                  # optional
  },
  "solution": [a0, a1, ..., a(n-1)],       # length: num_tasks, task j assigned to agent solution[j]
  "obj": <number>,
  "problem_type": "GAP"
}

We ALWAYS render inputs using PAIR lists:
- resource pairs: (agent_id, task_id, consumption)
- cost pairs:     (agent_id, task_id, cost)
- capacity pairs: (agent_id, capacity)

Diagonal notion does not apply (this is bipartite agent-task data).
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _remap_solution_gap(solution: Any, agent_id_map: Optional[Dict[int, Any]] = None) -> Any:
    """
    solution: list, length = num_tasks
      solution[j] = agent index assigned to task j
    If agent_id_map is provided, remap agent indices to labels.
    """
    if solution is None:
        return None
    if not isinstance(solution, list):
        return solution
    if agent_id_map is None:
        return solution
    out: List[Any] = []
    for a in solution:
        try:
            out.append(agent_id_map.get(int(a), a))
        except Exception:
            out.append(a)
    return out


def _build_labeled_gap_instance(
    consumption: List[List[float]],
    costs: List[List[float]],
    capacities: List[float],
    agent_id_map: Optional[Dict[int, Any]],
    task_id_map: Optional[Dict[int, Any]],
) -> Dict[str, Any]:
    m = len(consumption)  # agents
    n = len(consumption[0]) if consumption else 0  # tasks

    agents = [agent_id_map.get(i, i) if agent_id_map else i for i in range(m)]
    tasks = [task_id_map.get(j, j) if task_id_map else j for j in range(n)]

    resource_pairs: List[Dict[str, Any]] = []
    cost_pairs: List[Dict[str, Any]] = []
    capacity_pairs: List[Dict[str, Any]] = []

    for i in range(m):
        aid = agent_id_map.get(i, i) if agent_id_map else i
        cap = capacities[i] if i < len(capacities) else None
        if cap is not None:
            capacity_pairs.append({"agent_id": aid, "capacity": cap})

        for j in range(n):
            tid = task_id_map.get(j, j) if task_id_map else j
            resource_pairs.append({"agent_id": aid, "task_id": tid, "consumption": consumption[i][j]})
            cost_pairs.append({"agent_id": aid, "task_id": tid, "cost": costs[i][j]})

    return {
        "problem_type": "GAP",
        "num_agents": m,
        "num_tasks": n,
        "agents": agents,
        "tasks": tasks,
        "capacity_pairs": capacity_pairs,
        "resource_pairs": resource_pairs,
        "cost_pairs": cost_pairs,
    }


def gap_render_input(
    fmt: str,
    consumption: List[List[float]],
    costs: List[List[float]],
    capacities: List[float],
    agent_id_map: Optional[Dict[int, Any]],
    task_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    # defaults
    num_agents_key = "num_agents"
    num_tasks_key = "num_tasks"
    agents_key = "agents"
    tasks_key = "tasks"

    cap_agent_key, cap_val_key = "agent_id", "capacity"
    pair_agent_key, pair_task_key = "agent_id", "task_id"
    cons_val_key = "consumption"
    cost_val_key = "cost"

    # optional renaming via scenario_hint
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "num_agents":
                num_agents_key = new
            elif orig == "num_tasks":
                num_tasks_key = new
            elif orig == "agents":
                agents_key = new
            elif orig == "tasks":
                tasks_key = new

        for field in scenario_hint.get("capacity_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "agent_id":
                cap_agent_key = new
                pair_agent_key = new  # keep consistent
            elif orig == "capacity":
                cap_val_key = new

        for field in scenario_hint.get("pair_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "agent_id":
                pair_agent_key = new
                cap_agent_key = new
            elif orig == "task_id":
                pair_task_key = new
            elif orig == "consumption":
                cons_val_key = new
            elif orig == "cost":
                cost_val_key = new

    m = len(consumption)
    n = len(consumption[0]) if consumption else 0

    agent_ids = [agent_id_map.get(i, i) if agent_id_map else i for i in range(m)]
    task_ids = [task_id_map.get(j, j) if task_id_map else j for j in range(n)]
    agents_str = ", ".join(str(x) for x in agent_ids)
    tasks_str = ", ".join(str(x) for x in task_ids)

    capacity_pairs: List[Dict[str, Any]] = []
    resource_pairs: List[Dict[str, Any]] = []
    cost_pairs: List[Dict[str, Any]] = []

    for i in range(m):
        aid = agent_id_map.get(i, i) if agent_id_map else i
        cap = capacities[i] if i < len(capacities) else None
        if cap is not None:
            capacity_pairs.append({cap_agent_key: aid, cap_val_key: cap})
        for j in range(n):
            tid = task_id_map.get(j, j) if task_id_map else j
            resource_pairs.append({pair_agent_key: aid, pair_task_key: tid, cons_val_key: consumption[i][j]})
            cost_pairs.append({pair_agent_key: aid, pair_task_key: tid, cost_val_key: costs[i][j]})

    if fmt == "json":
        obj = {
            num_agents_key: m,
            num_tasks_key: n,
            agents_key: agent_ids,
            tasks_key: task_ids,
            "capacities": capacity_pairs,
            "resource": resource_pairs,
            "cost": cost_pairs,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_agents_key}={m}")
        lines.append(f"# {num_tasks_key}={n}")
        lines.append(f"# {agents_key}={agents_str}")
        lines.append(f"# {tasks_key}={tasks_str}")

        lines.append("")
        lines.append(f"{cap_agent_key},{cap_val_key}")
        for rec in capacity_pairs:
            lines.append(f"{rec[cap_agent_key]},{rec[cap_val_key]}")

        lines.append("")
        lines.append(f"{pair_agent_key},{pair_task_key},{cons_val_key}")
        for rec in resource_pairs:
            lines.append(f"{rec[pair_agent_key]},{rec[pair_task_key]},{rec[cons_val_key]}")

        lines.append("")
        lines.append(f"{pair_agent_key},{pair_task_key},{cost_val_key}")
        for rec in cost_pairs:
            lines.append(f"{rec[pair_agent_key]},{rec[pair_task_key]},{rec[cost_val_key]}")

        return "\n".join(lines)

    if fmt == "markdown_table":
        lines: List[str] = []

        # optional header/footer; NO line template
        if isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            if header_tmpl:
                lines.append(
                    _format_safe(
                        header_tmpl,
                        num_agents=m,
                        num_tasks=n,
                        agents=agents_str,
                        tasks=tasks_str,
                        **{num_agents_key: m, num_tasks_key: n, agents_key: agents_str, tasks_key: tasks_str},
                    )
                )
                lines.append("")

        lines.append("")
        lines.append(f"| {cap_agent_key} | {cap_val_key} |")
        lines.append("|---|---|")
        for rec in capacity_pairs:
            lines.append(f"| {rec[cap_agent_key]} | {rec[cap_val_key]} |")

        lines.append("")
        lines.append("")
        lines.append(f"| {pair_agent_key} | {pair_task_key} | {cons_val_key} |")
        lines.append("|---|---|---|")
        for rec in resource_pairs:
            lines.append(f"| {rec[pair_agent_key]} | {rec[pair_task_key]} | {rec[cons_val_key]} |")

        lines.append("")
        lines.append(f"| {pair_agent_key} | {pair_task_key} | {cost_val_key} |")
        lines.append("|---|---|---|")
        for rec in cost_pairs:
            lines.append(f"| {rec[pair_agent_key]} | {rec[pair_task_key]} | {rec[cost_val_key]} |")

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        num_agents=m,
                        num_tasks=n,
                        agents=agents_str,
                        tasks=tasks_str,
                        **{num_agents_key: m, num_tasks_key: n, agents_key: agents_str, tasks_key: tasks_str},
                    )
                )

        return "\n".join(lines)

    # fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        cap_line_tmpl = nl_style.get("capacity_item_fields_line_template", "") or ""

        # IMPORTANT: GAP NL uses ONE pair line template that includes BOTH consumption and cost
        pair_line_tmpl = nl_style.get("pair_item_fields_line_template", "") or ""

        parts: List[str] = []
        base_vars = {
            "num_agents": m,
            "num_tasks": n,
            "agents": agents_str,
            "tasks": tasks_str,
            num_agents_key: m,
            num_tasks_key: n,
            agents_key: agents_str,
            tasks_key: tasks_str,
        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        # capacities
        if cap_line_tmpl:
            for rec in capacity_pairs:
                parts.append(
                    _format_safe(
                        cap_line_tmpl,
                        **base_vars,
                        agent_id=rec.get(cap_agent_key, ""),
                        capacity=rec.get(cap_val_key, ""),
                    )
                )

        # agent-task pairs: include BOTH consumption and cost in the SAME line
        if pair_line_tmpl:
            # build quick lookup for cost by (agent_id, task_id)
            cost_map = {}
            for rec in cost_pairs:
                cost_map[(rec.get(pair_agent_key), rec.get(pair_task_key))] = rec.get(cost_val_key)

            for rec in resource_pairs:
                a = rec.get(pair_agent_key)
                t = rec.get(pair_task_key)
                parts.append(
                    _format_safe(
                        pair_line_tmpl,
                        **base_vars,
                        agent_id=a,
                        task_id=t,
                        consumption=rec.get(cons_val_key, ""),
                        cost=cost_map.get((a, t), ""),
                    )
                )

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)


def contextualize_instance_gap(
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

    consumption = core.get("resource_consumption")
    costs = core.get("assignment_costs")
    capacities = core.get("capacities", [])

    if not isinstance(consumption, list) or not consumption or not isinstance(consumption[0], list):
        return results
    if not isinstance(costs, list) or not costs or not isinstance(costs[0], list):
        return results

    m = len(consumption)
    n = len(consumption[0])

    if any(len(row) != n for row in consumption):
        return results
    if len(costs) != m or any(len(row) != n for row in costs):
        return results
    if not isinstance(capacities, list) or len(capacities) != m:
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

    # label maps
    # Agents use fixed "A1, A2, ..." labels (stable + readable)
    agent_labels = [f"A{i + 1}" for i in range(m)]
    task_labels = _build_labels(n, index_base)

    agent_id_map = {i: agent_labels[i] for i in range(m)}
    task_id_map = {j: task_labels[j] for j in range(n)}

    solution_variant = _remap_solution_gap(solution, agent_id_map=agent_id_map)

    input_text = gap_render_input(
        fmt=fmt,
        consumption=consumption,
        costs=costs,
        capacities=capacities,
        agent_id_map=agent_id_map,
        task_id_map=task_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_gap_instance(
        consumption=consumption,
        costs=costs,
        capacities=capacities,
        agent_id_map=agent_id_map,
        task_id_map=task_id_map,
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
    # GAP only
    desc = (
        "Generalized Assignment Problem (GAP): you are given a set of tasks and a set of agents. "
        "Each task must be assigned to exactly one agent, with no task skipped or duplicated, "
        "and each agent may be assigned multiple tasks.\n\n"

        "For every possible agent–task pair, the input specifies (1) the amount of the agent's "
        "capacity that the task would consume if assigned there, and (2) the cost incurred by "
        "that assignment. Each agent has a fixed total capacity.\n\n"

        "The sum of the resource consumptions of all tasks assigned to any agent must not exceed "
        "that agent’s capacity. Your goal is to find an assignment that satisfies all capacity "
        "constraints while minimizing the total assignment cost over all tasks."
    )

    baseline_template = (
        "You are given a Generalized Assignment Problem (GAP).\n\n"
        "Problem description:\n"
        "- There are multiple agents and multiple tasks.\n"
        "- Each task must be assigned to exactly one agent.\n"
        "- Each agent can take multiple tasks, but has a limited capacity.\n"
        "- For each agent–task pair, the input provides:\n"
        "  (1) resource consumption if that task is assigned to that agent\n"
        "  (2) assignment cost for assigning that task to that agent\n\n"
        "Objective:\n"
        "- Find an assignment that respects every agent's capacity.\n"
        "- Among all valid assignments, minimize the total assignment cost summed over all tasks.\n\n"
        "Important details:\n"
        "- The input is provided in PAIR format.\n"
        "- Capacity pairs specify each agent's total capacity.\n"
        "- Resource and cost pairs specify values for each (agent, task).\n\n"
        "You will receive the instance in the following PAIR format:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your decision using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [\n'
        '    <agent_id_for_first_task>,\n'
        '    <agent_id_for_second_task>,\n'
        '    ...,\n'
        '    <agent_id_for_last_task>\n'
        "  ]\n"
        "}\n"
        "```\n"
        "The list `solution` must have length equal to the number of tasks.\n"
        "The first element specifies the agent assigned to the first task,\n"
        "the second element specifies the agent assigned to the second task,\n"
        "and so on until the last task.\n"
        "Each entry must be a valid agent identifier from the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [\n'
        '    <agent_id_for_first_task>,\n'
        '    <agent_id_for_second_task>,\n'
        '    ...,\n'
        '    <agent_id_for_last_task>\n'
        "  ]\n"
        "}\n"
        "```\n"
    )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    # Minimal hint, no explicit "pairs" global field
    global_fields = [
        {"name": "num_agents", "description": "number of agents"},
        {"name": "num_tasks", "description": "number of tasks"},
        {"name": "agents", "description": "agent identifiers"},
        {"name": "tasks", "description": "task identifiers"},
    ]
    capacity_item_fields = [
        {"name": "agent_id", "description": "agent identifier"},
        {"name": "capacity", "description": "total capacity of this agent"},
    ]
    pair_item_fields = [
        {"name": "agent_id", "description": "agent identifier"},
        {"name": "task_id", "description": "task identifier"},
        {"name": "consumption", "description": "resource consumed if task is assigned to agent"},
        {"name": "cost", "description": "assignment cost if task is assigned to agent"},
    ]
    allowed = [
        "num_agents", "num_tasks", "agents", "tasks",
        "agent_id", "task_id", "capacity", "consumption", "cost",
    ]
    return {
        "global_fields": global_fields,
        "capacity_item_fields": capacity_item_fields,
        "pair_item_fields": pair_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize GAP instances into NL/JSON/CSV/Markdown (PAIR input only).")
    parser.add_argument("--problem_type", type=str, default="GAP", choices=["GAP"])
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
        contextualize_fn=contextualize_instance_gap,
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
python -m step2_contextualization.contextualize_gap \
  --problem_type GAP \
  --instance_dir ./step1_instance_creation/generated_data/GAP \
  --output_root_dir step2_contextualization/dataset/GAP \
  --k_per_call 20 \
  --n_target 50


'''
