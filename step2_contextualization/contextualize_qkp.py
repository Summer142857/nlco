# step2_contextualization/contextualize_qkp.py
import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for QKP (Quadratic Knapsack Problem).

Instance example:
{
  "linear_coeffs": [p_i ...],
  "quadratic_coeffs": [[q_ij ...], ...],   # often upper-triangular-ish
  "weights": [w_i ...],
  "capacity": C,
  "objective": <optional>,
  "solution": [item_ids...],              # optional
  "obj": <number>,
  "problem_type": "QKP"
}

We ALWAYS render input using PAIR lists:
- item profits: (item_id, linear_profit)
- item weights: (item_id, weight)
- quadratic profits: (item_i_id, item_j_id, quadratic_profit)
We keep diagonal (i==j) pairs if present (and nonzero or explicitly included).
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _remap_solution_qkp(solution: Any, item_id_map: Optional[Dict[int, Any]]) -> Any:
    if solution is None:
        return None
    if item_id_map is None:
        return solution
    if isinstance(solution, list):
        out = []
        for x in solution:
            try:
                out.append(item_id_map.get(int(x), x))
            except Exception:
                out.append(x)
        return out
    return solution


def _build_labeled_qkp_instance(
    linear: List[float],
    quad: List[List[float]],
    weights: List[float],
    capacity: float,
    item_id_map: Optional[Dict[int, Any]],
    include_zero_quadratic: bool = False,
) -> Dict[str, Any]:
    n = len(linear)
    items = [item_id_map.get(i, i) if item_id_map else i for i in range(n)]

    linear_pairs = [{"item_id": items[i], "linear_profit": linear[i]} for i in range(n)]
    weight_pairs = [{"item_id": items[i], "weight": weights[i]} for i in range(n)]

    quad_pairs: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            val = quad[i][j]
            if (not include_zero_quadratic) and (val == 0):
                continue
            quad_pairs.append(
                {"item_i_id": items[i], "item_j_id": items[j], "quadratic_profit": val}
            )

    return {
        "problem_type": "QKP",
        "num_items": n,
        "capacity": capacity,
        "items": items,
        "linear_pairs": linear_pairs,
        "weight_pairs": weight_pairs,
        "quadratic_pairs": quad_pairs,
    }


def qkp_render_input(
    fmt: str,
    linear: List[float],
    quad: List[List[float]],
    weights: List[float],
    capacity: float,
    item_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
    include_zero_quadratic: bool = False,
) -> str:
    # -------------------------
    # default field names
    # -------------------------
    num_key = "num_items"
    items_key = "items"
    cap_key = "capacity"

    item_id_key = "item_id"
    linear_key = "linear_profit"
    weight_key = "weight"

    qi_key = "item_i_id"
    qj_key = "item_j_id"
    qv_key = "quadratic_profit"

    # -------------------------
    # optional renaming (lightweight)
    # -------------------------
    if scenario_hint and isinstance(scenario_hint, dict):
        for f in scenario_hint.get("global_fields", []):
            orig, new = f.get("name"), f.get("new_name")
            if not new:
                continue
            if orig == "num_items":
                num_key = new
            elif orig == "items":
                items_key = new
            elif orig == "capacity":
                cap_key = new

        for f in scenario_hint.get("linear_item_fields", []):
            orig, new = f.get("name"), f.get("new_name")
            if not new:
                continue
            if orig == "item_id":
                item_id_key = new
            elif orig == "linear_profit":
                linear_key = new

        for f in scenario_hint.get("weight_item_fields", []):
            orig, new = f.get("name"), f.get("new_name")
            if not new:
                continue
            if orig == "item_id":
                item_id_key = new
            elif orig == "weight":
                weight_key = new

        for f in scenario_hint.get("quadratic_item_fields", []):
            orig, new = f.get("name"), f.get("new_name")
            if not new:
                continue
            if orig == "item_i_id":
                qi_key = new
            elif orig == "item_j_id":
                qj_key = new
            elif orig == "quadratic_profit":
                qv_key = new

    # -------------------------
    # build ids + pair records
    # -------------------------
    n = len(linear)
    item_ids = [item_id_map.get(i, i) if item_id_map else i for i in range(n)]
    items_str = ", ".join(str(x) for x in item_ids)

    linear_pairs = [{item_id_key: item_ids[i], linear_key: linear[i]} for i in range(n)]
    weight_pairs = [{item_id_key: item_ids[i], weight_key: weights[i]} for i in range(n)]

    quad_pairs: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = quad[i][j]
            if (not include_zero_quadratic) and (val == 0):
                continue
            quad_pairs.append({qi_key: item_ids[i], qj_key: item_ids[j], qv_key: val})

    # -------------------------
    # render
    # -------------------------
    if fmt == "json":
        obj = {
            num_key: n,
            cap_key: capacity,
            items_key: item_ids,
            "linear": linear_pairs,
            "weights": weight_pairs,
            "quadratic": quad_pairs,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_key}={n}")
        lines.append(f"# {cap_key}={capacity}")
        lines.append(f"# {items_key}={items_str}")
        lines.append("")
        lines.append(f"{item_id_key},{linear_key}")
        for r in linear_pairs:
            lines.append(f"{r[item_id_key]},{r[linear_key]}")
        lines.append("")
        lines.append(f"{item_id_key},{weight_key}")
        for r in weight_pairs:
            lines.append(f"{r[item_id_key]},{r[weight_key]}")
        lines.append("")
        lines.append(f"{qi_key},{qj_key},{qv_key}")
        for r in quad_pairs:
            lines.append(f"{r[qi_key]},{r[qj_key]},{r[qv_key]}")
        return "\n".join(lines)

    if fmt == "markdown_table":
        # header/footer allowed, but NO line templates
        lines: List[str] = []
        if isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            if header_tmpl:
                lines.append(
                    _format_safe(
                        header_tmpl,
                        num_items=n,
                        items=items_str,
                        capacity=capacity,
                        **{num_key: n, items_key: items_str, cap_key: capacity},
                    )
                )
                lines.append("")

        lines.append(f"| {item_id_key} | {linear_key} |")
        lines.append("|---|---|")
        for r in linear_pairs:
            lines.append(f"| {r[item_id_key]} | {r[linear_key]} |")

        lines.append("")
        lines.append(f"| {item_id_key} | {weight_key} |")
        lines.append("|---|---|")
        for r in weight_pairs:
            lines.append(f"| {r[item_id_key]} | {r[weight_key]} |")

        lines.append("")
        lines.append(f"| {qi_key} | {qj_key} | {qv_key} |")
        lines.append("|---|---|---|")
        for r in quad_pairs:
            lines.append(f"| {r[qi_key]} | {r[qj_key]} | {r[qv_key]} |")

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        num_items=n,
                        items=items_str,
                        capacity=capacity,
                        **{num_key: n, items_key: items_str, cap_key: capacity},
                    )
                )
        return "\n".join(lines)

    # fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        linear_tmpl = nl_style.get("linear_item_fields_line_template", "") or ""
        weight_tmpl = nl_style.get("weight_item_fields_line_template", "") or ""
        quad_tmpl = nl_style.get("quadratic_item_fields_line_template", "") or ""

        parts: List[str] = []
        base_vars = {
            "num_items": n,
            "items": items_str,
            "capacity": capacity,
            num_key: n,
            items_key: items_str,
            cap_key: capacity,
        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        if linear_tmpl:
            for r in linear_pairs:
                vars_ = dict(base_vars)
                vars_.update(
                    {
                        "item_id": r.get(item_id_key, ""),
                        "linear_profit": r.get(linear_key, ""),
                        item_id_key: r.get(item_id_key, ""),
                        linear_key: r.get(linear_key, ""),
                    }
                )
                parts.append(_format_safe(linear_tmpl, **vars_))

        if weight_tmpl:
            for r in weight_pairs:
                vars_ = dict(base_vars)
                vars_.update(
                    {
                        "item_id": r.get(item_id_key, ""),
                        "weight": r.get(weight_key, ""),
                        item_id_key: r.get(item_id_key, ""),
                        weight_key: r.get(weight_key, ""),
                    }
                )
                parts.append(_format_safe(weight_tmpl, **vars_))

        if quad_tmpl:
            for r in quad_pairs:
                vars_ = dict(base_vars)
                vars_.update(
                    {
                        "item_i_id": r.get(qi_key, ""),
                        "item_j_id": r.get(qj_key, ""),
                        "quadratic_profit": r.get(qv_key, ""),
                        qi_key: r.get(qi_key, ""),
                        qj_key: r.get(qj_key, ""),
                        qv_key: r.get(qv_key, ""),
                    }
                )
                parts.append(_format_safe(quad_tmpl, **vars_))

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)

    return ""


def contextualize_instance_qkp(
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

    core = inst.get("instance") if isinstance(inst.get("instance"), dict) else inst
    solution = inst.get("solution")
    obj = inst.get("obj")
    problem_type = inst.get("problem_type", task_name)

    linear = core.get("linear_coeffs")
    quad = core.get("quadratic_coeffs")
    weights = core.get("weights")
    capacity = core.get("capacity")

    if not isinstance(linear, list) or not isinstance(weights, list) or not isinstance(quad, list):
        return results
    if capacity is None:
        return results

    n = len(linear)
    if len(weights) != n:
        return results
    if len(quad) != n or any((not isinstance(row, list) or len(row) != n) for row in quad):
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

    # label map
    labels = _build_labels(n, index_base)
    item_id_map = {i: labels[i] for i in range(n)}

    solution_variant = _remap_solution_qkp(solution, item_id_map=item_id_map)

    input_text = qkp_render_input(
        fmt=fmt,
        linear=linear,
        quad=quad,
        weights=weights,
        capacity=capacity,
        item_id_map=item_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
        include_zero_quadratic=False,  # keep compact
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_qkp_instance(
        linear=linear,
        quad=quad,
        weights=weights,
        capacity=capacity,
        item_id_map=item_id_map,
        include_zero_quadratic=False,
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
    # QKP only
    desc = (
        "Quadratic Knapsack Problem (QKP): you are choosing a subset of items to carry, with a single capacity limit.\n\n"
        "Each item has a weight, and selecting it contributes a base profit. In addition, some pairs of items create extra "
        "profit (or bonus) when they are both selected.\n\n"
        "Your task is to pick items whose total weight stays within the capacity, while making the total profit (base profits "
        "plus all applicable pair bonuses) as large as possible."
    )

    baseline_template = (
        "You are given a Quadratic Knapsack Problem (QKP).\n\n"
        "Problem description:\n"
        "- There are n items.\n"
        "- Each item i has a weight w(i) and a base profit p(i).\n"
        "- There are also pair bonuses q(i, j): if you select both items i and j, you gain additional profit q(i, j).\n"
        "- Your selection must respect a single capacity limit: the total weight of selected items cannot exceed the capacity.\n\n"
        "Objective:\n"
        "- Choose a subset of items that maximizes total profit.\n"
        "- Total profit = sum of selected base profits + sum of all pair bonuses whose two items are both selected.\n\n"
        "Important details:\n"
        "- The instance is provided in PAIR format (item profits, item weights, and quadratic pair bonuses).\n"
        "- Quadratic pairs may include diagonal (i == i) entries; treat them as additional profit that applies when item i is selected.\n"
        "- Your answer must list the selected item identifiers exactly as they appear in the input.\n\n"
        "You will receive the instance in the following PAIR format:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your decision using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [<selected_item_id>, <selected_item_id>, ...]\n'
        "}\n"
        "```\n"
        "The list `solution` is the set of items you chose to take.\n"
        "Each identifier must match exactly those used in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [<selected_item_id>, <selected_item_id>, ...]\n'
        "}\n"
        "```\n"
    )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    # Minimal hint; no explicit “pairs” field in description
    global_fields = [
        {"name": "num_items", "description": "number of items"},
        {"name": "items", "description": "item identifiers"},
        {"name": "capacity", "description": "maximum total weight allowed"},
    ]
    linear_item_fields = [
        {"name": "item_id", "description": "item identifier"},
        {"name": "linear_profit", "description": "base profit of selecting the item"},
    ]
    weight_item_fields = [
        {"name": "item_id", "description": "item identifier"},
        {"name": "weight", "description": "weight of the item"},
    ]
    quadratic_item_fields = [
        {"name": "item_i_id", "description": "first item id in the pair"},
        {"name": "item_j_id", "description": "second item id in the pair"},
        {"name": "quadratic_profit", "description": "bonus profit if both items are selected"},
    ]
    allowed = [
        "num_items", "items", "capacity",
        "item_id", "linear_profit", "weight",
        "item_i_id", "item_j_id", "quadratic_profit",
    ]
    return {
        "global_fields": global_fields,
        "linear_item_fields": linear_item_fields,
        "weight_item_fields": weight_item_fields,
        "quadratic_item_fields": quadratic_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize QKP instances into NL/JSON/CSV/Markdown (PAIR input only).")
    parser.add_argument("--problem_type", type=str, default="QKP", choices=["QKP"])
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
        contextualize_fn=contextualize_instance_qkp,
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
python -m step2_contextualization.contextualize_qkp \
  --problem_type QKP \
  --instance_dir ./step1_instance_creation/generated_data/QKP \
  --output_root_dir step2_contextualization/dataset/QKP \
  --k_per_call 20 \
  --n_target 50


'''
