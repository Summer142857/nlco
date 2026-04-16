# step2_contextualization/contextualize_packing.py
import json
import random
from typing import Any, Dict, List, Optional, Tuple

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for packing / knapsack problems:

- 2SP : 2D Packing (bin_width/bin_height, items with width/height/demand, placement solution)
- BPP : Bin Packing (weights list, capacity, solution = bins of item indices)
- CSP : Cutting Stock (weights, demands, capacity, solution_patterns = list of patterns dict)
- KP  : 0/1 Knapsack (weights, profits, capacity, solution = binary vector)
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]



def _build_label_map(n: int, index_base) -> Dict[int, Any]:
    labels = make_labels(n, index_base)
    return {i: labels[i] for i in range(n)}


def _try_int(x: Any) -> Any:
    try:
        return int(x)
    except Exception:
        return x


def _get_renaming_maps(scenario_hint: Optional[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return:
      global_rename: canonical_global_name -> renamed_name
      item_rename:   canonical_item_name   -> renamed_name
    """
    global_rename: Dict[str, str] = {}
    item_rename: Dict[str, str] = {}
    if not scenario_hint:
        return global_rename, item_rename

    for f in scenario_hint.get("global_fields", []):
        name = f.get("name")
        new_name = f.get("new_name")
        if name and new_name:
            global_rename[name] = new_name

    for f in scenario_hint.get("item_fields", []):
        name = f.get("name")
        new_name = f.get("new_name")
        if name and new_name:
            item_rename[name] = new_name

    return global_rename, item_rename


def _ctx_with_aliases(ctx: Dict[str, Any], global_rename: Dict[str, str], item_rename: Dict[str, str]) -> Dict[str, Any]:
    """
    Add alias keys into ctx so that BOTH:
      - canonical placeholders like {bin_width}
      - renamed placeholders like {total_bin_width}
    can be resolved.

    Rule:
      if global_rename["bin_width"] == "total_bin_width"
      then ctx["total_bin_width"] = ctx["bin_width"]  (if exists)

    Same for item fields.
    """
    out = dict(ctx)

    for canon, renamed in global_rename.items():
        if canon in out and renamed not in out:
            out[renamed] = out[canon]

    for canon, renamed in item_rename.items():
        if canon in out and renamed not in out:
            out[renamed] = out[canon]

    return out


# ---------------------------------------------------------------------
# 1) 2SP
# ---------------------------------------------------------------------

def _2sp_solution_variant(solution: Any, item_id_map: Dict[int, Any]) -> Any:
    if not isinstance(solution, list):
        return solution

    out = []
    for p in solution:
        if not isinstance(p, dict):
            out.append(p)
            continue
        q = dict(p)
        it = q.get("item_type", None)
        if it is not None:
            it_i = int(it)
            q["item_type"] = item_id_map.get(it_i, it_i)
        out.append(q)
    return out



def _2sp_instance_variant(inst_core: Dict[str, Any], item_id_map: Dict[int, Any]) -> Dict[str, Any]:
    items = inst_core.get("items", [])
    items_slim = []
    for i, it in enumerate(items):
        items_slim.append({
            "item_id": item_id_map[i],
            "width": it.get("width"),
            "height": it.get("height"),
        })

    return {
        "problem_type": "2SP",
        "bin_width": inst_core.get("bin_width"),
        "bin_height": inst_core.get("bin_height"),
        "items": items_slim,
        "num_item_types": len(items_slim),
    }



def _2sp_render_input(
    fmt: str,
    inst_core: Dict[str, Any],
    item_id_map: Dict[int, Any],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:

    global_rename, item_rename = _get_renaming_maps(scenario_hint)

    items = inst_core.get("items", [])
    bw = inst_core.get("bin_width")
    bh = inst_core.get("bin_height")

    base_ctx = {
        "bin_width": bw,
        "bin_height": bh,
        "num_item_types": len(items),
    }
    base_ctx = _ctx_with_aliases(base_ctx, global_rename, item_rename)

    # keys for json/csv/md (still honor renaming)
    bin_w_key = global_rename.get("bin_width", "bin_width")
    bin_h_key = global_rename.get("bin_height", "bin_height")
    items_key = global_rename.get("items", "items")  # optional
    id_key = item_rename.get("item_type", "item_id")
    w_key = item_rename.get("width", "width")
    h_key = item_rename.get("height", "height")

    if fmt == "json":
        obj = {
            bin_w_key: bw,
            bin_h_key: bh,
            items_key: [{id_key: item_id_map[i], w_key: it.get("width"), h_key: it.get("height")}
                        for i, it in enumerate(items)],
        }

        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines = [
            f"# {bin_w_key}={bw}",
            f"# {bin_h_key}={bh}",
        ]
        cols = [id_key, w_key, h_key]
        lines.append(",".join(cols))
        for i, it in enumerate(items):
            lines.append(",".join(map(str, [item_id_map[i], it.get("width"), it.get("height")])))
        return "\n".join(lines)

    if fmt == "markdown_table":
        lines = [
            f"- **{bin_w_key}**: {bw}",
            f"- **{bin_h_key}**: {bh}",
            "",
        ]
        cols = [id_key, w_key, h_key]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for i, it in enumerate(items):
            lines.append("| " + " | ".join(map(str, [item_id_map[i], it.get("width"), it.get("height")])) + " |")
        return "\n".join(lines)

    if fmt == "nl" and isinstance(nl_style, dict):
        out: List[str] = []

        header_t = nl_style.get("header", "")
        line_t = nl_style.get("line_template", "")
        footer_t = nl_style.get("footer", "")

        if header_t:
            out.append(_format_safe(header_t, **base_ctx))

        if line_t:
            for i, it in enumerate(items):
                line_ctx = {
                    **base_ctx,
                    "item_type": item_id_map[i],
                    "width": it.get("width"),
                    "height": it.get("height"),
                }

                line_ctx = _ctx_with_aliases(line_ctx, global_rename, item_rename)
                out.append(_format_safe(line_t, **line_ctx))

        if footer_t:
            out.append(_format_safe(footer_t, **base_ctx))

        return "\n".join(out)

    return ""



# ---------------------------------------------------------------------
# 2) BPP
# ---------------------------------------------------------------------

def _bpp_solution_variant(solution: Any, item_id_map: Dict[int, Any]) -> Any:
    if not isinstance(solution, list):
        return solution
    out = []
    for b in solution:
        if isinstance(b, list):
            out.append([item_id_map.get(int(i), i) for i in b])
        else:
            out.append(b)
    return out


def _bpp_instance_variant(weights: List[Any], cap: Any, item_id_map: Dict[int, Any]) -> Dict[str, Any]:
    return {
        "problem_type": "BPP",
        "bin_capacity": cap,
        "n_items": len(weights),
        "items": [{"item_id": item_id_map[i], "weight": weights[i]} for i in range(len(weights))],
    }


def _bpp_render_input(
    fmt: str,
    weights: List[Any],
    cap: Any,
    item_id_map: Dict[int, Any],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    global_rename, item_rename = _get_renaming_maps(scenario_hint)

    # renamed keys for structured formats
    cap_key = global_rename.get("bin_capacity", "bin_capacity")
    n_key = global_rename.get("n_items", "n_items")
    item_key = item_rename.get("item_id", "item_id")
    w_key = item_rename.get("weight", "weight")

    rows = [{item_key: item_id_map[i], w_key: weights[i]} for i in range(len(weights))]

    # canonical ctx for NL + alias
    base_ctx = {
        "bin_capacity": cap,
        "n_items": len(weights),
    }
    base_ctx = _ctx_with_aliases(base_ctx, global_rename, item_rename)

    if fmt == "json":
        obj = {cap_key: cap, n_key: len(weights), "items": rows}
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines = [f"# {cap_key}={cap}", f"# {n_key}={len(weights)}", f"{item_key},{w_key}"]
        for r in rows:
            lines.append(f"{r[item_key]},{r[w_key]}")
        return "\n".join(lines)

    if fmt == "markdown_table":
        lines = [f"- **{cap_key}**: {cap}", f"- **{n_key}**: {len(weights)}", ""]
        cols = [item_key, w_key]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    if fmt == "nl" and isinstance(nl_style, dict):
        out: List[str] = []

        if nl_style.get("header"):
            out.append(_format_safe(nl_style["header"], **base_ctx))

        lt = nl_style.get("line_template", "")
        if lt:
            for r in rows:
                # provide BOTH canonical and renamed keys for per-item rendering
                line_ctx = {
                    **base_ctx,
                    "item_id": r[item_key],
                    "weight": r[w_key],
                    item_key: r[item_key],
                    w_key: r[w_key],
                }
                line_ctx = _ctx_with_aliases(line_ctx, global_rename, item_rename)
                out.append(_format_safe(lt, **line_ctx))

        if nl_style.get("footer"):
            out.append(_format_safe(nl_style["footer"], **base_ctx))

        return "\n".join(out)

    return ""


# ---------------------------------------------------------------------
# 3) CSP
# ---------------------------------------------------------------------

def _csp_solution_variant(solution_patterns: Any, item_id_map: Dict[int, Any]) -> Any:
    if not isinstance(solution_patterns, list):
        return solution_patterns
    out = []
    for p in solution_patterns:
        if not isinstance(p, dict):
            out.append(p)
            continue
        q = {}
        for k, v in p.items():
            kk = _try_int(k)
            if isinstance(kk, int):
                q[str(item_id_map.get(kk, kk))] = v
            else:
                q[str(kk)] = v
        out.append(q)
    return out


def _csp_instance_variant(inst_core: Dict[str, Any], item_id_map: Dict[int, Any]) -> Dict[str, Any]:
    cap = inst_core.get("bin_capacity")
    widths = inst_core.get("weights", [])
    demands = inst_core.get("demands", [])
    n = len(widths)

    items = []
    for i in range(n):
        items.append({
            "item_id": item_id_map[i],
            "width": widths[i],
            "demand": demands[i] if i < len(demands) else None,
        })

    return {
        "problem_type": "CSP",
        "bin_capacity": cap,
        "n_items": n,
        "items": items,
    }



def _csp_render_input(
    fmt: str,
    inst_core: Dict[str, Any],
    item_id_map: Dict[int, Any],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    global_rename, item_rename = _get_renaming_maps(scenario_hint)

    cap = inst_core.get("bin_capacity")
    widths = inst_core.get("weights", [])
    demands = inst_core.get("demands", [])
    n = len(widths)

    # renamed keys for structured formats
    cap_key = global_rename.get("bin_capacity", "bin_capacity")
    item_key = item_rename.get("item_id", "item_id")
    w_key = item_rename.get("width", "width")
    d_key = item_rename.get("demand", "demand")

    rows = []
    for i in range(n):
        rows.append({
            item_key: item_id_map[i],
            w_key: widths[i],
            d_key: demands[i] if i < len(demands) else None
        })

    # NL ctx: canonical + aliases
    base_ctx = {"bin_capacity": cap, "n_items": n}
    base_ctx = _ctx_with_aliases(base_ctx, global_rename, item_rename)

    if fmt == "json":
        obj = {cap_key: cap, "items": rows}
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines = [f"# {cap_key}={cap}", f"{item_key},{w_key},{d_key}"]
        for r in rows:
            lines.append(f"{r[item_key]},{r[w_key]},{r[d_key]}")
        return "\n".join(lines)

    if fmt == "markdown_table":
        lines = [f"- **{cap_key}**: {cap}", ""]
        cols = [item_key, w_key, d_key]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    if fmt == "nl" and isinstance(nl_style, dict):
        out: List[str] = []

        if nl_style.get("header"):
            out.append(_format_safe(nl_style["header"], **base_ctx))

        lt = nl_style.get("line_template", "")
        if lt:
            for r in rows:
                # provide BOTH canonical and renamed keys
                line_ctx = {
                    **base_ctx,
                    "item_id": r[item_key],
                    "width": r[w_key],
                    "demand": r[d_key],
                    item_key: r[item_key],
                    w_key: r[w_key],
                    d_key: r[d_key],
                }
                line_ctx = _ctx_with_aliases(line_ctx, global_rename, item_rename)
                out.append(_format_safe(lt, **line_ctx))

        if nl_style.get("footer"):
            out.append(_format_safe(nl_style["footer"], **base_ctx))

        return "\n".join(out)

    return ""


# ---------------------------------------------------------------------
# 4) KP
# ---------------------------------------------------------------------

def _kp_solution_variant(solution: Any, item_id_map: Dict[int, Any]) -> Any:
    # Expect solution to be a binary list; convert to {label: 0/1}
    if not isinstance(solution, list):
        return solution
    out = {}
    for i, v in enumerate(solution):
        out[str(item_id_map.get(i, i))] = int(v) if v is not None else v
    return out


def _kp_instance_variant(inst_core: Dict[str, Any], item_id_map: Dict[int, Any]) -> Dict[str, Any]:
    cap = inst_core.get("capacity")
    weights = inst_core.get("weights", [])
    profits = inst_core.get("profits", [])
    n = inst_core.get("n_items", len(weights))

    items = []
    for i in range(n):
        items.append({
            "item_id": item_id_map.get(i, i),
            "weight": weights[i] if i < len(weights) else None,
            "profit": profits[i] if i < len(profits) else None,
        })

    return {
        "problem_type": "KP",
        "capacity": cap,
        "n_items": n,
        "items": items,
    }



def _kp_render_input(
    fmt: str,
    inst_core: Dict[str, Any],
    item_id_map: Dict[int, Any],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    global_rename, item_rename = _get_renaming_maps(scenario_hint)

    cap = inst_core.get("capacity")
    weights = inst_core.get("weights", [])
    profits = inst_core.get("profits", [])
    n = len(weights)

    # renamed keys for structured formats
    cap_key = global_rename.get("capacity", "capacity")
    item_key = item_rename.get("item_id", "item_id")
    w_key = item_rename.get("weight", "weight")
    p_key = item_rename.get("profit", "profit")

    rows = []
    for i in range(n):
        rows.append({
            item_key: item_id_map[i],
            w_key: weights[i],
            p_key: profits[i] if i < len(profits) else None
        })

    # NL ctx
    base_ctx = {"capacity": cap, "n_items": n}
    base_ctx = _ctx_with_aliases(base_ctx, global_rename, item_rename)

    if fmt == "json":
        obj = {cap_key: cap, "items": rows}
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines = [f"# {cap_key}={cap}", f"{item_key},{w_key},{p_key}"]
        for r in rows:
            lines.append(f"{r[item_key]},{r[w_key]},{r[p_key]}")
        return "\n".join(lines)

    if fmt == "markdown_table":
        lines = [f"- **{cap_key}**: {cap}", ""]
        cols = [item_key, w_key, p_key]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    if fmt == "nl" and isinstance(nl_style, dict):
        out: List[str] = []

        if nl_style.get("header"):
            out.append(_format_safe(nl_style["header"], **base_ctx))

        lt = nl_style.get("line_template", "")
        if lt:
            for r in rows:
                line_ctx = {
                    **base_ctx,
                    "item_id": r[item_key],
                    "weight": r[w_key],
                    "profit": r[p_key],
                    item_key: r[item_key],
                    w_key: r[w_key],
                    p_key: r[p_key],
                }
                line_ctx = _ctx_with_aliases(line_ctx, global_rename, item_rename)
                out.append(_format_safe(lt, **line_ctx))

        if nl_style.get("footer"):
            out.append(_format_safe(nl_style["footer"], **base_ctx))

        return "\n".join(out)

    return ""



def contextualize_instance_packing(
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

    problem_type = inst.get("problem_type", task_name).upper()
    solution = inst.get("solution")
    obj = inst.get("obj", inst.get("objective"))

    # choose template
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

    # dispatch by problem type
    if problem_type == "2SP":
        inst_core = inst.get("instance", inst)
        if not isinstance(inst_core, dict) or "items" not in inst_core:
            return results

        item_id_map = _build_label_map(len(inst_core.get("items", [])), index_base)

        input_text = _2sp_render_input(fmt, inst_core, item_id_map, nl_style, scenario_hint)
        instance_variant = _2sp_instance_variant(inst_core, item_id_map)
        solution_variant = _2sp_solution_variant(solution, item_id_map)

    elif problem_type == "BPP":
        weights = inst.get("instance")
        cap = inst.get("bin_capacity")
        if not isinstance(weights, list) or cap is None:
            return results
        item_id_map = _build_label_map(len(weights), index_base)
        input_text = _bpp_render_input(fmt, weights, cap, item_id_map, nl_style, scenario_hint)
        instance_variant = _bpp_instance_variant(weights, cap, item_id_map)
        solution_variant = _bpp_solution_variant(solution, item_id_map)
        inst_core = weights

    elif problem_type == "CSP":
        inst_core = inst
        if not isinstance(inst_core, dict) or "weights" not in inst_core or "demands" not in inst_core:
            return results
        item_id_map = _build_label_map(len(inst_core.get("weights", [])), index_base)
        input_text = _csp_render_input(fmt, inst_core, item_id_map, nl_style, scenario_hint)
        instance_variant = _csp_instance_variant(inst_core, item_id_map)
        solution_variant = _csp_solution_variant(inst.get("solution_patterns"), item_id_map)
        solution = inst.get("solution_patterns")

    elif problem_type == "KP":
        inst_core = inst
        if not isinstance(inst_core, dict) or "weights" not in inst_core or "profits" not in inst_core:
            return results
        item_id_map = _build_label_map(len(inst_core.get("weights", [])), index_base)
        input_text = _kp_render_input(fmt, inst_core, item_id_map, nl_style, scenario_hint)
        instance_variant = _kp_instance_variant(inst_core, item_id_map)
        solution_variant = _kp_solution_variant(solution, item_id_map)

    else:
        return results

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    record = build_output_record(
        task_id=task_name,
        difficulty_tier=difficulty_tier or "UNKNOWN",
        example_index=context_index,
        prompt=full_instruction,
        surface_format=fmt,
        indexing_scheme=index_base,
        instance_canonical_json=inst_core,
        reference_solution_canonical_json=solution,
        reference_objective_value=obj,
        instance_surface_json=instance_variant,
        reference_solution_surface_json=solution_variant,
    )
    results.append(record)
    return results


def _get_problem_specs(problem_type: str) -> Dict[str, str]:
    pt = problem_type.upper()

    if pt == "2SP":

        desc = (
            "2D Strip Packing Problem (2SP / 2D-SP): Given a rectangular strip with a fixed bin width "
            "and an unbounded or variable height, and a set of rectangular item types (each with a fixed width and height), "
            "the goal is to place exactly one copy of each item type into the strip without any overlap. "
            "All items must be axis-aligned and lie entirely within the strip boundaries.\n\n"
            "Each item is positioned by specifying the coordinates of its bottom-left corner within the strip. "
            "The horizontal coordinate determines how far the item is placed from the left boundary, and the vertical "
            "coordinate determines how far the item is placed from the bottom boundary. The height of an item refers to "
            "the vertical size of the rectangle itself.\n\n"
            "Objective: Minimize the used height of the packing, which is defined as the highest vertical position reached "
            "by the top edge of any placed item."
        )

        baseline_template = (
            "You are given a 2D Strip Packing Problem (2SP).\n"
            "The input describes a rectangular strip with a fixed width and a variable or limited height, "
            "along with a list of item types, each specified by a fixed width and height.\n\n"
            "**You must place exactly one copy of each item type into the strip.** "
            "**All items must be axis-aligned, lie entirely within the strip boundaries, and must not overlap.**\n\n"
            "Each item is positioned by specifying the coordinates of its bottom-left corner within the strip. "
            "The horizontal coordinate indicates the distance from the left boundary, and the vertical coordinate "
            "indicates the distance from the bottom boundary. The height of an item refers to the vertical size of "
            "the rectangle itself.\n\n"
            "The objective is to minimize the used height of the packing, defined as the highest vertical position "
            "reached by the top edge of any placed item.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your placement in **exactly** the following JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [\n'
            '    {"item_type": <item_type_id>, "x": 0, "y": 0, "width": 10, "height": 5},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "```\n"
            "Each object represents one placed rectangle, defined by its bottom-left corner coordinates and its size.\n"
            "Use the item_type identifiers exactly as they appear in the input.\n"
            "**Do not include explanations, comments, or any extra keys.**"
        )

        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [{"item_type": <item_type_id>, "x": 0, "y": 0, "width": 10, "height": 5}, ...]\n'
            "}\n"
            "```\n"
        )


    elif pt == "BPP":
        desc = (
            "Bin Packing Problem (BPP): Given a set of items with weights and identical bins with a fixed capacity, "
            "the goal is to assign each item to exactly one bin such that the total weight in any bin does not "
            "exceed the capacity, while minimizing the number of bins used."
        )

        baseline_template = (
            "You are given a Bin Packing Problem (BPP).\n"
            "The input specifies a list of items with weights and a bin capacity.\n\n"
            "**Each item must be assigned to exactly one bin.**\n"
            "**The sum of item weights in any bin must not exceed the bin capacity.**\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to partition the items into bins so that all constraints are satisfied "
            "and the total number of bins is as small as possible.\n\n"
            "Return your answer in **exactly** the following JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [[item_id, item_id, ...], [item_id, ...], ...]\n'
            "}\n"
            "```\n"
            "Each inner list represents one bin and contains the identifiers of items placed in that bin.\n"
            "Use the item identifiers exactly as shown in the input.\n"
            "**Do not include explanations or extra keys.**"
        )

        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [[item_id, ...], ...]\n'
            "}\n"
            "```\n"
        )

    elif pt == "CSP":
        desc = (
            "Cutting Stock Problem (CSP): Given a stock material of fixed length (capacity) and a set of item types, "
            "each associated with a required size and a demand, the objective is to cut the stock into pieces such that "
            "all demands are satisfied while minimizing the number of stock rolls used."
        )

        baseline_template = (
            "You are given a Cutting Stock Problem (CSP).\n"
            "The input specifies a stock length (capacity) and several item types, each with a width and a demand.\n\n"
            "**Each cutting pattern must respect the stock length capacity.**\n"
            "**All item demands must be satisfied exactly.**\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to propose a set of cutting patterns that together satisfy all demands "
            "using as few stock rolls as possible.\n\n"
            "Return your answer in **exactly** the following JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [\n'
            '    {"<item_id>": 2, "<item_id>": 1},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "```\n"
            "Each dictionary represents one cutting pattern: keys are item identifiers and values are "
            "the number of pieces of that item cut in this pattern.\n"
            "Use item identifiers exactly as in the input.\n"
            "**Do not include explanations or extra keys.**"
        )

        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [\n'
            '    {"<item_id>": 2, "<item_id>": 1},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "```\n"
        )

    else:  # KP
        desc = (
            "0/1 Knapsack Problem (KP): Given a set of items with weights and profits and a knapsack with limited "
            "capacity, the goal is to select a subset of items that maximizes total profit without exceeding "
            "the capacity. Each item can be selected at most once."
        )

        baseline_template = (
            "You are given a 0/1 Knapsack Problem (KP).\n"
            "The input specifies a knapsack capacity and a list of items, each with a weight and a profit.\n\n"
            "**Each item may be selected at most once.**\n"
            "**The total weight of selected items must not exceed the capacity.**\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to decide which items to take so that the total profit is maximized.\n\n"
            "Return your answer in **exactly** the following JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [0, 1, 0, 1, ...]\n'
            "}\n"
            "```\n"
            "This is a binary vector where solution[i] = 1 means item i is selected.\n"
            "Use the same item order as in the input.\n"
            "**Do not include explanations or extra keys.**"
        )

        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [0, 1, 0, 1, ...]\n'
            "}\n"
            "```\n"
        )

    return {
        "task_description": desc,
        "baseline_template": baseline_template,
        "output_format": output_format,
    }

def _get_hint(problem_type: str) -> Dict[str, Any]:
    pt = problem_type.upper()

    if pt == "2SP":
        global_fields = [
            {"name": "bin_width", "description": "bin width"},
            {"name": "bin_height", "description": "bin height"},
        ]
        item_fields = [
            {"name": "width", "description": "item width"},
            {"name": "height", "description": "item height"},
            {"name": "item_type", "description": "identifier of an item"},
        ]
        allowed = ["bin_width", "bin_height",  "item_type", "width", "height"]

    elif pt == "BPP":
        global_fields = [
            {"name": "bin_capacity", "description": "bin capacity"},
            {"name": "n_items", "description": "total number of items"},
        ]
        item_fields = [
            {"name": "item_id", "description": "identifier of an item"},
            {"name": "weight", "description": "weight of the item"},
        ]
        allowed = ["bin_capacity", "n_items", "item_id", "weight"]

    elif pt == "CSP":
        global_fields = [
            {"name": "bin_capacity", "description": "stock length / capacity"},
        ]
        item_fields = [
            {"name": "item_id", "description": "identifier of an item type"},
            {"name": "width", "description": "piece width"},
            {"name": "demand", "description": "required number of pieces"},
        ]
        allowed = ["bin_capacity", "item_id", "width", "demand"]

    else:  # KP
        global_fields = [
            {"name": "capacity", "description": "knapsack capacity"},
        ]
        item_fields = [
            {"name": "item_id", "description": "identifier of an item"},
            {"name": "weight", "description": "item weight"},
            {"name": "profit", "description": "item profit"},
        ]
        allowed = ["capacity", "item_id", "weight", "profit"]

    return {"global_fields": global_fields, "item_fields": item_fields, "allowed_placeholders": allowed}



def run_packing_step2(
    problem_type: str,
    instance_dir: str,
    output_root_dir: str,
    k_per_call: int = 20,
    n_target: int = 50,
    scale: str | None = None,
    load_cached_contexts: bool = True,
    contexts_audit_path: str | None = None,
) -> str:

    problem_type = problem_type.upper()
    task_name = problem_type

    specs = _get_problem_specs(problem_type)
    desc = specs["task_description"]
    baseline_template = specs["baseline_template"]
    output_format = specs["output_format"]

    llm = OpenAILlmService()
    prompter = PromptLoader()
    hint = _get_hint(problem_type)

    generate_dataset_for_task(
        task_name=task_name,
        task_description=desc,
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_instance_packing,
        llm=llm,
        prompter=prompter,
        hint=hint,
        baseline_template=baseline_template,
        output_format=output_format,
        k_per_call=int(k_per_call),
        n_target=int(n_target),
        scale=scale,
        load_cached_contexts=load_cached_contexts,
        contexts_audit_path=contexts_audit_path,
    )

    return str(output_root_dir)



def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Contextualize packing instances (2SP, BPP, CSP, KP) into NL/JSON/CSV/Markdown."
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="BPP",
        choices=["2SP", "BPP", "CSP", "KP"],
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        default=None,
        help="Directory containing raw JSON instances (default: ./step1_instance_creation/generated_data/<problem_type>).",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=None,
        help="Root directory to save contextualized data (default: step2_contextualization/dataset/<problem_type>).",
    )
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
        contextualize_fn=contextualize_instance_packing,
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
Example runs:

python -m step2_contextualization.contextualize_packing \
  --problem_type 2SP \
  --instance_dir ./step1_instance_creation/generated_data/2SP \
  --output_root_dir step2_contextualization/dataset/2SP \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_packing \
  --problem_type BPP \
  --instance_dir ./step1_instance_creation/generated_data/BPP \
  --output_root_dir step2_contextualization/dataset/BPP \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_packing \
  --problem_type CSP \
  --instance_dir ./step1_instance_creation/generated_data/CSP \
  --output_root_dir step2_contextualization/dataset/CSP \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_packing \
  --problem_type KP \
  --instance_dir ./step1_instance_creation/generated_data/KP \
  --output_root_dir step2_contextualization/dataset/KP \
  --k_per_call 20 \
  --n_target 50
"""
