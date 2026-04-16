# step2_contextualization/contextualize_pdp.py

import json
import random
from typing import Dict, Any, List, Optional, Tuple

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for Pickup-and-Delivery Problem (PDP):

This dataset variant is:
- Single-vehicle PDP with Time Windows (1-PDP-TW)
- All nodes must be visited exactly once
- Precedence constraints: pickup must occur before delivery for each pair
- Time windows per node
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]



def _build_node_label_map(num_nodes: int, index_base) -> Dict[int, Any]:
    labels = make_labels(num_nodes, index_base)
    return {i: labels[i] for i in range(num_nodes)}


def _remap_pairs(pairs: Any, node_id_map: Dict[int, Any]) -> Any:
    if pairs is None:
        return None
    if not isinstance(pairs, list):
        return pairs
    out = []
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            p, d = pair
            out.append([node_id_map.get(int(p), p), node_id_map.get(int(d), d)])
        else:
            out.append(pair)
    return out


def _remap_solution_pdp(
    solution: Any,
    node_id_map: Dict[int, Any],
) -> Any:
    """
    Your current solution format example:
      solution = [[0, ..., 0]]   (list of routes, even if num_vehicles=1)

    We remap every int node id to label.
    """
    if solution is None:
        return None

    def remap_route(route: List[int]) -> List[Any]:
        return [node_id_map.get(int(v), v) for v in route]

    if isinstance(solution, list) and solution:
        if isinstance(solution[0], list):
            return [remap_route(r) for r in solution]
        else:
            return [remap_route(solution)]
    return solution


def _build_labeled_pdp_instance(
    inst_core: Dict[str, Any],
    node_id_map: Dict[int, Any],
    problem_type: str,
) -> Dict[str, Any]:
    coords = inst_core.get("coordinates", [])
    depot = inst_core.get("depot", 0)
    tw = inst_core.get("time_windows")
    pairs = inst_core.get("pickup_delivery_pairs")
    num_vehicles = inst_core.get("num_vehicles")

    nodes = []
    for i, xy in enumerate(coords):
        x, y = xy
        rec = {
            "id": node_id_map.get(i, i),
            "x": x,
            "y": y,
        }
        if tw is not None and i < len(tw):
            rec["tw_start"] = tw[i][0]
            rec["tw_end"] = tw[i][1]
        nodes.append(rec)

    inst_variant: Dict[str, Any] = {
        "problem_type": problem_type,
        "num_nodes": len(coords),
        "depot": node_id_map.get(int(depot), depot),
        "nodes": nodes,
    }
    if num_vehicles is not None:
        inst_variant["num_vehicles"] = num_vehicles
    if pairs is not None:
        inst_variant["pickup_delivery_pairs"] = _remap_pairs(pairs, node_id_map)

    # keep optional metadata if present
    for k in ["total_distance", "objective"]:
        if k in inst_core:
            inst_variant[k] = inst_core[k]

    return inst_variant


from typing import Dict, Any, List

def pdp_render_nl_with_style(
    inst_core: Dict[str, Any],
    node_id_map: Dict[int, Any],
    nl_style: Dict[str, Any],
    problem_type: str,
) -> str:

    coords = inst_core.get("coordinates", []) or []
    depot = inst_core.get("depot", 0)
    tw = inst_core.get("time_windows")
    pairs = inst_core.get("pickup_delivery_pairs")
    num_vehicles = inst_core.get("num_vehicles")

    num_nodes = len(coords)

    # ---------- templates (compat with your nl_styles keys) ----------
    header_t = nl_style.get("header") or ""
    footer_t = nl_style.get("footer") or ""

    node_line_t = (
        nl_style.get("node_item_fields_line_template")
        or nl_style.get("node_line_template")
        or nl_style.get("line_template")
        or ""
    )

    pair_line_t = (
        nl_style.get("pair_item_fields_line_template")
        or nl_style.get("pair_line_template")
        or ""
    )

    base_ctx = {
        "num_nodes": num_nodes,
        "depot": node_id_map.get(int(depot), depot),
    }

    out: List[str] = []

    # ---------- header ----------
    if header_t:
        out.append(_format_safe(header_t, **base_ctx))

    # ---------- nodes section ----------
    if node_line_t:
        for i in range(num_nodes):
            x, y = coords[i]
            ctx = {
                **base_ctx,
                "node_id": node_id_map.get(i, i),
                "x": x,
                "y": y,
                "tw_start": tw[i][0] if tw is not None and i < len(tw) else None,
                "tw_end": tw[i][1] if tw is not None and i < len(tw) else None,
            }
            out.append(_format_safe(node_line_t, **ctx))

    # ---------- pairs section ----------
    if isinstance(pairs, list) and pairs:
        if pair_line_t:
            for idx, pair in enumerate(pairs):
                # be defensive if pair isn't exactly length-2
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    continue
                p, d = pair
                ctx = {
                    **base_ctx,
                    "pair_index": idx,
                    "pickup": node_id_map.get(int(p), p),
                    "delivery": node_id_map.get(int(d), d),
                }
                out.append(_format_safe(pair_line_t, **ctx))
        else:
            # fallback ONLY when no pair template exists
            out.append(
                f"Pickup-delivery precedence pairs: {_remap_pairs(pairs, node_id_map)}"
            )

    # ---------- footer ----------
    if footer_t:
        out.append(_format_safe(footer_t, **base_ctx))

    return "\n".join(out)



import json
from typing import Dict, Any, List, Optional

def pdp_render_input(
    fmt: str,
    inst_core: Dict[str, Any],
    node_id_map: Dict[int, Any],
    nl_style: Optional[Dict[str, Any]],
    problem_type: str,
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    coords = inst_core.get("coordinates", [])
    depot = inst_core.get("depot", 0)
    tw = inst_core.get("time_windows")
    pairs = inst_core.get("pickup_delivery_pairs")

    num_nodes = len(coords)

    # ---------------- default keys ----------------
    num_nodes_key = "num_nodes"
    depot_key = "depot"
    pairs_key = "pickup_delivery_pairs"

    node_id_key = "node_id"
    x_key = "x"
    y_key = "y"
    tw_start_key = "tw_start"
    tw_end_key = "tw_end"

    pair_pickup_key = "pickup"
    pair_delivery_key = "delivery"

    # ---------------- scenario_hint renaming ----------------
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "num_nodes":
                num_nodes_key = new
            elif orig == "depot":
                depot_key = new
            elif orig == "pickup_delivery_pairs":
                pairs_key = new
            # NOTE: ignore num_vehicles on purpose

        # node item fields
        for field in scenario_hint.get("node_item_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "node_id":
                node_id_key = new
            elif orig == "x":
                x_key = new
            elif orig == "y":
                y_key = new
            elif orig == "tw_start":
                tw_start_key = new
            elif orig == "tw_end":
                tw_end_key = new

        # pair item fields
        for field in scenario_hint.get("pair_item_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "pickup":
                pair_pickup_key = new
            elif orig == "delivery":
                pair_delivery_key = new

    # ---------------- build node records ----------------
    node_records: List[Dict[str, Any]] = []
    for i in range(num_nodes):
        x, y = coords[i]
        rec: Dict[str, Any] = {
            node_id_key: node_id_map.get(i, i),
            x_key: x,
            y_key: y,
        }
        if tw is not None and i < len(tw):
            rec[tw_start_key] = tw[i][0]
            rec[tw_end_key] = tw[i][1]
        node_records.append(rec)

    # ---------------- build pair records ----------------
    pair_records: List[Dict[str, Any]] = []
    if isinstance(pairs, list):
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                p, d = pair
                pair_records.append(
                    {
                        pair_pickup_key: node_id_map.get(int(p), p),
                        pair_delivery_key: node_id_map.get(int(d), d),
                    }
                )

    # ---------------- JSON ----------------
    if fmt == "json":
        parts: List[str] = []

        # ---- JSON 1: nodes ----
        nodes_obj: Dict[str, Any] = {
            num_nodes_key: num_nodes,
            depot_key: node_id_map.get(int(depot), depot),
            "nodes": node_records,
        }
        parts.append(json.dumps(nodes_obj, ensure_ascii=False, indent=2))

        # ---- JSON 2: pickup-delivery pairs ----
        if pair_records:
            pairs_obj = {
                pairs_key: pair_records
            }
            parts.append(json.dumps(pairs_obj, ensure_ascii=False, indent=2))

        # two JSON objects, separated by blank line
        return "\n\n".join(parts)

    # ---------------- CSV ----------------
    if fmt == "csv":
        # Output as "all CSV" (multi-table CSV), NOT JSON-in-comment.
        # No num_vehicles anywhere.
        lines: List[str] = []

        # 1) GLOBALS table
        lines.append("# GLOBALS")
        lines.append("key,value")
        lines.append(f"{num_nodes_key},{num_nodes}")
        lines.append(f"{depot_key},{node_id_map.get(int(depot), depot)}")
        lines.append("")  # separator

        # 2) PAIRS table
        lines.append("# PAIRS")
        lines.append(f"{pair_pickup_key},{pair_delivery_key}")
        for pr in pair_records:
            lines.append(f"{pr.get(pair_pickup_key,'')},{pr.get(pair_delivery_key,'')}")
        lines.append("")

        # 3) NODES table
        cols = [node_id_key, x_key, y_key]
        if tw is not None:
            cols += [tw_start_key, tw_end_key]

        lines.append("# NODES")
        lines.append(",".join(cols))
        for rec in node_records:
            lines.append(",".join(str(rec.get(c, "")) for c in cols))

        return "\n".join(lines)

    # ---------------- Markdown table ----------------
    if fmt == "markdown_table":
        lines: List[str] = []

        lines.append(f"- **{num_nodes_key}**: {num_nodes}")
        lines.append(f"- **{depot_key}**: {node_id_map.get(int(depot), depot)}")
        lines.append("")

        # pairs as a table (NOT python dict list)
        if pair_records:
            lines.append(f"**{pairs_key}**")
            lines.append(f"| {pair_pickup_key} | {pair_delivery_key} |")
            lines.append("|---|---|")
            for pr in pair_records:
                lines.append(
                    f"| {pr.get(pair_pickup_key, '')} | {pr.get(pair_delivery_key, '')} |"
                )
            lines.append("")

        # nodes table
        cols = [node_id_key, x_key, y_key]
        if tw is not None:
            cols += [tw_start_key, tw_end_key]

        lines.append(f"**nodes**")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for rec in node_records:
            lines.append("| " + " | ".join(str(rec.get(c, "")) for c in cols) + " |")

        return "\n".join(lines)

    # ---------------- NL ----------------
    if fmt == "nl" and isinstance(nl_style, dict):
        return pdp_render_nl_with_style(
            inst_core=inst_core,
            node_id_map=node_id_map,
            nl_style=nl_style,
            problem_type=problem_type,
        )

    # fallback
    return json.dumps(
        {
            "nodes": node_records,
            "pairs": pair_records,
            depot_key: node_id_map.get(int(depot), depot),
            num_nodes_key: num_nodes,
        },
        ensure_ascii=False,
        indent=2,
    )

def contextualize_instance_pdp(
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

    inst_core = inst.get("instance")
    solution = inst.get("solution")
    obj = inst.get("obj")
    problem_type = inst.get("problem_type", task_name)

    if inst_core is None:
        return results

    coords = inst_core.get("coordinates", [])
    num_nodes = len(coords)
    if num_nodes == 0:
        return results

    # choose context index
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

    node_id_map = _build_node_label_map(num_nodes, index_base)

    solution_variant = _remap_solution_pdp(solution, node_id_map)

    input_text = pdp_render_input(
        fmt=fmt,
        inst_core=inst_core,
        node_id_map=node_id_map,
        nl_style=nl_style,
        problem_type=problem_type,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_pdp_instance(
        inst_core=inst_core,
        node_id_map=node_id_map,
        problem_type=problem_type,
    )

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

    # We keep only PDP here (single file)
    desc = (
        "Pickup and Delivery Problem with Time Windows (PDP-TW, single vehicle): "
        "Given a depot and a set of locations with coordinates and time windows, "
        "plus a list of pickup-delivery precedence pairs (pickup must be visited before its delivery), "
        "find a single tour that starts and ends at the depot, visits every location exactly once, "
        "respects all time windows, satisfies all precedence constraints, and minimizes total travel distance."
    )

    baseline_template = (
        "You are given a Pickup and Delivery Problem with Time Windows instance (single vehicle).\n"
        "The input describes:\n"
        "- A depot node where the route must start and end\n"
        "- A set of locations with coordinates and time windows\n"
        "- A list of pickup-delivery precedence pairs (pickup must be visited before delivery)\n\n"
        "You will receive the instance in the following format:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Your task is to propose a single route that:\n"
        "1) Starts and ends at the depot\n"
        "2) Visits every non-depot node exactly once\n"
        "3) Respects every node's time window (waiting is allowed)\n"
        "4) For every (pickup, delivery) pair, visits pickup before delivery\n"
        "and aims to minimize total travel distance.\n\n"
        "Return your decision in **exactly** this JSON format:\n"
        "```json\n"
        "{\n"
        '  "solution": [depot_id, ..., depot_id]\n'
        "}\n"
        "```\n"
        "Use **exactly** the same node identifiers that appear in the input. "
        "Do not include explanations or extra keys."
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [depot_id, ..., depot_id]\n'
        "}\n"
        "```\n"
    )

    return {
        "task_description": desc,
        "baseline_template": baseline_template,
        "output_format": output_format,
    }



def _get_hint(problem_type: str) -> Dict[str, Any]:
    """
    Two item groups:
      - node_item_fields
      - pair_item_fields
    This matches your facility script style (multiple item groups).
    """
    global_fields = [
        {"name": "num_nodes", "description": "total number of locations (including depot)"},
        {"name": "depot", "description": "identifier of the depot node"},
    ]

    node_item_fields = [
        {"name": "node_id", "description": "identifier of a location"},
        {"name": "x", "description": "x coordinate"},
        {"name": "y", "description": "y coordinate"},
        {"name": "tw_start", "description": "earliest allowed arrival time"},
        {"name": "tw_end", "description": "latest allowed arrival time"},
    ]

    pair_item_fields = [
        {"name": "pickup", "description": "pickup node identifier (must be visited first)"},
        {"name": "delivery", "description": "delivery node identifier (must be visited after pickup)"},
    ]

    allowed = [
        "num_nodes",
        "depot",
        "node_id",
        "x",
        "y",
        "tw_start",
        "tw_end",
        "pickup",
        "delivery",
        "pair_index",
    ]

    return {
        "global_fields": global_fields,
        "node_item_fields": node_item_fields,
        "pair_item_fields": pair_item_fields,
        "allowed_placeholders": allowed,
    }


# ----------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Contextualize PDP instances into NL/JSON/CSV/Markdown."
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="PDP",
        choices=["PDP"],
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        default=None,
        help="Directory containing raw JSON instances (default: ./step1_instance_creation/generated_data/PDP).",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=None,
        help="Root directory to save contextualized data (default: step2_contextualization/dataset/PDP).",
    )
    parser.add_argument(
        "--k_per_call",
        type=int,
        default=20,
        help="How many contextualized examples to generate per LLM call.",
    )
    parser.add_argument(
        "--n_target",
        type=int,
        default=50,
        help="Total number of contextualized examples to generate (per split).",
    )

    args = parser.parse_args()
    problem_type = args.problem_type
    task_name = problem_type

    specs = _get_problem_specs(problem_type)
    desc = specs["task_description"]
    baseline_template = specs["baseline_template"]
    output_format = specs["output_format"]

    instance_dir = args.instance_dir or "./step1_instance_creation/generated_data/PDP"
    output_root_dir = args.output_root_dir or "step2_contextualization/dataset/PDP"

    llm = OpenAILlmService()
    prompter = PromptLoader()

    hint = _get_hint(problem_type)

    generate_dataset_for_task(
        task_name=task_name,
        task_description=desc,
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_instance_pdp,
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
Example:

python -m step2_contextualization.contextualize_pdp \
  --problem_type PDP \
  --instance_dir ./step1_instance_creation/generated_data/PDP \
  --output_root_dir step2_contextualization/dataset/PDP \
  --k_per_call 20 \
  --n_target 50
"""
