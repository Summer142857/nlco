import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for QAP (Quadratic Assignment Problem).

Instance format example:
{
  "instance": {
    "distance_matrix": [[...],[...],...],
    "flow_matrix": [[...],[...],...],
    "objective": <number>   # optional
  },
  "solution": [perm...],     # permutation: facility i assigned to location perm[i]
  "obj": <number>,
  "problem_type": "QAP"
}

We ALWAYS render input using pair lists (from_id,to_id,value).
We provide TWO pair blocks:
- distance pairs: (loc_a, loc_b, distance)
- flow pairs:     (fac_a, fac_b, flow)
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _remap_solution_qap(solution: Any, facility_id_map: Optional[Dict[int, Any]] = None, location_id_map: Optional[Dict[int, Any]] = None) -> Any:
    """
    solution is a permutation list:
      solution[i] = assigned location index for facility i
    We remap facility index by position (implicitly), and remap location ids by value.
    Output stays as a list, but values become mapped location labels if mapping provided.
    """
    if solution is None:
        return None
    if not isinstance(solution, list):
        return solution
    if location_id_map is None:
        return solution
    out = []
    for loc in solution:
        try:
            out.append(location_id_map.get(int(loc), loc))
        except Exception:
            out.append(loc)
    return out


def _build_labeled_qap_instance(
    dist_mat: List[List[float]],
    flow_mat: List[List[float]],
    facility_id_map: Optional[Dict[int, Any]],
    location_id_map: Optional[Dict[int, Any]],
    omit_diagonal: bool = False,
) -> Dict[str, Any]:
    n = len(dist_mat)
    facilities = [facility_id_map.get(i, i) if facility_id_map else i for i in range(n)]
    locations = [location_id_map.get(i, i) if location_id_map else i for i in range(n)]

    dist_pairs: List[Dict[str, Any]] = []
    flow_pairs: List[Dict[str, Any]] = []

    for i in range(n):
        li = location_id_map.get(i, i) if location_id_map else i
        for j in range(n):
            if omit_diagonal and i == j:
                continue
            lj = location_id_map.get(j, j) if location_id_map else j
            dist_pairs.append({"from_id": li, "to_id": lj, "distance": dist_mat[i][j]})

    for i in range(n):
        fi = facility_id_map.get(i, i) if facility_id_map else i
        for j in range(n):
            if omit_diagonal and i == j:
                continue
            fj = facility_id_map.get(j, j) if facility_id_map else j
            flow_pairs.append({"from_id": fi, "to_id": fj, "flow": flow_mat[i][j]})

    return {
        "problem_type": "QAP",
        "num_facilities": n,
        "num_locations": n,
        "facilities": facilities,
        "locations": locations,
        "distance_pairs": dist_pairs,
        "flow_pairs": flow_pairs,
    }


def qap_render_input(
    fmt: str,
    dist_mat: List[List[float]],
    flow_mat: List[List[float]],
    facility_id_map: Optional[Dict[int, Any]],
    location_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
    omit_diagonal: bool = True,
) -> str:
    # defaults (keep it simple; allow renaming of only core column names)
    num_key = "n"
    facilities_key = "facilities"
    locations_key = "locations"

    dist_from_key, dist_to_key, dist_val_key = "from_id", "to_id", "distance"
    flow_from_key, flow_to_key, flow_val_key = "from_id", "to_id", "flow"

    # optional renaming
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig in ["n", "num_nodes"]:
                num_key = new
            elif orig == "facilities":
                facilities_key = new
            elif orig == "locations":
                locations_key = new

        for field in scenario_hint.get("distance_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "from_id":
                dist_from_key = new
            elif orig == "to_id":
                dist_to_key = new
            elif orig == "distance":
                dist_val_key = new

        for field in scenario_hint.get("flow_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "from_id":
                flow_from_key = new
            elif orig == "to_id":
                flow_to_key = new
            elif orig == "flow":
                flow_val_key = new

    n = len(dist_mat)
    fac_ids = [facility_id_map.get(i, i) if facility_id_map else i for i in range(n)]
    loc_ids = [location_id_map.get(i, i) if location_id_map else i for i in range(n)]
    fac_str = ", ".join(str(x) for x in fac_ids)
    loc_str = ", ".join(str(x) for x in loc_ids)

    dist_pairs: List[Dict[str, Any]] = []
    flow_pairs: List[Dict[str, Any]] = []

    for i in range(n):
        li = location_id_map.get(i, i) if location_id_map else i
        for j in range(n):
            if omit_diagonal and i == j:
                continue
            lj = location_id_map.get(j, j) if location_id_map else j
            dist_pairs.append({dist_from_key: li, dist_to_key: lj, dist_val_key: dist_mat[i][j]})

    for i in range(n):
        fi = facility_id_map.get(i, i) if facility_id_map else i
        for j in range(n):
            if omit_diagonal and i == j:
                continue
            fj = facility_id_map.get(j, j) if facility_id_map else j
            flow_pairs.append({flow_from_key: fi, flow_to_key: fj, flow_val_key: flow_mat[i][j]})

    if fmt == "json":
        obj = {
            num_key: n,
            facilities_key: fac_ids,
            locations_key: loc_ids,
            "distance": dist_pairs,
            "flow": flow_pairs,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_key}={n}")
        lines.append(f"# {facilities_key}={fac_str}")
        lines.append(f"# {locations_key}={loc_str}")
        lines.append("")
        lines.append(f"{dist_from_key},{dist_to_key},{dist_val_key}")
        for rec in dist_pairs:
            lines.append(f"{rec[dist_from_key]},{rec[dist_to_key]},{rec[dist_val_key]}")
        lines.append("")
        lines.append(f"{flow_from_key},{flow_to_key},{flow_val_key}")
        for rec in flow_pairs:
            lines.append(f"{rec[flow_from_key]},{rec[flow_to_key]},{rec[flow_val_key]}")
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
                        n=n,
                        facilities=fac_str,
                        locations=loc_str,
                        **{num_key: n, facilities_key: fac_str, locations_key: loc_str},
                    )
                )
                lines.append("")

        lines.append(f"| {dist_from_key} | {dist_to_key} | {dist_val_key} |")
        lines.append("|---|---|---|")
        for rec in dist_pairs:
            lines.append(f"| {rec[dist_from_key]} | {rec[dist_to_key]} | {rec[dist_val_key]} |")

        lines.append("")
        lines.append(f"| {flow_from_key} | {flow_to_key} | {flow_val_key} |")
        lines.append("|---|---|---|")
        for rec in flow_pairs:
            lines.append(f"| {rec[flow_from_key]} | {rec[flow_to_key]} | {rec[flow_val_key]} |")

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        n=n,
                        facilities=fac_str,
                        locations=loc_str,
                        **{num_key: n, facilities_key: fac_str, locations_key: loc_str},
                    )
                )

        return "\n".join(lines)

    # fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        dist_line_tmpl = nl_style.get("distance_item_fields_line_template", "") or ""
        flow_line_tmpl = nl_style.get("flow_item_fields_line_template", "") or ""

        parts: List[str] = []
        base_vars = {
            "n": n,
            "facilities": fac_str,
            "locations": loc_str,
            num_key: n,
            facilities_key: fac_str,
            locations_key: loc_str,
        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        if dist_line_tmpl:
            for rec in dist_pairs:
                parts.append(
                    _format_safe(
                        dist_line_tmpl,
                        **base_vars,
                        from_id=rec[dist_from_key],
                        to_id=rec[dist_to_key],
                        distance=rec[dist_val_key],
                    )
                )

        if flow_line_tmpl:
            for rec in flow_pairs:
                parts.append(
                    _format_safe(
                        flow_line_tmpl,
                        **base_vars,
                        from_id=rec[flow_from_key],
                        to_id=rec[flow_to_key],
                        flow=rec[flow_val_key],
                    )
                )

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)


def contextualize_instance_qap(
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
    dist_mat = core.get("distance_matrix")
    flow_mat = core.get("flow_matrix")
    if not isinstance(dist_mat, list) or not dist_mat or not isinstance(dist_mat[0], list):
        return results
    if not isinstance(flow_mat, list) or not flow_mat or not isinstance(flow_mat[0], list):
        return results

    n = len(dist_mat)
    if any(len(row) != n for row in dist_mat):
        return results
    if len(flow_mat) != n or any(len(row) != n for row in flow_mat):
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

    # label maps (facilities and locations can share same label space; keep separate for clarity)
    fac_labels = [f"F{j + 1}" for j in range(n)]

    loc_labels = _build_labels(n, index_base)
    facility_id_map = {i: fac_labels[i] for i in range(n)}
    location_id_map = {i: loc_labels[i] for i in range(n)}

    solution_variant = _remap_solution_qap(solution, facility_id_map=facility_id_map, location_id_map=location_id_map)

    input_text = qap_render_input(
        fmt=fmt,
        dist_mat=dist_mat,
        flow_mat=flow_mat,
        facility_id_map=facility_id_map,
        location_id_map=location_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
        omit_diagonal=True,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_qap_instance(
        dist_mat=dist_mat,
        flow_mat=flow_mat,
        facility_id_map=facility_id_map,
        location_id_map=location_id_map,
        omit_diagonal=True,
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
    # QAP only
    desc = (
        "Quadratic Assignment Problem (QAP): you are asked to assign a set of facilities to a set of locations. "
        "Each facility must be assigned to exactly one location, and each location must receive exactly one facility.\n\n"
        "The input provides two kinds of directed pair information: distances between pairs of locations, and flows between "
        "pairs of facilities. When two facilities are assigned to two locations, their contribution to the total cost depends "
        "on how much flow exists between the facilities and how far apart the chosen locations are.\n\n"
        "Your task is to choose an assignment (a permutation of locations) that minimizes the total cost induced by all such "
        "facility–location interactions."
    )

    baseline_template = (
        "You are given a Quadratic Assignment Problem (QAP).\n\n"
        "Problem description:\n"
        "- There are n facilities and n locations.\n"
        "- You must assign each facility to exactly one location, and each location to exactly one facility.\n"
        "- The instance provides:\n"
        "  (1) Distance pairs between locations: (from_id, to_id, distance)\n"
        "  (2) Flow pairs between facilities:  (from_id, to_id, flow)\n"
        "- Distances and flows are directed in the input (do not assume symmetry).\n\n"
        "Objective:\n"
        "- Minimize the total assignment cost induced by your assignment.\n"
        "- Intuitively, if two facilities have high flow, assigning them to distant locations is expensive.\n\n"
        "Important details:\n"
        "- The input is provided in PAIR format (diagonal pairs are omitted).\n"
        "- Your answer must be a valid one-to-one assignment, i.e., a permutation of the location identifiers.\n\n"
        "You will receive the instance in the following PAIR format:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your decision using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [\n'
        '    <location_id_for_first_facility>,\n'
        '    <location_id_for_second_facility>,\n'
        '    ...,\n'
        '    <location_id_for_last_facility>\n'
        "  ]\n"
        "}\n"
        "```\n"
        "The list `solution` must have length n.\n"
        "The first element specifies the location assigned to the first facility,\n"
        "the second element specifies the location assigned to the second facility,\n"
        "and so on until the last facility.\n"
        "Each location identifier must appear exactly once in `assignment`.\n"
        "All identifiers must match exactly those used in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [\n'
        '    <location_id_for_first_facility>,\n'
        '    <location_id_for_second_facility>,\n'
        '    ...,\n'
        '    <location_id_for_last_facility>\n'
        "  ]\n"
        "}\n"
        "```\n"
    )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    # Minimal hint, no explicit "pairs" field; only core column semantics
    global_fields = [
        {"name": "n", "description": "number of facilities/locations"},
        {"name": "facilities", "description": "facility identifiers"},
        {"name": "locations", "description": "location identifiers"},
    ]
    distance_item_fields = [
        {"name": "from_id", "description": "location id (source)"},
        {"name": "to_id", "description": "location id (target)"},
        {"name": "distance", "description": "distance between two locations"},
    ]
    flow_item_fields = [
        {"name": "from_id", "description": "facility id (source)"},
        {"name": "to_id", "description": "facility id (target)"},
        {"name": "flow", "description": "flow between two facilities"},
    ]
    allowed = ["n", "facilities", "locations", "from_id", "to_id", "distance", "flow"]
    return {
        "global_fields": global_fields,
        "distance_item_fields": distance_item_fields,
        "flow_item_fields": flow_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize QAP instances into NL/JSON/CSV/Markdown (PAIR input only).")
    parser.add_argument("--problem_type", type=str, default="QAP", choices=["QAP"])
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
        contextualize_fn=contextualize_instance_qap,
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

python -m step2_contextualization.contextualize_qap \
  --problem_type QAP \
  --instance_dir ./step1_instance_creation/generated_data/QAP \
  --output_root_dir step2_contextualization/dataset/QAP \
  --k_per_call 20 \
  --n_target 50
'''
