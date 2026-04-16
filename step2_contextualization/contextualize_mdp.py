# step2_contextualization/contextualize_mdp.py
import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for MDP (Maximum Diversity Problem).

Instance example:
{
  "instance": {
    "distance_matrix": [[...],[...],...],
    "m": 3,
    "objective": <optional>
  },
  "solution": [selected_ids...],
  "obj": <number>,
  "problem_type": "MDP"
}

We ALWAYS render input using PAIR lists (from_id,to_id,distance).
Diagonal pairs (i==j) are omitted by default.
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _remap_solution_mdp(solution: Any, node_id_map: Optional[Dict[int, Any]]) -> Any:
    if solution is None:
        return None
    if node_id_map is None:
        return solution
    if isinstance(solution, list):
        out = []
        for x in solution:
            try:
                out.append(node_id_map.get(int(x), x))
            except Exception:
                out.append(x)
        return out
    return solution


def _build_labeled_mdp_instance(
    dist_mat: List[List[float]],
    m: int,
    node_id_map: Optional[Dict[int, Any]],
    omit_diagonal: bool = True,
) -> Dict[str, Any]:
    n = len(dist_mat)
    nodes = [node_id_map.get(i, i) if node_id_map else i for i in range(n)]
    pairs: List[Dict[str, Any]] = []

    for i in range(n):
        fi = node_id_map.get(i, i) if node_id_map else i
        for j in range(n):
            if omit_diagonal and i == j:
                continue
            tj = node_id_map.get(j, j) if node_id_map else j
            pairs.append({"from_id": fi, "to_id": tj, "distance": dist_mat[i][j]})

    return {
        "problem_type": "MDP",
        "num_nodes": n,
        "m": m,
        "nodes": nodes,
        "distance_pairs": pairs,
    }


def mdp_render_input(
    fmt: str,
    dist_mat: List[List[float]],
    m: int,
    node_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
    omit_diagonal: bool = True,
) -> str:
    # defaults
    num_nodes_key = "num_nodes"
    nodes_key = "nodes"
    m_key = "m"
    from_id_key = "from_id"
    to_id_key = "to_id"
    dist_key = "distance"

    # renaming via scenario_hint (optional)
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "num_nodes":
                num_nodes_key = new
            elif orig == "nodes":
                nodes_key = new
            elif orig == "m":
                m_key = new

        for field in scenario_hint.get("pair_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "from_id":
                from_id_key = new
            elif orig == "to_id":
                to_id_key = new
            elif orig == "distance":
                dist_key = new

    n = len(dist_mat)
    node_ids = [node_id_map.get(i, i) if node_id_map else i for i in range(n)]
    nodes_str = ", ".join(str(x) for x in node_ids)

    pair_records: List[Dict[str, Any]] = []
    for i in range(n):
        fi = node_id_map.get(i, i) if node_id_map else i
        for j in range(n):
            if omit_diagonal and i == j:
                continue
            tj = node_id_map.get(j, j) if node_id_map else j
            pair_records.append({from_id_key: fi, to_id_key: tj, dist_key: dist_mat[i][j]})

    if fmt == "json":
        obj = {
            num_nodes_key: n,
            m_key: m,
            nodes_key: node_ids,
            "data": pair_records,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_nodes_key}={n}")
        lines.append(f"# {m_key}={m}")
        lines.append(f"# {nodes_key}={nodes_str}")
        lines.append(f"{from_id_key},{to_id_key},{dist_key}")
        for rec in pair_records:
            lines.append(f"{rec[from_id_key]},{rec[to_id_key]},{rec[dist_key]}")
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
                        num_nodes=n,
                        nodes=nodes_str,
                        m=m,
                        **{num_nodes_key: n, nodes_key: nodes_str, m_key: m},
                    )
                )
                lines.append("")

        lines.append(f"| {from_id_key} | {to_id_key} | {dist_key} |")
        lines.append("|---|---|---|")
        for rec in pair_records:
            lines.append(f"| {rec[from_id_key]} | {rec[to_id_key]} | {rec[dist_key]} |")

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        num_nodes=n,
                        nodes=nodes_str,
                        m=m,
                        **{num_nodes_key: n, nodes_key: nodes_str, m_key: m},
                    )
                )
        return "\n".join(lines)

    # fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        pair_line_tmpl = nl_style.get("pair_item_fields_line_template", "") or ""

        parts: List[str] = []
        base_vars = {
            "num_nodes": n,
            "nodes": nodes_str,
            "m": m,
            num_nodes_key: n,
            nodes_key: nodes_str,
            m_key: m,
        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        if pair_line_tmpl:
            for rec in pair_records:
                vars_ = dict(base_vars)
                vars_.update(
                    {
                        "from_id": rec.get(from_id_key, ""),
                        "to_id": rec.get(to_id_key, ""),
                        "distance": rec.get(dist_key, ""),
                        from_id_key: rec.get(from_id_key, ""),
                        to_id_key: rec.get(to_id_key, ""),
                        dist_key: rec.get(dist_key, ""),
                    }
                )
                parts.append(_format_safe(pair_line_tmpl, **vars_))

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)

    return ""


def contextualize_instance_mdp(
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
    m = core.get("m")

    if not isinstance(dist_mat, list) or not dist_mat or not isinstance(dist_mat[0], list):
        return results
    if not isinstance(m, int):
        return results

    n = len(dist_mat)
    if any(len(row) != n for row in dist_mat):
        return results
    if not (1 <= m <= n):
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

    labels = _build_labels(n, index_base)
    node_id_map = {i: labels[i] for i in range(n)}

    solution_variant = _remap_solution_mdp(solution, node_id_map=node_id_map)

    input_text = mdp_render_input(
        fmt=fmt,
        dist_mat=dist_mat,
        m=m,
        node_id_map=node_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
        omit_diagonal=True,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_mdp_instance(
        dist_mat=dist_mat,
        m=m,
        node_id_map=node_id_map,
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
    # MDP only
    desc = (
        "Maximum Diversity Problem (MDP): you need to pick a fixed-size subset of items so the chosen set is as spread out as possible.\n\n"
        "The input provides a distance (or dissimilarity) between every pair of items. A set is considered more diverse when the "
        "items inside it are far from each other.\n\n"
        "Your task is to select exactly m items that maximizes the total distance summed over all pairs within the selected set."
    )

    baseline_template = (
        "You are given a Maximum Diversity Problem (MDP).\n\n"
        "Problem description:\n"
        "- There are n items.\n"
        "- You must choose exactly m distinct items.\n"
        "- The instance provides a distance value for each ordered pair (i, j).\n\n"
        "Objective:\n"
        "- Maximize the total distance among the selected items.\n"
        "- Concretely, you want the chosen items to be as spread out (diverse) as possible.\n\n"
        "Important details:\n"
        "- The input is provided in PAIR format.\n"
        "- Distance values may be symmetric, but do not assume symmetry unless the data shows it.\n"
        "- Diagonal pairs (i == j) are omitted from the input.\n"
        "- Your answer must list exactly m item identifiers, with no duplicates.\n\n"
        "You will receive the instance in the following PAIR format:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your decision using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [<selected_item>, <selected_item>, ...]\n'
        "}\n"
        "```\n"
        "The list `solution` must contain exactly m distinct items, using identifiers exactly as they appear in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [<selected_item>, <selected_item>, ...]\n'
        "}\n"
        "```\n"
    )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    global_fields = [
        {"name": "num_nodes", "description": "number of items"},
        {"name": "nodes", "description": "item identifiers"},
        {"name": "m", "description": "number of items to select"},
    ]
    pair_item_fields = [
        {"name": "from_id", "description": "item id (first)"},
        {"name": "to_id", "description": "item id (second)"},
        {"name": "distance", "description": "distance / dissimilarity between the two items"},
    ]
    allowed = ["num_nodes", "nodes", "m", "from_id", "to_id", "distance"]
    return {
        "global_fields": global_fields,
        "pair_item_fields": pair_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize MDP instances into NL/JSON/CSV/Markdown (PAIR input only).")
    parser.add_argument("--problem_type", type=str, default="MDP", choices=["MDP"])
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
        contextualize_fn=contextualize_instance_mdp,
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
python -m step2_contextualization.contextualize_mdp \
  --problem_type MDP \
  --instance_dir ./step1_instance_creation/generated_data/MDP \
  --output_root_dir step2_contextualization/dataset/MDP \
  --k_per_call 20 \
  --n_target 50

"""
