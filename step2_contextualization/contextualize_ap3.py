# step2_contextualization/contextualize_ap3.py
import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for AP3 (Three-Index Assignment Problem).

Instance example:
{
  "instance": {
    "cost_tensor": [[[...]]],   # n x n x n
    "objective": <optional>
  },
  "solution": [[i,j,k], ...],  # one triple per i (i appears exactly once)
  "obj": <number>,
  "problem_type": "AP3"
}

We render input by expanding cost_tensor into triplets (i, j, k, cost).
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)

def _matrix2d_to_markdown(
    mat: List[List[float]],
    row_ids: List[Any],
    col_ids: List[Any],
    corner_label: str,
) -> str:
    """
    Render a 2D matrix as Markdown table.
    First row/col are ids.
    """
    if not mat:
        return ""
    nrow = len(mat)
    ncol = len(mat[0]) if nrow > 0 else 0

    lines: List[str] = []
    lines.append("| " + " | ".join([corner_label] + [str(x) for x in col_ids]) + " |")
    lines.append("|" + "|".join(["---"] * (ncol + 1)) + "|")

    for r in range(nrow):
        row = mat[r]
        vals = []
        for c in range(ncol):
            try:
                vals.append(str(row[c]))
            except Exception:
                vals.append("0")
        lines.append("| " + " | ".join([str(row_ids[r])] + vals) + " |")

    return "\n".join(lines)
def _build_ap3_tensor_desc(
    n_key: str,
    ids_key: str,
    i_key: str,
    j_key: str,
    k_key: str,
    cost_key: str,
) -> str:
    return (
        f"Meaning: costs form a 3D tensor. For each fixed {i_key}=i, you are given a {j_key}×{k_key} "
        f"matrix whose entry at row {j_key}=j and column {k_key}=k equals {cost_key}(i,j,k). "
        f"You must pick exactly one (j,k) for every i, with all {j_key} and all {k_key} used exactly once."
    )

def _remap_solution_ap3(solution: Any, id_map: Optional[Dict[int, Any]]) -> Any:
    """
    solution expected as list of triples [[i,j,k], ...]
    remap i/j/k via id_map (if provided)
    """
    if solution is None or id_map is None:
        return solution
    if not isinstance(solution, list):
        return solution

    out = []
    for t in solution:
        if not (isinstance(t, list) and len(t) == 3):
            out.append(t)
            continue
        i, j, k = t
        try:
            ii = id_map.get(int(i), i)
        except Exception:
            ii = i
        try:
            jj = id_map.get(int(j), j)
        except Exception:
            jj = j
        try:
            kk = id_map.get(int(k), k)
        except Exception:
            kk = k
        out.append([ii, jj, kk])
    return out


def _build_labeled_ap3_instance(
    cost_tensor: List[List[List[float]]],
    id_map: Optional[Dict[int, Any]],
) -> Dict[str, Any]:
    n = len(cost_tensor)
    ids = [id_map.get(i, i) if id_map else i for i in range(n)]

    triplets: List[Dict[str, Any]] = []
    for i in range(n):
        ii = id_map.get(i, i) if id_map else i
        for j in range(n):
            jj = id_map.get(j, j) if id_map else j
            for k in range(n):
                kk = id_map.get(k, k) if id_map else k
                triplets.append({"i": ii, "j": jj, "k": kk, "cost": cost_tensor[i][j][k]})

    return {
        "problem_type": "AP3",
        "n": n,
        "ids": ids,
        "costs": triplets,
    }


def ap3_render_input(
    fmt: str,
    cost_tensor: List[List[List[float]]],
    id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    # defaults
    n_key = "n"
    ids_key = "ids"

    i_key, j_key, k_key, cost_key = "i", "j", "k", "cost"

    # optional renaming
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig in ["n", "num_nodes", "num_items"]:
                n_key = new
            elif orig in ["ids", "items", "indices"]:
                ids_key = new

        for field in scenario_hint.get("cost_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "i":
                i_key = new
            elif orig == "j":
                j_key = new
            elif orig == "k":
                k_key = new
            elif orig in ["cost", "value", "weight"]:
                cost_key = new

    n = len(cost_tensor)
    ids = [id_map.get(i, i) if id_map else i for i in range(n)]
    ids_str = ", ".join(str(x) for x in ids)

    cost_recs: List[Dict[str, Any]] = []
    for i in range(n):
        ii = id_map.get(i, i) if id_map else i
        for j in range(n):
            jj = id_map.get(j, j) if id_map else j
            for k in range(n):
                kk = id_map.get(k, k) if id_map else k
                cost_recs.append({i_key: ii, j_key: jj, k_key: kk, cost_key: cost_tensor[i][j][k]})
    desc = _build_ap3_tensor_desc(
        n_key=n_key,
        ids_key=ids_key,
        i_key=i_key,
        j_key=j_key,
        k_key=k_key,
        cost_key=cost_key,
    )

    # For each i, build a j×k markdown matrix
    blocks: List[str] = []
    for i in range(n):
        ii = id_map.get(i, i) if id_map else i
        # build j×k matrix for fixed i
        mat_jk = []
        for j in range(n):
            row = []
            for k in range(n):
                row.append(cost_tensor[i][j][k])
            mat_jk.append(row)

        block = (
                f"## {i_key}={ii}\n"
                + _matrix2d_to_markdown(
            mat=mat_jk,
            row_ids=ids,  # rows are j ids
            col_ids=ids,  # cols are k ids
            corner_label=f"{j_key}\\{k_key}",
        )
        )
        blocks.append(block)

    tensor_md = "\n\n".join(blocks)

    if fmt == "json":
        obj = {n_key: n, ids_key: ids}
        json_part = json.dumps(obj, ensure_ascii=False, indent=2)
        return (
                json_part
                + "\n\n"
                + f"# {desc}\n"
                + f"# {cost_key}_tensor (per-{i_key} markdown matrices)\n\n"
                + tensor_md
        )

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {n_key}={n}")
        lines.append(f"# {ids_key}={ids_str}")
        lines.append("")
        lines.append(f"# {desc}")
        lines.append(f"# {cost_key}_tensor (per-{i_key} markdown matrices)")
        lines.append("")
        lines.append(tensor_md)
        return "\n".join(lines)

    if fmt == "markdown_table":
        lines: List[str] = []

        if isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            if header_tmpl:
                lines.append(
                    _format_safe(
                        header_tmpl,
                        n=n,
                        ids=ids_str,
                        **{n_key: n, ids_key: ids_str},
                    )
                )
                lines.append("")

        lines.append(f"*{desc}*")
        lines.append("")
        lines.append(f"**{cost_key}_tensor (per-{i_key} markdown matrices)**")
        lines.append("")
        lines.append(tensor_md)

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        n=n,
                        ids=ids_str,
                        **{n_key: n, ids_key: ids_str},
                    )
                )
        return "\n".join(lines)

    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""

        parts: List[str] = []
        base_vars = {"n": n, "ids": ids_str, n_key: n, ids_key: ids_str}

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        parts.append(desc)
        parts.append(f"{cost_key}_tensor (per-{i_key} markdown matrices):\n{tensor_md}")

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)


def contextualize_instance_ap3(
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

    cost_tensor = core.get("cost_tensor")
    if not (isinstance(cost_tensor, list) and cost_tensor and isinstance(cost_tensor[0], list) and isinstance(cost_tensor[0][0], list)):
        return results

    n = len(cost_tensor)
    # validate cube shape
    if any(len(cost_tensor[i]) != n for i in range(n)):
        return results
    for i in range(n):
        if any(len(cost_tensor[i][j]) != n for j in range(n)):
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
    id_map = {i: labels[i] for i in range(n)}

    solution_variant = _remap_solution_ap3(solution, id_map=id_map)

    input_text = ap3_render_input(
        fmt=fmt,
        cost_tensor=cost_tensor,
        id_map=id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_ap3_instance(cost_tensor=cost_tensor, id_map=id_map)

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
    # AP3 only
    desc = (
        "Three-Index Assignment Problem (AP3): you need to create a one-to-one matching across three index sets of equal size.\n\n"
        "Think of choosing exactly one triple (i, j, k) for each i. The choices must be consistent: every j is used exactly once "
        "and every k is used exactly once.\n\n"
        "Each triple (i, j, k) has a given cost, and your task is to pick the set of triples with the smallest total cost."
    )

    baseline_template = (
        "You are given a Three-Index Assignment Problem (AP3).\n\n"
        "Problem description:\n"
        "- There are n items in each of three index sets (call them i, j, k).\n"
        "- For every combination (i, j, k), the instance provides a cost c(i, j, k).\n"
        "- You must choose exactly one (j, k) partner for every i.\n"
        "- Each j must be used exactly once across all chosen triples.\n"
        "- Each k must be used exactly once across all chosen triples.\n\n"
        "Objective:\n"
        "- Minimize the total cost summed over the n chosen triples.\n\n"
        "You will receive the instance below:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your decision using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [\n'
        '    [<i_id>, <j_id>, <k_id>],\n'
        '    ...\n'
        "  ]\n"
        "}\n"
        "```\n"
        "Rules for `solution`:\n"
        "- It must contain exactly n triples.\n"
        "- Every i identifier must appear exactly once as the first element of a triple.\n"
        "- Every j identifier must appear exactly once as the second element of a triple.\n"
        "- Every k identifier must appear exactly once as the third element of a triple.\n"
        "- All identifiers must match exactly those used in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [\n'
        '    [<i_id>, <j_id>, <k_id>],\n'
        '    ...\n'
        "  ]\n"
        "}\n"
        "```\n"
    )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    global_fields = [
        {"name": "n", "description": "problem size (same size for i/j/k sets)"},
        {"name": "ids", "description": "identifiers for indices"},
    ]
    cost_item_fields = [
        {"name": "i", "description": "first index id"},
        {"name": "j", "description": "second index id"},
        {"name": "k", "description": "third index id"},
        {"name": "cost", "description": "cost of choosing triple (i,j,k)"},
    ]
    allowed = ["n", "ids", "i", "j", "k", "cost"]
    return {
        "global_fields": global_fields,
        "cost_item_fields": cost_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize AP3 instances into NL/JSON/CSV/Markdown.")
    parser.add_argument("--problem_type", type=str, default="AP3", choices=["AP3"])
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
        contextualize_fn=contextualize_instance_ap3,
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
python -m step2_contextualization.contextualize_ap3 \
  --problem_type AP3 \
  --instance_dir ./step1_instance_creation/generated_data/AP3 \
  --output_root_dir step2_contextualization/dataset/AP3 \
  --k_per_call 20 \
  --n_target 50


'''
