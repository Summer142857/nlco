import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for LOP (Linear Ordering Problem).

LOP instance format:
{
  "instance": [[w_00, w_01, ...], [...], ...],   # square matrix
  "solution": [perm...]                          # permutation of node ids
  "obj": <number>
  "problem_type": "LOP"
}

We ALWAYS render input in pair mode:
- one (from_id,to_id,weight) per line
- diagonal pairs (i==j) omitted
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)
def _lop_matrix_to_markdown(
    inst_core: List[List[float]],
    node_ids: List[Any],
    id_label: str = "from\\to",
) -> str:
    """
    Render LOP weight matrix as Markdown table.
    First row/col are item ids (node_ids).
    """
    n = len(inst_core)
    if n == 0:
        return ""

    col_ids = [str(x) for x in node_ids]
    row_ids = [str(x) for x in node_ids]

    lines: List[str] = []
    lines.append("| " + " | ".join([id_label] + col_ids) + " |")
    lines.append("|" + "|".join(["---"] * (n + 1)) + "|")

    for i in range(n):
        row = inst_core[i]
        vals = []
        for j in range(n):
            try:
                vals.append(str(row[j]))
            except Exception:
                vals.append("0")
        lines.append("| " + " | ".join([row_ids[i]] + vals) + " |")

    return "\n".join(lines)


def _build_lop_matrix_desc(
    from_id_key: str,
    to_id_key: str,
    weight_key: str,
    num_nodes_key: int,
    global_nodes_key: str,
) -> str:
    """
    One short context-aware sentence describing the LOP matrix semantics.
    Uses renamed keys so it adapts to scenario_hint automatically.
    """
    return (
        f"Meaning: this is a directed {weight_key} matrix over items in {global_nodes_key} "
        f"({num_nodes_key} total). Entry at row {from_id_key}=i and column {to_id_key}=j "
        f"is the reward gained if i is placed BEFORE j in the final ordering (diagonal can be ignored)."
    )


def _remap_solution_lop(solution: Any, node_id_map: Optional[Dict[int, Any]] = None) -> Any:
    if solution is None or node_id_map is None:
        return solution
    if isinstance(solution, list):
        return [node_id_map.get(int(i), i) for i in solution]
    return solution


def _build_labeled_lop_instance(inst_core: List[List[float]], node_id_map: Optional[Dict[int, Any]]) -> Dict[str, Any]:
    n = len(inst_core)
    node_ids = [node_id_map.get(i, i) if node_id_map else i for i in range(n)]

    pairs: List[Dict[str, Any]] = []
    for i in range(n):
        fi = node_id_map.get(i, i) if node_id_map else i
        row = inst_core[i]
        for j in range(n):
            if i == j:
                continue
            tj = node_id_map.get(j, j) if node_id_map else j
            pairs.append({"from_id": fi, "to_id": tj, "weight": row[j]})

    return {
        "problem_type": "LOP",
        "num_nodes": n,
        "nodes": node_ids,
        "pairs": pairs,  # pair-mode view for debug/downstream
    }


def lop_render_input(
    fmt: str,
    inst_core: List[List[float]],
    node_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    # defaults
    num_nodes_key = "num_nodes"
    global_nodes_key = "nodes"
    from_id_key = "from_id"
    to_id_key = "to_id"
    weight_key = "weight"

    # renaming via scenario_hint (optional)
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "num_nodes":
                num_nodes_key = new
            elif orig == "nodes":
                global_nodes_key = new
            # NOTE: no "pairs" renaming anymore

        for field in scenario_hint.get("pair_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "from_id":
                from_id_key = new
            elif orig == "to_id":
                to_id_key = new
            elif orig == "weight":
                weight_key = new

    n = len(inst_core)
    node_ids = [node_id_map.get(i, i) if node_id_map else i for i in range(n)]
    nodes_str = ", ".join(str(x) for x in node_ids)

    pair_records: List[Dict[str, Any]] = []
    for i in range(n):
        fi = node_id_map.get(i, i) if node_id_map else i
        row = inst_core[i]
        for j in range(n):
            if i == j:
                continue
            tj = node_id_map.get(j, j) if node_id_map else j
            pair_records.append({from_id_key: fi, to_id_key: tj, weight_key: row[j]})
    # markdown matrix + description (context-aware via possibly-renamed keys)
    matrix_md = _lop_matrix_to_markdown(
        inst_core=inst_core,
        node_ids=node_ids,
        id_label=f"{from_id_key}\\{to_id_key}",
    )
    matrix_desc = _build_lop_matrix_desc(
        from_id_key=from_id_key,
        to_id_key=to_id_key,
        weight_key=weight_key,
        num_nodes_key=n,
        global_nodes_key=global_nodes_key,
    )

    if fmt == "json":
        obj = {
            num_nodes_key: n,
            global_nodes_key: node_ids,
        }
        json_part = json.dumps(obj, ensure_ascii=False, indent=2)

        return (
                json_part
                + "\n\n"
                + f"# {matrix_desc}\n"
                + f"# {weight_key} \n"
                + matrix_md
        )

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_nodes_key}={n}")
        lines.append(f"# {global_nodes_key}={nodes_str}")

        lines.append("")
        lines.append(f"# {matrix_desc}")
        lines.append(f"# {weight_key}")
        lines.append(matrix_md)

        return "\n".join(lines)

    if fmt == "markdown_table":
        lines: List[str] = []

        if isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            if header_tmpl:
                lines.append(
                    _format_safe(
                        header_tmpl,
                        num_nodes=n,
                        nodes=nodes_str,
                        **{num_nodes_key: n, global_nodes_key: nodes_str},
                    )
                )
                lines.append("")

        lines.append(f"*{matrix_desc}*")
        lines.append("")
        lines.append(f"**{weight_key})**")
        lines.append("")
        lines.append(matrix_md)

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        num_nodes=n,
                        nodes=nodes_str,
                        **{num_nodes_key: n, global_nodes_key: nodes_str},
                    )
                )

        return "\n".join(lines)

    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""

        parts: List[str] = []
        base_vars = {
            "num_nodes": n,
            "nodes": nodes_str,
            num_nodes_key: n,
            global_nodes_key: nodes_str,
        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        parts.append(matrix_desc)
        parts.append(f"{weight_key}:\n{matrix_md}")

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)


def contextualize_instance_lop(
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

    if not isinstance(inst_core, list) or not inst_core or not isinstance(inst_core[0], list):
        return results

    n = len(inst_core)
    if any(len(row) != n for row in inst_core):
        return results  # must be square

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

    solution_variant = _remap_solution_lop(solution, node_id_map=node_id_map)

    input_text = lop_render_input(
        fmt=fmt,
        inst_core=inst_core,
        node_id_map=node_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_lop_instance(inst_core, node_id_map=node_id_map)

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
    # LOP only
    desc = (
        "Linear Ordering Problem (LOP): you are asked to arrange a set of items into a single, complete sequence. "
        "Each item must appear exactly once.\n\n"
        "For any two different items, placing one item before the other produces a specific benefit. "
        "These benefits are given as directed pairs in the input, where each pair specifies the benefit of placing "
        "one item before another.\n\n"
        "Your task is to find the ordering of all items that makes the total accumulated benefit as large as possible."
    )

    baseline_template = (
        "You are given a Linear Ordering Problem (LOP).\n\n"
        "Problem description:\n"
        "- There are n items that must be arranged into a single total order.\n"
        "- For every ordered pair of distinct items (i, j), there is a weight w(i, j).\n"
        "- If item i is placed BEFORE item j in your final ordering, you earn reward w(i, j).\n"
        "- Your objective is to MAXIMIZE the total reward summed over all ordered pairs consistent with your ordering.\n\n"
        "Important details:\n"
        "- Weights are directed: w(i, j) is generally different from w(j, i).\n"
        "- Diagonal entries (i == j) are omitted from the input.\n"
        "- Your answer must be a permutation containing each item exactly once.\n\n"
        "You will receive the instance in the following PAIR format:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your decision using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [<first_item>, <second_item>, ..., <last_item>]\n'
        "}\n"
        "```\n"
        "The list `solution` must contain each item exactly once, using identifiers exactly as they appear in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [<first_item>, <second_item>, ..., <last_item>]\n'
        "}\n"
        "```\n"
    )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    # LOP hint for template generator / field renaming
    global_fields = [
        {"name": "num_nodes", "description": "total number of items to order"},
        {"name": "nodes", "description": "list of all item identifiers"},
    ]
    pair_item_fields = [
        {"name": "from_id", "description": "source item identifier"},
        {"name": "to_id", "description": "target item identifier"},
        {"name": "weight", "description": "reward if from_id is placed before to_id"},
    ]
    allowed = ["num_nodes", "nodes", "from_id", "to_id", "weight"]
    return {"global_fields": global_fields, "pair_item_fields": pair_item_fields, "allowed_placeholders": allowed}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize LOP instances into NL/JSON/CSV/Markdown (PAIR input only).")
    parser.add_argument("--problem_type", type=str, default="LOP", choices=["LOP"])
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
        contextualize_fn=contextualize_instance_lop,
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

python -m step2_contextualization.contextualize_lop \
  --problem_type LOP \
  --instance_dir ./step1_instance_creation/generated_data/LOP \
  --output_root_dir step2_contextualization/dataset/LOP \
  --k_per_call 20 \
  --n_target 50
'''
