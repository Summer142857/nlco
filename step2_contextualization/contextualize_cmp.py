# step2_contextualization/contextualize_cmp.py
import json
import random
from typing import Any, Dict, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for CMP (Cutwidth Minimization Problem).

Instance example:
{
  "instance": {
    "num_nodes": 12,
    "num_edges": 11,
    "edges": [[u,v], ...],
    "objective": <optional>
  },
  "solution": [perm...],
  "obj": <number>,
  "problem_type": "CMP"
}

We render input as edge pairs (u, v).
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _remap_solution_cmp(solution: Any, node_id_map: Optional[Dict[int, Any]]) -> Any:
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


def _build_labeled_cmp_instance(
    n: int,
    edges: List[List[int]],
    node_id_map: Optional[Dict[int, Any]],
) -> Dict[str, Any]:
    nodes = [node_id_map.get(i, i) if node_id_map else i for i in range(n)]
    edge_pairs: List[Dict[str, Any]] = []
    for e in edges:
        if not (isinstance(e, list) and len(e) == 2):
            continue
        u, v = e[0], e[1]
        uu = node_id_map.get(int(u), u) if node_id_map else u
        vv = node_id_map.get(int(v), v) if node_id_map else v
        edge_pairs.append({"u": uu, "v": vv})
    return {
        "problem_type": "CMP",
        "num_nodes": n,
        "nodes": nodes,
        "num_edges": len(edge_pairs),
        "edges": edge_pairs,
    }


def cmp_render_input(
    fmt: str,
    n: int,
    edges: List[List[int]],
    node_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    # defaults
    num_nodes_key = "num_nodes"
    num_edges_key = "num_edges"
    nodes_key = "nodes"

    # edge field names (for templates / table headers)
    u_key = "u"
    v_key = "v"

    # optional renaming
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "num_nodes":
                num_nodes_key = new
            elif orig == "num_edges":
                num_edges_key = new
            elif orig == "nodes":
                nodes_key = new

        for field in scenario_hint.get("edge_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig in ["u", "from_id", "node_u"]:
                u_key = new
            elif orig in ["v", "to_id", "node_v"]:
                v_key = new

    node_ids = [node_id_map.get(i, i) if node_id_map else i for i in range(n)]
    nodes_str = ", ".join(str(x) for x in node_ids)

    edge_recs: List[Dict[str, Any]] = []
    for e in edges:
        if not (isinstance(e, list) and len(e) == 2):
            continue
        u, v = e[0], e[1]
        uu = node_id_map.get(int(u), u) if node_id_map else u
        vv = node_id_map.get(int(v), v) if node_id_map else v
        edge_recs.append({u_key: uu, v_key: vv})

    m = len(edge_recs)

    if fmt == "json":
        obj = {
            num_nodes_key: n,
            num_edges_key: m,
            nodes_key: node_ids,
            "edges": edge_recs,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_nodes_key}={n}")
        lines.append(f"# {num_edges_key}={m}")
        lines.append(f"# {nodes_key}={nodes_str}")
        lines.append(f"{u_key},{v_key}")
        for r in edge_recs:
            lines.append(f"{r[u_key]},{r[v_key]}")
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
                        num_edges=m,
                        nodes=nodes_str,
                        **{num_nodes_key: n, num_edges_key: m, nodes_key: nodes_str},
                    )
                )
                lines.append("")

        lines.append(f"| {u_key} | {v_key} |")
        lines.append("|---|---|")
        for r in edge_recs:
            lines.append(f"| {r[u_key]} | {r[v_key]} |")

        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        num_nodes=n,
                        num_edges=m,
                        nodes=nodes_str,
                        **{num_nodes_key: n, num_edges_key: m, nodes_key: nodes_str},
                    )
                )

        return "\n".join(lines)

    # fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        edge_line_tmpl = nl_style.get("edge_item_fields_line_template", "") or ""

        parts: List[str] = []
        base_vars = {
            "num_nodes": n,
            "num_edges": m,
            "nodes": nodes_str,
            num_nodes_key: n,
            num_edges_key: m,
            nodes_key: nodes_str,
        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        if edge_line_tmpl:
            for r in edge_recs:
                vars_ = dict(base_vars)
                # support both generic placeholders and renamed keys
                vars_.update(
                    {
                        "u": r.get(u_key, ""),
                        "v": r.get(v_key, ""),
                        u_key: r.get(u_key, ""),
                        v_key: r.get(v_key, ""),
                    }
                )
                parts.append(_format_safe(edge_line_tmpl, **vars_))
        else:
            # fallback: one edge per line
            for r in edge_recs:
                parts.append(f"{r[u_key]} {r[v_key]}")

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)

    return ""


def contextualize_instance_cmp(
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

    n = core.get("num_nodes")
    edges = core.get("edges")

    if not isinstance(n, int) or n <= 0:
        return results
    if not isinstance(edges, list):
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

    solution_variant = _remap_solution_cmp(solution, node_id_map=node_id_map)

    input_text = cmp_render_input(
        fmt=fmt,
        n=n,
        edges=edges,
        node_id_map=node_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_cmp_instance(
        n=n,
        edges=edges,
        node_id_map=node_id_map,
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
    # CMP only
    desc = (
        "Cutwidth Minimization Problem (CMP): you need to place all nodes into a single linear order (a permutation).\n\n"
        "Given an undirected graph, look at every cut position in the order: nodes placed earlier are on the left, nodes placed later "
        "are on the right. The cut size at that position is the number of edges that connect a left node to a right node.\n\n"
        "Your task is to find an ordering that makes the largest cut size (over all positions) as small as possible."
    )

    baseline_template = (
        "You are given a Cutwidth Minimization Problem (CMP).\n\n"
        "Problem description:\n"
        "- There are n nodes that must be arranged into a single total order (a permutation).\n"
        "- The instance provides a list of edges between nodes.\n\n"
        "Objective:\n"
        "- Consider the ordering from left to right.\n"
        "- For each position k (after placing the first k nodes), count how many edges connect a placed node to an unplaced node.\n"
        "- The cutwidth of an ordering is the maximum of this count over all k.\n"
        "- Your goal is to MINIMIZE that maximum value.\n\n"
        "Important details:\n"
        "- Treat edges as undirected connections between two nodes.\n"
        "- Your answer must be a permutation containing each node exactly once.\n\n"
        "You will receive the instance below:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your decision using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": [<first_node>, <second_node>, ..., <last_node>]\n'
        "}\n"
        "```\n"
        "The list `solution` must contain each node exactly once, using identifiers exactly as they appear in the input.\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [<first_node>, <second_node>, ..., <last_node>]\n'
        "}\n"
        "```\n"
    )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    global_fields = [
        {"name": "num_nodes", "description": "number of nodes"},
        {"name": "num_edges", "description": "number of edges"},
        {"name": "nodes", "description": "node identifiers"},
    ]
    edge_item_fields = [
        {"name": "u", "description": "one endpoint of the edge"},
        {"name": "v", "description": "the other endpoint of the edge"},
    ]
    allowed = ["num_nodes", "num_edges", "nodes", "u", "v"]
    return {
        "global_fields": global_fields,
        "edge_item_fields": edge_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize CMP instances into NL/JSON/CSV/Markdown.")
    parser.add_argument("--problem_type", type=str, default="CMP", choices=["CMP"])
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
        contextualize_fn=contextualize_instance_cmp,
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
python -m step2_contextualization.contextualize_cmp \
  --problem_type CMP \
  --instance_dir ./step1_instance_creation/generated_data/CMP \
  --output_root_dir step2_contextualization/dataset/CMP \
  --k_per_call 20 \
  --n_target 50

'''
