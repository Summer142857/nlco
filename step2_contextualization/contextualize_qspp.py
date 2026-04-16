import json
import random
from typing import Any, Dict, List, Optional, Tuple

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for QSPP (Quadratic Shortest Path Problem).

Instance format example:
{
  "instance": {
    "nodes": [...],
    "edges": [{"from": u, "to": v, "var_index": k}, ...],
    "objective": {
      "constant": c,
      "linear": [..],             # indexed by var_index
      "quadratic": [[..],[..],..] # indexed by var_index (KEEP DIAGONAL)
    },
    "source": s,
    "target": t
  },
  "solution": [node0, node1, ..., nodeM],  # path as node sequence
  "obj": <number>,
  "problem_type": "QSPP"
}

We ALWAYS render input in PAIR / record form:
- edge list: (from, to, var_index)
- linear list: (var_index, linear_cost)
- quadratic list: (var_i, var_j, quadratic_cost)  # diagonal INCLUDED
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _remap_path_solution(solution: Any, node_id_map: Optional[Dict[int, Any]]) -> Any:
    if solution is None:
        return None
    if not isinstance(solution, list) or node_id_map is None:
        return solution
    out = []
    for x in solution:
        try:
            out.append(node_id_map.get(int(x), x))
        except Exception:
            out.append(x)
    return out


def _build_qspp_pairs(
    edges: List[Dict[str, Any]],
    linear: List[float],
    quadratic: List[List[float]],
    node_id_map: Optional[Dict[int, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    # edge records
    edge_recs: List[Dict[str, Any]] = []
    for e in edges:
        u = e.get("from")
        v = e.get("to")
        k = e.get("var_index")
        if u is None or v is None or k is None:
            continue
        u2 = node_id_map.get(int(u), u) if node_id_map else u
        v2 = node_id_map.get(int(v), v) if node_id_map else v
        edge_recs.append({"from": u2, "to": v2, "var_index": int(k)})

    # linear records (by var_index)
    lin_recs: List[Dict[str, Any]] = []
    for k, val in enumerate(linear):
        lin_recs.append({"var_index": int(k), "linear_cost": val})

    # quadratic records (KEEP DIAGONAL)
    quad_recs: List[Dict[str, Any]] = []
    n = len(linear)
    for i in range(n):
        row = quadratic[i]
        for j in range(n):
            quad_recs.append({"var_i": int(i), "var_j": int(j), "quadratic_cost": row[j]})

    return edge_recs, lin_recs, quad_recs
def _infer_edge_unit_name(edge_var_key: str) -> str:
    """
    Infer a human-friendly unit name from the edge id field key.
    Examples:
      segment_id -> segment
      link_id -> link
      var_index -> edge
    """
    k = (edge_var_key or "").strip().lower()
    for suffix in ["_id", "_index", "_idx"]:
        if k.endswith(suffix):
            k = k[: -len(suffix)]
            break
    k = k.replace("-", "_").replace(" ", "_")

    # common fallbacks
    if k in {"var", "varindex", "var_index"}:
        return "edge"
    if not k:
        return "edge"
    return k


def _build_quadratic_matrix_desc(
    quadratic_group_key: str,
    edge_var_key: str = "var_index",
    row_key: str = "var_i",
    col_key: str = "var_j",
) -> str:
    """
    Description for the quadratic matrix under the convention:
    - SUM over ALL ordered pairs (i, j) that are both selected
    - symmetric terms are therefore counted twice when Q is symmetric (i.e., Q[i][j] and Q[j][i])
    - diagonal terms (i == j) ARE INCLUDED
    """
    unit = _infer_edge_unit_name(edge_var_key)
    return (
        f"Meaning: the {quadratic_group_key} matrix is assumed symmetric and contributes to the objective as a sum over "
        f"ALL ordered pairs ({row_key}, {col_key}). If two {unit}s with IDs i and j are both used in the chosen path, "
        f"then quadratic_costs[i][j] is added to the total. This includes diagonal terms (i == j), so selecting edge i "
        f"also adds quadratic_costs[i][i]. Because the matrix is symmetric, the interaction between two distinct edges "
        f"i and j is counted twice in the ordered-pair sum: quadratic_costs[i][j] + quadratic_costs[j][i] (= 2 * quadratic_costs[i][j])."
    )


def _quadratic_to_markdown_matrix(
    quadratic: List[List[float]],
    row_key: str = "var_i",
    col_key: str = "var_j",
    id_label: Optional[str] = None,
) -> str:
    """
    Render quadratic cost matrix as a Markdown table.
    First row/col are ids. Diagonal included.
    """
    n = len(quadratic)
    if n == 0:
        return ""

    # ids are 0..n-1 (var indices)
    col_ids = [str(j) for j in range(n)]
    row_ids = [str(i) for i in range(n)]

    corner = id_label or f"{row_key}\\{col_key}"

    lines: List[str] = []
    # header row
    lines.append("| " + " | ".join([corner] + col_ids) + " |")
    lines.append("|" + "|".join(["---"] * (n + 1)) + "|")

    # body
    for i in range(n):
        row = quadratic[i]
        # safety: if row length mismatched, fallback pad/truncate
        vals = []
        for j in range(n):
            try:
                vals.append(str(row[j]))
            except Exception:
                vals.append("0")
        lines.append("| " + " | ".join([row_ids[i]] + vals) + " |")

    return "\n".join(lines)

def qspp_render_input(
    fmt: str,
    nodes: List[Any],
    edges: List[Dict[str, Any]],
    objective: Dict[str, Any],
    source: Any,
    target: Any,
    node_id_map: Optional[Dict[int, Any]],
    nl_style: Optional[Dict[str, Any]],
    scenario_hint: Optional[Dict[str, Any]] = None,
) -> str:
    # -----------------------------
    # DEFAULT FIELD NAMES (ALIGN WITH _get_hint())
    # -----------------------------
    num_nodes_key = "num_nodes"
    nodes_key = "nodes"
    num_edges_key = "num_edges"
    source_key = "source_node"
    target_key = "target_node"

    edge_group_key = "edges"
    linear_group_key = "linear_costs"
    quadratic_group_key = "quadratic_costs"

    # edge item fields
    edge_from_key, edge_to_key, edge_var_key = "edge_start_node", "edge_end_node", "edge_id"
    # linear item fields
    lin_var_key, lin_val_key = "edge_id", "linear_cost"
    # quadratic item fields
    quad_i_key, quad_j_key, quad_val_key = "edge_i_id", "edge_j_id", "quadratic_cost"

    # -----------------------------
    # OPTIONAL RENAME VIA scenario_hint (robust to old/new names)
    # -----------------------------
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "num_nodes":
                num_nodes_key = new
            elif orig == "nodes":
                nodes_key = new
            elif orig == "num_edges":
                num_edges_key = new
            elif orig in ("source_node", "source"):
                source_key = new
            elif orig in ("target_node", "target"):
                target_key = new
            elif orig == "edges":
                edge_group_key = new
            elif orig == "linear_costs":
                linear_group_key = new
            elif orig == "quadratic_costs":
                quadratic_group_key = new

        for field in scenario_hint.get("edge_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig in ("edge_start_node", "from", "from_node_id"):
                edge_from_key = new
            elif orig in ("edge_end_node", "to", "to_node_id"):
                edge_to_key = new
            elif orig in ("edge_id", "var_index", "edge_index", "edge_idx"):
                edge_var_key = new

        for field in scenario_hint.get("linear_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig in ("edge_id", "var_index", "edge_index", "edge_idx"):
                lin_var_key = new
            elif orig in ("linear_cost",):
                lin_val_key = new

        for field in scenario_hint.get("quadratic_item_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig in ("edge_i_id", "var_i"):
                quad_i_key = new
            elif orig in ("edge_j_id", "var_j"):
                quad_j_key = new
            elif orig in ("quadratic_cost",):
                quad_val_key = new

    linear = objective.get("linear", []) or []
    quadratic = objective.get("quadratic", []) or []

    # map nodes/source/target for display
    node_ids = [node_id_map.get(int(x), x) if node_id_map else x for x in nodes]
    src2 = node_id_map.get(int(source), source) if node_id_map else source
    tgt2 = node_id_map.get(int(target), target) if node_id_map else target

    edge_recs, lin_recs, _ = _build_qspp_pairs(edges, linear, quadratic, node_id_map=node_id_map)

    quad_md = _quadratic_to_markdown_matrix(
        quadratic=quadratic,
        row_key=quad_i_key,
        col_key=quad_j_key,
    )
    quad_desc = _build_quadratic_matrix_desc(
        quadratic_group_key=quadratic_group_key,
        edge_var_key=edge_var_key,
        row_key=quad_i_key,
        col_key=quad_j_key,
    )

    nodes_str = ", ".join(str(x) for x in node_ids)

    # ---------------- JSON ----------------
    if fmt == "json":
        obj = {
            num_nodes_key: len(node_ids),
            num_edges_key: len(edge_recs),
            nodes_key: node_ids,
            source_key: src2,
            target_key: tgt2,
            edge_group_key: [
                {edge_from_key: r["from"], edge_to_key: r["to"], edge_var_key: r["var_index"]}
                for r in edge_recs
            ],
            linear_group_key: [
                {lin_var_key: r["var_index"], lin_val_key: r["linear_cost"]}
                for r in lin_recs
            ],
        }

        json_part = json.dumps(obj, ensure_ascii=False, indent=2)
        return (
            json_part
            + "\n\n"
            + f"# {quad_desc}\n"
            + f"# {quadratic_group_key}\n"
            + quad_md
        )

    # ---------------- CSV ----------------
    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_nodes_key}={len(node_ids)}")
        lines.append(f"# {num_edges_key}={len(edge_recs)}")
        lines.append(f"# {nodes_key}={nodes_str}")
        lines.append(f"# {source_key}={src2}")
        lines.append(f"# {target_key}={tgt2}")

        lines.append("")
        lines.append(f"{edge_from_key},{edge_to_key},{edge_var_key}")
        for r in edge_recs:
            lines.append(f"{r['from']},{r['to']},{r['var_index']}")

        lines.append("")
        lines.append(f"{lin_var_key},{lin_val_key}")
        for r in lin_recs:
            lines.append(f"{r['var_index']},{r['linear_cost']}")

        lines.append("")
        lines.append(f"# {quad_desc}")
        lines.append(f"# {quadratic_group_key}")
        lines.append(quad_md)

        return "\n".join(lines)

    # ---------------- MARKDOWN TABLE ----------------
    if fmt == "markdown_table":
        lines: List[str] = []

        # header (if any)
        if isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            if header_tmpl:
                lines.append(
                    _format_safe(
                        header_tmpl,
                        num_nodes=len(node_ids),
                        num_edges=len(edge_recs),
                        nodes=nodes_str,
                        source_node=src2,
                        target_node=tgt2,

                    )
                )
                lines.append("")

        lines.append(f"| {edge_from_key} | {edge_to_key} | {edge_var_key} |")
        lines.append("|---|---|---|")
        for r in edge_recs:
            lines.append(f"| {r['from']} | {r['to']} | {r['var_index']} |")

        lines.append("")
        lines.append(f"| {lin_var_key} | {lin_val_key} |")
        lines.append("|---|---|")
        for r in lin_recs:
            lines.append(f"| {r['var_index']} | {r['linear_cost']} |")

        lines.append("")
        lines.append(f"*{quad_desc}*")
        lines.append("")
        lines.append(f"**{quadratic_group_key}**")
        lines.append("")
        lines.append(quad_md)

        # footer (if any)
        if isinstance(nl_style, dict):
            footer_tmpl = nl_style.get("footer", "") or ""
            if footer_tmpl:
                lines.append("")
                lines.append(
                    _format_safe(
                        footer_tmpl,
                        num_nodes=len(node_ids),
                        num_edges=len(edge_recs),
                        nodes=nodes_str,
                        source_node=src2,
                        target_node=tgt2,

                    )
                )
        return "\n".join(lines)

    # ---------------- NL ----------------
    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        edge_line_tmpl = nl_style.get("edge_item_fields_line_template", "") or ""
        lin_line_tmpl = nl_style.get("linear_item_fields_line_template", "") or ""
        quad_line_tmpl = nl_style.get("quadratic_item_fields_line_template", "") or ""

        parts: List[str] = []

        # Provide BOTH new placeholders (hint-aligned) + old ones (compat)
        base_vars = {
            # hint-aligned canonical placeholders
            "num_nodes": len(node_ids),
            "num_edges": len(edge_recs),
            "nodes": nodes_str,
            "source_node": src2,
            "target_node": tgt2,

        }

        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **base_vars))

        if edge_line_tmpl:
            for r in edge_recs:
                parts.append(
                    _format_safe(
                        edge_line_tmpl,
                        **base_vars,
                        # canonical keys (from/to/var_index) for template convenience
                        **{"edge_start_node": r["from"], "edge_end_node": r["to"], "edge_id": r["var_index"]},
                        # hint-aligned keys if template uses them

                    )
                )

        if lin_line_tmpl:
            print(lin_line_tmpl)
            for r in lin_recs:
                parts.append(
                    _format_safe(
                        lin_line_tmpl,
                        **base_vars,
                        edge_id=r["var_index"],
                        linear_cost=r["linear_cost"],
                    )
                )

        # quadratic block (markdown matrix)
        parts.append(quad_desc)
        parts.append(f"{quadratic_group_key}:\n{quad_md}")

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **base_vars))

        return "\n".join(parts)

    return ""



def contextualize_instance_qspp(
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

    nodes = core.get("nodes")
    edges = core.get("edges")
    objective = core.get("objective")
    source = core.get("source")
    target = core.get("target")

    if not isinstance(nodes, list) or not nodes:
        return results
    if not isinstance(edges, list) or not edges:
        return results
    if not isinstance(objective, dict):
        return results
    if source is None or target is None:
        return results

    linear = objective.get("linear")
    quadratic = objective.get("quadratic")
    if not isinstance(linear, list) or not linear:
        return results
    if not isinstance(quadratic, list) or len(quadratic) != len(linear):
        return results
    if any((not isinstance(row, list) or len(row) != len(linear)) for row in quadratic):
        return results

    # validate var_index range
    m = len(linear)
    for e in edges:
        if not isinstance(e, dict):
            return results
        k = e.get("var_index")
        if k is None:
            return results
        try:
            kk = int(k)
        except Exception:
            return results
        if kk < 0 or kk >= m:
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

    # node labels
    node_labels = _build_labels(len(nodes), index_base)
    # map original node ids (assume 0..n-1 as in your data) to labels
    node_id_map = {}
    for i, nid in enumerate(nodes):
        try:
            node_id_map[int(nid)] = node_labels[i]
        except Exception:
            node_id_map[nid] = node_labels[i]

    solution_variant = _remap_path_solution(solution, node_id_map=node_id_map)

    input_text = qspp_render_input(
        fmt=fmt,
        nodes=nodes,
        edges=edges,
        objective=objective,
        source=source,
        target=target,
        node_id_map=node_id_map,
        nl_style=nl_style,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    # optional: lightweight labeled view
    # build labeled pair/record view (same structure as render_input uses)
    edge_recs_v, lin_recs_v, quad_recs_v = _build_qspp_pairs(
        edges=edges,
        linear=linear,
        quadratic=quadratic,
        node_id_map=node_id_map,
    )

    instance_variant = {
        "problem_type": "QSPP",
        "num_nodes": len(nodes),
        "num_edges": len(edges),

        # labeled nodes + endpoints
        "nodes": [node_id_map.get(int(x), x) if node_id_map else x for x in nodes],
        "source": node_id_map.get(int(source), source) if node_id_map else source,
        "target": node_id_map.get(int(target), target) if node_id_map else target,

        # objective (keep everything, including diagonal)
        "objective": {
            "constant": objective.get("constant", 0.0),
            "linear": lin_recs_v,  # list of {var_index, linear_cost}
            "quadratic": quad_recs_v,  # list of {var_i, var_j, quadratic_cost}
        },

        # graph edges in pair/record form
        "edges": edge_recs_v,  # list of {from, to, var_index}

        # helpful for debugging / reversibility
        "node_id_map": node_id_map,  # original node id -> label
    }

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
    desc = (
        "Quadratic Shortest Path Problem (QSPP).\n\n"
        "You are given a directed graph with a designated source node and a designated target node. "
        "Your task is to choose exactly one directed path from the source to the target.\n\n"
        "Graph structure:\n"
        "- Nodes represent locations in the graph.\n"
        "- Directed edges connect pairs of nodes.\n"
        "- Each directed edge has a unique identifier (edge_id).\n"
        "- IMPORTANT: edge_id identifies an EDGE, not a node. Edge identifiers must never appear in the output.\n\n"
        "Costs:\n"
        "- Each edge_id has a base (linear) cost, incurred if that edge is used in the path.\n"
        "- In addition, quadratic interaction costs are defined between pairs of edges.\n"
        "- A quadratic cost is incurred if and only if both corresponding edges are used in the path.\n\n"
        "Quadratic costs are defined for all ordered pairs (i, j) of edge IDs, including diagonal terms (i == j):\n"
        "- Using edge i contributes quadratic_costs[i][i].\n"
        "- Using two distinct edges i and j contributes quadratic_costs[i][j] + quadratic_costs[j][i].\n\n"
        "Objective:\n"
        "- Minimize the total cost, defined as the sum of all linear edge costs and all quadratic interaction costs "
        "over the selected path.\n\n"
        "Output requirement:\n"
        "- Output a single valid directed path as a sequence of NODE identifiers only.\n"
        "- The first node must be the source, and the last node must be the target.\n"
        "- Each consecutive pair of nodes must correspond to an existing directed edge.\n"
        "- Do NOT include edge_id or any cost-related identifiers in the output."
    )

    baseline_template = (
        "You are given a Quadratic Shortest Path Problem (QSPP).\n\n"
        "Problem description:\n"
        "- The input specifies a directed graph.\n"
        "- A source node and a target node are given.\n"
        "- You must choose exactly one directed path from the source node to the target node.\n\n"
        "Graph structure:\n"
        "- The graph is defined by a list of directed edges.\n"
        "- Each edge record includes (edge_start_node, edge_end_node, edge_id).\n"
        "- IMPORTANT: edge_id is an EDGE identifier used ONLY for costs. It is NOT a node ID.\n\n"
        "Costs:\n"
        "- Each edge_id has a base (linear) cost.\n"
        "- Quadratic costs are defined for all ordered pairs (i, j) of edge IDs, including diagonal terms (i == j).\n"
        "- If both edges i and j are selected in the path, then quadratic_costs[i][j] contributes to the objective.\n\n"
        "Objective:\n"
        "- Minimize: total_cost = (sum of linear costs of selected edges)\n"
        "            + (sum of quadratic_costs[i][j] over all ordered pairs (i, j) of selected edges).\n\n"
        "Input format:\n"
        "{{INSTANCE_INPUT}}\n\n"
        "Return your decision using exactly the following JSON format (no extra keys, no explanations):\n"
        "```json\n"
        "{\n"
        '  "solution": []\n'
        "}\n"
        "```\n"
        "The list `solution` must be a valid directed path expressed as a sequence of **node identifiers only**.\n"
        "Each entry in the list must be a node ID from the input graph.\n"
        "The first node must be the given source node, and the last node must be the given target node.\n"
        "**Do NOT include edge_id (or any edge identifiers) in the solution.**\n"
        "**Do not include explanations or any extra keys.**"
    )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": []\n'
        "}\n"
        "```\n"
    )

    return {
        "task_description": desc,
        "baseline_template": baseline_template,
        "output_format": output_format,
    }

def _get_hint(problem_type: str) -> Dict[str, Any]:
    # Minimal hint. Focus: make every field name self-explanatory and unambiguous.

    global_fields = [
        {
            "name": "num_nodes",
            "description": "Total number of nodes in the directed graph."
        },
        {
            "name": "num_edges",
            "description": "Total number of directed edges in the graph."
        },
        {
            "name": "nodes",
            "description": (
                "List of valid node identifiers. "
                "The solution path must be an ordered list of node IDs chosen from this list."
            ),
        },
        {
            "name": "source_node",
            "description": "Source node ID. The solution path must start at this node."
        },
        {
            "name": "target_node",
            "description": "Target node ID. The solution path must end at this node."
        },
    ]

    edge_item_fields = [
        {
            "name": "edge_start_node",
            "description": "Start (tail) node ID of this directed edge."
        },
        {
            "name": "edge_end_node",
            "description": "End (head) node ID of this directed edge."
        },
        {
            "name": "edge_id",
            "description": (
                "Unique identifier of this directed edge. "
                "This ID is used to reference linear costs and quadratic interaction costs."
            ),
        },
    ]

    linear_item_fields = [
        {
            "name": "edge_id",
            "description": (
                "Edge identifier. Must match an edge_id listed in the edges section."
            ),
        },
        {
            "name": "linear_cost",
            "description": (
                "Base cost incurred if and only if this edge is used in the chosen path."
            ),
        },
    ]

    quadratic_item_fields = [
        {
            "name": "edge_i_id",
            "description": (
                "Identifier of the first edge in a quadratic interaction term."
            ),
        },
        {
            "name": "edge_j_id",
            "description": (
                "Identifier of the second edge in a quadratic interaction term."
            ),
        },
        {
            "name": "quadratic_cost",
            "description": (
                "Additional cost incurred if both edges are used in the path "
                "(diagonal terms represent self-interaction)."
            ),
        },
    ]

    allowed = [
        "num_nodes",
        "num_edges",
        "nodes",
        "source_node",
        "target_node",
        "edge_start_node",
        "edge_end_node",
        "edge_id",
        "linear_cost",
        "edge_i_id",
        "edge_j_id",
        "quadratic_cost",
    ]

    return {
        "global_fields": global_fields,
        "edge_item_fields": edge_item_fields,
        "linear_item_fields": linear_item_fields,
        "quadratic_item_fields": quadratic_item_fields,
        "allowed_placeholders": allowed,
    }



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Contextualize QSPP instances into NL/JSON/CSV/Markdown (PAIR input only).")
    parser.add_argument("--problem_type", type=str, default="QSPP", choices=["QSPP"])
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
        contextualize_fn=contextualize_instance_qspp,
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
python -m step2_contextualization.contextualize_qspp \
  --problem_type QSPP \
  --instance_dir ./step1_instance_creation/generated_data/QSPP \
  --output_root_dir step2_contextualization/dataset/QSPP \
  --k_per_call 20 \
  --n_target 50


'''
