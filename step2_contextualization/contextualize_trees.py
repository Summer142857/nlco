import json
import random
from typing import Dict, Any, List, Tuple

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for tree-like graph problems:
- STP  : Steiner Tree Problem
- SFP  : Steiner Forest Problem (problem_type = 'SFP', instance['problem_type'] = 'SF')
- KMST : k-Minimum Spanning Tree Problem
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


# ----------------------------------------------------------------------
# 1. Solution remapping & variant building
# ----------------------------------------------------------------------


def _build_node_label_map(num_nodes: int, index_base) -> Dict[int, Any]:
    """
    Build a mapping from internal 1..num_nodes to human-friendly labels.
    We assume the instance generator has relabeled nodes to 1..num_nodes.
    """
    node_labels = make_labels(num_nodes, index_base)
    return {i + 1: node_labels[i] for i in range(len(node_labels))}


def _remap_solution_edges(
    solution: list | None,
    id_map: Dict[int, Any],
) -> list:
    """
    Remap a list of edge solutions to new node labels.

    Input:
        solution = [[u, v], [u, v], ...]
    Output:
        [[u_label, v_label], ...]
    """
    if not solution:
        return []

    remapped = []
    for u, v in solution:
        remapped.append([
            id_map.get(int(u), u),
            id_map.get(int(v), v),
        ])
    return remapped



def _build_labeled_tree_instance(
    inst_core: Dict[str, Any],
    id_map: Dict[int, Any],
    problem_type: str,
) -> Dict[str, Any]:
    """
    Build a variant of the instance where node ids are replaced
    by human-friendly labels. This is only for 'instance_variant' in
    the contextualized record; the original 'instance' is kept as-is.
    """
    pt = problem_type.upper()
    num_nodes = inst_core.get("num_nodes")
    num_edges = inst_core.get("num_edges")
    edges = inst_core.get("edges", [])
    density = inst_core.get("density")
    source_file = inst_core.get("source_file")

    # Remap edges
    edges_variant: List[Dict[str, Any]] = []
    for e in edges:
        u = e.get("u")
        v = e.get("v")
        w = e.get("w", 1.0)
        edges_variant.append(
            {
                "u": id_map.get(int(u), u),
                "v": id_map.get(int(v), v),
                "w": w,
            }
        )

    inst_variant: Dict[str, Any] = {
        "problem_type": problem_type,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "edges": edges_variant,
    }
    if density is not None:
        inst_variant["density"] = density
    if source_file is not None:
        inst_variant["source_file"] = source_file

    # Extra fields per problem type
    if pt == "STP":
        terminals = inst_core.get("terminals", [])
        inst_variant["terminals"] = [id_map.get(int(t), t) for t in terminals]
    elif pt == "SFP":
        # Note: inner instance["problem_type"] is "SF", outer task is "SFP"
        # We take "terminal_groups" and "terminals" if present
        term_groups = inst_core.get("terminal_groups", [])
        term_groups_variant: List[List[Any]] = []
        for g in term_groups:
            term_groups_variant.append([id_map.get(int(t), t) for t in g])
        inst_variant["terminal_groups"] = term_groups_variant

        terminals = inst_core.get("terminals", [])
        inst_variant["terminals"] = [id_map.get(int(t), t) for t in terminals]
        num_groups = inst_core.get("num_groups")
        if num_groups is not None:
            inst_variant["num_groups"] = num_groups
    elif pt == "KMST":
        k = inst_core.get("k")
        if k is not None:
            inst_variant["k"] = k

    return inst_variant


# ----------------------------------------------------------------------
# 2. Input rendering (JSON / CSV / Markdown / NL)
# ----------------------------------------------------------------------


def tree_render_nl_with_style(
    inst_core: Dict[str, Any],
    node_id_map: Dict[int, Any],
    nl_style: Dict[str, Any],
    problem_type: str,
) -> str:
    """
    Render the instance in natural language using an NL style template.

    We enforce 'one edge per line' in the main body:
    line_template will normally refer to {u}, {v}, {w}.
    """
    line_t = nl_style.get("line_template")
    header_t = nl_style.get("header")
    footer_t = nl_style.get("footer")

    out: List[str] = []

    num_nodes = inst_core.get("num_nodes")
    num_edges = inst_core.get("num_edges")
    density = inst_core.get("density", None)

    base_ctx = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "problem_type": problem_type,
    }

    # Extra fields per problem
    pt = problem_type.upper()
    if pt == "STP":
        terminals = inst_core.get("terminals", [])
        base_ctx["terminals"] = " ".join(str(node_id_map.get(int(t), t)) for t in terminals)
        base_ctx["num_terminals"] = len(terminals)
    elif pt == "SFP":
        term_groups = inst_core.get("terminal_groups", [])
        # simple textual description
        groups_text = []
        for idx, g in enumerate(term_groups, start=1):
            labels = [str(node_id_map.get(int(t), t)) for t in g]
            groups_text.append(f"Group {idx}: {' '.join(labels)}")
        base_ctx["terminal_groups"] = "\n".join(groups_text)
        base_ctx["num_groups"] = len(term_groups)
    elif pt == "KMST":
        k = inst_core.get("k")
        base_ctx["k"] = k

    # header
    if header_t:
        out.append(_format_safe(header_t, **base_ctx))

    # edges, one line per edge
    edges = inst_core.get("edges", [])
    for e in edges:
        u = e.get("u")
        v = e.get("v")
        w = e.get("w", 1.0)
        u_label = node_id_map.get(int(u), u)
        v_label = node_id_map.get(int(v), v)
        ctx = {
            **base_ctx,
            "u": u_label,
            "v": v_label,
            "w": w,
        }
        if line_t:
            out.append(_format_safe(line_t, **ctx))

    # footer
    if footer_t:
        out.append(_format_safe(footer_t, **base_ctx))

    return "\n".join(out)


def tree_render_input(
    fmt: str,
    inst_core: Dict[str, Any],
    node_id_map: Dict[int, Any],
    nl_style: Dict[str, Any] | None,
    problem_type: str,
    scenario_hint: Dict[str, Any] | None = None,
) -> str:
    """
    Render the instance into different input formats.

    - JSON: keys may be renamed according to scenario_hint.global_fields / item_fields
    - CSV / markdown_table: one edge per row
    - NL: uses tree_render_nl_with_style + nl_style
    """

    num_nodes = inst_core.get("num_nodes")
    num_edges = inst_core.get("num_edges")
    edges = inst_core.get("edges", [])
    density = inst_core.get("density", None)
    pt = problem_type.upper()

    # Original field names
    num_nodes_key = "num_nodes"
    num_edges_key = "num_edges"
    edges_key = "edges"
    terminals_key = "terminals"
    terminal_groups_key = "terminal_groups"
    k_key = "k"

    # For item (edge) fields
    u_key = "u"
    v_key = "v"
    w_key = "w"

    # Apply scenario hint renaming
    if scenario_hint and isinstance(scenario_hint, dict):
        # global_fields: num_nodes, num_edges, terminals, terminal_groups, k, ...
        for field in scenario_hint.get("global_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "num_nodes":
                num_nodes_key = new
            elif orig == "num_edges":
                num_edges_key = new
            elif orig == "terminals":
                terminals_key = new
            elif orig == "terminal_groups":
                terminal_groups_key = new
            elif orig == "k":
                k_key = new

        # item_fields: u, v, w
        for field in scenario_hint.get("item_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "u":
                u_key = new
            elif orig == "v":
                v_key = new
            elif orig == "w":
                w_key = new

    # Build labeled edges (using node_id_map)
    labeled_edges: List[Dict[str, Any]] = []
    for e in edges:
        u = e.get("u")
        v = e.get("v")
        w = e.get("w", 1.0)
        labeled_edges.append(
            {
                u_key: node_id_map.get(int(u), u),
                v_key: node_id_map.get(int(v), v),
                w_key: w,
            }
        )

    # JSON format
    if fmt == "json":
        obj: Dict[str, Any] = {
            num_nodes_key: num_nodes,
            num_edges_key: num_edges,
            edges_key: labeled_edges,
        }


        if pt == "STP":
            terminals = inst_core.get("terminals", [])
            obj[terminals_key] = [node_id_map.get(int(t), t) for t in terminals]
        elif pt == "SFP":
            term_groups = inst_core.get("terminal_groups", [])
            term_groups_labeled: List[List[Any]] = []
            for g in term_groups:
                term_groups_labeled.append(
                    [node_id_map.get(int(t), t) for t in g]
                )
            obj[terminal_groups_key] = term_groups_labeled
        elif pt == "KMST":
            k_val = inst_core.get("k")
            if k_val is not None:
                obj[k_key] = k_val

        return json.dumps(obj, ensure_ascii=False, indent=2)

    # CSV format
    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_nodes_key}={num_nodes}")
        lines.append(f"# {num_edges_key}={num_edges}")

        if pt == "STP":
            terminals = inst_core.get("terminals", [])
            term_str = " ".join(
                str(node_id_map.get(int(t), t)) for t in terminals
            )
            lines.append(f"# {terminals_key}={term_str}")
        elif pt == "SFP":
            term_groups = inst_core.get("terminal_groups", [])
            groups_strs = []
            for idx, g in enumerate(term_groups, start=1):
                labels = [str(node_id_map.get(int(t), t)) for t in g]
                groups_strs.append(f"G{idx}: {' '.join(labels)}")
            lines.append(f"# {terminal_groups_key}=" + " | ".join(groups_strs))
        elif pt == "KMST":
            k_val = inst_core.get("k")
            if k_val is not None:
                lines.append(f"# {k_key}={k_val}")

        lines.append(f"{u_key},{v_key},{w_key}")
        for e in labeled_edges:
            lines.append(f"{e[u_key]},{e[v_key]},{e[w_key]}")

        return "\n".join(lines)

    if fmt == "markdown_table":
        if isinstance(nl_style, dict):
            line_t = nl_style.get("line_template")
            header_t = nl_style.get("header")
            footer_t = nl_style.get("footer")

            out: List[str] = []

            base_ctx = {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "problem_type": problem_type,
            }

            # Extra fields per problem (same semantics as NL)
            if pt == "STP":
                terminals = inst_core.get("terminals", [])
                base_ctx["terminals"] = " ".join(str(node_id_map.get(int(t), t)) for t in terminals)
                base_ctx["num_terminals"] = len(terminals)
            elif pt == "SFP":
                term_groups = inst_core.get("terminal_groups", [])
                groups_text = []
                for idx, g in enumerate(term_groups, start=1):
                    labels = [str(node_id_map.get(int(t), t)) for t in g]
                    groups_text.append(f"Group {idx}: {' '.join(labels)}")
                base_ctx["terminal_groups"] = "\n".join(groups_text)
                base_ctx["num_groups"] = len(term_groups)
            elif pt == "KMST":
                base_ctx["k"] = inst_core.get("k")  # keep field for templates that reference it

            if header_t:
                out.append(_format_safe(header_t, **base_ctx))

            edges = inst_core.get("edges", [])
            for e in edges:
                u = e.get("u")
                v = e.get("v")
                w = e.get("w", 1.0)
                ctx = {
                    **base_ctx,
                    "u": node_id_map.get(int(u), u),
                    "v": node_id_map.get(int(v), v),
                    "w": w,
                }
                if line_t:
                    out.append(_format_safe(line_t, **ctx))

            if footer_t:
                out.append(_format_safe(footer_t, **base_ctx))

            return "\n".join(out)


    # NL when fmt == "nl"
    if fmt == "nl" and isinstance(nl_style, dict):
        return tree_render_nl_with_style(
            inst_core=inst_core,
            node_id_map=node_id_map,
            nl_style=nl_style,
            problem_type=problem_type,
        )

    # Fallback: JSON
    return tree_render_input(
        fmt="json",
        inst_core=inst_core,
        node_id_map=node_id_map,
        nl_style=nl_style,
        problem_type=problem_type,
        scenario_hint=scenario_hint,
    )


# ----------------------------------------------------------------------
# 3. Contextualize one instance
# ----------------------------------------------------------------------


def contextualize_instance_trees(
    inst: Dict[str, Any],
    task_name: str,
    contexts: List[Dict[str, Any]],
    precomputed_templates: Dict[int, str],
    nl_styles: Dict[tuple, Dict[str, Any]],
    scenario_hints: Dict[int, Dict[str, Any]],
    instance_idx: int = 0,
    difficulty_tier: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Contextualize a single tree-like graph instance (STP/SFP/KMST).
    """
    rng = random.Random()
    results: List[Dict[str, Any]] = []

    inst_core = inst.get("instance")
    solution = inst.get("solution")
    solution = [[e["u"], e["v"]] for e in solution]
    obj = inst.get("obj")
    problem_type = inst.get("problem_type", task_name).upper()

    if inst_core is None:
        return results

    num_nodes = inst_core.get("num_nodes")
    if num_nodes is None:
        return results

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

    # Build mapping internal_id (1..n) -> human label
    node_id_map = _build_node_label_map(num_nodes, index_base)

    solution_variant = _remap_solution_edges(solution=solution, id_map=node_id_map)

    input_text = tree_render_input(
        fmt=fmt,
        inst_core=inst_core,
        node_id_map=node_id_map,
        nl_style=nl_style,
        problem_type=problem_type,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_tree_instance(
        inst_core=inst_core,
        id_map=node_id_map,
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


# ----------------------------------------------------------------------
# 4. Problem specs (description + output JSON format)
# ----------------------------------------------------------------------


def _get_problem_specs(problem_type: str) -> Dict[str, str]:
    pt = problem_type.upper()

    if pt == "STP":
        desc = (
            "Steiner Tree Problem (STP): You are given a connected weighted graph and a set of terminal nodes. "
            "The goal is to select a subset of edges that forms a connected subgraph spanning all terminals, "
            "with minimum total edge weight. Non-terminal nodes may be used as Steiner nodes to reduce cost."
        )
        baseline_template = (
            "You are given a Steiner Tree Problem (STP).\n\n"
            "Problem description:\n"
            "- The input describes a connected, weighted, undirected graph.\n"
            "- A subset of nodes are terminals; all terminals must be connected in the final solution.\n"
            "- You may include non-terminal nodes as Steiner nodes if it reduces total cost.\n\n"
            "Decision:\n"
            "- Choose a set of undirected edges from the input graph.\n\n"
            "Feasibility constraints:\n"
            "- The chosen edges must form a connected subgraph that spans all terminals.\n"
            "- Every chosen edge must appear in the input (same endpoints).\n\n"
            "Objective:\n"
            "- Minimize the total weight (sum of edge weights) of the chosen edges.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Reply using exactly the following JSON structure (no extra keys, no explanations):\n"
            "```json\n"
            "{\n"
            '  "solution": [[u1, v1], [u2, v2], ...]\n'
            "}\n"
            "```\n"
            "Notes:\n"
            "- Each [u, v] denotes an undirected edge. Use the exact node identifiers from the input.\n"
            "- Do not add new nodes/edges. Do not include explanations.\n"
        )

    elif pt == "SFP":

        desc = (
            "Steiner Forest Problem (SFP): You are given a connected weighted graph and multiple groups of terminals. "
            "Each group of terminals must be connected in the final forest, "
            "and importantly, different groups do not need to be connected to each other."
            "The objective is to choose a minimum-cost set of edges that connects all terminals within "
            "each group."
        )

        baseline_template = (
            "You are given a Steiner Forest Problem (SFP).\n\n"
            "Problem description:\n"
            "- The input describes a connected, weighted, undirected graph.\n"
            "- Terminals are divided into multiple groups.\n"
            "- For each group, all terminals in that group must be connected by the chosen edges.\n"
            "- Different groups do NOT need to be connected to each other.\n\n"
            "Decision:\n"
            "- Choose a set of undirected edges from the input graph.\n\n"
            "Feasibility constraints:\n"
            "- For every terminal group, the chosen edges must connect all terminals within that group.\n"
            "- Every chosen edge must appear in the input (same endpoints).\n\n"
            "Objective:\n"
            "- Minimize the total weight (sum of edge weights) of the chosen edges.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Reply using exactly the following JSON structure (no extra keys, no explanations):\n"
            "```json\n"
            "{\n"
            '  "solution": [[u1, v1], [u2, v2], ...]\n'
            "}\n"
            "```\n"
            "Notes:\n"
            "- Each [u, v] denotes an undirected edge. Use the exact node identifiers from the input.\n"
            "- Do not add new nodes/edges. Do not include explanations.\n"
        )

    else:  # KMST
        desc = (
            "k-Minimum Spanning Tree Problem (k-MST): You are given a connected weighted graph and an integer k. "
            "The goal is to choose a tree (connected acyclic subgraph) that spans exactly k nodes and has minimum total edge weight."
        )
        baseline_template = (
            "You are given a k-Minimum Spanning Tree problem (k-MST).\n\n"
            "Problem description:\n"
            "- The input describes a connected, weighted, undirected graph.\n"
            "- The input also specifies a target number of distinct nodes that the tree must span.\n\n"
            "Decision:\n"
            "- Choose a set of undirected edges from the input graph.\n\n"
            "Feasibility constraints:\n"
            "- The chosen edges must form a connected, cycle-free subgraph (a tree).\n"
            "- The tree must span exactly the required number of DISTINCT nodes.\n"
            "- Every chosen edge must appear in the input (same endpoints).\n\n"
            "Objective:\n"
            "- Minimize the total weight (sum of edge weights) of the chosen edges.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Reply using exactly the following JSON structure (no extra keys, no explanations):\n"
            "```json\n"
            "{\n"
            '  "solution": [[u1, v1], [u2, v2], ...]\n'
            "}\n"
            "```\n"
            "Notes:\n"
            "- Each [u, v] denotes an undirected edge. Use the exact node identifiers from the input.\n"
            "- Ensure the solution is a tree and spans exactly the required number of nodes.\n"
            "- Do not include explanations.\n"
        )

    output_format = (
        "```json\n"
        "{\n"
        '  "solution": [[u1, v1], [u2, v2], ...]\n'
        "}\n"
        "```\n"
    )

    return {
        "task_description": desc,
        "baseline_template": baseline_template,
        "output_format": output_format,
    }


# ----------------------------------------------------------------------
# 5. Hints for NL templates (field names for STP/SFP/KMST)
# ----------------------------------------------------------------------


def _get_hint(problem_type: str) -> Dict[str, Any]:
    """
    Generate NL-style hint for templating according to problem type.
    Ensures each problem has appropriate global_fields, item_fields,
    and allowed_placeholders without leaking irrelevant placeholders.
    """
    pt = problem_type.upper()

    if pt == "STP":
        global_fields = [
            {"name": "num_nodes", "description": "total number of nodes in the graph"},
            {"name": "num_edges", "description": "total number of edges in the graph"},
            {"name": "terminals", "description": "list of terminal nodes that must be connected"},
        ]
        item_fields = [
            {"name": "u", "description": "one endpoint node of an edge"},
            {"name": "v", "description": "the other endpoint node of an edge"},
            {"name": "w", "description": "weight/cost of the edge between u and v"},
        ]
        allowed = ["num_nodes", "num_edges", "terminals", "u", "v", "w"]

    elif pt == "SFP":
        global_fields = [
            {"name": "num_nodes", "description": "total number of nodes in the graph"},
            {"name": "num_edges", "description": "total number of edges in the graph"},
            {
                "name": "terminal_groups",
                "description": "groups of terminals where each group must be internally connected",
            },
        ]
        item_fields = [
            {"name": "u", "description": "one endpoint node of an edge"},
            {"name": "v", "description": "the other endpoint node of an edge"},
            {"name": "w", "description": "weight/cost of the edge between u and v"},
        ]
        allowed = ["num_nodes", "num_edges", "terminal_groups", "u", "v", "w"]

    else:  # KMST
        global_fields = [
            {"name": "num_nodes", "description": "total number of nodes in the graph"},
            {"name": "num_edges", "description": "total number of edges in the graph"},
            {"name": "k", "description": "target number of nodes that the tree must span"},
        ]
        item_fields = [
            {"name": "u", "description": "one endpoint node of an edge"},
            {"name": "v", "description": "the other endpoint node of an edge"},
            {"name": "w", "description": "weight/cost of the edge between u and v"},
        ]
        allowed = ["num_nodes", "num_edges", "k", "u", "v", "w"]

    return {
        "global_fields": global_fields,
        "item_fields": item_fields,
        "allowed_placeholders": allowed,
    }


# ----------------------------------------------------------------------
# 6. CLI entry
# ----------------------------------------------------------------------

def run_tree_step2(
    problem_type: str,
    instance_dir: str,
    output_root_dir: str,
    k_per_call: int = 20,
    n_target: int = 50,
    scale: str | None = None,
    load_cached_contexts: bool = True,
    contexts_audit_path: str | None = None,
) -> str:



    specs = _get_problem_specs(problem_type)
    desc = specs["task_description"]
    baseline_template = specs["baseline_template"]
    output_format = specs["output_format"]

    llm = OpenAILlmService()
    prompter = PromptLoader()
    hint = _get_hint(problem_type)

    generate_dataset_for_task(
        task_name=problem_type,
        task_description=desc,
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_instance_trees,
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

    return output_root_dir



'''
["STP", "SFP", "KMST"]
python -m step2_contextualization.contextualize_trees \
  --problem_type STP \
  --instance_dir ./step1_instance_creation/generated_data/STP \
  --output_root_dir step2_contextualization/dataset/STP

python -m step2_contextualization.contextualize_trees \
  --problem_type SFP \
  --instance_dir ./step1_instance_creation/generated_data/SFP \
  --output_root_dir step2_contextualization/dataset/SFP
python -m step2_contextualization.contextualize_trees \
  --problem_type KMST \
  --instance_dir ./step1_instance_creation/generated_data/KMST \
  --output_root_dir step2_contextualization/dataset/KMST

'''
