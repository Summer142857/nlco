import json
import random
from typing import Dict, Any, List

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader


FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]



def _remap_solution_graph(
    solution,
    problem_type: str,
    node_labels: List[Any],
):

    if not solution:
        return solution

    node_map = {i + 1: node_labels[i] for i in range(len(node_labels))}
    pt = problem_type.upper()

    # ---------- MIS / MVC / MCP / MDS ----------
    if pt in {"MIS", "MVC", "MCP", "MDS"}:
        return [node_map[int(v)] for v in solution]

    # ---------- MAXCUT ----------
    if pt == "MAXCUT":
        remapped = []
        for group in solution:
            remapped.append([node_map[int(v)] for v in group])
        return remapped

    # ---------- GCP ----------
    if pt == "GCP":
        remapped = []
        for color_group in solution:
            remapped.append([node_map[int(v)] for v in color_group])
        return remapped

    return solution




def graph_render_nl_with_style(
    inst_core: Dict[str, Any],
    node_labels: List[Any],
    nl_style: Dict[str, Any],
    problem_type: str,
) -> str:


    line_t = nl_style.get("line_template")
    header_t = nl_style.get("header")
    footer_t = nl_style.get("footer")

    out: List[str] = []

    num_nodes = inst_core.get("num_nodes")
    num_edges = inst_core.get("num_edges")
    edges = inst_core.get("edges", [])

    node_map = {i + 1: node_labels[i] for i in range(len(node_labels))}

    base_ctx = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "problem_type": problem_type,
    }

    # header
    if header_t:
        out.append(_format_safe(header_t, **base_ctx))

    for e in edges:
        u_raw = e["u"]
        v_raw = e["v"]
        u_lbl = node_map.get(int(u_raw), u_raw)
        v_lbl = node_map.get(int(v_raw), v_raw)

        ctx = {
            **base_ctx,
            "u": u_lbl,
            "v": v_lbl,
        }
        if line_t:
            out.append(_format_safe(line_t, **ctx))
        else:
            out.append(f"({u_lbl}, {v_lbl})")

    # footer
    if footer_t:
        out.append(_format_safe(footer_t, **base_ctx))

    return "\n".join(out)

def _humanize_label(raw: str) -> str:

    if not isinstance(raw, str):
        return str(raw)

    label = raw

    if label.startswith("total_"):
        label = label[len("total_"):]

    label = label.replace("_", " ")

    return label


def graph_render_input(
    fmt: str,
    inst_core: Dict[str, Any],
    node_labels: List[Any],
    nl_style: Dict[str, Any] | None,
    problem_type: str,
    scenario_hint: Dict[str, Any] | None = None,
) -> str:


    num_nodes = inst_core.get("num_nodes")
    num_edges = inst_core.get("num_edges")
    edges = inst_core.get("edges", [])

    num_nodes_key = "num_nodes"
    num_edges_key = "num_edges"
    u_key = "u"
    v_key = "v"

    if scenario_hint and isinstance(scenario_hint, dict):
        # global_fields: num_nodes, num_edges
        for field in scenario_hint.get("global_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "num_nodes":
                num_nodes_key = new
            elif orig == "num_edges":
                num_edges_key = new

        # item_fields: u, v
        for field in scenario_hint.get("item_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "u":
                u_key = new
            elif orig == "v":
                v_key = new

    node_map = {i + 1: node_labels[i] for i in range(len(node_labels))}

    labeled_edges = []
    for e in edges:
        u_raw = e["u"]
        v_raw = e["v"]
        u_lbl = node_map.get(int(u_raw), u_raw)
        v_lbl = node_map.get(int(v_raw), v_raw)
        labeled_edges.append({u_key: u_lbl, v_key: v_lbl})

    # ============= JSON =============
    if fmt == "json":
        obj: Dict[str, Any] = {
            num_nodes_key: num_nodes,
            num_edges_key: num_edges,
            "edges": labeled_edges,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    # ============= CSV =============
    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_nodes_key}={num_nodes}")
        lines.append(f"# {num_edges_key}={num_edges}")
        lines.append(f"{u_key},{v_key}")
        for e in labeled_edges:
            lines.append(f"{e[u_key]},{e[v_key]}")
        return "\n".join(lines)

    # ============= Markdown table =============
    if fmt == "markdown_table":
        lines: List[str] = []

        gf = {f["name"]: f for f in scenario_hint.get("global_fields", [])}

        num_nodes_label = gf.get("num_nodes", {}).get("new_name", num_nodes_key)
        num_edges_label = gf.get("num_edges", {}).get("new_name", num_edges_key)

        num_nodes_phrase = _humanize_label(num_nodes_label)
        num_edges_phrase = _humanize_label(num_edges_label)

        intro = (
                f"There are {num_nodes} {num_nodes_phrase} in total "
                f"and {num_edges} {num_edges_phrase}."
        )
        lines.append(intro)

        lines.append("")
        lines.append(f"| {u_key} | {v_key} |")
        lines.append("|---|---|")
        for e in labeled_edges:
            lines.append(f"| {e[u_key]} | {e[v_key]} |")
        return "\n".join(lines)

    # ============= NL =============
    if fmt == "nl" and isinstance(nl_style, dict):
        return graph_render_nl_with_style(
            inst_core=inst_core,
            node_labels=node_labels,
            nl_style=nl_style,
            problem_type=problem_type,
        )




def contextualize_instance_graph(
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

    num_nodes = inst_core.get("num_nodes")
    if not num_nodes:
        return results

    K = len(contexts)
    if K == 0:
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

    node_labels = make_labels(num_nodes, index_base)

    input_text = graph_render_input(
        fmt=fmt,
        inst_core=inst_core,
        node_labels=node_labels,
        nl_style=nl_style,
        problem_type=problem_type,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    node_map = {i + 1: node_labels[i] for i in range(len(node_labels))}
    labeled_edges = []
    for e in inst_core.get("edges", []):
        u_raw = e["u"]
        v_raw = e["v"]
        u_lbl = node_map.get(int(u_raw), u_raw)
        v_lbl = node_map.get(int(v_raw), v_raw)
        labeled_edges.append({"u": u_lbl, "v": v_lbl})

    instance_variant = {
        "num_nodes": inst_core.get("num_nodes"),
        "num_edges": inst_core.get("num_edges"),
        "edges": labeled_edges,
    }

    solution_variant = _remap_solution_graph(
        solution=solution,
        problem_type=problem_type,
        node_labels=node_labels,
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



from typing import Dict

def _get_problem_specs(problem_type: str) -> Dict[str, str]:
    pt = problem_type.upper()

    # MIS: Maximum Independent Set
    if pt == "MIS":
        desc = (
            "Maximum Independent Set (MIS): Given an undirected graph, "
            "select a largest possible set of vertices such that no two selected vertices "
            "are adjacent (no edge connects any pair of chosen vertices)."
        )
        baseline_template = (
            "You are given an instance of the Maximum Independent Set (MIS) problem.\n"
            "The input describes an undirected graph using a list of edges.\n\n"
            "Your goal is to select as many vertices as possible so that no two chosen vertices "
            "are connected by an edge.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  "solution": ["vertex_id", "vertex_id", ...]\n'
            "}\n"
            "```\n"
            "Here \"solution\" is the list of vertex IDs you choose as the independent set. "
            "Your output must use exactly the same vertex identifiers that appear in the input "
            "(whatever numbers or labels are used there). Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": ["vertex_id", "vertex_id", ...]\n'
            "}\n"
            "```\n"
        )

    # MVC: Minimum Vertex Cover
    elif pt == "MVC":
        desc = (
            "Minimum Vertex Cover (MVC): Given an undirected graph, "
            "select a vertex cover with the minimum possible number of vertices, "
            "such that every edge has at least one endpoint in the selected set."
        )

        baseline_template = (
            "You are given an instance of the Minimum Vertex Cover (MVC) problem.\n"
            "The input describes an undirected graph using a list of edges.\n\n"
            "**Goal:** Your task is to find a vertex cover of **minimum size**.\n"
            "That is, select as **few vertices as possible** so that **every edge** "
            "in the graph has **at least one endpoint** in the selected set.\n\n"
            "Formally, among all vertex sets that cover all edges, you must choose one "
            "with the **smallest possible cardinality**.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  "solution": ["vertex_id", "vertex_id", ...]\n'
            "}\n"
            "```\n"
            "Here \"solution\" is the list of vertex IDs you choose as the vertex cover.\n"
            "- Every edge must have at least one endpoint in this list.\n"
            "- The list should contain as **few vertices as possible**.\n\n"
            "Your output must use exactly the same vertex identifiers that appear in the input "
            "(whatever numbers or labels are used there).\n"
            "Do **not** include explanations, comments, or extra keys."
        )

        output_format = (
            "```json\n"
            "{\n"
            '  "solution": ["vertex_id", "vertex_id", ...]\n'
            "}\n"
            "```\n"
        )


    # MCP: Maximum Clique
    elif pt == "MCP":
        desc = (
            "Maximum Clique Problem (MCP): Given an undirected graph, "
            "select a largest possible set of vertices such that every pair of selected vertices "
            "is connected by an edge (they form a clique)."
        )
        baseline_template = (
            "You are given an instance of the Maximum Clique Problem (MCP).\n"
            "The input describes an undirected graph using a list of edges.\n\n"
            "Your goal is to select a largest possible set of vertices such that every pair of "
            "chosen vertices is adjacent (they form a clique).\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  "solution": ["vertex_id", "vertex_id", ...]\n'
            "}\n"
            "```\n"
            "Here \"solution\" is the list of vertex IDs you choose as the clique. "
            "Your output must use exactly the same vertex identifiers that appear in the input. "
            "Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": ["vertex_id", "vertex_id", ...]\n'
            "}\n"
            "```\n"
        )

    # MAXCUT
    elif pt == "MAXCUT":
        desc = (
            "Maximum Cut (MaxCut): Given an undirected graph, partition the vertices into two groups "
            "so that the number of edges crossing between the groups is as large as possible."
        )
        baseline_template = (
            "You are given an instance of the Maximum Cut (MaxCut) problem.\n"
            "The input describes an undirected graph using a list of edges.\n\n"
            "Your goal is to partition the vertices into two disjoint groups so that the number of edges "
            "with endpoints in different groups is maximized.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  "solution": [\n'
            '    ["vertex_id", "vertex_id", ...],\n'
            '    ["vertex_id", "vertex_id", ...]\n'
            "  ]\n"
            "}\n"
            "```\n"
            "Here \"solution\" is a list of two inner lists: the first inner list contains all vertex IDs "
            "assigned to group 1, and the second inner list contains all vertex IDs assigned to group 2. "
            "Every vertex that appears in the input must appear **in exactly one** of these two inner lists. "
            "Use exactly the same vertex identifiers as in the input. Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [\n'
            '    ["vertex_id", "vertex_id", ...],\n'
            '    ["vertex_id", "vertex_id", ...]\n'
            "  ]\n"
            "}\n"
            "```\n"
        )

    # GCP: Graph Coloring
    elif pt == "GCP":
        desc = (
            "Graph Coloring Problem (GCP): Given an undirected graph, assign a color (integer label) "
            "to each vertex so that no edge has both endpoints with the same color. "
            "The objective is to use as few colors as possible."
        )
        baseline_template = (
            "You are given an instance of the Graph Coloring Problem (GCP).\n"
            "The input describes an undirected graph using a list of edges.\n\n"
            "Your goal is to assign colors to vertices so that adjacent vertices never share the "
            "same color, and the total number of distinct colors is as small as possible.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  "solution": [\n'
            '    ["vertex_id", "vertex_id", ...],\n'
            '    ["vertex_id", "vertex_id", ...],\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "```\n"
            "Here \"solution\" is a list of color classes: each inner list contains the vertex IDs that "
            "share the same color. Different inner lists represent different colors. Every vertex that "
            "appears in the input must appear in exactly one of these inner lists. Use exactly the same "
            "vertex identifiers as in the input. Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [\n'
            '    ["vertex_id", "vertex_id", ...],\n'
            '    ["vertex_id", "vertex_id", ...],\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "```\n"
        )

    # MDS: Minimum Dominating Set
    else:  # MDS
        desc = (
            "Minimum Dominating Set (MDS): Given an undirected graph, select a smallest possible set of vertices "
            "such that every vertex in the graph is either in the selected set or adjacent to at least one selected vertex."
        )
        baseline_template = (
            "You are given an instance of the Minimum Dominating Set (MDS) problem.\n"
            "The input describes an undirected graph using a list of edges.\n\n"
            "Your goal is to select a smallest possible set of vertices such that every vertex is either "
            "in the chosen set or has at least one neighbor in the chosen set.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  "solution": ["vertex_id", "vertex_id", ...]\n'
            "}\n"
            "```\n"
            "Here \"solution\" is the list of vertex IDs you choose as the dominating set. "
            "Your output must use exactly the same vertex identifiers that appear in the input. "
            "Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": ["vertex_id", "vertex_id", ...]\n'
            "}\n"
            "```\n"
        )

    return {
        "task_description": desc,
        "baseline_template": baseline_template,
        "output_format": output_format,
    }




def _get_hint(problem_type: str) -> Dict[str, Any]:

    global_fields = [
        {
            "name": "num_nodes",
            "description": "total number of vertices in the graph",
        },
        {
            "name": "num_edges",
            "description": "total number of edges in the graph",
        },
    ]
    item_fields = [
        {
            "name": "u",
            "description": "one endpoint of an undirected edge (vertex identifier)",
        },
        {
            "name": "v",
            "description": "the other endpoint of the edge (vertex identifier)",
        },
    ]
    allowed = ["num_nodes", "num_edges", "u", "v"]

    return {
        "global_fields": global_fields,
        "item_fields": item_fields,
        "allowed_placeholders": allowed,
    }


def run_graph_step2(
    problem_type: str,
    instance_dir: str,
    output_root_dir: str,
    k_per_call: int = 20,
    n_target: int = 50,
    scale: str | None = None,
    load_cached_contexts: bool = True,
    contexts_audit_path: str | None = None,
):
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
        contextualize_fn=contextualize_instance_graph,
        llm=llm,
        prompter=prompter,
        hint=hint,
        baseline_template=baseline_template,
        output_format=output_format,
        k_per_call=k_per_call,
        n_target=n_target,
        scale=scale,
        load_cached_contexts=load_cached_contexts,
        contexts_audit_path=contexts_audit_path,
    )
    return output_root_dir


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Contextualize graph instances (MIS, MVC, MCP, MAXCUT, GCP, MDS) into NL/JSON/CSV/Markdown."
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="MIS",
        choices=["MIS", "MVC", "MCP", "MAXCUT", "GCP", "MDS"],
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        default=None,
        help="Directory containing raw JSON instances (default: ./generated_data/<problem_type>).",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=None,
        help="Root directory to save contextualized data (default: step2_contextualization/dataset/<problem_type>).",
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
        help="Total number of contextualized examples to generate (for context generation).",
    )

    args = parser.parse_args()
    problem_type = args.problem_type
    task_name = problem_type

    specs = _get_problem_specs(problem_type)
    desc = specs["task_description"]
    baseline_template = specs["baseline_template"]
    output_format = specs["output_format"]

    if args.instance_dir is not None:
        instance_dir = args.instance_dir
    else:
        instance_dir = os.path.join("generated_data", problem_type)

    if args.output_root_dir is not None:
        output_root_dir = args.output_root_dir
    else:
        output_root_dir = os.path.join("step2_contextualization", "dataset", problem_type)

    llm = OpenAILlmService()
    prompter = PromptLoader()

    hint = _get_hint(problem_type)

    generate_dataset_for_task(
        task_name=task_name,
        task_description=desc,
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_instance_graph,
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

python -m step2_contextualization.contextualize_graphs \
  --problem_type MIS \
  --instance_dir step1_instance_creation/generated_data/MIS \
  --output_root_dir step2_contextualization/dataset/MIS \
  --k_per_call 20 \
  --n_target 50
python -m step2_contextualization.contextualize_graphs \
  --problem_type MVC \
  --instance_dir step1_instance_creation/generated_data/MVC \
  --output_root_dir step2_contextualization/dataset/MVC \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_graphs \
  --problem_type MCP \
  --instance_dir step1_instance_creation/generated_data/MCP \
  --output_root_dir step2_contextualization/dataset/MCP \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_graphs \
  --problem_type MAXCUT \
  --instance_dir step1_instance_creation/generated_data/MAXCUT \
  --output_root_dir step2_contextualization/dataset/MAXCUT \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_graphs \
  --problem_type GCP \
  --instance_dir step1_instance_creation/generated_data/GCP \
  --output_root_dir step2_contextualization/dataset/GCP \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_graphs \
  --problem_type MDS \
  --instance_dir step1_instance_creation/generated_data/MDS \
  --output_root_dir step2_contextualization/dataset/MDS \
  --k_per_call 20 \
  --n_target 50


'''
