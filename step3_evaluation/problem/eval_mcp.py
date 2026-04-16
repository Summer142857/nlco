from typing import Any, Dict, List, Set, Tuple, FrozenSet
import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL8 = "Global8"  # local graph labeling / adjacency constraints


def mcp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse MCP instance from `instance_variant` in label space.

    Expected instance_variant format:
      {
        "num_nodes": 25,
        "num_edges": 65,
        "edges": [{"u": 19, "v": 1}, ...]
      }

    We DO NOT assume nodes are 0..num_nodes-1.
    Instead, we infer node labels directly from the edge list.

    Returns:
      {
        "nodes": set of node labels (Any),
        "edges": set of frozenset({u, v}) for undirected adjacency checks
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for MCP.")

    edges_raw = inst_var.get("edges", [])
    if not isinstance(edges_raw, list):
        raise ValueError("MCP instance_variant['edges'] must be a list.")

    nodes: Set[Any] = set()
    edges: Set[FrozenSet[Any]] = set()

    for e in edges_raw:
        u = e.get("u", None)
        v = e.get("v", None)
        if u is None or v is None:
            continue
        if u == v:
            continue
        nodes.add(u)
        nodes.add(v)
        edges.add(frozenset({u, v}))


    extra_nodes = inst_var.get("nodes", None)
    if isinstance(extra_nodes, list):
        for n in extra_nodes:
            if isinstance(n, dict) and "id" in n:
                nodes.add(n["id"])
            else:
                nodes.add(n)

    return {"nodes": nodes, "edges": edges}



def mcp_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check MCP (Maximum Clique) feasibility + unified violation pattern.

    Patterns (aligned with taxonomy: Global_8):
      - FormatError: wrong type / unknown labels / malformed "set-like" output (duplicates)
      - Global8: clique adjacency violation (a chosen pair is not an edge)
    """
    nodes: Set[Any] = instance["nodes"]
    edges: Set[FrozenSet[Any]] = instance["edges"]

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of node identifiers."

    # Allow empty clique (objective = 0) as feasible
    if len(solution) == 0:
        return True, PATTERN_OK, "Feasible (empty) clique."

    # Duplicates => malformed set-like selection
    if len(solution) != len(set(solution)):
        return False, PATTERN_FORMAT, "Some nodes are repeated in the solution; clique must not repeat nodes."

    chosen = set(solution)

    # Unknown labels
    unknown = chosen - nodes
    if unknown:
        return False, PATTERN_FORMAT, f"Solution contains unknown node labels: {unknown!r}"

    # ---- Global8: clique constraint (all pairs must be edges) ----
    sol_list = list(solution)
    n = len(sol_list)
    for i in range(n):
        for j in range(i + 1, n):
            u, v = sol_list[i], sol_list[j]
            if frozenset({u, v}) not in edges:
                return False, PATTERN_GLOBAL8, f"Not a clique: missing edge between {u!r} and {v!r}."

    return True, PATTERN_OK, "Feasible MCP clique."


def mcp_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    MCP objective: maximize clique size.
    Count unique nodes to be robust to accidental duplicates.
    """
    if not isinstance(solution, list):
        return float("-inf")
    return float(len(set(solution)))
