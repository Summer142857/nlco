from typing import Any, Dict, List, Set, FrozenSet, Tuple
import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL3 = "Global3"  # coverage / hitting (domination)


def mds_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse MDS instance from `instance_variant`.

    Expected instance_variant format:
      {
        "num_nodes": 9,
        "num_edges": 9,
        "edges": [{"u": 3, "v": 0}, ...]
      }

    We work in label space (int or str). We infer nodes from edges.
    If num_nodes is provided AND labels look like small ints, we also
    attempt to include isolated nodes by assuming the node universe is 0..num_nodes-1.
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for MDS.")

    edges_raw = inst_var.get("edges", [])
    num_nodes = inst_var.get("num_nodes", None)

    nodes: Set[Any] = set()
    edges: Set[FrozenSet[Any]] = set()

    for e in edges_raw:
        u = e["u"]
        v = e["v"]
        if u == v:
            continue
        nodes.add(u)
        nodes.add(v)
        edges.add(frozenset({u, v}))

    # Optional: include isolated nodes if labels look like 0..num_nodes-1 ints
    if isinstance(num_nodes, int) and num_nodes > 0:
        # if all current nodes are ints and within [0, num_nodes-1], assume that universe
        if all(isinstance(x, int) for x in nodes) and all(0 <= x < num_nodes for x in nodes):
            nodes |= set(range(num_nodes))

    # Build adjacency map
    adj: Dict[Any, Set[Any]] = {v: set() for v in nodes}
    for e in edges:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)

    return {
        "nodes": nodes,  # set of labels
        "edges": edges,  # undirected edges
        "adj": adj,      # adjacency dict
    }



def mds_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check MDS feasibility + unified violation pattern.

    Patterns for MDS (per taxonomy Global_3):
      - FormatError: invalid type / unknown labels / duplicates
      - Global3: coverage/domination violation (some node not dominated, or empty set when nodes exist)
    """
    nodes: Set[Any] = instance["nodes"]
    adj: Dict[Any, Set[Any]] = instance["adj"]

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of node identifiers."

    # Empty dominating set is feasible only if the graph has no nodes
    if len(solution) == 0:
        if len(nodes) == 0:
            return True, PATTERN_OK, "Feasible (empty graph, empty dominating set)."
        return False, PATTERN_GLOBAL3, "Solution set is empty; non-empty graph cannot be dominated by an empty set."

    # Duplicates => malformed set-like selection
    if len(solution) != len(set(solution)):
        return False, PATTERN_FORMAT, "Some nodes are repeated in the solution; dominating set must not repeat nodes."

    chosen: Set[Any] = set(solution)

    unknown = chosen - nodes
    if unknown:
        return False, PATTERN_FORMAT, f"Solution contains unknown node labels: {unknown!r}"

    # ---- Global3: domination / coverage ----
    for v in nodes:
        if v in chosen:
            continue
        nbrs = adj.get(v, set())
        if not (nbrs & chosen):
            return False, PATTERN_GLOBAL3, f"Node {v!r} is not dominated by the chosen set."

    return True, PATTERN_OK, "Feasible dominating set."


def mds_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    MDS objective (to MINIMIZE): number of selected nodes (cameras).
    Count unique nodes to be robust to duplicates.
    """
    if not isinstance(solution, list):
        return float("inf")
    return float(len(set(solution)))
