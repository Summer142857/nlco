from typing import Any, Dict, List, Set, FrozenSet, Tuple
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL8 = "Global8"  # local graph labeling (edge coverage constraints)


# -------- MVC-specific logic --------

def mvc_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse MVC instance from `instance_variant`.

    Expected instance_variant format (already labeled), e.g.:
      {
        "num_nodes": 20,
        "num_edges": 36,
        "edges": [{"u": 16, "v": 6}, ...]
      }

    We work directly in label space (int or str).
    Return:
      {
        "nodes": set(labels),
        "edges": set(frozenset({u,v}))
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for MVC.")

    edges_raw = inst_var.get("edges", [])

    nodes: Set[Any] = set()
    edges: Set[FrozenSet[Any]] = set()

    for e in edges_raw:
        u = e["u"]
        v = e["v"]
        if u == v:
            # self-loop can be ignored; if kept, vertex cover would force selecting u
            continue
        nodes.add(u)
        nodes.add(v)
        edges.add(frozenset({u, v}))  # undirected

    return {
        "nodes": nodes,
        "edges": edges,
    }




def mvc_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check MVC feasibility + unified violation pattern.

    Patterns for MVC (per taxonomy Global_8):
      - FormatError: wrong type / unknown labels
      - Global8: edge not covered (local edge constraint violated)

    Note: duplicates in output are ignored (treated as a set).
    """
    nodes: Set[Any] = instance["nodes"]
    edges: Set[FrozenSet[Any]] = instance["edges"]

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of node identifiers."

    chosen = set(solution)

    # ---- FormatError: unknown labels ----
    unknown = chosen - nodes
    if unknown:
        return False, PATTERN_FORMAT, f"Solution contains unknown node labels: {unknown!r}"

    # ---- Global8: edge coverage ----
    uncovered = []
    for e in edges:
        if not (e & chosen):  # no endpoint chosen
            uncovered.append(tuple(e))
            if len(uncovered) >= 5:
                break

    if uncovered:
        return False, PATTERN_GLOBAL8, f"Uncovered edges exist (showing up to 5): {uncovered!r}"

    return True, PATTERN_OK, "Feasible MVC solution."


def mvc_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    MVC objective: minimize the number of selected vertices.
    Return |set(solution)| as float.
    """
    if not isinstance(solution, list):
        return float("inf")
    return float(len(set(solution)))
