import asyncio
import argparse
import json
from typing import Any, Dict, List, Set, FrozenSet, Tuple

import pandas as pd

from step3_evaluation.utils.eval_generic import (
    evaluate_csv_generic_async,
)
from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL8 = "Global8"  # local graph labeling (independence constraint)


# -------- MIS-specific logic --------

def mis_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse MIS instance from `instance_variant`.

    instance_variant is expected to look like:
      {
        "num_nodes": 8,
        "num_edges": 15,
        "edges": [
          {"u": 0, "v": 2}, {"u": "C", "v": "A"}, ...
        ]
      }

    We work directly in the label space (int or str).
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant'.")

    edges_raw = inst_var["edges"]

    nodes: Set[Any] = set()
    edges: Set[FrozenSet[Any]] = set()

    for e in edges_raw:
        u = e["u"]
        v = e["v"]
        if u == v:
            continue
        nodes.add(u)
        nodes.add(v)
        edges.add(frozenset({u, v}))  # undirected edge

    return {
        "nodes": nodes,   # set of labels (str or int)
        "edges": edges,   # set of frozenset({u, v})
    }



def mis_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check MIS feasibility + unified violation pattern.

    Patterns for MIS (per taxonomy Global_8):
      - FormatError: wrong type / unknown labels / malformed selection
      - Global8: violates independence constraint (some edge has both endpoints selected)

    Note: The empty set is a valid independent set (objective=0). This function treats it as feasible.
    """
    nodes: Set[Any] = instance["nodes"]
    edges: Set[FrozenSet[Any]] = instance["edges"]

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of labels."

    # Standard feasibility: empty independent set is feasible
    if len(solution) == 0:
        return True, PATTERN_OK, "Feasible (empty) independent set."

    # Optional: treat duplicates as format error (set-like solution expected)
    if len(solution) != len(set(solution)):
        return False, PATTERN_FORMAT, "Some labels are repeated; independent set should not repeat nodes."

    chosen = set(solution)

    # Unknown labels
    unknown = chosen - nodes
    if unknown:
        return False, PATTERN_FORMAT, f"Solution contains unknown labels: {unknown!r}"

    # ---- Global8: independence constraint ----
    for e in edges:
        if e.issubset(chosen):
            return False, PATTERN_GLOBAL8, f"Edge {set(e)!r} has both endpoints selected."

    return True, PATTERN_OK, "Feasible MIS solution."


def mis_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    MIS objective: maximize the number of selected nodes.
    We count unique labels to be robust to duplicates.
    """
    return float(len(set(solution)))



