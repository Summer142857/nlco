
import asyncio
import argparse
import json
from typing import Any, Dict, List, Set, FrozenSet, Tuple

import pandas as pd


from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL5 = "Global5"  # partitioning (2-way partition)

def maxcut_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse MAXCUT instance from `instance_variant`.

    instance_variant expected format, in *label* space, e.g.:

      {
        "num_nodes": 11,
        "num_edges": 19,
        "edges": [
          {"u": "G", "v": "A"},
          {"u": "G", "v": "B"},
          ...
        ]
      }

    We extract:
      - nodes: set of all vertex labels
      - edges: set of frozenset({u, v}) as undirected edges
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for MAXCUT.")

    edges_raw = inst_var.get("edges", [])
    num_nodes = inst_var.get("num_nodes", 0)

    nodes: Set[Any] = set()
    edges: Set[FrozenSet[Any]] = set()

    for e in edges_raw:
        u = e["u"]
        v = e["v"]
        if u == v:
            # self-loops don't matter for MaxCut, skip
            continue
        nodes.add(u)
        nodes.add(v)
        edges.add(frozenset({u, v}))

    # Infer missing nodes if integer range implies them
    if nodes and all(isinstance(n, int) for n in nodes):
        min_n, max_n = min(nodes), max(nodes)
        # If we have fewer nodes than num_nodes, we might need to fill in isolated ones.
        # Heuristic: if min is 1, assume 1..num_nodes. If min is 0, assume 0..num_nodes-1.
        if len(nodes) < num_nodes:
            if min_n >= 1:
                # Assume 1-based
                expected = set(range(1, num_nodes + 1))
                if nodes.issubset(expected):
                    nodes = expected
            else:
                # Assume 0-based
                expected = set(range(0, num_nodes))
                if nodes.issubset(expected):
                    nodes = expected
    elif not nodes and num_nodes > 0:
        # No edges, but num_nodes given. Assume 0-based integers.
        nodes = set(range(num_nodes))

    return {
        "nodes": nodes,   # set of labels (str/int)
        "edges": edges,   # set of frozenset({u, v})
    }


def _normalize_solution(
    solution: List[Any],
    instance_nodes: Set[Any],
):
    """
    Normalize solution to match instance node labels.
    Handles:
      - Exact match
      - String <-> Integer conversion
      - 0-based <-> 1-based indexing shift
    Returns (normalized_partition, error_message).
    """
    if not isinstance(solution, list) or len(solution) != 2:
        return None, "Solution must be a list of two groups."

    group_0, group_1 = solution
    if not isinstance(group_0, list) or not isinstance(group_1, list):
        return None, "Each group in the solution must be a list of labels."

    sol_nodes_flat = set(group_0) | set(group_1)
    
    # 1. Check exact match
    if sol_nodes_flat == instance_nodes:
        return [set(group_0), set(group_1)], None

    # 2. Check if solution is subset (missing nodes) - handled in feasibility, 
    # but here we want to see if we can map labels first.
    
    # Try converting solution to instance types (e.g. sol ints -> inst strings)
    # or sol strings -> inst ints
    
    # Helper to try mapping
    def try_map(mapper):
        try:
            mapped_0 = {mapper(x) for x in group_0}
            mapped_1 = {mapper(x) for x in group_1}
            return mapped_0, mapped_1
        except:
            return None, None

    # Case A: Instance has strings, Solution has ints
    # e.g. Inst {'1', '2'}, Sol {1, 2}
    if all(isinstance(x, str) for x in instance_nodes):
        # Try str(x)
        m0, m1 = try_map(str)
        if m0 is not None and (m0 | m1).issubset(instance_nodes):
             return [m0, m1], None
    
    # Case B: Instance has ints, Solution has strings
    # e.g. Inst {1, 2}, Sol {'1', '2'}
    if all(isinstance(x, int) for x in instance_nodes):
        # Try int(x)
        m0, m1 = try_map(int)
        if m0 is not None and (m0 | m1).issubset(instance_nodes):
             return [m0, m1], None

    # Case C: Index shift (Ints only)
    if all(isinstance(x, int) for x in instance_nodes) and all(isinstance(x, int) for x in sol_nodes_flat):
        # Try +1 (Sol 0-based -> Inst 1-based)
        m0, m1 = try_map(lambda x: x + 1)
        if m0 is not None and (m0 | m1) == instance_nodes:
            return [m0, m1], None
            
        # Try -1 (Sol 1-based -> Inst 0-based)
        m0, m1 = try_map(lambda x: x - 1)
        if m0 is not None and (m0 | m1) == instance_nodes:
            return [m0, m1], None

    # If we are here, we couldn't find a global transform that makes sets match perfectly.
    # But maybe it's just a partial match (missing nodes)?
    # We return the raw sets and let feasibility check complain about missing/unknown.
    return [set(group_0), set(group_1)], None


def maxcut_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check MAXCUT feasibility in label space + unified violation pattern.

    Feasibility here is purely the existence of a valid 2-way partition of vertices.

    Patterns:
      - FormatError: wrong top-level structure / wrong types / unknown labels
      - Global5: partitioning violation (duplicate vertex / missing vertex)
    """
    nodes: Set[Any] = instance["nodes"]

    # Normalize solution
    norm_res, err = _normalize_solution(solution, nodes)
    if err:
        return False, PATTERN_FORMAT, err
    
    group_0_set, group_1_set = norm_res

    # ---- Global5: each vertex in exactly one group (no duplicates) ----
    intersection = group_0_set & group_1_set
    if intersection:
        return False, PATTERN_GLOBAL5, f"Vertices {list(intersection)!r} appear in both partitions."

    union = group_0_set | group_1_set

    # ---- FormatError: unknown labels ----
    unknown = union - nodes
    if unknown:
        return False, PATTERN_FORMAT, f"Solution contains unknown vertex labels: {list(unknown)!r}"

    # ---- Global5: missing vertices ----
    missing = nodes - union
    if missing:
        return False, PATTERN_GLOBAL5, f"Some vertices are missing from the partition: {list(missing)!r}"

    return True, PATTERN_OK, "Feasible MAXCUT partition."



def maxcut_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    MAXCUT objective: maximize the number of edges crossing the cut.

    We count edges (u, v) such that u and v are assigned to different groups.

    Assumes the solution has already passed `maxcut_check_feasibility`.
    """
    nodes: Set[Any] = instance["nodes"]
    edges: Set[FrozenSet[Any]] = instance["edges"]

    # Normalize solution again (safe because feasibility passed)
    norm_res, err = _normalize_solution(solution, nodes)
    if err or norm_res is None:
        return float("-inf") # Should not happen if feasible
        
    group_0_set, group_1_set = norm_res

    # build assignment map
    assignment: Dict[Any, int] = {}
    for v in group_0_set:
        assignment[v] = 0
    for v in group_1_set:
        assignment[v] = 1

    cut_edges = 0
    for e in edges:
        u, v = tuple(e)
        gu = assignment.get(u, None)
        gv = assignment.get(v, None)
        # If feasibility passed, both gu and gv must be 0 or 1
        if gu is None or gv is None:
            # if somehow missing, just treat as not contributing
            continue
        if gu != gv:
            cut_edges += 1

    return float(cut_edges)
