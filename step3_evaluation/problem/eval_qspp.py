from typing import Any, Dict, List, Tuple
import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # path structure


from typing import Any, Dict, List, Tuple
import pandas as pd
from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # path structure


def qspp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse QSPP instance from `instance_variant` (CURRENT FLAT + SPARSE OBJECTIVE FORMAT).

    instance_variant objective:
      - linear:  [{"var_index": k, "linear_cost": c}, ...]
      - quadratic:[{"var_i": i, "var_j": j, "quadratic_cost": q}, ...]
    """
    inst = parse_json_field(row["instance_variant"])
    if inst is None:
        raise ValueError("Failed to parse 'instance_variant' for QSPP.")

    nodes = inst.get("nodes", [])
    edges = inst.get("edges", [])
    source = inst.get("source", None)
    target = inst.get("target", None)

    if not nodes:
        raise ValueError("QSPP instance has empty 'nodes'.")
    if not edges:
        raise ValueError("QSPP instance has empty 'edges'.")
    if source is None:
        raise ValueError("QSPP instance missing 'source'.")
    if target is None:
        raise ValueError("QSPP instance missing 'target'.")

    # num_vars determined by var_index range
    max_var = max(e.get("var_index", -1) for e in edges)
    if max_var < 0:
        raise ValueError("QSPP edges missing 'var_index'.")
    n_vars = max_var + 1

    objective = inst.get("objective", {})
    constant = float(objective.get("constant", 0.0))

    # ---- linear: sparse list -> dense vector ----
    linear_sparse = objective.get("linear", [])
    if not isinstance(linear_sparse, list) or len(linear_sparse) == 0:
        raise ValueError("QSPP objective missing 'linear' coefficients (sparse list).")

    linear = [0.0] * n_vars
    for item in linear_sparse:
        if not isinstance(item, dict) or "var_index" not in item or "linear_cost" not in item:
            raise ValueError(f"Malformed linear term: {item!r}")
        k = int(item["var_index"])
        c = float(item["linear_cost"])
        if k < 0 or k >= n_vars:
            raise ValueError(f"Linear var_index {k} out of range [0, {n_vars-1}].")
        linear[k] = c

    # ---- quadratic: sparse list -> dense matrix ----
    quad_sparse = objective.get("quadratic", [])
    if not isinstance(quad_sparse, list) or len(quad_sparse) == 0:
        raise ValueError("QSPP objective missing 'quadratic' coefficients (sparse list).")

    quadratic = [[0.0] * n_vars for _ in range(n_vars)]
    for item in quad_sparse:
        if not isinstance(item, dict) or "var_i" not in item or "var_j" not in item or "quadratic_cost" not in item:
            raise ValueError(f"Malformed quadratic term: {item!r}")
        i = int(item["var_i"])
        j = int(item["var_j"])
        q = float(item["quadratic_cost"])
        if i < 0 or i >= n_vars or j < 0 or j >= n_vars:
            raise ValueError(f"Quadratic index ({i},{j}) out of range for n_vars={n_vars}.")
        quadratic[i][j] = q

    return {
        "nodes": list(nodes),
        "edges": list(edges),
        "constant": constant,
        "linear": linear,
        "quadratic": quadratic,
        "source": source,
        "target": target,
    }


def qspp_check_feasibility(
    solution: List[int],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Patterns:
      - FormatError: invalid solution type, unknown nodes, malformed instance edges
      - Global1: path violations (wrong start/end, missing directed edge)
    """
    nodes: List[int] = instance["nodes"]
    edges: List[Dict[str, int]] = instance["edges"]
    source: int = instance["source"]
    target: int = instance["target"]

    node_set = set(nodes)

    # Build directed edge set; malformed edges -> FormatError
    edge_set = set()
    for e in edges:
        if not isinstance(e, dict) or "from" not in e or "to" not in e:
            return False, PATTERN_FORMAT, f"Instance error: malformed edge record {e!r}"
        edge_set.add((e["from"], e["to"]))

    # ---- format ----
    if not isinstance(solution, list) or len(solution) == 0:
        return False, PATTERN_FORMAT, "Solution must be a non-empty list of node indices."

    for v in solution:
        if v not in node_set:
            return False, PATTERN_FORMAT, f"Invalid node index {v} (valid nodes: {sorted(node_set)})."

    # ---- Global1: start/end ----
    if solution[0] != source:
        return False, PATTERN_GLOBAL1, f"Path must start at source node {source}, but starts at {solution[0]}."
    if solution[-1] != target:
        return False, PATTERN_GLOBAL1, f"Path must end at target node {target}, but ends at {solution[-1]}."

    # ---- Global1: directed adjacency ----
    for i in range(len(solution) - 1):
        u, v = solution[i], solution[i + 1]
        if (u, v) not in edge_set:
            return False, PATTERN_GLOBAL1, f"No edge exists from node {u} to node {v} at position {i} in path."

    return True, PATTERN_OK, "Feasible QSPP path."


def qspp_objective_model(
    solution: List[int],
    instance: Dict[str, Any],
) -> float:
    """
    Objective:
      constant
      + sum(linear[var_idx] for edges on path)
      + sum(quadratic[i][j] for i in edge_indices for j in edge_indices)
    """
    if not isinstance(solution, list) or len(solution) < 2:
        return float("inf")

    edges: List[Dict[str, int]] = instance["edges"]
    constant: float = instance["constant"]
    linear: List[float] = instance["linear"]
    quadratic: List[List[float]] = instance["quadratic"]

    # (from,to) -> var_index
    edge_to_var: Dict[Tuple[int, int], int] = {}
    for e in edges:
        if "from" not in e or "to" not in e or "var_index" not in e:
            return float("inf")
        edge_to_var[(e["from"], e["to"])] = int(e["var_index"])

    # node path -> edge var_indices
    edge_indices: List[int] = []
    for u, v in zip(solution[:-1], solution[1:]):
        idx = edge_to_var.get((u, v), None)
        if idx is None:
            return float("inf")
        edge_indices.append(idx)

    # linear
    linear_cost = 0.0
    for idx in edge_indices:
        if idx < 0 or idx >= len(linear):
            return float("inf")
        linear_cost += linear[idx]

    # quadratic (full pair sum, i x j)
    quad_cost = 0.0
    for i in edge_indices:
        if i < 0 or i >= len(quadratic):
            return float("inf")
        row_i = quadratic[i]
        for j in edge_indices:
            if j < 0 or j >= len(row_i):
                return float("inf")
            quad_cost += row_i[j]

    return float(constant + linear_cost + quad_cost)
