from typing import Any, Dict, List, Tuple
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"


def lop_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse LOP (Linear Ordering Problem) instance from `instance_variant`.

    Detects both matrix-style instances (`instance` field) and
    pair-based definitions (`nodes`/`pairs`). Node identifiers can be
    ints or strings; they are remapped to 0..n-1 internally.

    Returns:
      {
        "cost_matrix": n×n matrix of costs/weights,
        "n": problem size,
        "node_id2idx": mapping from original node ids to remapped ints
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for LOP.")

    num_nodes = inst_var.get("num_nodes", None)
    matrix = inst_var.get("instance", None)

    if matrix is not None:
        if not matrix:
            raise ValueError("LOP instance_variant has empty 'instance'.")
        n = len(matrix)
        if num_nodes is not None and int(num_nodes) != n:
            raise ValueError(
                f"Reported num_nodes ({num_nodes}) does not match matrix size ({n})."
            )
        if n == 0:
            raise ValueError("LOP cost_matrix has size 0.")
        for i, row_data in enumerate(matrix):
            if not isinstance(row_data, list) or len(row_data) != n:
                raise ValueError(f"LOP cost_matrix row {i} has incorrect size.")
        mapped_nodes = list(range(n))
        node_id2idx = {i: i for i in range(n)}
        cost_matrix = [
            [float(matrix[i][j]) for j in range(n)] for i in range(n)
        ]
    else:
        nodes = inst_var.get("nodes", [])
        pairs = inst_var.get("pairs", [])
        if not nodes:
            raise ValueError("LOP instance_variant missing 'nodes' definition.")
        if not pairs:
            raise ValueError("LOP instance_variant missing 'pairs' definition.")
        n = len(nodes)
        if num_nodes is not None and int(num_nodes) != n:
            raise ValueError(f"Reported num_nodes ({num_nodes}) does not match number of nodes ({n}).")
        if len(set(nodes)) != n:
            raise ValueError("LOP instance 'nodes' contains duplicate identifiers.")
        node_id2idx = {node: idx for idx, node in enumerate(nodes)}
        cost_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for pair in pairs:
            u = pair.get("from_id")
            v = pair.get("to_id")
            w = pair.get("weight")
            if u not in node_id2idx or v not in node_id2idx:
                raise ValueError(f"LOP pair references unknown node id(s): {u}, {v}.")
            cost_matrix[node_id2idx[u]][node_id2idx[v]] = float(w)

    return {
        "cost_matrix": cost_matrix,
        "n": n,
        "node_id2idx": node_id2idx,
    }


def _resolve_node_index(raw_node: Any, node_id2idx: Dict[Any, int]) -> Any:
    if node_id2idx:
        return node_id2idx.get(raw_node)
    if isinstance(raw_node, int):
        return raw_node
    return None


def lop_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check LOP feasibility + unified violation pattern.

    Expected solution format:
      solution = [8, 6, 5, 0, 1, 7, 2, 3, 4]  # permutation representing ordering

    Patterns for LOP (per taxonomy Global_4):
      - FormatError: invalid format/types, invalid indices
      - Global4: assignment violations (not a valid permutation)

    Constraints:
      1) solution is a list of n integers.
      2) solution must be a valid permutation of [0, 1, ..., n-1].
      3) each element appears exactly once in the ordering (Global4).
    """
    cost_matrix: List[List[float]] = instance["cost_matrix"]
    n: int = instance["n"]
    node_id2idx: Dict[Any, int] = instance.get("node_id2idx", {})

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list representing the ordering."

    if len(solution) != n:
        return False, PATTERN_GLOBAL1, (
            f"Solution must have exactly {n} elements, but has {len(solution)}."
        )

    seen_elements = set()
    for pos, element in enumerate(solution):
        mapped_idx = _resolve_node_index(element, node_id2idx)
        if mapped_idx is None:
            return False, PATTERN_FORMAT, (
                f"Position {pos} has unknown node id {element!r}."
            )
        if not isinstance(mapped_idx, int):
            return False, PATTERN_FORMAT, (
                f"Position {pos} has non-integer element: {element!r}."
            )
        if mapped_idx < 0 or mapped_idx >= n:
            return False, PATTERN_FORMAT, (
                f"Position {pos} has invalid element {mapped_idx} (valid range: 0-{n-1})."
            )
        if mapped_idx in seen_elements:
            return False, PATTERN_GLOBAL1, (
                f"Position {pos} has element {mapped_idx}, "
                f"but element {mapped_idx} already appears earlier in the ordering."
            )
        seen_elements.add(mapped_idx)

    missing_elements = set(range(n)) - seen_elements
    if missing_elements:
        return False, PATTERN_GLOBAL1, (
            f"Some elements are missing from the ordering: {sorted(missing_elements)}."
        )

    return True, PATTERN_OK, "Feasible LOP solution."


def lop_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    LOP objective: maximize the sum of costs for all pairs (i, j) where i < j in the ordering.

    Objective = sum(cost_matrix[solution[i]][solution[j]] for all i < j)
    """
    if not isinstance(solution, list):
        return float("inf")

    cost_matrix: List[List[float]] = instance["cost_matrix"]
    n: int = instance["n"]
    node_id2idx: Dict[Any, int] = instance.get("node_id2idx", {})

    if len(solution) != n:
        return float("inf")

    mapped_solution = []
    for pos, element in enumerate(solution):
        mapped_idx = _resolve_node_index(element, node_id2idx)
        if mapped_idx is None or not isinstance(mapped_idx, int):
            return float("inf")
        if mapped_idx < 0 or mapped_idx >= n:
            return float("inf")
        mapped_solution.append(mapped_idx)

    total_cost = 0.0
    for i in range(n):
        elem_i = mapped_solution[i]
        for j in range(i + 1, n):
            elem_j = mapped_solution[j]
            total_cost += cost_matrix[elem_i][elem_j]

    return float(total_cost)

