from typing import Any, Dict, List, Tuple
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL4 = "Global4"  # assignment constraints (each index used exactly once)


def _build_index_map(ids: List[Any], n: int) -> Dict[Any, int]:
    if len(ids) != n:
        raise ValueError(f"'ids' length {len(ids)} does not match n={n}.")
    mapping: Dict[Any, int] = {}
    for idx, identifier in enumerate(ids):
        if identifier in mapping:
            raise ValueError(f"Duplicate identifier '{identifier}' in 'ids'.")
        mapping[identifier] = idx
    return mapping


def _normalize_value(
    value: Any, *, field: str, node_map: Dict[Any, int], n: int
) -> int:
    if node_map:
        if value not in node_map:
            raise ValueError(f"{field} value {value!r} not present in identifiers.")
        idx = node_map[value]
    elif isinstance(value, int):
        idx = value
    elif isinstance(value, float) and value.is_integer():
        idx = int(value)
    elif isinstance(value, str) and value.isdigit():
        idx = int(value)
    else:
        raise ValueError(f"{field} value {value!r} is not integer-like.")
    if idx < 0 or idx >= n:
        raise ValueError(f"{field} index {idx} out of bounds 0-{n-1}.")
    return idx


def ap3_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse AP3 instances (matrix or identifier/cost list) and normalize node ids.
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for AP3.")

    data = inst_var
    if isinstance(data.get("instance"), dict):
        data = data["instance"]

    n_raw = data.get("n")
    tensor = data.get("cost_tensor")
    ids = data.get("ids")
    costs = data.get("costs")

    if n_raw is None:
        if tensor is not None:
            n = len(tensor)
        else:
            raise ValueError("AP3 instance missing 'n'.")
    else:
        n = int(n_raw)
    if n <= 0:
        raise ValueError("AP3 'n' must be positive.")

    node_id2idx: Dict[Any, int] = {}
    if tensor is not None:
        if len(tensor) != n:
            raise ValueError("AP3 tensor size mismatch.")
        for i, matrix in enumerate(tensor):
            if not isinstance(matrix, list) or len(matrix) != n:
                raise ValueError(f"AP3 tensor layer {i} has incorrect size.")
            for j, row_data in enumerate(matrix):
                if not isinstance(row_data, list) or len(row_data) != n:
                    raise ValueError(f"AP3 tensor slice [{i}][{j}] has incorrect size.")
        cost_tensor = [
            [[float(tensor[i][j][k]) for k in range(n)] for j in range(n)]
            for i in range(n)
        ]
        node_id2idx = {i: i for i in range(n)}
    elif ids is not None and costs is not None:
        node_id2idx = _build_index_map(ids, n)
        cost_tensor = [[[0.0 for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for entry in costs:
            i_raw = entry.get("i")
            j_raw = entry.get("j")
            k_raw = entry.get("k")
            cost_raw = entry.get("cost")
            if i_raw is None or j_raw is None or k_raw is None or cost_raw is None:
                raise ValueError("AP3 cost entry missing i/j/k/cost.")
            i = _normalize_value(i_raw, field="i", node_map=node_id2idx, n=n)
            j = _normalize_value(j_raw, field="j", node_map=node_id2idx, n=n)
            k = _normalize_value(k_raw, field="k", node_map=node_id2idx, n=n)
            cost_tensor[i][j][k] = float(cost_raw)
    else:
        raise ValueError("AP3 instance must include 'cost_tensor' or both 'ids' and 'costs'.")

    return {
        "cost_tensor": cost_tensor,
        "n": n,
        "node_id2idx": node_id2idx,
    }


def _map_assignment(assignment: Any, *, pos: int, node_map: Dict[Any, int], n: int) -> Tuple[int, int, int]:
    if not isinstance(assignment, list) or len(assignment) != 3:
        raise ValueError(f"Assignment {pos} must be length 3 list.")
    i = _normalize_value(assignment[0], field=f"assignment[{pos}][0]", node_map=node_map, n=n)
    j = _normalize_value(assignment[1], field=f"assignment[{pos}][1]", node_map=node_map, n=n)
    k = _normalize_value(assignment[2], field=f"assignment[{pos}][2]", node_map=node_map, n=n)
    return i, j, k


def ap3_check_feasibility(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    cost_tensor: List[List[List[float]]] = instance["cost_tensor"]
    n: int = instance["n"]
    node_map: Dict[Any, int] = instance.get("node_id2idx", {})

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of assignments."

    if len(solution) != n:
        return False, PATTERN_GLOBAL4, (
            f"Solution must have exactly {n} assignments, but has {len(solution)}."
        )

    used_i = set()
    used_j = set()
    used_k = set()

    for idx, assignment in enumerate(solution):
        try:
            i, j, k = _map_assignment(assignment, pos=idx, node_map=node_map, n=n)
        except ValueError as err:
            return False, PATTERN_FORMAT, str(err)

        if i in used_i:
            return False, PATTERN_GLOBAL4, f"Assignment {idx} uses i={i} twice."
        if j in used_j:
            return False, PATTERN_GLOBAL4, f"Assignment {idx} uses j={j} twice."
        if k in used_k:
            return False, PATTERN_GLOBAL4, f"Assignment {idx} uses k={k} twice."

        used_i.add(i)
        used_j.add(j)
        used_k.add(k)

    all_indices = set(range(n))
    missing_i = all_indices - used_i
    missing_j = all_indices - used_j
    missing_k = all_indices - used_k

    if missing_i:
        return False, PATTERN_GLOBAL4, f"Some i indices are missing: {sorted(missing_i)}."
    if missing_j:
        return False, PATTERN_GLOBAL4, f"Some j indices are missing: {sorted(missing_j)}."
    if missing_k:
        return False, PATTERN_GLOBAL4, f"Some k indices are missing: {sorted(missing_k)}."

    return True, PATTERN_OK, "Feasible AP3 solution."


def _map_assignment_safe(assignment: Any, *, pos: int, node_map: Dict[Any, int], n: int) -> Tuple[bool, str, Tuple[int, int, int]]:
    if not isinstance(assignment, list) or len(assignment) != 3:
        return False, "Assignment must be a list of 3 elements.", (-1, -1, -1)
    try:
        mapped = _map_assignment(assignment, pos=pos, node_map=node_map, n=n)
    except ValueError as err:
        return False, str(err), (-1, -1, -1)
    return True, "", mapped


def ap3_objective_model(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> float:
    if not isinstance(solution, list):
        return float("inf")

    cost_tensor: List[List[List[float]]] = instance["cost_tensor"]
    n: int = instance["n"]
    node_map: Dict[Any, int] = instance.get("node_id2idx", {})

    total_cost = 0.0
    for idx, assignment in enumerate(solution):
        ok, msg, (i, j, k) = _map_assignment_safe(assignment, pos=idx, node_map=node_map, n=n)
        if not ok:
            return float("inf")
        total_cost += cost_tensor[i][j][k]

    return float(total_cost)
