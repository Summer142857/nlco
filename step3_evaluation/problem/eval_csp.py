from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL2 = "Global2"  # capacity / demand violations


def _build_item_map(ids: List[Any]) -> Dict[Any, int]:
    item_map: Dict[Any, int] = {}
    for idx, identifier in enumerate(ids):
        if identifier in item_map:
            raise ValueError(f"Duplicate identifier '{identifier}' in CSP items[].item_id.")
        item_map[identifier] = idx
    return item_map


def csp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    New-format-only CSP instance parser.

    Required instance_variant format:
      {
        "problem_type": "CSP",
        "bin_capacity": 150.0,
        "n_items": 22,                 # optional (ignored)
        "items": [{"item_id":"A","width":73,"demand":1}, ...]
      }

    Returns:
      {
        "weights": [float],
        "demands": [int],
        "bin_capacity": float,
        "item_id2idx": {item_id -> idx}
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for CSP.")

    bin_capacity = inst_var.get("bin_capacity", None)
    if bin_capacity is None:
        raise ValueError("CSP instance_variant missing 'bin_capacity'.")

    items = inst_var.get("items", None)
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("CSP instance_variant must contain a non-empty 'items' list.")

    weights: List[float] = []
    demands: List[int] = []
    ids: List[Any] = []

    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"CSP instance_variant.items[{i}] must be a dict.")

        item_id = it.get("item_id", None)
        width = it.get("width", None)
        demand = it.get("demand", None)

        if item_id is None:
            raise ValueError(f"CSP instance_variant.items[{i}] missing 'item_id'.")
        if width is None:
            raise ValueError(f"CSP instance_variant.items[{i}] missing 'width'.")
        if demand is None:
            raise ValueError(f"CSP instance_variant.items[{i}] missing 'demand'.")

        ids.append(item_id)
        weights.append(float(width))
        demands.append(int(demand))

    item_map = _build_item_map(ids)

    return {
        "weights": weights,
        "demands": demands,
        "bin_capacity": float(bin_capacity),
        "item_id2idx": item_map,
        "ids": ids,  # optional debug
    }


def csp_check_feasibility(
    solution: Any,
    instance: Dict[str, Any],
    *,
    exact: bool = True,
) -> Tuple[bool, str, str]:
    """
    New-format-only CSP feasibility.

    Expected solution format:
      solution = [
        {"A": 1, "B": 2},
        {"C": 1},
        ...
      ]

    Rules:
      - keys must be valid item_id appearing in instance_variant.items[].item_id
      - counts must be positive integers (int-like allowed: "2", 2.0)
      - each pattern length <= bin_capacity
      - demand must be satisfied exactly (default) or at least (exact=False)
    """
    weights: List[float] = instance["weights"]
    demands: List[int] = instance["demands"]
    bin_capacity = float(instance["bin_capacity"])
    item_map: Dict[Any, int] = instance["item_id2idx"]
    num_items = len(weights)

    if not isinstance(solution, list) or len(solution) == 0:
        return False, PATTERN_FORMAT, "Solution must be a non-empty list of patterns."

    produced = [0] * num_items

    for pidx, pattern in enumerate(solution):
        if not isinstance(pattern, dict) or len(pattern) == 0:
            return False, PATTERN_FORMAT, f"Pattern {pidx} must be a non-empty dictionary."

        used_len = 0.0

        for item_key, count in pattern.items():

            # ---- canonicalize key type ----
            key = item_key
            #print(key, type(key))
            if key not in item_map:
                #print(key)
                # try str<->int normalization
                if isinstance(key, str):
                    try:
                        k2 = int(key)
                        if k2 in item_map:
                            key = k2
                        else:
                            return False, PATTERN_FORMAT, \
                                f"Pattern {pidx} contains unknown item_id {item_key!r}. Must use identifiers from input."
                    except Exception:
                        return False, PATTERN_FORMAT, \
                            f"Pattern {pidx} contains unknown item_id {item_key!r}. Must use identifiers from input."
                else:
                    k2 = str(key)
                    if k2 in item_map:
                        key = k2
                    else:
                        return False, PATTERN_FORMAT, \
                            f"Pattern {pidx} contains unknown item_id {item_key!r}. Must use identifiers from input."

            item_idx = item_map[key]

            # allow int-like
            if isinstance(count, bool) or count is None:
                return False, PATTERN_FORMAT, f"Pattern {pidx} has invalid count {count!r} for item {item_key!r}."
            try:
                c = int(count)
            except Exception:
                return False, PATTERN_FORMAT, f"Pattern {pidx} count {count!r} is not integer-like for item {item_key!r}."
            if c <= 0:
                return False, PATTERN_FORMAT, f"Pattern {pidx} has non-positive count {c} for item {item_key!r}."

            produced[item_idx] += c
            used_len += weights[item_idx] * c

        if used_len > bin_capacity + 1e-9:
            return (
                False,
                PATTERN_GLOBAL2,
                f"Pattern {pidx} exceeds capacity: used={used_len:.6f} > capacity={bin_capacity:.6f}",
            )

    for i in range(num_items):
        if exact:
            if produced[i] != demands[i]:
                return (
                    False,
                    PATTERN_GLOBAL2,
                    f"Demand mismatch for item_id={instance['ids'][i]!r}: produced={produced[i]}, demanded={demands[i]} (exact required)",
                )
        else:
            if produced[i] < demands[i]:
                return (
                    False,
                    PATTERN_GLOBAL2,
                    f"Demand not satisfied for item_id={instance['ids'][i]!r}: produced={produced[i]}, demanded={demands[i]}",
                )

    return True, PATTERN_OK, "Feasible CSP solution."


def csp_objective_model(solution: Any, instance: Dict[str, Any]) -> float:
    """
    Objective: minimize number of rolls = number of patterns.
    """
    if not isinstance(solution, list) or len(solution) == 0:
        return float("inf")
    return float(len(solution))
