from typing import Any, Dict, List, Optional, Tuple, Union
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL2 = "Global2"  # budgeted subset (capacity/budget)

EPS = 1e-9


def _letter_to_index(label: str) -> Optional[int]:
    if not label:
        return None
    idx = 0
    for ch in label.upper():
        if ch < "A" or ch > "Z":
            return None
        idx = idx * 26 + (ord(ch) - ord("A"))
    return idx


def _to_int_strict(x: Any, *, field: str) -> Tuple[Optional[int], Optional[str]]:
    """Convert x to int, only if it's integer-like (e.g., 1, '1', 1.0)."""
    if isinstance(x, bool):
        return None, f"'{field}' must be an integer, got bool."
    if isinstance(x, int):
        return x, None
    if isinstance(x, float):
        if not math.isfinite(x):
            return None, f"'{field}' must be finite, got {x}."
        if abs(x - round(x)) > EPS:
            return None, f"'{field}' must be an integer-like number, got {x}."
        return int(round(x)), None
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None, f"'{field}' is empty."
        try:
            xf = float(s)
        except ValueError:
            return None, f"'{field}' must be an integer-like string, got {x!r}."
        if not math.isfinite(xf):
            return None, f"'{field}' must be finite, got {x!r}."
        if abs(xf - round(xf)) > EPS:
            return None, f"'{field}' must be an integer-like value, got {x!r}."
        return int(round(xf)), None
    return None, f"'{field}' must be an integer-like value, got {type(x).__name__}."


def _to_float_finite(x: Any, *, field: str) -> Tuple[Optional[float], Optional[str]]:
    """Convert x to float, must be finite."""
    if x is None:
        return None, f"Missing '{field}'."
    if isinstance(x, bool):
        return None, f"'{field}' must be a number, got bool."
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None, f"'{field}' must be numeric, got {x!r}."
    if not math.isfinite(xf):
        return None, f"'{field}' must be finite, got {x!r}."
    return xf, None


def kp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse 0/1 Knapsack Problem (KP) instance from `instance_variant`.

    Supports:
      (A) nested format:
        {"instance": {"problem_type":"KP", "capacity": C, "weights":[...], "profits":[...], "n_items": n}}
      (B) flat format:
        {"problem_type":"KP", "capacity": C, "weights":[...], "profits":[...], "n_items": n}

    Return (normalized):
      {
        "n_items": int,
        "capacity": float,
        "weights": [float],
        "profits": [float]
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for KP.")

    instance = inst_var.get("instance", None)
    if instance is None:
        instance = inst_var
    if not isinstance(instance, dict) or not instance:
        raise ValueError("KP instance_variant has no usable instance dictionary.")

    cap_raw = instance.get("capacity", None)
    items_raw = instance.get("items", None)
    weights_raw = instance.get("weights", None)
    profits_raw = instance.get("profits", None)
    n_raw = instance.get("n_items", None)

    if cap_raw is None:
        raise ValueError("KP instance missing 'capacity'.")
    if items_raw is None and (weights_raw is None or profits_raw is None):
        raise ValueError("KP instance missing 'items' or both 'weights'/'profits'.")
    if n_raw is None:
        # allow inference
        if isinstance(weights_raw, list):
            n_raw = len(weights_raw)
        else:
            raise ValueError("KP instance missing 'n_items'.")

    n, err = _to_int_strict(n_raw, field="n_items")
    if err:
        raise ValueError(f"Invalid n_items: {err}")
    if n <= 0:
        raise ValueError(f"Invalid n_items: must be positive, got {n}.")

    cap, err = _to_float_finite(cap_raw, field="capacity")
    if err:
        raise ValueError(f"Invalid capacity: {err}")
    if cap < -EPS:
        raise ValueError(f"capacity must be >= 0, got {cap}.")

    weights: List[float] = []
    profits: List[float] = []
    item_id2idx: Dict[Any, int] = {}

    if isinstance(items_raw, list) and items_raw and isinstance(items_raw[0], dict):
        if n_raw is not None and int(n_raw) != len(items_raw):
            raise ValueError(f"'n_items' ({n_raw}) does not match len(items) ({len(items_raw)}).")
        for idx, entry in enumerate(items_raw):
            if not isinstance(entry, dict):
                raise ValueError("Each item entry must be a dict.")
            item_id = entry.get("item_id")
            weight = entry.get("weight")
            profit = entry.get("profit")
            if item_id is None or weight is None or profit is None:
                raise ValueError("Item entry missing 'item_id', 'weight', or 'profit'.")
            if item_id in item_id2idx:
                raise ValueError(f"Duplicate item_id '{item_id}'.")
            item_id2idx[item_id] = idx
            weights.append(float(weight))
            profits.append(float(profit))
    else:
        if not isinstance(weights_raw, list) or len(weights_raw) != n:
            raise ValueError(f"'weights' must be a list of length n_items={n}.")
        if not isinstance(profits_raw, list) or len(profits_raw) != n:
            raise ValueError(f"'profits' must be a list of length n_items={n}.")
        for i in range(n):
            w, err = _to_float_finite(weights_raw[i], field=f"weights[{i}]")
            if err:
                raise ValueError(err)
            if w < -EPS:
                raise ValueError(f"weights[{i}] is negative: {w}.")
            weights.append(float(w))

            p, err = _to_float_finite(profits_raw[i], field=f"profits[{i}]")
            if err:
                raise ValueError(err)
            profits.append(float(p))

    return {
        "n_items": n,
        "capacity": float(cap),
        "weights": weights,
        "profits": profits,
        "item_id2idx": item_id2idx,
    }


def _map_item_key(key: Any, *, field: str, n: int, id_map: Optional[Dict[Any, int]] = None) -> Tuple[Optional[int], Optional[str]]:
    if id_map and key in id_map:
        return id_map[key], None
    if isinstance(key, str):
        stripped = key.strip()
        if stripped.isdigit():
            idx = int(stripped)
        else:
            letter_idx = _letter_to_index(stripped)
            if letter_idx is None:
                return None, f"{field} '{key}' is not a recognized identifier."
            idx = letter_idx
    elif isinstance(key, int):
        idx = key
    elif isinstance(key, float):
        if not math.isfinite(key):
            return None, f"{field} '{key}' must be finite."
        if abs(key - round(key)) > EPS:
            return None, f"{field} '{key}' must be integer-like."
        idx = int(round(key))
    else:
        return None, f"{field} '{key}' is not integer-like."
    if idx < 0 or idx >= n:
        return None, f"{field} index {idx} out of range 0-{n-1}."
    return idx, None


def _extract_selected_items(solution: Any, *, n_items: Optional[int] = None, id_map: Optional[Dict[Any, int]] = None) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Accept solution formats:
      - list of items: [0, 2, 4]
      - dict: {"selected_items":[...]}, {"items":[...]}, {"selection":[...]}
      - dict with binary vector: {"x":[0/1,...]} or {"selected":[0/1,...]}
    """
    if isinstance(solution, list):
        return solution, None
    if isinstance(solution, dict):
        for key in ("selected_items", "items", "selection", "chosen_items"):
            if key in solution:
                if not isinstance(solution[key], list):
                    return None, f"Solution '{key}' must be a list."
                return solution[key], None
        for key in ("x", "selected_vector", "selected"):
            if key in solution:
                if not isinstance(solution[key], list):
                    return None, f"Solution '{key}' must be a list."
                return solution[key], None
        if n_items is None:
            return None, "Solution dict must contain selected items list or binary vector."
        entries: List[int] = []
        seen = set()
        for key, val in solution.items():
            if key in ("selected_items", "items", "selection", "chosen_items", "x", "selected_vector", "selected"):
                continue
            if isinstance(val, bool):
                v = int(val)
            else:
                v, err = _to_int_strict(val, field=f"value for key {key}")
                if err:
                    return None, err
            if v not in (0, 1):
                return None, f"Solution value for key {key!r} must be 0 or 1."
            idx, err = _map_item_key(key, field="Solution key", n=n_items, id_map=id_map)
            if err:
                return None, err
            if idx in seen:
                return None, f"Item {key!r} appears multiple times."
            seen.add(idx)
            if v == 1:
                entries.append(idx)
        return entries, None
    return None, "Solution must be a list or a dict."


def _normalize_selected_items(raw_list: List[Any], n: int) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Normalize selection into list of unique item indices.
    If raw_list length == n and all values are 0/1-like, interpret as binary vector.
    Otherwise interpret as list of indices (0-based; accept 1-based if unambiguous).
    """
    if len(raw_list) == 0:
        return [], None

    # binary vector case
    if len(raw_list) == n:
        bin_vals: List[int] = []
        ok = True
        for k, x in enumerate(raw_list):
            v, err = _to_int_strict(x, field=f"x[{k}]")
            if err:
                ok = False
                break
            if v not in (0, 1):
                ok = False
                break
            bin_vals.append(v)
        if ok:
            return [i for i, v in enumerate(bin_vals) if v == 1], None

    # indices case
    ints: List[int] = []
    for k, x in enumerate(raw_list):
        v, err = _to_int_strict(x, field=f"selected_items[{k}]")
        if err:
            return None, err
        ints.append(v)

    is_one_based = (all(1 <= v <= n for v in ints) and all(v != 0 for v in ints))
    sel: List[int] = []
    seen = set()
    for v in ints:
        idx = v - 1 if is_one_based else v
        if idx < 0 or idx >= n:
            base = "1-based" if is_one_based else "0-based"
            return None, f"Selected item {v} ({base}) is out of range."
        if idx in seen:
            return None, f"Item {v} appears multiple times in selection."
        seen.add(idx)
        sel.append(idx)

    return sel, None


def kp_check_feasibility(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Check KP feasibility.

    Constraints:
      1) selected items are valid indices (or binary vector).
      2) total weight <= capacity.
    """
    n: int = instance["n_items"]
    cap: float = float(instance["capacity"])
    weights: List[float] = instance["weights"]

    item_map: Dict[Any, int] = instance.get("item_id2idx", {})
    raw_list, err = _extract_selected_items(solution, n_items=n, id_map=item_map)
    if err:
        return False, PATTERN_FORMAT, err
    assert raw_list is not None

    sel, err = _normalize_selected_items(raw_list, n)
    if err:
        return False, PATTERN_FORMAT, err
    assert sel is not None

    total_w = 0.0
    for i in sel:
        total_w += float(weights[i])

    if total_w > cap + EPS:
        return False, PATTERN_GLOBAL2, f"Capacity violated: total weight {total_w:.6f} exceeds capacity {cap:.6f}."

    return True, PATTERN_OK, "Feasible KP solution."


def kp_objective_model(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    KP objective: maximize sum_i profits[i] * x_i.
    For a minimization-based evaluator, return NEGATIVE profit.

    Assumes solution is feasible; if not well-formed, return +inf (never raise).
    """
    try:
        n = int(instance.get("n_items", -1))
        if n <= 0:
            return float("inf")
        cap = float(instance.get("capacity", float("nan")))
        if not math.isfinite(cap):
            return float("inf")

        weights = instance.get("weights", None)
        profits = instance.get("profits", None)
        if not isinstance(weights, list) or not isinstance(profits, list):
            return float("inf")
        if len(weights) != n or len(profits) != n:
            return float("inf")

        item_map: Dict[Any, int] = instance.get("item_id2idx", {})
        raw_list, err = _extract_selected_items(solution, n_items=n, id_map=item_map)
        if err or raw_list is None:
            return float("inf")
        sel, err = _normalize_selected_items(raw_list, n)
        if err or sel is None:
            return float("inf")

        total_w = 0.0
        profit = 0.0
        for i in sel:
            total_w += float(weights[i])
            profit += float(profits[i])

        if total_w > cap + EPS:
            return float("inf")

        return float(profit)
    except Exception:
        return float("inf")
