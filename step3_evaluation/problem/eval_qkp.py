from typing import Any, Dict, List, Tuple, Optional, Union
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL2 = "Global2"  # budgeted subset (knapsack capacity)

EPS = 1e-9


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


def qkp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Quadratic Knapsack Problem (QKP) instance from `instance_variant`.

    Expected flat instance_variant format:
      {
        "problem_type": "QKP",
        "num_items": n,
        "capacity": C,
        "items": [0,1,...,n-1] or ['A','B',...] (optional),
        "linear_pairs": [{"item_id": i, "linear_profit": p_i}, ...],
        "weight_pairs": [{"item_id": i, "weight": w_i}, ...],
        "quadratic_pairs": [{"item_i_id": i, "item_j_id": j, "quadratic_profit": q_ij}, ...]
      }

    Notes:
      - Items can be identified by integers (0-indexed, 1-indexed) or strings ('A', 'B', etc.)
      - Internally normalized to 0-based indices
      - Quadratic terms may be sparse. Stored as dict-of-dicts: quadratic[i][j] = q_ij.
      - Diagonal terms i==j are allowed.
      - If data includes both (i,j) and (j,i), both will be counted by objective_model.

    Return (normalized):
      {
        "num_items": int,
        "capacity": float,
        "weights": [w_i] length n,
        "linear_profits": [p_i] length n,
        "quadratic": Dict[int, Dict[int, float]],
        "item_ids": list of original item identifiers,
        "item_id_to_index": mapping from identifier -> 0-based index
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for QKP.")

    instance = inst_var.get("instance", None)
    if instance is None:
        instance = inst_var
    if not isinstance(instance, dict) or not instance:
        raise ValueError("QKP instance_variant has no usable instance dictionary.")

    n_raw = instance.get("num_items", None)
    cap_raw = instance.get("capacity", None)
    linear_pairs = instance.get("linear_pairs", None)
    weight_pairs = instance.get("weight_pairs", None)
    quad_pairs = instance.get("quadratic_pairs", None)

    if n_raw is None:
        raise ValueError("QKP instance missing 'num_items'.")
    if cap_raw is None:
        raise ValueError("QKP instance missing 'capacity'.")
    if linear_pairs is None:
        raise ValueError("QKP instance missing 'linear_pairs'.")
    if weight_pairs is None:
        raise ValueError("QKP instance missing 'weight_pairs'.")
    if quad_pairs is None:
        raise ValueError("QKP instance missing 'quadratic_pairs'.")

    n, err = _to_int_strict(n_raw, field="num_items")
    if err:
        raise ValueError(f"Invalid num_items: {err}")
    if n <= 0:
        raise ValueError(f"Invalid num_items: must be positive, got {n}.")

    cap, err = _to_float_finite(cap_raw, field="capacity")
    if err:
        raise ValueError(f"Invalid capacity: {err}")
    if cap < -EPS:
        raise ValueError(f"Invalid capacity: must be >= 0, got {cap}.")

    if not isinstance(linear_pairs, list):
        raise ValueError("'linear_pairs' must be a list.")
    if not isinstance(weight_pairs, list):
        raise ValueError("'weight_pairs' must be a list.")
    if not isinstance(quad_pairs, list):
        raise ValueError("'quadratic_pairs' must be a list.")

    # Build item_id_to_index mapping from the "items" field if provided
    item_ids: List[Any] = []
    item_id_to_index: Dict[Any, int] = {}
    items_list = instance.get("items", None)
    if items_list is not None and isinstance(items_list, list):
        if len(items_list) != n:
            raise ValueError(f"'items' list length {len(items_list)} doesn't match num_items={n}.")
        for idx, item_id in enumerate(items_list):
            if isinstance(item_id, str):
                item_id = item_id.strip()
                if item_id == "":
                    raise ValueError(f"items[{idx}] is empty string.")
            if item_id in item_id_to_index:
                raise ValueError(f"Duplicate item_id: {item_id!r}.")
            item_id_to_index[item_id] = idx
            item_ids.append(item_id)
    else:
        # Default to 0-based indices
        item_ids = list(range(n))
        item_id_to_index = {i: i for i in range(n)}

    def resolve_item_id(item_id: Any, field: str) -> int:
        """Resolve item_id to 0-based index."""
        # First check if it's in the mapping
        if item_id in item_id_to_index:
            return item_id_to_index[item_id]
        # For strings, also check stripped version
        if isinstance(item_id, str):
            s = item_id.strip()
            if s in item_id_to_index:
                return item_id_to_index[s]
        # Try to interpret as integer index
        i, err = _to_int_strict(item_id, field=field)
        if err is None:
            if 0 <= i < n:
                return i
            raise ValueError(f"{field}={i} is out of range 0-{n-1}.")
        raise ValueError(f"{field}={item_id!r} is not a valid item identifier.")

    # linear profits
    linear_profits: List[Optional[float]] = [None] * n
    for k, rec in enumerate(linear_pairs):
        if not isinstance(rec, dict):
            raise ValueError(f"linear_pairs[{k}] must be a dict.")
        i = resolve_item_id(rec.get("item_id", None), field=f"linear_pairs[{k}].item_id")
        p, err = _to_float_finite(rec.get("linear_profit", None), field=f"linear_pairs[{k}].linear_profit")
        if err:
            raise ValueError(err)
        linear_profits[i] = float(p)

    missing_lp = [i for i, v in enumerate(linear_profits) if v is None]
    if missing_lp:
        raise ValueError(f"Missing linear profits for items {missing_lp}.")

    # weights
    weights: List[Optional[float]] = [None] * n
    for k, rec in enumerate(weight_pairs):
        if not isinstance(rec, dict):
            raise ValueError(f"weight_pairs[{k}] must be a dict.")
        i = resolve_item_id(rec.get("item_id", None), field=f"weight_pairs[{k}].item_id")
        w, err = _to_float_finite(rec.get("weight", None), field=f"weight_pairs[{k}].weight")
        if err:
            raise ValueError(err)
        if w < -EPS:
            raise ValueError(f"weight_pairs[{k}].weight is negative: {w}.")
        weights[i] = float(w)

    missing_w = [i for i, v in enumerate(weights) if v is None]
    if missing_w:
        raise ValueError(f"Missing weights for items {missing_w}.")

    # quadratic profits (sparse)
    quadratic: Dict[int, Dict[int, float]] = {}
    for k, rec in enumerate(quad_pairs):
        if not isinstance(rec, dict):
            raise ValueError(f"quadratic_pairs[{k}] must be a dict.")
        i = resolve_item_id(rec.get("item_i_id", None), field=f"quadratic_pairs[{k}].item_i_id")
        j = resolve_item_id(rec.get("item_j_id", None), field=f"quadratic_pairs[{k}].item_j_id")
        q, err = _to_float_finite(rec.get("quadratic_profit", None), field=f"quadratic_pairs[{k}].quadratic_profit")
        if err:
            raise ValueError(err)

        if i not in quadratic:
            quadratic[i] = {}
        if j in quadratic[i]:
            raise ValueError(f"Duplicate quadratic pair for (i,j)=({i},{j}).")
        quadratic[i][j] = float(q)

    return {
        "num_items": n,
        "capacity": float(cap),
        "weights": [float(x) for x in weights],  # type: ignore[arg-type]
        "linear_profits": [float(x) for x in linear_profits],  # type: ignore[arg-type]
        "quadratic": quadratic,
        "item_ids": item_ids,
        "item_id_to_index": item_id_to_index,
    }


def _extract_selected_items(solution: Any) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Accept solution formats:
      - list of items: [0, 2, 5]
      - dict: {"selected_items": [...]}, {"items": [...]}, {"selection": [...]}
      - dict with binary vector: {"x": [0/1,...]} or {"selected": [0/1,...]}
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
        return None, "Solution dict must contain selected items list (selected_items/items/selection) or binary vector (x/selected)."
    return None, "Solution must be a list or a dict."


def _normalize_selected_items(
    raw_list: List[Any],
    n: int,
    item_id_to_index: Optional[Dict[Any, int]] = None,
) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Normalize selection into list of unique item indices.
    
    Priority:
    1. If raw_list length == n and all values are 0/1-like, interpret as binary vector.
    2. Try to look up each value in item_id_to_index mapping (handles string IDs, 0-indexed, 1-indexed)
    3. If not found, interpret as direct 0-based index
    """
    if item_id_to_index is None:
        item_id_to_index = {}
    
    if len(raw_list) == 0:
        return [], None

    # Possible binary vector
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

    # Otherwise, interpret as item identifiers or indices
    sel: List[int] = []
    seen = set()
    
    for k, x in enumerate(raw_list):
        idx: Optional[int] = None
        
        # First priority: check if x exists in item_id_to_index
        if x in item_id_to_index:
            idx = item_id_to_index[x]
        elif isinstance(x, str):
            # For strings, also check stripped version
            s = x.strip()
            if s in item_id_to_index:
                idx = item_id_to_index[s]
        
        # Second priority: try to interpret as integer index
        if idx is None:
            v, err = _to_int_strict(x, field=f"selected_items[{k}]")
            if err:
                return None, err
            # Check if it's a valid direct index
            if 0 <= v < n:
                idx = v
            else:
                return None, f"Selected item {x!r} (index {v}) is out of range 0-{n-1}."
        
        if idx < 0 or idx >= n:
            return None, f"Selected item {x!r} maps to invalid index {idx}."
        if idx in seen:
            return None, f"Item {x!r} (index {idx}) appears multiple times in selection."
        
        seen.add(idx)
        sel.append(idx)

    return sel, None


def qkp_check_feasibility(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check QKP feasibility + unified violation pattern.

    Constraints:
      1) selected items are valid indices (or binary vector).
      2) total weight <= capacity.

    Patterns (taxonomy for QKP: Global_2):
      - FormatError: malformed selection / invalid indices / instance errors
      - Global2: capacity violated
    """
    n: int = instance["num_items"]
    cap: float = float(instance["capacity"])
    weights: List[float] = instance["weights"]
    item_id_to_index: Dict[Any, int] = instance.get("item_id_to_index", {})

    raw_list, err = _extract_selected_items(solution)
    if err:
        return False, PATTERN_FORMAT, err
    assert raw_list is not None

    sel, err = _normalize_selected_items(raw_list, n, item_id_to_index)
    if err:
        return False, PATTERN_FORMAT, err
    assert sel is not None

    total_w = 0.0
    for i in sel:
        try:
            w = float(weights[i])
        except Exception as e:
            return False, PATTERN_FORMAT, f"Instance error: cannot read weight for item {i}: {e!r}"
        if w < -EPS:
            return False, PATTERN_FORMAT, f"Instance error: negative weight for item {i}."
        total_w += w

    if total_w > cap + EPS:
        return False, PATTERN_GLOBAL2, (
            f"Capacity violated: total weight {total_w:.6f} exceeds capacity {cap:.6f}."
        )

    return True, PATTERN_OK, "Feasible QKP solution."


def qkp_objective_model(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    QKP objective: maximize
        sum_i p_i x_i + sum_{i,j} q_{i,j} x_i x_j
    For a minimization-based evaluator, return the NEGATIVE profit.

    Assumes solution is feasible; if not well-formed, return +inf (never raise).
    """
    try:
        n = int(instance.get("num_items", -1))
        if n <= 0:
            return float("inf")
        cap = float(instance.get("capacity", float("nan")))
        if not math.isfinite(cap):
            return float("inf")
        weights = instance.get("weights", None)
        lin = instance.get("linear_profits", None)
        quad = instance.get("quadratic", None)
        item_id_to_index = instance.get("item_id_to_index", {})
        if not isinstance(weights, list) or not isinstance(lin, list) or not isinstance(quad, dict):
            return float("inf")
        if len(weights) != n or len(lin) != n:
            return float("inf")

        raw_list, err = _extract_selected_items(solution)
        if err or raw_list is None:
            return float("inf")
        sel, err = _normalize_selected_items(raw_list, n, item_id_to_index)
        if err or sel is None:
            return float("inf")

        # capacity check; if violated => inf
        total_w = 0.0
        for i in sel:
            total_w += float(weights[i])
        if total_w > cap + EPS:
            return float("inf")

        xset = set(sel)

        profit = 0.0
        for i in xset:
            profit += float(lin[i])

        # Add quadratic terms only for pairs provided.
        for i, row in quad.items():
            if i not in xset:
                continue
            if not isinstance(row, dict):
                return float("inf")
            for j, qij in row.items():
                if j in xset and i != j:
                    profit += float(qij)

        return float(profit)
    except Exception:
        return float("inf")
