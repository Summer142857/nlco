from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL2 = "Global2"  # budgeted subset (capacity/budget)
PATTERN_GLOBAL5 = "Global5"  # partitioning (each item exactly once)

def _letter_to_index(label: str) -> Optional[int]:
    """
    Convert alphabetical label (e.g., 'A', 'B', ..., 'AA') to 0-based index.
    Returns None if the label is not uppercase letters.
    """
    if not label:
        return None
    idx = 0
    for ch in label.upper():
        if ch < "A" or ch > "Z":
            return None
        idx = idx * 26 + (ord(ch) - ord("A"))
    return idx

def _to_item_index(
    value: Any, *, num_items: int, item_map: Optional[Dict[Any, int]] = None
) -> Tuple[Optional[int], Optional[str]]:
    if item_map and value in item_map:
        return item_map[value], None
    if isinstance(value, bool):
        return None, "Item index must not be a boolean."
    if isinstance(value, int):
        idx = value
    elif isinstance(value, float):
        if not value.is_integer():
            return None, f"Item index must be integer-like, got {value}."
        idx = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            idx = int(stripped)
        else:
            letter_idx = _letter_to_index(stripped)
            if letter_idx is None:
                return None, f"Item identifier {value!r} is not recognized."
            idx = letter_idx
    else:
        return None, f"Item identifier {value!r} is not an int or string."
    if idx < 0 or idx >= num_items:
        return None, f"Item index {idx} out of range 0-{num_items-1}."
    return idx, None

def bpp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse BPP instance from `instance_variant`.

    Expected `instance_variant` format:
      {
        "instance": [60.0, 48.0, 40.0, 66.0, ...],
        "solution": [[4, 6, 7], [1, 9], ...],  # (recorded, not used here)
        "obj": 5,                               # (recorded, not used here)
        "bin_capacity": 150.0,
        "problem_type": "BPP"
      }

    Return:
      {
        "items": [weight1, weight2, ...],  # list of item weights
        "bin_capacity": capacity
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for BPP.")

    items_field = inst_var.get("items", None)
    weights: List[float] = []
    item_map: Dict[Any, int] = {}
    n_items_raw = inst_var.get("n_items", None)

    if items_field is None:
        # fallback to legacy weights list
        items_field = inst_var.get("weights", [])
    if not items_field:
        raise ValueError("BPP instance_variant has empty 'items'.")

    if isinstance(items_field, list) and isinstance(items_field[0], dict):
        if n_items_raw is not None and int(n_items_raw) != len(items_field):
            raise ValueError(
                f"BPP instance n_items={n_items_raw} does not match len(items)={len(items_field)}."
            )
        for idx, entry in enumerate(items_field):
            if not isinstance(entry, dict):
                raise ValueError("Each BPP item entry must be a dictionary.")
            item_id = entry.get("item_id")
            weight = entry.get("weight")
            if item_id is None or weight is None:
                raise ValueError("BPP item entry missing 'item_id' or 'weight'.")
            if item_id in item_map:
                raise ValueError(f"Duplicate item_id '{item_id}'.")
            item_map[item_id] = idx
            weights.append(float(weight))
    elif isinstance(items_field, list):
        weights = [float(w) for w in items_field]
    else:
        raise ValueError("BPP 'items' must be a list.")

    bin_capacity = inst_var.get("bin_capacity", None)
    if bin_capacity is None:
        raise ValueError("BPP instance_variant missing 'bin_capacity'.")

    return {
        "items": weights,
        "bin_capacity": float(bin_capacity),
        "item_id2idx": item_map,
    }


def bpp_check_feasibility(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check BPP feasibility + output a unified violation pattern.

    Patterns used for BPP (per taxonomy Global_{2,5}):
      - Global2: capacity/budget violation (bin exceeds capacity)
      - Global5: partitioning violation (missing items / duplicates)
      - FormatError: wrong format/types/out-of-range indices, etc.

    Expected solution format:
      solution = [
        [item_idx1, item_idx2, ...],  # bin 1
        [item_idx3, item_idx4, ...],  # bin 2
        ...
      ]
    """
    items: List[float] = instance["items"]
    bin_capacity = float(instance["bin_capacity"])
    num_items = len(items)

    if not isinstance(solution, list) or len(solution) == 0:
        return False, PATTERN_FORMAT, "Solution must be a non-empty list of bins (list[list[int]])."

    seen_items: List[int] = []

    item_map: Dict[Any, int] = instance.get("item_id2idx", {})
    for bidx, bin_items in enumerate(solution):
        if not isinstance(bin_items, list):
            return False, PATTERN_FORMAT, f"Bin {bidx} must be a list of item indices."

        converted_items: List[int] = []
        for item_idx in bin_items:
            idx, err = _to_item_index(item_idx, num_items=num_items, item_map=item_map)
            if err:
                return False, PATTERN_FORMAT, f"Bin {bidx}: {err}"
            converted_items.append(idx)

        bin_weight = sum(items[idx] for idx in converted_items)
        if bin_weight > bin_capacity + 1e-9:
            return (
                False,
                PATTERN_GLOBAL2,
                f"Bin {bidx} exceeds capacity: weight={bin_weight:.6f} > capacity={bin_capacity:.6f}",
            )

        seen_items.extend(converted_items)

    seen_set = set(seen_items)
    all_items = set(range(num_items))

    missing = all_items - seen_set
    if missing:
        return False, PATTERN_GLOBAL5, f"Some items are not packed: missing indices {sorted(missing)!r}"

    if len(seen_items) != len(seen_set):
        cnt: Dict[int, int] = {}
        for idx in seen_items:
            cnt[idx] = cnt.get(idx, 0) + 1
        dup = [k for k, v in cnt.items() if v > 1]
        return False, PATTERN_GLOBAL5, f"Some items are packed more than once: indices {sorted(dup)!r}"

    return True, PATTERN_OK, "Feasible BPP solution."


def bpp_objective_model(
    solution: List[List[Any]],
    instance: Dict[str, Any],
) -> float:
    """
    BPP objective: minimize the number of bins used.

    Assumes solution is feasible. If not well-formed, return inf.
    """
    if not isinstance(solution, list) or len(solution) == 0:
        return float("inf")

    items: List[float] = instance["items"]
    num_items = len(items)
    item_map: Dict[Any, int] = instance.get("item_id2idx", {})

    for bidx, bin_items in enumerate(solution):
        if not isinstance(bin_items, list):
            return float("inf")
        for item_idx in bin_items:
            idx, err = _to_item_index(item_idx, num_items=num_items, item_map=item_map)
            if err:
                return float("inf")

    return float(len(solution))

