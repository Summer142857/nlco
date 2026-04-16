from typing import Any, Dict, List, Optional, Tuple
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL7 = "Global7"  # no-overlap constraint

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


def _resolve_item_type(
    x: Any,
    *,
    num_items: int,
    item_id_to_index: Dict[Any, int],
    field: str,
) -> Tuple[Optional[int], Optional[str]]:
    """Interpret x as an item index or identifier.
    
    Priority:
    1. First check if x exists as an item_id in item_id_to_index
    2. If x is a string, also check the stripped version
    3. Only if not found as item_id, try to interpret as a direct index (0-based)
    """
    # First priority: check if x exists as an item_id (handles both 0-indexed and 1-indexed cases)
    if x in item_id_to_index:
        return item_id_to_index[x], None
    
    # For strings, also check stripped version
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None, f"{field} is empty."
        if s in item_id_to_index:
            return item_id_to_index[s], None
        
        # Try converting string to int and check if it is an ID
        idx_val, err = _to_int_strict(x, field=field)
        if err is None and idx_val in item_id_to_index:
             return item_id_to_index[idx_val], None

    # If x is int, try converting to string and check if it is an ID
    if isinstance(x, int):
        s_val = str(x)
        if s_val in item_id_to_index:
            return item_id_to_index[s_val], None
    
    # Second priority: try to convert to integer and use as direct index
    idx, err = _to_int_strict(x, field=field)
    if err is None:
        if idx < 0 or idx >= num_items:
            return None, f"{field}={idx} is out of range 0-{num_items-1}."
        return idx, None

    return None, err or f"{field} must be a valid item index or id."


def twosp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse 2SP (2D Strip Packing) instance from `instance_variant`.

    Expected `instance_variant` format:
      {
        "items": [
          {"item_id": "A", "width": 12, "height": 10, "demand": 1},
          {"item_id": 0, "width": 33, "height": 95},
          ...
        ],
        "bin_width": 200,
        "bin_height": 19,
        "is_strip_packing": false,
        "problem_type": "2SP"
      }

    Return:
      {
        "items": list of items with width, height, demand,
        "bin_width": width of the strip/bin,
        "bin_height": height constraint (or max height for strip packing),
        "is_strip_packing": whether this is strip packing variant,
        "item_ids": ordered list of provided item identifiers,
        "item_id_to_index": mapping from identifier -> index
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for 2SP.")

    items = inst_var.get("items", [])
    bin_width_raw = inst_var.get("bin_width", None)
    bin_height_raw = inst_var.get("bin_height", None)
    is_strip_packing = inst_var.get("is_strip_packing", False)

    if not items:
        raise ValueError("2SP instance has empty 'items'.")
    if bin_width_raw is None:
        raise ValueError("2SP instance missing 'bin_width'.")
    if bin_height_raw is None:
        raise ValueError("2SP instance missing 'bin_height'.")

    bin_width, err = _to_int_strict(bin_width_raw, field="bin_width")
    if err:
        raise ValueError(err)
    bin_height, err = _to_int_strict(bin_height_raw, field="bin_height")
    if err:
        raise ValueError(err)

    parsed_items: List[Dict[str, Any]] = []
    item_ids: List[Any] = []
    item_id_to_index: Dict[Any, int] = {}

    for k, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{k}] must be a dict.")
        if "item_id" not in item:
            raise ValueError(f"items[{k}].item_id is missing.")
        item_id = item["item_id"]
        if isinstance(item_id, str):
            item_id = item_id.strip()
            if item_id == "":
                raise ValueError(f"items[{k}].item_id is empty.")
        if item_id in item_id_to_index:
            raise ValueError(f"Duplicate item_id: {item_id!r}.")
        item_id_to_index[item_id] = k
        item_ids.append(item_id)

        width, err = _to_int_strict(item.get("width", None), field=f"items[{k}].width")
        if err:
            raise ValueError(err)
        height, err = _to_int_strict(item.get("height", None), field=f"items[{k}].height")
        if err:
            raise ValueError(err)
        if width < 0 or height < 0:
            raise ValueError(f"items[{k}] width/height must be non-negative.")

        demand_raw = item.get("demand", 1)
        demand, err = _to_int_strict(demand_raw, field=f"items[{k}].demand")
        if err:
            raise ValueError(err)

        parsed_items.append(
            {
                "width": width,
                "height": height,
                "demand": demand,
            }
        )

    return {
        "items": parsed_items,
        "bin_width": bin_width,
        "bin_height": bin_height,
        "is_strip_packing": bool(is_strip_packing),
        "item_ids": item_ids,
        "item_id_to_index": item_id_to_index,
    }


def twosp_check_feasibility(
    solution: List[Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check 2SP feasibility + unified violation pattern.

    Expected solution format:
      solution = [
        {"item_type": 0 or "A", "x": 23, "y": 9, "width": 12, "height": 10},
        {"item_type": 1 or "B", "x": 3, "y": 0, "width": 4, "height": 4},
        ...
      ]

    Patterns for 2SP (per taxonomy Global_7):
      - FormatError: invalid format/types, invalid item_type, dimension mismatch
      - Global7: no-overlap violations, out-of-bounds, wrong demand count

    Constraints:
      1) solution is a list of placements with item_type, x, y, width, height.
      2) each placement must reference a valid item_type.
      3) width and height must match the item's dimensions.
      4) all placements must be within bounds.
      5) no two items may overlap (Global7).
      6) each item must be placed exactly as many times as its demand.
    """
    items: List[Dict[str, int]] = instance["items"]
    bin_width: int = instance["bin_width"]
    bin_height: int = instance["bin_height"]
    is_strip_packing: bool = instance["is_strip_packing"]

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of placements."

    if len(solution) == 0:
        # Check if all items have 0 demand
        total_demand = sum(item["demand"] for item in items)
        if total_demand == 0:
            return True, PATTERN_OK, "Feasible 2SP solution (no items to place)."
        return False, PATTERN_GLOBAL7, "Solution is empty but items need to be placed."

    item_id_to_index: Dict[Any, int] = instance.get("item_id_to_index", {})
    placements: List[Dict[str, Any]] = []
    item_type_counts = {}

    for i, placement in enumerate(solution):
        if not isinstance(placement, dict):
            return False, PATTERN_FORMAT, f"Placement {i} must be a dictionary."

        required_keys = ["item_type", "x", "y", "width", "height"]
        for key in required_keys:
            if key not in placement:
                return False, PATTERN_FORMAT, f"Placement {i} missing '{key}' key."

        item_type = placement["item_type"]
        x = placement["x"]
        y = placement["y"]
        width = placement["width"]
        height = placement["height"]

        # Type checking
        item_idx, err = _resolve_item_type(
            item_type,
            num_items=len(items),
            item_id_to_index=item_id_to_index,
            field=f"placements[{i}].item_type",
        )
        if err:
            return False, PATTERN_FORMAT, err
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False, PATTERN_FORMAT, f"Placement {i} has non-numeric x or y."
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            return False, PATTERN_FORMAT, f"Placement {i} has non-numeric width or height."

        # Validate resolved index
        if item_idx < 0 or item_idx >= len(items):
            return False, PATTERN_FORMAT, (
                f"Placement {i} has invalid item_type {item_type} "
                f"(valid range: 0-{len(items)-1})."
            )

        # Check dimensions match
        expected_width = items[item_idx]["width"]
        expected_height = items[item_idx]["height"]
        if width != expected_width or height != expected_height:
            return False, PATTERN_FORMAT, (
                f"Placement {i} dimensions ({width}x{height}) do not match "
                f"item_type {item_type} dimensions ({expected_width}x{expected_height})."
            )

        # Count placements per item type
        item_type_counts[item_idx] = item_type_counts.get(item_idx, 0) + 1

        placements.append({
            "item_type": item_idx,
            "x": float(x),
            "y": float(y),
            "width": float(width),
            "height": float(height),
            "x2": float(x) + float(width),
            "y2": float(y) + float(height),
        })

    # ---- Global7: Check bounds ----
    for i, p in enumerate(placements):
        # Check if placement is within bin bounds
        if p["x"] < 0 or p["y"] < 0:
            return False, PATTERN_GLOBAL7, (
                f"Placement {i} has negative coordinates: (x={p['x']}, y={p['y']})."
            )

        if p["x2"] > bin_width:
            return False, PATTERN_GLOBAL7, (
                f"Placement {i} exceeds bin width: "
                f"x={p['x']}, width={p['width']}, x2={p['x2']} > {bin_width}."
            )

        # For strip packing with unlimited height, we don't check y2 against bin_height
        # Otherwise, check height bounds
        if not is_strip_packing and p["y2"] > bin_height:
            return False, PATTERN_GLOBAL7, (
                f"Placement {i} exceeds bin height: "
                f"y={p['y']}, height={p['height']}, y2={p['y2']} > {bin_height}."
            )

    # ---- Global7: Check no overlap ----
    for i in range(len(placements)):
        for j in range(i + 1, len(placements)):
            p1 = placements[i]
            p2 = placements[j]

            # Two rectangles overlap if they intersect in both x and y dimensions
            # No overlap if: p1.x2 <= p2.x OR p2.x2 <= p1.x OR p1.y2 <= p2.y OR p2.y2 <= p1.y
            x_overlap = not (p1["x2"] <= p2["x"] or p2["x2"] <= p1["x"])
            y_overlap = not (p1["y2"] <= p2["y"] or p2["y2"] <= p1["y"])

            if x_overlap and y_overlap:
                return False, PATTERN_GLOBAL7, (
                    f"Placements {i} and {j} overlap: "
                    f"[{p1['x']},{p1['y']},{p1['x2']},{p1['y2']}] vs "
                    f"[{p2['x']},{p2['y']},{p2['x2']},{p2['y2']}]."
                )

    # ---- Check demand satisfaction ----
    for item_idx, item in enumerate(items):
        expected_count = item["demand"]
        actual_count = item_type_counts.get(item_idx, 0)
        if actual_count != expected_count:
            return False, PATTERN_GLOBAL7, (
                f"Item {item_idx} has demand {expected_count} but is placed {actual_count} times."
            )

    return True, PATTERN_OK, "Feasible 2SP solution."


def twosp_objective_model(
    solution: List[Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    2SP objective: minimize the maximum height used (for strip packing).

    Objective = max(y + height for all placements)

    For bounded bin packing variant, the objective might be different,
    but we use height as the standard objective for 2SP.

    Assumes solution is feasible. If not well-formed, return inf.
    """
    if not isinstance(solution, list):
        return float("inf")

    if len(solution) == 0:
        return 0.0

    max_height = 0.0
    for placement in solution:
        if not isinstance(placement, dict):
            return float("inf")
        if "y" not in placement or "height" not in placement:
            return float("inf")

        y = placement["y"]
        height = placement["height"]

        if not isinstance(y, (int, float)) or not isinstance(height, (int, float)):
            return float("inf")

        top_edge = float(y) + float(height)
        max_height = max(max_height, top_edge)

    return max_height

