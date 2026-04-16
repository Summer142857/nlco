from typing import Any, Dict, List, Tuple, Optional, Union
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL5 = "Global5"  # partitioning / exact cover

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


def spp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Set Partitioning Problem (SPP) instance from `instance_variant`.

    Return (normalized):
      {
        "num_elements": int,
        "num_sets": int,
        "set_ids": [str],                 # length num_sets
        "sets": [ [int, ...], ... ],      # list of element lists (0-based element ids)
        "costs": [float, ...],            # cost per set, aligned with sets
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for SPP.")

    instance = inst_var.get("instance", None)
    if instance is None:
        instance = inst_var
    if not isinstance(instance, dict) or not instance:
        raise ValueError("SPP instance_variant has no usable instance dictionary.")

    ne_raw = instance.get("num_elements", None)
    ns_raw = instance.get("num_sets", None)
    sets_raw = instance.get("sets", None)

    if ne_raw is None:
        raise ValueError("SPP instance missing 'num_elements'.")
    if ns_raw is None:
        raise ValueError("SPP instance missing 'num_sets'.")
    if sets_raw is None:
        raise ValueError("SPP instance missing 'sets'.")

    ne, err = _to_int_strict(ne_raw, field="num_elements")
    if err:
        raise ValueError(f"Invalid num_elements: {err}")
    ns, err = _to_int_strict(ns_raw, field="num_sets")
    if err:
        raise ValueError(f"Invalid num_sets: {err}")

    if ne <= 0:
        raise ValueError(f"Invalid num_elements: must be positive, got {ne}.")
    if ns <= 0:
        raise ValueError(f"Invalid num_sets: must be positive, got {ns}.")

    if not isinstance(sets_raw, list) or len(sets_raw) == 0:
        raise ValueError("'sets' must be a non-empty list.")
    if len(sets_raw) != ns:
        raise ValueError(f"Length mismatch: num_sets={ns} but len(sets)={len(sets_raw)}.")

    set_ids: List[str] = []
    sets: List[List[int]] = []
    costs: List[float] = []

    # Detect element type (string vs int) and base (0 vs 1)
    all_elements = set()
    is_string_elements = False
    
    for srec in sets_raw:
        if isinstance(srec, dict) and "elements" in srec:
            for e in srec["elements"]:
                if isinstance(e, str):
                    is_string_elements = True
                all_elements.add(e)

    element_to_idx = {}
    if is_string_elements:
        sorted_elements = sorted(list(all_elements), key=lambda x: str(x))
        for idx, elem in enumerate(sorted_elements):
            # Map to 0-based index for internal consistency with existing logic
            element_to_idx[elem] = idx
    else:
        # Integer elements
        # Check if 0-based or 1-based
        if all_elements:
            min_val = min(all_elements)
            is_one_based = (min_val == 1)
        else:
            is_one_based = False # Default

    seen_ids = set()
    for i, srec in enumerate(sets_raw):
        if not isinstance(srec, dict):
            raise ValueError(f"sets[{i}] must be a dict with keys 'id', 'elements', 'cost'.")

        sid = srec.get("id", None)
        if not isinstance(sid, str) or sid.strip() == "":
            raise ValueError(f"sets[{i}].id must be a non-empty string.")
        sid = sid.strip()
        if sid in seen_ids:
            raise ValueError(f"Duplicate set id: {sid!r}.")
        seen_ids.add(sid)

        elems_raw = srec.get("elements", None)
        if not isinstance(elems_raw, list) or len(elems_raw) == 0:
            raise ValueError(f"sets[{i}].elements must be a non-empty list.")

        elems: List[int] = []
        seen_elems = set()
        for k, x in enumerate(elems_raw):
            if is_string_elements:
                if x not in element_to_idx:
                     raise ValueError(f"sets[{i}] has unknown element {x}.")
                e = element_to_idx[x]
            else:
                e, err = _to_int_strict(x, field=f"sets[{i}].elements[{k}]")
                if err:
                    raise ValueError(err)
                
                # Normalize to 0-based
                if is_one_based:
                    e = e - 1

            if e < 0 or e >= ne:
                raise ValueError(
                    f"sets[{i}].elements[{k}] (normalized {e}) is out of range 0-{ne-1}."
                )
            if e in seen_elems:
                raise ValueError(f"sets[{i}] (id={sid!r}) has duplicate element {e}.")
            seen_elems.add(e)
            elems.append(e)

        c, err = _to_float_finite(srec.get("cost", None), field=f"sets[{i}].cost")
        if err:
            raise ValueError(err)
        # cost may be 0; negative costs are allowed in some variants, but usually costs are nonnegative.
        # We'll allow negative unless you want to forbid it. If you want to forbid, uncomment:
        # if c < -EPS: raise ValueError(f"sets[{i}].cost is negative: {c}.")

        set_ids.append(sid)
        sets.append(elems)
        costs.append(float(c))

    return {
        "num_elements": ne,
        "num_sets": ns,
        "set_ids": set_ids,
        "sets": sets,
        "costs": costs,
    }


def _extract_selected_sets(solution: Any) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Accept solution in either of these forms:
      - {"selected_sets": [...]}
      - {"sets": [...]}
      - {"chosen_sets": [...]}
      - a list directly: [...]
    """
    if isinstance(solution, list):
        return solution, None
    if isinstance(solution, dict):
        for key in ("selected_sets", "sets", "chosen_sets", "selected", "selection"):
            if key in solution:
                if not isinstance(solution[key], list):
                    return None, f"Solution '{key}' must be a list."
                return solution[key], None
        return None, "Solution dict must contain one of: selected_sets / sets / chosen_sets / selected / selection."
    return None, "Solution must be either a list or a dictionary containing a list of selected sets."


def _selected_indices_from_solution(
    raw_list: List[Any],
    set_ids: List[str],
    ns: int,
) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Convert a raw selection list into set indices.
    Accept:
      - set ids (strings)
      - indices (0-based, or 1-based if all in 1..ns and none is 0)
    Ensures uniqueness.
    """
    if len(raw_list) == 0:
        return [], None

    by_ids = any(isinstance(x, str) for x in raw_list)

    selected_indices: List[int] = []
    seen = set()

    if by_ids:
        if not isinstance(set_ids, list) or len(set_ids) != ns:
            return None, "Instance error: missing/invalid set_ids mapping."
        id_to_idx = {sid: i for i, sid in enumerate(set_ids)}

        for k, x in enumerate(raw_list):
            if not isinstance(x, str):
                return None, f"Selected set at position {k} must be a string id (got {type(x).__name__})."
            sid = x.strip()
            if sid == "":
                return None, f"Selected set id at position {k} is empty."
            if sid not in id_to_idx:
                return None, f"Unknown set id {sid!r} in selection."
            idx = id_to_idx[sid]
            if idx in seen:
                return None, f"Set {sid!r} appears multiple times in selection."
            seen.add(idx)
            selected_indices.append(idx)
    else:
        ints: List[int] = []
        for k, x in enumerate(raw_list):
            v, err2 = _to_int_strict(x, field=f"selected_sets[{k}]")
            if err2:
                return None, err2
            ints.append(v)

        is_one_based = (all(1 <= v <= ns for v in ints) and all(v != 0 for v in ints))
        for v in ints:
            idx = v - 1 if is_one_based else v
            if idx < 0 or idx >= ns:
                base = "1-based" if is_one_based else "0-based"
                return None, f"Selected set index {v} ({base}) is out of range."
            if idx in seen:
                return None, f"Set index {v} appears multiple times in selection."
            seen.add(idx)
            selected_indices.append(idx)

    return selected_indices, None



def spp_check_feasibility(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check Set Partitioning (SPP) feasibility + unified violation pattern.

    Elements are 0..num_elements-1.

    Accepted solution formats:
      - List of set INDICES (0..num_sets-1) or 1-based indices (if unambiguous)
      - List of set IDS (strings)
      - Dict wrapper like {"selected_sets": [...]}

    SPP constraint (Global5):
      - Each element must be covered by EXACTLY ONE selected set.
    """
    ne: int = instance["num_elements"]
    ns: int = instance["num_sets"]
    set_ids: List[str] = instance.get("set_ids", [])
    sets: List[List[int]] = instance["sets"]

    raw_list, err = _extract_selected_sets(solution)
    if err:
        return False, PATTERN_FORMAT, err
    assert raw_list is not None

    selected_indices, err = _selected_indices_from_solution(raw_list, set_ids, ns)
    if err:
        return False, PATTERN_FORMAT, err
    assert selected_indices is not None

    if len(selected_indices) == 0:
        return False, PATTERN_GLOBAL5, "Solution selects no sets; cannot partition elements."

    # Count coverage occurrences
    counts = [0] * ne
    for idx in selected_indices:
        try:
            elems = sets[idx]
        except Exception as e:
            return False, PATTERN_FORMAT, f"Instance error: cannot read sets[{idx}]: {e!r}"

        for e in elems:
            if not isinstance(e, int):
                return False, PATTERN_FORMAT, f"Instance error: element label {e!r} is not an int."
            if e < 0 or e >= ne:
                return False, PATTERN_FORMAT, f"Instance error: element {e} out of range 0-{ne-1}."
            counts[e] += 1

    missing = [e for e, c in enumerate(counts) if c == 0]
    multi = [e for e, c in enumerate(counts) if c > 1]

    if missing:
        return False, PATTERN_GLOBAL5, f"SPP violated: elements not covered {missing}."
    if multi:
        return False, PATTERN_GLOBAL5, f"SPP violated: elements covered more than once {multi}."

    return True, PATTERN_OK, "Feasible SPP solution."


def spp_objective_model(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    SPP objective: minimize total cost of selected sets.

    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    ns = instance.get("num_sets", None)
    set_ids = instance.get("set_ids", None)
    costs = instance.get("costs", None)
    if not isinstance(ns, int) or not isinstance(set_ids, list) or not isinstance(costs, list):
        return float("inf")
    if len(set_ids) != ns or len(costs) != ns:
        return float("inf")

    raw_list, err = _extract_selected_sets(solution)
    if err or raw_list is None:
        return float("inf")

    selected_indices, err = _selected_indices_from_solution(raw_list, set_ids, ns)
    if err or selected_indices is None:
        return float("inf")

    total = 0.0
    for idx in selected_indices:
        c, err2 = _to_float_finite(costs[idx], field="costs")
        if err2:
            return float("inf")
        total += c

    return float(total)
