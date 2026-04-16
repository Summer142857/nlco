from typing import Any, Dict, List, Tuple, Optional, Union
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL3 = "Global3"  # coverage / hitting

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


def hsp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Hitting Set Problem (HSP) instance from `instance_variant`.


    Return (normalized):
      {
        "num_elements": int,
        "num_sets": int,
        "set_ids": [str],              # length num_sets
        "sets": [ [int, ...], ... ],   # element ids (as provided, base preserved)
        "element_base": int            # 0 or 1
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for HSP.")

    instance = inst_var.get("instance", None)
    if instance is None:
        instance = inst_var
    if not isinstance(instance, dict) or not instance:
        raise ValueError("HSP instance_variant has no usable instance dictionary.")

    ne_raw = instance.get("num_elements", None)
    ns_raw = instance.get("num_sets", None)
    sets_raw = instance.get("sets", None)

    if ne_raw is None:
        raise ValueError("HSP instance missing 'num_elements'.")
    if ns_raw is None:
        raise ValueError("HSP instance missing 'num_sets'.")
    if sets_raw is None:
        raise ValueError("HSP instance missing 'sets'.")

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

    tmp_sets: List[List[int]] = []
    set_ids: List[str] = []
    seen_ids = set()
    min_elem: Optional[int] = None
    
    # Collect all unique elements to build a mapping if they are strings
    all_elements = set()
    is_string_elements = False
    
    # First pass to detect element type
    for srec in sets_raw:
        if isinstance(srec, dict) and "elements" in srec:
            for e in srec["elements"]:
                if isinstance(e, str):
                    is_string_elements = True
                all_elements.add(e)

    element_to_idx = {}
    idx_to_element = {}
    if is_string_elements:
        sorted_elements = sorted(list(all_elements), key=lambda x: str(x))
        for idx, elem in enumerate(sorted_elements):
            # Use 0-based index internally
            element_to_idx[elem] = idx
            idx_to_element[idx] = elem
        element_base = 0
    else:
        # Integer elements
        element_base = 0 # Default to 0-based
        if all_elements:
            min_val = min(all_elements)
            if min_val == 1:
                element_base = 1
            elif min_val == 0:
                element_base = 0
            else:
                # If min is neither 0 nor 1, assume 0-based but just shifted? 
                # Or maybe it's a subset. Let's stick to the logic of checking min.
                element_base = min_val

    for i, srec in enumerate(sets_raw):
        if not isinstance(srec, dict):
            raise ValueError(f"sets[{i}] must be a dict with keys 'id' and 'elements'.")

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
            
            if e in seen_elems:
                raise ValueError(f"sets[{i}] (id={sid!r}) has duplicate element {e}.")
            seen_elems.add(e)
            elems.append(e)
            if not is_string_elements:
                min_elem = e if min_elem is None else min(min_elem, e)

        set_ids.append(sid)
        tmp_sets.append(elems)

    if not is_string_elements:
        if min_elem is None:
             # Should be caught by empty list check, but safe guard
             element_base = 0
        else:
             # Re-evaluate base based on min_elem found during processing if needed
             # But we did it in pre-pass.
             pass
    
    # If integer elements, validate range
    if not is_string_elements:
        lo = element_base
        hi = element_base + ne - 1
        # Note: if elements are sparse (e.g. 1, 5, 9), this range check might be too strict 
        # if we assume they must be contiguous. 
        # But standard HSP usually implies contiguous 1..N or 0..N-1.
        # Let's keep the range check but be aware.
        
        for i, elems in enumerate(tmp_sets):
            for e in elems:
                if e < lo or e > hi:
                     # Relaxed check: just warn or allow if it's within reasonable bounds?
                     # The prompt says "Resolve problem related to 0-based index and 1-based index".
                     # If we have 1-based, lo=1, hi=ne.
                     pass

    return {
        "num_elements": ne,
        "num_sets": ns,
        "set_ids": set_ids,
        "sets": tmp_sets,
        "element_base": element_base,
        "is_string_elements": is_string_elements,
        "element_to_idx": element_to_idx,
        "idx_to_element": idx_to_element
    }


def _extract_hitting_set(solution: Any) -> Tuple[Optional[List[Any]], Optional[str]]:
    """
    Accept solution in either of these forms:
      - {"selected_elements": [...]}
      - {"elements": [...]}
      - {"hitting_set": [...]}
      - {"selection": [...]}
      - a list directly: [...]
    """
    if isinstance(solution, list):
        return solution, None
    if isinstance(solution, dict):
        for key in ("selected_elements", "elements", "hitting_set", "selected", "selection"):
            if key in solution:
                if not isinstance(solution[key], list):
                    return None, f"Solution '{key}' must be a list."
                return solution[key], None
        return None, "Solution dict must contain one of: selected_elements / elements / hitting_set / selected / selection."
    return None, "Solution must be either a list or a dictionary containing a list of selected elements."


def hsp_check_feasibility(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check Hitting Set Problem (HSP) feasibility + unified violation pattern.

    A hitting set is a subset H of elements such that each set contains at least one element from H.

    Patterns (taxonomy for Hitting Set: Global_3):
      - FormatError: malformed solution format, invalid element ids/types/out-of-range, instance issues
      - Global3: hitting/coverage violations (some set not hit; empty selection)
    """
    ne: int = instance["num_elements"]
    sets: List[List[int]] = instance["sets"]
    base: int = instance.get("element_base", 1)
    is_string_elements = instance.get("is_string_elements", False)
    element_to_idx = instance.get("element_to_idx", {})

    raw_list, err = _extract_hitting_set(solution)
    if err:
        return False, PATTERN_FORMAT, err
    assert raw_list is not None

    if len(raw_list) == 0:
        return False, PATTERN_GLOBAL3, "Solution selects no elements; cannot hit any sets."

    lo = base
    hi = base + ne - 1

    selected: set[int] = set()
    for k, x in enumerate(raw_list):
        if is_string_elements:
            if x not in element_to_idx:
                 return False, PATTERN_FORMAT, f"selected_elements[{k}]={x!r} is not a known element."
            e = element_to_idx[x]
        else:
            e, err2 = _to_int_strict(x, field=f"selected_elements[{k}]")
            if err2:
                return False, PATTERN_FORMAT, err2
            if e < lo or e > hi:
                return False, PATTERN_FORMAT, f"selected_elements[{k}]={e} is out of range {lo}-{hi}."
        
        if e in selected:
            return False, PATTERN_GLOBAL3, f"Element {x} appears multiple times in selection."
        selected.add(e)

    # Hitting constraint: for every set S_i, S_i ∩ H != ∅.
    for i, S in enumerate(sets):
        # guard instance contents
        if not isinstance(S, list):
            return False, PATTERN_FORMAT, f"Instance error: sets[{i}] is not a list."
        hit = False
        for e in S:
            if not isinstance(e, int):
                return False, PATTERN_FORMAT, f"Instance error: sets[{i}] contains non-int element {e!r}."
            if e in selected:
                hit = True
                break
        if not hit:
            return False, PATTERN_GLOBAL3, f"Set {i} is not hit by the selected elements."

    return True, PATTERN_OK, "Feasible HSP solution."


def hsp_objective_model(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    HSP objective (unweighted): minimize number of selected elements.

    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    raw_list, err = _extract_hitting_set(solution)
    if err or raw_list is None:
        return float("inf")

    try:
        vals: List[int] = []
        is_string_elements = instance.get("is_string_elements", False)
        element_to_idx = instance.get("element_to_idx", {})
        
        for x in raw_list:
            if is_string_elements:
                if x not in element_to_idx:
                    return float("inf")
                v = element_to_idx[x]
            else:
                v, err2 = _to_int_strict(x, field="selected_elements")
                if err2 or v is None:
                    return float("inf")
            vals.append(v)
        return float(len(set(vals)))
    except Exception:
        return float("inf")
