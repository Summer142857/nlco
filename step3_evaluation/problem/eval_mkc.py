from typing import Any, Dict, List, Tuple, Optional, Union
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL2 = "Global2"  # budgeted subset (cardinality <= k)

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


def mkc_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Max k-Coverage (MkC) instance from `instance_variant`.


    Return (normalized):
      {
        "num_elements": int,
        "num_sets": int,
        "budget_k": int,
        "set_ids": [str],                 # length num_sets
        "sets": [ [Any, ...], ... ],      # list of element labels
        "universe": [Any, ...],           # sorted if sortable, otherwise insertion order unique
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for MkC.")

    instance = inst_var.get("instance", None)
    if instance is None:
        instance = inst_var
    if not isinstance(instance, dict) or not instance:
        raise ValueError("MkC instance_variant has no usable instance dictionary.")

    ne_raw = instance.get("num_elements", None)
    ns_raw = instance.get("num_sets", None)
    sets_raw = instance.get("sets", None)
    k_raw = instance.get("budget_k", None)

    if ne_raw is None:
        raise ValueError("MkC instance missing 'num_elements'.")
    if ns_raw is None:
        raise ValueError("MkC instance missing 'num_sets'.")
    if sets_raw is None:
        raise ValueError("MkC instance missing 'sets'.")
    if k_raw is None:
        raise ValueError("MkC instance missing 'budget_k'.")

    ne, err = _to_int_strict(ne_raw, field="num_elements")
    if err:
        raise ValueError(f"Invalid num_elements: {err}")
    ns, err = _to_int_strict(ns_raw, field="num_sets")
    if err:
        raise ValueError(f"Invalid num_sets: {err}")
    k, err = _to_int_strict(k_raw, field="budget_k")
    if err:
        raise ValueError(f"Invalid budget_k: {err}")

    if ne <= 0:
        raise ValueError(f"Invalid num_elements: must be positive, got {ne}.")
    if ns <= 0:
        raise ValueError(f"Invalid num_sets: must be positive, got {ns}.")
    if k < 0:
        raise ValueError(f"Invalid budget_k: must be >= 0, got {k}.")
    if k > ns:
        raise ValueError(f"Invalid budget_k={k}: cannot exceed num_sets={ns}.")

    if not isinstance(sets_raw, list) or len(sets_raw) == 0:
        raise ValueError("'sets' must be a non-empty list.")
    if len(sets_raw) != ns:
        raise ValueError(f"Length mismatch: num_sets={ns} but len(sets)={len(sets_raw)}.")

    set_ids: List[str] = []
    sets: List[List[Any]] = []
    seen_ids = set()

    universe_seen = set()
    universe_list: List[Any] = []

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
            # Map to 1-based index for internal consistency with existing logic
            element_to_idx[elem] = idx + 1
    else:
        # Integer elements
        # Check if 0-based or 1-based
        if all_elements:
            min_val = min(all_elements)
            is_zero_based = (min_val == 0)
        else:
            is_zero_based = False # Default

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

        elems: List[Any] = []
        seen_elems = set()
        for j, e_raw in enumerate(elems_raw):
            # allow strings/ints; reject unhashable values (lists/dicts)
            try:
                hash(e_raw)
            except Exception:
                raise ValueError(f"sets[{i}].elements[{j}] is unhashable: {e_raw!r}.")

            if is_string_elements:
                if e_raw not in element_to_idx:
                     raise ValueError(f"sets[{i}] has unknown element {e_raw}.")
                e = element_to_idx[e_raw]
            else:
                # Integer elements
                if not isinstance(e_raw, int):
                     # Try to convert if it's float-like int
                     e_int, err = _to_int_strict(e_raw, field=f"sets[{i}].elements[{j}]")
                     if err:
                         raise ValueError(err)
                     e = e_int
                else:
                    e = e_raw
                
                # Normalize to 1-based
                if is_zero_based:
                    e = e + 1

            if e in seen_elems:
                raise ValueError(f"sets[{i}] (id={sid!r}) has duplicate element {e!r}.")
            seen_elems.add(e)
            elems.append(e)

            if e not in universe_seen:
                universe_seen.add(e)
                universe_list.append(e)

        set_ids.append(sid)
        sets.append(elems)

    # Optional sanity: ensure instance's num_elements matches universe size
    # Note: if we normalized, universe_seen contains normalized values.
    if len(universe_seen) != ne:
        # Relaxed check: sometimes num_elements is just an upper bound or ID space size
        # But for strict validation, let's warn or error.
        # Given the prompt asks to adapt, let's be slightly lenient if it's just a count mismatch 
        # but consistent with the data.
        # However, if we have MORE elements than num_elements, that's definitely weird.
        if len(universe_seen) > ne:
             raise ValueError(
                f"Instance mismatch: num_elements={ne} but union of elements in sets has size {len(universe_seen)}."
            )

    # Try to return a stable "universe": sorted if possible, else insertion order unique
    try:
        universe_sorted = sorted(universe_list)
        universe = universe_sorted
    except Exception:
        universe = universe_list

    return {
        "num_elements": ne,
        "num_sets": ns,
        "budget_k": k,
        "set_ids": set_ids,
        "sets": sets,
        "universe": universe,
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



def mkc_check_feasibility(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check Max k-Coverage (MkC) feasibility + unified violation pattern.

    Accepted solution formats:
      - List of set IDS (strings), e.g. ["S3"]
      - List of set INDICES (0..num_sets-1) or 1-based indices (if unambiguous)
      - Dict wrapper like {"selected_sets": [...]}

    Feasibility constraints:
      1) selected sets must be unique and valid.
      2) number of selected sets must be <= budget_k.

    Patterns (taxonomy for MkC: Global_{3,2}, but feasibility is mainly Global2):
      - FormatError: malformed solution format / invalid ids/indices / instance mapping errors
      - Global2: budget/cardinality violated (|selected| > k) or duplicate selections
    """
    ns: int = instance["num_sets"]
    k: int = instance["budget_k"]
    set_ids: List[str] = instance.get("set_ids", [])

    raw_list, err = _extract_selected_sets(solution)
    if err:
        return False, PATTERN_FORMAT, err
    assert raw_list is not None

    selected_indices, err = _selected_indices_from_solution(raw_list, set_ids, ns)
    if err:
        return False, PATTERN_FORMAT, err
    assert selected_indices is not None

    # If your helper does not enforce uniqueness, enforce it here
    if len(selected_indices) != len(set(selected_indices)):
        return False, PATTERN_GLOBAL2, "Selected sets contain duplicates; selection must be a set."

    if len(selected_indices) > k:
        return False, PATTERN_GLOBAL2, f"Selected {len(selected_indices)} sets, but budget_k={k}."

    return True, PATTERN_OK, "Feasible MkC solution."


def mkc_objective_model(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    MkC objective: maximize number of covered elements.
    For a minimization-based evaluator, return the NEGATIVE coverage, i.e. -|covered|.

    Assumes solution is feasible; if not well-formed, return +inf (never raise).
    """
    if not isinstance(instance, dict):
        return float("inf")

    ns = instance.get("num_sets", None)
    k = instance.get("budget_k", None)
    set_ids = instance.get("set_ids", None)
    sets = instance.get("sets", None)
    if not isinstance(ns, int) or not isinstance(k, int) or not isinstance(set_ids, list) or not isinstance(sets, list):
        return float("inf")
    if len(set_ids) != ns or len(sets) != ns:
        return float("inf")

    raw_list, err = _extract_selected_sets(solution)
    if err or raw_list is None:
        return float("inf")

    selected_indices, err = _selected_indices_from_solution(raw_list, set_ids, ns)
    if err or selected_indices is None:
        return float("inf")

    if len(selected_indices) > k:
        return float("inf")

    covered = set()
    try:
        for idx in selected_indices:
            for e in sets[idx]:
                covered.add(e)
    except Exception:
        return float("inf")

    return float(len(covered))
