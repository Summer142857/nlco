from typing import Any, Dict, List, Tuple, Optional
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL2 = "Global2"  # budgeted subset (cardinality: choose p facilities)
PATTERN_GLOBAL4 = "Global4"  # assignment (each node assigned to a chosen facility)


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


def _resolve_node_index(
    x: Any,
    *,
    n: int,
    node_id_to_index: Dict[Any, int],
    field: str,
) -> Tuple[Optional[int], Optional[str]]:
    """Interpret x as a node index or identifier."""
    # 1. Try as integer (index or numeric ID)
    idx, err_int = _to_int_strict(x, field=field)
    if err_int is None:
        # a) Check if it matches a node ID (int or str)
        if idx in node_id_to_index:
            return node_id_to_index[idx], None
        if str(idx) in node_id_to_index:
            return node_id_to_index[str(idx)], None
        
        # b) Check if it is a valid 0-based index
        if 0 <= idx < n:
            return idx, None
            
        return None, f"{field}={idx} is out of range 0-{n-1} and not a known node ID."

    # 2. Try as string ID
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None, f"{field} is empty."
        
        # Direct string match
        if s in node_id_to_index:
            return node_id_to_index[s], None
            
        # Try converting string to int to match int IDs
        try:
            f = float(s)
            if f.is_integer():
                i = int(f)
                if i in node_id_to_index:
                    return node_id_to_index[i], None
        except ValueError:
            pass
            
        return None, f"{field}: unknown node id {s!r}."

    # 3. Try direct lookup (for other types)
    mapped = node_id_to_index.get(x)
    if mapped is not None:
        return mapped, None

    return None, err_int


def pcenter_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse p-center (PCENTER) instance from `instance_variant`.

    Expected instance format:
      {
        "num_nodes": int,
        "num_open": int,  # this is 'p'
        "sites": [
          {"id": str, "distances": {str: float, ...}},
          ...
        ],
        "objective": float (optional)
      }

    Return:
      {
        "distance_matrix": ...,
        "p": int,
        "n_nodes": int
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for PCENTER.")

    num_nodes_raw = inst_var.get("num_nodes", None)
    num_open_raw = inst_var.get("num_open", None)
    sites = inst_var.get("sites", None)

    if num_nodes_raw is None:
        raise ValueError("PCENTER instance missing 'num_nodes'.")
    if num_open_raw is None:
        raise ValueError("PCENTER instance missing 'num_open'.")
    if sites is None:
        raise ValueError("PCENTER instance missing 'sites'.")
    if not isinstance(sites, list) or len(sites) == 0:
        raise ValueError("'sites' must be a non-empty list.")

    n, err = _to_int_strict(num_nodes_raw, field="num_nodes")
    if err:
        raise ValueError(f"Invalid num_nodes: {err}")
    if n <= 0:
        raise ValueError(f"Invalid num_nodes: must be positive, got {n}.")

    p, err = _to_int_strict(num_open_raw, field="num_open")
    if err:
        raise ValueError(f"Invalid num_open: {err}")
    if p <= 0:
        raise ValueError(f"Invalid num_open: must be positive, got {p}.")
    if p > n:
        raise ValueError(f"Invalid num_open={p}: cannot exceed num_nodes={n}.")

    if len(sites) != n:
        raise ValueError(f"'sites' has {len(sites)} entries, expected num_nodes={n}.")

    # Build node name to index mapping
    node_name_to_idx: Dict[Any, int] = {}
    node_ids: List[Any] = []
    for idx, site in enumerate(sites):
        if not isinstance(site, dict):
            raise ValueError(f"Site {idx} must be a dictionary.")
        site_id = site.get("id", None)
        if site_id is None:
            raise ValueError(f"Site {idx} missing 'id'.")
        if isinstance(site_id, str):
            site_id = site_id.strip()
            if site_id == "":
                raise ValueError(f"Site {idx} 'id' is empty.")
        if site_id in node_name_to_idx:
            raise ValueError(f"Duplicate site id '{site_id}'.")
        node_name_to_idx[site_id] = idx
        node_ids.append(site_id)

    # Build distance matrix
    dist: List[List[float]] = [[0.0] * n for _ in range(n)]

    for idx, site in enumerate(sites):
        distances_dict = site.get("distances", None)
        if distances_dict is None:
            raise ValueError(f"Site {idx} missing 'distances'.")
        if not isinstance(distances_dict, dict):
            raise ValueError(f"Site {idx} 'distances' must be a dictionary.")

        site_id = site["id"]
        expected_node_names = set(node_name_to_idx.keys())

        for target_name, dist_value_raw in distances_dict.items():
            if target_name not in node_name_to_idx:
                raise ValueError(f"Site {idx} ('{site_id}') has distance to unknown node '{target_name}'.")
            target_idx = node_name_to_idx[target_name]

            dist_value, err = _to_float_finite(dist_value_raw, field=f"sites[{idx}].distances['{target_name}']")
            if err:
                raise ValueError(err)
            if dist_value < -EPS:
                raise ValueError(f"sites[{idx}].distances['{target_name}'] is negative: {dist_value}.")

            dist[idx][target_idx] = dist_value

        # Check that all nodes are present in distances dict
        got_node_names = set(distances_dict.keys())
        if got_node_names != expected_node_names:
            missing = sorted(expected_node_names - got_node_names)
            extra = sorted(got_node_names - expected_node_names)
            if missing:
                raise ValueError(f"Site {idx} ('{site_id}') missing distances to nodes: {missing}.")
            raise ValueError(f"Site {idx} ('{site_id}') has extra distances to nodes: {extra}.")

    return {
        "distance_matrix": dist,
        "p": p,
        "n_nodes": n,
        "node_ids": node_ids,
        "node_id_to_index": node_name_to_idx,
    }


def pcenter_check_feasibility(
    solution: Dict[str, Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check PCENTER feasibility + unified violation pattern.

    Patterns for p-center (taxonomy Global_{4,2}):
      - FormatError: invalid structure/types/lengths/out-of-range ids
      - Global2: violates 'choose exactly p facilities' (cardinality)
      - Global4: assignment/linking violated (assigned to non-selected facility)
    """
    n: int = instance["n_nodes"]
    p: int = instance["p"]

    if not isinstance(solution, dict):
        return False, PATTERN_FORMAT, "Solution must be a dictionary."

    fac = solution.get("selected", None)
    assign = solution.get("assignments", None)

    if not isinstance(fac, list):
        return False, PATTERN_FORMAT, "Solution 'selected' must be a list."
    if not isinstance(assign, list):
        return False, PATTERN_FORMAT, "Solution 'assignments' must be a list."

    if len(assign) != n:
        return False, PATTERN_FORMAT, f"Solution has {len(assign)} assignments, expected n_nodes={n}."

    # ---- Global2: must select exactly p facilities ----
    if len(fac) != p:
        return False, PATTERN_GLOBAL2, f"Solution has {len(fac)} facilities, expected p={p}."

    node_id_to_index: Dict[Any, int] = instance.get("node_id_to_index", {})
    node_ids: List[Any] = instance.get("node_ids", [])

    fac_set = set()
    for k, x in enumerate(fac):
        f_idx, err = _resolve_node_index(
            x,
            n=n,
            node_id_to_index=node_id_to_index,
            field=f"selected[{k}]",
        )
        if err:
            return False, PATTERN_FORMAT, err
        if f_idx is None:
            return False, PATTERN_FORMAT, f"facilities[{k}] resolved to None."

        fid_label = node_ids[f_idx]
        if f_idx in fac_set:
            return False, PATTERN_GLOBAL2, f"Facility {fid_label!r} appears multiple times in facilities."
        fac_set.add(f_idx)

    if len(fac_set) != p:
        return False, PATTERN_GLOBAL2, f"Selected facilities are not {p} unique nodes (got {len(fac_set)} unique)."

    # ---- Global4: each node assigned to a selected facility ----
    for i, x in enumerate(assign):
        a_idx, err = _resolve_node_index(
            x,
            n=n,
            node_id_to_index=node_id_to_index,
            field=f"assignments[{i}]",
        )
        if err:
            return False, PATTERN_FORMAT, err
        if a_idx is None:
            return False, PATTERN_FORMAT, f"assignments[{i}] resolved to None."

        if a_idx not in fac_set:
            aid_label = node_ids[a_idx]
            return False, PATTERN_GLOBAL4, (
                f"Node {i} assigned to {aid_label} (index {a_idx}), but {aid_label} is not a selected facility."
            )

    return True, PATTERN_OK, "Feasible PCENTER solution."


def pcenter_objective_model(
    solution: Dict[str, Any],
    instance: Dict[str, Any],
) -> float:
    """
    PCENTER objective: minimize the maximum assignment distance:
        max_i dist[i][a_i]
    where a_i is the facility serving node i.

    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    if not isinstance(solution, dict):
        return float("inf")

    dist = instance.get("distance_matrix", None)
    n = instance.get("n_nodes", None)
    p = instance.get("p", None)
    if not isinstance(dist, list) or not isinstance(n, int) or not isinstance(p, int):
        return float("inf")

    fac = solution.get("selected", None)
    assign = solution.get("assignments", None)
    if not isinstance(fac, list) or not isinstance(assign, list):
        return float("inf")
    if len(assign) != n:
        return float("inf")

    node_id_to_index: Dict[Any, int] = instance.get("node_id_to_index", {})
    fac_set = set()
    field_base = "selected"
    for k, x in enumerate(fac):
        f_idx, err = _resolve_node_index(
            x,
            n=n,
            node_id_to_index=node_id_to_index,
            field=f"{field_base}[{k}]",
        )
        if err or f_idx is None:
            return float("inf")
        if f_idx < 0 or f_idx >= n or f_idx in fac_set:
            return float("inf")
        fac_set.add(f_idx)

    worst = 0.0
    for i, x in enumerate(assign):
        a_idx, err = _resolve_node_index(
            x,
            n=n,
            node_id_to_index=node_id_to_index,
            field=f"assignments[{i}]",
        )
        if err or a_idx is None:
            return float("inf")
        if a_idx < 0 or a_idx >= n:
            return float("inf")
        if a_idx not in fac_set:
            return float("inf")
        row = dist[i]
        if not isinstance(row, list) or a_idx >= len(row):
            return float("inf")
        dij, err = _to_float_finite(row[a_idx], field="distance")
        if err:
            return float("inf")
        if dij > worst:
            worst = dij

    return float(worst)
