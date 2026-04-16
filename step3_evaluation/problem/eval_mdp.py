from typing import Any, Dict, List, Tuple, Optional, Union
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL2 = "Global2"  # budgeted subset (cardinality / subset structure)

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


def mdp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Maximum Diversity Problem (MDP) instance from `instance_variant`.

    Supports:
      (A) nested: {"instance": {...}}
      (B) flat: {...}

    Expected fields:
      {
        "problem_type": "MDP" (optional),
        "num_nodes": n,
        "m": k,  # number of nodes to select (exactly k)
        "nodes": [node_id_0, ...],  # hashable ids (ints or strings like 'A')
        "distance_pairs": [{"from_id": u, "to_id": v, "distance": d_uv}, ...]
      }

    Notes:
      - We treat distances as UNDIRECTED for objective (sum over i<j).
      - If both directions are provided and they disagree (beyond EPS), we raise.
      - If only one direction is provided for a pair, we use it.
      - Distances for pairs among nodes must be provided (either direction); otherwise we raise.

    Return (normalized):
      {
        "num_nodes": int,
        "m": int,
        "nodes": List[Any],
        "dist": List[List[float]]  # n x n, symmetric, zeros on diagonal
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for MDP.")

    instance = inst_var.get("instance", None)
    if instance is None:
        instance = inst_var
    if not isinstance(instance, dict) or not instance:
        raise ValueError("MDP instance_variant has no usable instance dictionary.")

    n_raw = instance.get("num_nodes", None)
    m_raw = instance.get("m", None)
    nodes = instance.get("nodes", None)
    pairs = instance.get("distance_pairs", None)

    if n_raw is None:
        raise ValueError("MDP instance missing 'num_nodes'.")
    if m_raw is None:
        raise ValueError("MDP instance missing 'm'.")
    if nodes is None:
        raise ValueError("MDP instance missing 'nodes'.")
    if pairs is None:
        raise ValueError("MDP instance missing 'distance_pairs'.")

    n, err = _to_int_strict(n_raw, field="num_nodes")
    if err:
        raise ValueError(f"Invalid num_nodes: {err}")
    k, err = _to_int_strict(m_raw, field="m")
    if err:
        raise ValueError(f"Invalid m: {err}")
    if n <= 0:
        raise ValueError(f"num_nodes must be positive, got {n}.")
    if k <= 0:
        raise ValueError(f"m must be positive, got {k}.")
    if k > n:
        raise ValueError(f"m={k} cannot exceed num_nodes={n}.")

    if not isinstance(nodes, list) or len(nodes) != n:
        raise ValueError(f"'nodes' must be a list of length num_nodes={n}.")

    node_ids: List[Any] = []
    seen = set()
    for i, nid in enumerate(nodes):
        try:
            hash(nid)
        except Exception:
            raise ValueError(f"nodes[{i}] is unhashable: {nid!r}.")
        if nid in seen:
            raise ValueError(f"Duplicate node id {nid!r} in nodes.")
        seen.add(nid)
        node_ids.append(nid)

    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    if not isinstance(pairs, list) or len(pairs) == 0:
        raise ValueError("'distance_pairs' must be a non-empty list.")

    directed: Dict[Tuple[int, int], float] = {}
    for t, rec in enumerate(pairs):
        if not isinstance(rec, dict):
            raise ValueError(f"distance_pairs[{t}] must be a dict.")
        u = rec.get("from_id", None)
        v = rec.get("to_id", None)
        if u not in node_to_idx or v not in node_to_idx:
            raise ValueError(f"distance_pairs[{t}] has unknown from_id/to_id: ({u!r},{v!r}).")
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        d, err = _to_float_finite(rec.get("distance", None), field=f"distance_pairs[{t}].distance")
        if err:
            raise ValueError(err)
        if d < -EPS:
            raise ValueError(f"distance_pairs[{t}].distance is negative: {d}.")
        key = (ui, vi)
        if key in directed:
            if abs(directed[key] - float(d)) > EPS:
                raise ValueError(f"Duplicate directed distance for ({u!r}->{v!r}) with conflicting values.")
        directed[key] = float(d)

    dist: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    missing_undirected: List[Tuple[Any, Any]] = []

    for i in range(n):
        for j in range(i + 1, n):
            dij = directed.get((i, j), None)
            dji = directed.get((j, i), None)
            if dij is None and dji is None:
                missing_undirected.append((node_ids[i], node_ids[j]))
                continue
            if dij is None:
                d = float(dji)  # type: ignore[arg-type]
            elif dji is None:
                d = float(dij)
            else:
                if abs(float(dij) - float(dji)) > EPS:
                    raise ValueError(
                        f"Asymmetric distances for pair ({node_ids[i]!r},{node_ids[j]!r}): "
                        f"d(i->j)={dij}, d(j->i)={dji}."
                    )
                d = float(dij)
            dist[i][j] = d
            dist[j][i] = d

    if missing_undirected:
        sample = missing_undirected[:10]
        raise ValueError(
            f"Missing distances for {len(missing_undirected)} undirected pairs. Sample: {sample}"
        )

    return {
        "num_nodes": n,
        "m": k,
        "nodes": node_ids,
        "dist": dist,
    }


def _extract_selected_nodes(solution: Any) -> Tuple[Optional[Any], Optional[str]]:
    """
    Accept solution formats:
      - list of node ids: [0,1,2,4,7] or ['A','C','E']
      - dict wrapper: {"selected_nodes":[...]}, {"nodes":[...]}, {"selection":[...]}, {"chosen_nodes":[...]}
      - binary vector: {"x":[0/1,...]} or {"selected":[0/1,...]} or {"selected_vector":[...]}
    """
    if isinstance(solution, list):
        return solution, None

    if isinstance(solution, dict):
        for key in ("selected_nodes", "nodes", "selection", "chosen_nodes"):
            if key in solution:
                return solution[key], None
        for key in ("x", "selected_vector", "selected"):
            if key in solution:
                return solution[key], None
        return None, "Solution dict must contain a selected node list or a binary vector."

    return None, "Solution must be a list or a dict."


def _normalize_selected_nodes(raw: Any, instance: Dict[str, Any]) -> Tuple[Optional[List[int]], Optional[str], str]:
    """
    Normalize selection into a list of node indices (0..n-1), unique, length m.

    Interpretation rules:
      - If raw is a list of length n and all entries are 0/1-like:
           * interpret as binary vector ONLY IF it selects exactly m nodes
           * otherwise fall back to interpreting as ids/indices (avoids false rejection).
      - Otherwise treat entries as node IDs / indices:
           * if entry matches a node id directly => use that node
           * else if int-like:
                - if int value matches a node id (when node ids are ints) => use that node
                - else if 0 <= value < n => treat as positional index
                - else if 1 <= value <= n and (value-1) valid => allow 1-based positional index
    """
    # ---- instance fields ----
    if "num_nodes" not in instance or "nodes" not in instance:
        return None, "Instance error: missing 'num_nodes' or 'nodes'.", PATTERN_FORMAT
    if "m" not in instance:
        return None, "Instance error: missing required subset size 'm'.", PATTERN_FORMAT

    n_raw = instance["num_nodes"]
    m_raw = instance["m"]
    nodes = instance["nodes"]

    n, err = _to_int_strict(n_raw, field="num_nodes")
    if err:
        return None, f"Instance error: {err}", PATTERN_FORMAT
    m, err = _to_int_strict(m_raw, field="m")
    if err:
        return None, f"Instance error: {err}", PATTERN_FORMAT
    assert n is not None and m is not None

    if not isinstance(nodes, list):
        return None, "Instance error: 'nodes' must be a list.", PATTERN_FORMAT
    if len(nodes) != n:
        return None, f"Instance error: len(nodes)={len(nodes)} but num_nodes={n}.", PATTERN_FORMAT
    if m < 0 or m > n:
        return None, f"Instance error: m={m} must be in [0, {n}].", PATTERN_FORMAT

    node_to_idx = {nid: i for i, nid in enumerate(nodes)}

    if not isinstance(raw, list):
        return None, "Selected nodes must be a list (or vector).", PATTERN_FORMAT

    if len(raw) == 0:
        return None, "Selected nodes list is empty.", PATTERN_GLOBAL2

    # ---- Try binary-vector interpretation, but ONLY accept if it matches m ----
    if len(raw) == n:
        bin_vals: List[int] = []
        all_binary = True
        for i, x in enumerate(raw):
            v, e = _to_int_strict(x, field=f"x[{i}]")
            if e:
                all_binary = False
                break
            assert v is not None
            if v not in (0, 1):
                all_binary = False
                break
            bin_vals.append(v)

        if all_binary:
            sel_bin = [i for i, v in enumerate(bin_vals) if v == 1]
            if len(sel_bin) == m:
                # accept as binary vector
                return sel_bin, None, PATTERN_OK
            # else: DO NOT error; fall through to interpret as IDs/indices

    # ---- Interpret as explicit list of node IDs/indices ----
    sel_idx: List[int] = []
    seen: set[int] = set()

    for t, x in enumerate(raw):
        idx: Optional[int] = None

        # Direct id match
        if x in node_to_idx:
            idx = node_to_idx[x]
        else:
            # Try int-like
            v, e = _to_int_strict(x, field=f"selected_nodes[{t}]")
            if e is None and v is not None:
                # If node IDs are integers and v is a node id
                if v in node_to_idx:
                    idx = node_to_idx[v]
                # 0-based positional index
                elif 0 <= v < n:
                    idx = v
                # 1-based positional index (common ambiguity)
                elif 1 <= v <= n:
                    idx = v - 1

        if idx is None:
            return None, f"Unknown node {x!r} in selection.", PATTERN_FORMAT

        if idx in seen:
            return None, f"Duplicate node {nodes[idx]!r} in selection.", PATTERN_GLOBAL2

        seen.add(idx)
        sel_idx.append(idx)

    if len(sel_idx) != m:
        return None, f"Selection has {len(sel_idx)} nodes, expected m={m}.", PATTERN_GLOBAL2

    return sel_idx, None, PATTERN_OK


def mdp_check_feasibility(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Feasibility for MDP (taxonomy: Global2):
      1) select exactly m distinct nodes from the instance's nodes list.

    Returns:
      (feasible, pattern, message) where pattern in {OK, FormatError, Global2}.
    """
    raw, err = _extract_selected_nodes(solution)
    if err:
        return False, PATTERN_FORMAT, err
    assert raw is not None

    sel, err, pat = _normalize_selected_nodes(raw, instance)
    if err:
        return False, pat, err
    assert sel is not None

    return True, PATTERN_OK, "Feasible MDP solution."



def mdp_objective_model(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    MDP objective: maximize sum of pairwise distances among selected nodes:
        sum_{i<j, i,j in S} d(i,j)

    If solution is not well-formed or infeasible, return +inf (never raise).
    """
    try:
        n_raw = instance.get("num_nodes", None)
        k_raw = instance.get("m", None)
        n, err = _to_int_strict(n_raw, field="num_nodes")
        if err or n is None or n <= 0:
            return float("inf")
        k, err = _to_int_strict(k_raw, field="m")
        if err or k is None or k <= 0:
            return float("inf")

        dist = instance.get("dist", None)
        nodes = instance.get("nodes", None)
        if not isinstance(nodes, list) or len(nodes) != n:
            return float("inf")
        if not isinstance(dist, list) or len(dist) != n:
            return float("inf")
        for r in dist:
            if not isinstance(r, list) or len(r) != n:
                return float("inf")

        raw, err = _extract_selected_nodes(solution)
        if err or raw is None:
            return float("inf")

        # NEW signature: (sel, err, pat)
        sel, err, _pat = _normalize_selected_nodes(raw, instance)
        if err or sel is None or len(sel) != k:
            return float("inf")

        total = 0.0
        # sum over i<j
        for a in range(k):
            i = sel[a]
            row = dist[i]
            for b in range(a + 1, k):
                j = sel[b]
                total += float(row[j])

        # minimization convention
        return float(total)

    except Exception:
        return float("inf")
