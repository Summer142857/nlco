from typing import Any, Dict, List, Tuple, Optional
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


EPS = 1e-9

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL1 = "Global1"  # permutation structure


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
            # allow "1" and also "1.0" but reject "1.2"
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


def _convert_edge_endpoint(raw: Any, *, field: str, node_id2idx: Optional[Dict[Any, int]], n: int) -> int:
    if node_id2idx is not None:
        if raw not in node_id2idx:
            raise ValueError(f"{field} {raw!r} is not listed among CMP nodes.")
        return node_id2idx[raw]
    val, err = _to_int_strict(raw, field=field)
    if err:
        raise ValueError(err)
    if val < 0 or val >= n:
        raise ValueError(f"{field}={val} is out of range 0-{n-1}.")
    return val


def cmp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Cutwidth Minimization Problem (CMP) instance from `instance_variant`.

    Supports:
      - legacy instances where edges are [[u, v], ...] with integers,
      - newer instances specifying nodes list (possibly str ids) and
        edges as dicts {'u': ..., 'v': ...}.

    Returns:
      {
        "name": str,
        "num_nodes": int,
        "num_edges": int,
        "edges": [[u, v], ...],  # ints in 0..n-1
        "node_id2idx": mapping (or None)
      }
    """
    instance = parse_json_field(row["instance_variant"])
    if instance is None:
        raise ValueError("Failed to parse 'instance_variant' for CMP.")

    name = instance.get("name", "")
    num_nodes_raw = instance.get("num_nodes", None)
    num_edges_raw = instance.get("num_edges", None)
    edges_raw = instance.get("edges", None)
    nodes_raw = instance.get("nodes", None)

    if num_nodes_raw is None:
        raise ValueError("CMP instance missing 'num_nodes'.")
    if edges_raw is None:
        raise ValueError("CMP instance missing 'edges'.")
    if not isinstance(edges_raw, list):
        raise ValueError("CMP instance 'edges' must be a list.")

    n, err = _to_int_strict(num_nodes_raw, field="num_nodes")
    if err:
        raise ValueError(f"Invalid num_nodes: {err}")
    if n <= 0:
        raise ValueError(f"Invalid num_nodes: must be positive, got {n}.")

    node_id2idx: Optional[Dict[Any, int]] = None
    if nodes_raw is not None:
        if not isinstance(nodes_raw, list):
            raise ValueError("CMP instance 'nodes' must be a list.")
        if len(nodes_raw) != n:
            raise ValueError(f"nodes list has length {len(nodes_raw)}, expected num_nodes={n}.")
        node_id2idx = {}
        for idx, node_id in enumerate(nodes_raw):
            if node_id in node_id2idx:
                raise ValueError(f"Duplicate node id '{node_id}' in CMP 'nodes'.")
            node_id2idx[node_id] = idx

    edges: List[List[int]] = []
    for idx, edge_raw in enumerate(edges_raw):
        if isinstance(edge_raw, (list, tuple)) and len(edge_raw) == 2:
            u = _convert_edge_endpoint(edge_raw[0], field=f"edges[{idx}][0]", node_id2idx=node_id2idx, n=n)
            v = _convert_edge_endpoint(edge_raw[1], field=f"edges[{idx}][1]", node_id2idx=node_id2idx, n=n)
        elif isinstance(edge_raw, dict):
            if "u" not in edge_raw or "v" not in edge_raw:
                raise ValueError(f"Edge {idx} missing 'u' or 'v'.")
            u = _convert_edge_endpoint(edge_raw["u"], field=f"edges[{idx}]['u']", node_id2idx=node_id2idx, n=n)
            v = _convert_edge_endpoint(edge_raw["v"], field=f"edges[{idx}]['v']", node_id2idx=node_id2idx, n=n)
        else:
            raise ValueError(f"Edge {idx} must be a 2-element list or a dict with 'u'/'v'.")
        if u == v:
            raise ValueError(f"Edge {idx} is a self-loop on node {u}, which CMP forbids.")
        edges.append([u, v])

    if num_edges_raw is not None:
        m, err = _to_int_strict(num_edges_raw, field="num_edges")
        if err:
            raise ValueError(f"Invalid num_edges: {err}")
        if m != len(edges):
            raise ValueError(f"Reported num_edges={m} differs from actual count {len(edges)}.")

    return {
        "name": name,
        "num_nodes": n,
        "num_edges": len(edges),
        "edges": edges,
        "node_id2idx": node_id2idx,
    }


def _map_solution_element(
    raw: Any,
    *,
    pos: int,
    node_id2idx: Optional[Dict[Any, int]],
) -> Tuple[Optional[int], Optional[str]]:
    if node_id2idx is not None and raw in node_id2idx:
        return node_id2idx[raw], None
    mapped, err = _to_int_strict(raw, field=f"solution[{pos}]")
    if err:
        return None, err
    return mapped, None




def cmp_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check CMP feasibility + unified violation pattern.

    Patterns for CMP (per taxonomy Global_1):
      - FormatError: wrong container/length/types/out-of-range ids
      - Global1: not a permutation (duplicates or missing nodes)
    """
    n: int = instance["num_nodes"]

    node_id2idx = instance.get("node_id2idx")

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list (a node ordering)."

    if len(solution) != n:
        return False, PATTERN_FORMAT, f"Solution has length {len(solution)}, expected num_nodes={n}."

    seen = set()
    for pos, x in enumerate(solution):
        node, err = _map_solution_element(x, pos=pos, node_id2idx=node_id2idx)
        if err:
            return False, PATTERN_FORMAT, err
        if node is None:
            return False, PATTERN_FORMAT, f"solution[{pos}] has invalid value {x!r}."
        if node < 0 or node >= n:
            return False, PATTERN_FORMAT, f"solution[{pos}]={node} is out of range 0-{n-1}."
        if node in seen:
            return False, PATTERN_GLOBAL1, f"Node {node} appears multiple times in solution."
        seen.add(node)

    if len(seen) != n:
        missing = sorted(set(range(n)) - seen)
        return False, PATTERN_GLOBAL1, f"Solution is missing nodes {missing}."

    return True, PATTERN_OK, "Feasible CMP solution."


def cmp_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    CMP objective: minimize CUTWIDTH of the ordering.

    For an ordering pi of nodes, the cut after position k (0..n-2) splits nodes into:
      Left = {pi[0..k]}, Right = {pi[k+1..n-1]}
    The cut size is the number of edges with one endpoint in Left and the other in Right.
    Cutwidth is the maximum cut size across all k.

    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    if not isinstance(solution, list) or len(solution) == 0:
        return float("inf")

    n = instance.get("num_nodes", None)
    edges = instance.get("edges", None)
    if not isinstance(n, int) or not isinstance(edges, list):
        return float("inf")
    if len(solution) != n:
        return float("inf")

    # Build position map; if not a permutation, return inf.
    pos: Dict[int, int] = {}
    node_id2idx = instance.get("node_id2idx")
    for i, x in enumerate(solution):
        node, err = _map_solution_element(x, pos=i, node_id2idx=node_id2idx)
        if err or node is None:
            return float("inf")
        if node in pos or node < 0 or node >= n:
            return float("inf")
        pos[node] = i
    if len(pos) != n:
        return float("inf")

    # Difference-array sweep:
    # Each edge (u,v) with positions a<b crosses all cuts k where a<=k<b.
    diff = [0] * (n + 1)
    for e in edges:
        if not (isinstance(e, list) and len(e) == 2):
            return float("inf")
        u, err = _to_int_strict(e[0], field="edge_u")
        if err or u is None:
            return float("inf")
        v, err = _to_int_strict(e[1], field="edge_v")
        if err or v is None:
            return float("inf")
        if u not in pos or v not in pos:
            return float("inf")
        a = pos[u]
        b = pos[v]
        if a == b:
            continue
        if a > b:
            a, b = b, a
        diff[a] += 1
        diff[b] -= 1

    cur = 0
    best = 0
    for k in range(n - 1):
        cur += diff[k]
        if cur > best:
            best = cur

    return float(best)
