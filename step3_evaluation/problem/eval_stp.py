from typing import Any, Dict, List, Set, Tuple
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL6 = "Global6"  # connectivity (tree)


def _resolve_node_id(node_id: Any, node_id2idx: Dict[Any, int]) -> int:
    """
    Resolve node_id to internal index using node_id2idx mapping.
    Tries exact match, then string->int or int->string conversion.
    Returns -1 if not found.
    """
    if node_id in node_id2idx:
        return node_id2idx[node_id]
    
    # Try converting int to string
    if isinstance(node_id, int):
        s = str(node_id)
        if s in node_id2idx:
            return node_id2idx[s]
            
    # Try converting string to int
    if isinstance(node_id, str):
        try:
            # Handle "1.0" -> 1 if necessary, but usually IDs are int-like
            f = float(node_id)
            if f.is_integer():
                i = int(f)
                if i in node_id2idx:
                    return node_id2idx[i]
        except ValueError:
            pass
            
    return -1


def stp_parse_instance(row: pd.Series) -> Dict[str, Any]: # PATCHED to expose node_id2idx for downstream use
    """
    Parse STP instance from `instance_variant`, mapping node ids (letters/numbers) to 0-based contiguous integer indices.

    Handles edge/terminal node id as string or int.

    Return:
        {
          "num_nodes": number of nodes,
          "edges": list of edge dicts with "u", "v", "w" (all u,v are remapped ints 0..n-1),
          "terminals": list of remapped ints
        }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for STP.")

    num_nodes = inst_var.get("num_nodes", None)
    edges = inst_var.get("edges", [])
    terminals = inst_var.get("terminals", [])

    if num_nodes is None:
        raise ValueError("STP instance missing 'num_nodes'.")
    if not edges:
        raise ValueError("STP instance has empty 'edges'.")
    if not terminals:
        raise ValueError("STP instance has empty 'terminals'.")

    # Gather all unique node ids (from edges and terminals), preserving deterministic order
    node_set = []
    node_id_set = set()
    for e in edges:
        for k in ["u", "v"]:
            node = e[k]
            if node not in node_id_set:
                node_set.append(node)
                node_id_set.add(node)
    for t in terminals:
        if t not in node_id_set:
            node_set.append(t)
            node_id_set.add(t)
    # Check num_nodes vs detected nodes
    if len(node_set) != int(num_nodes):
        raise ValueError(f"Number of unique node ids ({len(node_set)}) does not match reported num_nodes ({num_nodes}).")

    # Build mapping: node_id (str/int) -> new_id (int 0..n-1)
    node_id2idx = {node_id: idx for idx, node_id in enumerate(node_set)}

    remapped_edges = [
        {"u": node_id2idx[e["u"]], "v": node_id2idx[e["v"]], "w": float(e["w"])}
        for e in edges
    ]
    remapped_terminals = [node_id2idx[t] for t in terminals]
    return {
        "num_nodes": int(num_nodes),
        "edges": remapped_edges,
        "terminals": remapped_terminals,
        "node_id2idx": node_id2idx  # expose mapping for use in feasibility check
    }




def stp_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check STP feasibility + unified violation pattern.

    Patterns for STP (per taxonomy Global_6):
      - FormatError: invalid format/types/ids, edge not in instance
      - Global6: connectivity/tree violations (empty but needs connection, not a tree, terminals not connected)
    """
    num_nodes: int = instance["num_nodes"]
    edges: List[Dict[str, Any]] = instance["edges"]
    terminals: List[int] = instance["terminals"]

    # Create edge set for validation (both directions for undirected)
    valid_edges = set()
    for edge in edges:
        u, v = edge["u"], edge["v"]
        valid_edges.add((u, v))
        valid_edges.add((v, u))

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of edges."

    if len(solution) == 0:
        # Empty solution only valid if there's 0 or 1 terminal
        if len(terminals) <= 1:
            return True, PATTERN_OK, "Feasible STP solution (empty tree for single/no terminal)."
        return False, PATTERN_GLOBAL6, "Solution is empty but multiple terminals need to be connected."

    # Validate all edges in solution
    solution_edges: List[Tuple[int, int]] = []
    # Optional: obtain node_id2idx mapping if present. Otherwise, fallback to identity
    node_id2idx = instance.get("node_id2idx", None)
    for i, edge in enumerate(solution):
        # Accept either dict style (old) or list/tuple of length 2 (new)
        if isinstance(edge, dict):
            u, v = edge["u"], edge["v"]
        elif (isinstance(edge, (list, tuple)) and len(edge) == 2):
            u, v = edge[0], edge[1]
        else:
            return False, PATTERN_FORMAT, f"Solution edge {i} must be a dictionary or 2-element list/tuple."

        # Map node id to int index if mapping exists
        if node_id2idx:
            u_idx = _resolve_node_id(u, node_id2idx)
            v_idx = _resolve_node_id(v, node_id2idx)
            if u_idx == -1 or v_idx == -1:
                return False, PATTERN_FORMAT, f"Solution edge {i} has u or v not in known node ids."
        else:
            u_idx, v_idx = u, v

        if not isinstance(u_idx, int) or not isinstance(v_idx, int):
            return False, PATTERN_FORMAT, f"Solution edge {i} has non-integer node indices."

        if u_idx < 0 or u_idx >= num_nodes or v_idx < 0 or v_idx >= num_nodes:
            return False, PATTERN_FORMAT, (
                f"Solution edge {i} has node index out of range: (u={u_idx}, v={v_idx}), "
                f"valid range is 0..{num_nodes-1}."
            )

        # Check if edge exists in instance (either direction)
        if (u_idx, v_idx) not in valid_edges:
            return False, PATTERN_FORMAT, f"Solution edge ({u_idx}, {v_idx}) does not exist in instance."

        solution_edges.append((u_idx, v_idx))

    # Build adjacency list from solution edges (undirected)
    adjacency: Dict[int, List[int]] = {}
    nodes_in_solution = set()
    for u, v in solution_edges:
        nodes_in_solution.add(u)
        nodes_in_solution.add(v)
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    num_solution_nodes = len(nodes_in_solution)
    num_solution_edges = len(solution_edges)

    # ---- Global6: tree edge count property ----
    if num_solution_edges != num_solution_nodes - 1:
        return False, PATTERN_GLOBAL6, (
            f"Solution does not form a tree: {num_solution_nodes} nodes "
            f"but {num_solution_edges} edges (should be {num_solution_nodes - 1})."
        )

    # ---- Global6: connectivity ----
    if num_solution_nodes > 0:
        start_node = next(iter(nodes_in_solution))
        visited = set()
        stack = [start_node]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)

        if len(visited) != num_solution_nodes:
            return False, PATTERN_GLOBAL6, (
                f"Solution edges do not form a connected tree: "
                f"only {len(visited)} of {num_solution_nodes} nodes are reachable."
            )

    # ---- Global6: terminals must be connected (present in the tree) ----
    for terminal in terminals:
        if terminal not in nodes_in_solution:
            return False, PATTERN_GLOBAL6, f"Terminal node {terminal} is not connected in the solution tree."

    return True, PATTERN_OK, "Feasible STP solution."

def stp_objective_model(
    solution: List[Dict[str, int]],
    instance: Dict[str, Any],
) -> float:
    """
    STP objective: minimize the total weight of edges in the Steiner tree.

    Objective = sum(weight of each edge in solution)

    Assumes solution is feasible. If not well-formed, return inf.
    """
    if not isinstance(solution, list):
        return float("inf")

    edges: List[Dict[str, Any]] = instance["edges"]

    # Create edge weight lookup (both directions for undirected)
    edge_weights = {}
    for edge in edges:
        u, v, w = edge["u"], edge["v"], edge["w"]
        edge_weights[(u, v)] = w
        edge_weights[(v, u)] = w

    # Compute total weight
    total_weight = 0.0
    node_id2idx = instance.get("node_id2idx", None)
    for edge in solution:
        # Accept dict-style (old) or 2-element list/tuple (new)
        if isinstance(edge, dict) and "u" in edge and "v" in edge:
            u, v = edge["u"], edge["v"]
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            u, v = edge[0], edge[1]
        else:
            return float("inf")
        # Convert node id to mapped int idx if mapping is present
        if node_id2idx:
            u_idx = _resolve_node_id(u, node_id2idx)
            v_idx = _resolve_node_id(v, node_id2idx)
            if u_idx == -1 or v_idx == -1:
                return float("inf")
        else:
            u_idx, v_idx = u, v
        # Look up weight (try both directions)
        if (u_idx, v_idx) in edge_weights:
            total_weight += edge_weights[(u_idx, v_idx)]
        elif (v_idx, u_idx) in edge_weights:
            total_weight += edge_weights[(v_idx, u_idx)]
        else:
            return float("inf")  # Invalid edge

    return float(total_weight)

