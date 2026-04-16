from typing import Any, Dict, List, Set, Tuple
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL6 = "Global6"  # connectivity (forest/tree + terminal connectivity)


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
            # Handle "1.0" -> 1 if necessary
            f = float(node_id)
            if f.is_integer():
                i = int(f)
                if i in node_id2idx:
                    return node_id2idx[i]
        except ValueError:
            pass
            
    return -1


def sfp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse SFP (Steiner Forest Problem) instance from `instance_variant`.

    Expected `instance_variant` format:
      {
        "problem_type": "SFP",
        "num_nodes": 10,
        "num_edges": 11,
        "edges": [
          {"u": 1, "v": 2, "w": 385.0},
          {"u": 1, "v": 9, "w": 1512.0},
          ...
        ],
        "terminal_groups": [
          [1, 9],
          [3, 5]
        ],
        "terminals": [1, 3, 5, 9],
        "num_groups": 2,
        ...
      }

    Return:
      {
        "num_nodes": number of nodes,
        "edges": list of edge dicts with "u", "v", "w",
        "terminal_groups": list of terminal groups, each group must be internally connected
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for SFP.")

    num_nodes = inst_var.get("num_nodes", None)
    edges = inst_var.get("edges", [])
    terminal_groups = inst_var.get("terminal_groups", [])
    terminals = inst_var.get("terminals", [])

    if num_nodes is None:
        raise ValueError("SFP instance missing 'num_nodes'.")
    if not edges:
        raise ValueError("SFP instance has empty 'edges'.")
    if not terminal_groups:
        raise ValueError("SFP instance has empty 'terminal_groups'.")

    # Build deterministic list of node identifiers encountered in the instance
    node_sequence = []
    node_id_set = set()

    def add_node(node_id: Any) -> None:
        if node_id not in node_id_set:
            node_sequence.append(node_id)
            node_id_set.add(node_id)

    for edge in edges:
        add_node(edge["u"])
        add_node(edge["v"])
    for group in terminal_groups:
        for node_id in group:
            add_node(node_id)
    for terminal in terminals:
        add_node(terminal)

    if len(node_sequence) != int(num_nodes):
        raise ValueError(
            f"Number of unique node ids ({len(node_sequence)}) does not match reported num_nodes ({num_nodes})."
        )

    node_id2idx = {node_id: idx for idx, node_id in enumerate(node_sequence)}
    remapped_edges = [
        {"u": node_id2idx[e["u"]], "v": node_id2idx[e["v"]], "w": float(e["w"])}
        for e in edges
    ]
    remapped_terminal_groups = [
        [node_id2idx[node_id] for node_id in group] for group in terminal_groups
    ]
    remapped_terminals = [node_id2idx[terminal] for terminal in terminals]

    return {
        "num_nodes": int(num_nodes),
        "edges": remapped_edges,
        "terminal_groups": remapped_terminal_groups,
        "terminals": remapped_terminals,
        "node_id2idx": node_id2idx,
    }


def sfp_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check SFP feasibility + unified violation pattern.

    Patterns for SFP (per taxonomy Global_6):
      - FormatError: invalid format/types, invalid node ids, edge not in instance
      - Global6: connectivity/forest violations (cycles, missing terminals, disconnected terminal groups)
    """
    num_nodes: int = instance["num_nodes"]
    edges: List[Dict[str, Any]] = instance["edges"]
    terminal_groups: List[List[int]] = instance["terminal_groups"]
    node_id2idx = instance.get("node_id2idx", None)

    # Build valid undirected edge set
    valid_edges = set()
    for edge in edges:
        u, v = edge["u"], edge["v"]
        valid_edges.add((u, v))
        valid_edges.add((v, u))

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of edges."

    # Empty solution is feasible only if all terminal groups are empty or singleton
    if len(solution) == 0:
        for i, group in enumerate(terminal_groups):
            if len(group) > 1:
                return False, PATTERN_GLOBAL6, (
                    f"Solution is empty but terminal group {i} has {len(group)} terminals to connect."
                )
        return True, PATTERN_OK, "Feasible SFP solution (empty forest for singleton/no terminals)."

    # Validate all edges in solution
    solution_edges: List[Tuple[int, int]] = []
    for i, edge in enumerate(solution):
        if isinstance(edge, dict):
            if "u" not in edge or "v" not in edge:
                return False, PATTERN_FORMAT, f"Solution edge {i} missing 'u' or 'v' key."
            u_raw, v_raw = edge["u"], edge["v"]
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            u_raw, v_raw = edge[0], edge[1]
        else:
            return False, PATTERN_FORMAT, f"Solution edge {i} must be a dictionary or 2-element list/tuple."

        if node_id2idx:
            u = _resolve_node_id(u_raw, node_id2idx)
            v = _resolve_node_id(v_raw, node_id2idx)
            if u == -1 or v == -1:
                return False, PATTERN_FORMAT, f"Solution edge {i} has u or v not in known node ids."
        else:
            u, v = u_raw, v_raw

        if not isinstance(u, int) or not isinstance(v, int):
            return False, PATTERN_FORMAT, f"Solution edge {i} has non-integer node indices."

        if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
            return False, PATTERN_FORMAT, (
                f"Solution edge {i} has node index out of range: (u={u}, v={v}), "
                f"valid range is 0..{num_nodes-1}."
            )

        # Edge existence in instance (undirected)
        if (u, v) not in valid_edges:
            return False, PATTERN_FORMAT, f"Solution edge ({u}, {v}) does not exist in instance."

        solution_edges.append((u, v))

    # Build adjacency list from solution edges (undirected)
    adjacency: Dict[int, List[int]] = {}
    nodes_in_solution = set()
    for u, v in solution_edges:
        nodes_in_solution.add(u)
        nodes_in_solution.add(v)
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    # ---- Global6: forest (acyclic) check via component count property ----
    # For an undirected forest: |E| = |V| - (#components)
    num_solution_nodes = len(nodes_in_solution)
    num_solution_edges = len(solution_edges)

    visited_global = set()
    num_components = 0

    for node in nodes_in_solution:
        if node in visited_global:
            continue
        num_components += 1
        queue = [node]
        while queue:
            curr = queue.pop(0)
            if curr in visited_global:
                continue
            visited_global.add(curr)
            for nbr in adjacency.get(curr, []):
                if nbr not in visited_global:
                    queue.append(nbr)

    expected_edges = num_solution_nodes - num_components
    if num_solution_edges != expected_edges:
        return False, PATTERN_GLOBAL6, (
            f"Solution does not form a forest: {num_solution_nodes} nodes, "
            f"{num_components} components, but {num_solution_edges} edges "
            f"(should be {expected_edges} for a forest)."
        )

    # ---- Global6: each terminal group must be internally connected ----
    for group_idx, group in enumerate(terminal_groups):
        if len(group) <= 1:
            continue

        # All terminals must be present in the chosen subgraph to be connectable
        for terminal in group:
            if terminal not in nodes_in_solution:
                return False, PATTERN_GLOBAL6, (
                    f"Terminal group {group_idx} has terminal {terminal} that is not in the solution."
                )

        start_terminal = group[0]
        visited = set()
        queue = [start_terminal]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for nbr in adjacency.get(node, []):
                if nbr not in visited:
                    queue.append(nbr)

        for terminal in group:
            if terminal not in visited:
                return False, PATTERN_GLOBAL6, (
                    f"Terminal group {group_idx} is not connected: terminal {terminal} "
                    f"is not reachable from terminal {start_terminal}."
                )

    return True, PATTERN_OK, "Feasible SFP solution."


def sfp_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    SFP objective: minimize the total weight of edges in the Steiner forest.

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
    node_id2idx = instance.get("node_id2idx", None)
    total_weight = 0.0
    for edge in solution:
        if isinstance(edge, dict) and "u" in edge and "v" in edge:
            u_raw, v_raw = edge["u"], edge["v"]
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            u_raw, v_raw = edge[0], edge[1]
        else:
            return float("inf")

        if node_id2idx:
            u_idx = _resolve_node_id(u_raw, node_id2idx)
            v_idx = _resolve_node_id(v_raw, node_id2idx)
            if u_idx == -1 or v_idx == -1:
                return float("inf")
        else:
            u_idx, v_idx = u_raw, v_raw

        # Look up weight (try both directions)
        if (u_idx, v_idx) in edge_weights:
            total_weight += edge_weights[(u_idx, v_idx)]
        elif (v_idx, u_idx) in edge_weights:
            total_weight += edge_weights[(v_idx, u_idx)]
        else:
            return float("inf")  # Invalid edge

    return float(total_weight)

