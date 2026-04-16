from typing import Any, Dict, List, Set, Tuple
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL6 = "Global6"  # connectivity (tree structure + k-node requirement)


def kmst_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse k-MST (k-Minimum Spanning Tree) instance from `instance_variant`.

    Handle node ids that may be strings or ints by remapping them to a
    contiguous 0..n-1 range (and expose the mapping).

    Expected `instance_variant` format:
      {
        "instance": {
          "problem_type": "KMST",
          "num_nodes": 15,
          "num_edges": 17,
          "edges": [
            {"u": 10, "v": 6, "w": 710.0},
            {"u": 10, "v": 11, "w": 2367.0},
            ...
          ],
          "k": 6,
          "source_file": "...",
          "density": 0.161...
        },
        "solution": [
          {"u": 3, "v": 14},
          {"u": 4, "v": 13},
          ...
        ],
        "obj": 6377.0,
        "problem_type": "KMST"
      }

    Return:
      {
        "num_nodes": number of nodes,
        "edges": list of edge dicts with "u", "v", "w",
        "k": target number of nodes to span,
        "node_id2idx": mapping from original ids to remapped ints
      }
    """
    instance = parse_json_field(row["instance_variant"])
    if instance is None:
        raise ValueError("Failed to parse 'instance_variant' for k-MST.")

    num_nodes = instance.get("num_nodes", None)
    edges = instance.get("edges", [])
    k = instance.get("k", None)

    if num_nodes is None:
        raise ValueError("k-MST instance missing 'num_nodes'.")
    if not edges:
        raise ValueError("k-MST instance has empty 'edges'.")
    if k is None:
        raise ValueError("k-MST instance missing 'k'.")

    node_sequence = []
    node_id_set = set()

    def add_node(node_id: Any) -> None:
        if node_id not in node_id_set:
            node_sequence.append(node_id)
            node_id_set.add(node_id)

    for edge in edges:
        add_node(edge["u"])
        add_node(edge["v"])

    if len(node_sequence) != int(num_nodes):
        raise ValueError(
            f"Number of unique node ids ({len(node_sequence)}) does not match reported num_nodes ({num_nodes})."
        )

    node_id2idx = {node_id: idx for idx, node_id in enumerate(node_sequence)}
    remapped_edges = [
        {"u": node_id2idx[e["u"]], "v": node_id2idx[e["v"]], "w": float(e["w"])}
        for e in edges
    ]

    return {
        "num_nodes": int(num_nodes),
        "edges": remapped_edges,
        "k": int(k),
        "node_id2idx": node_id2idx,
    }


def kmst_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check k-MST feasibility + unified violation pattern.

    Expected solution format:
      solution = [
        {"u": 3, "v": 14},
        {"u": 4, "v": 13},
        ...
      ]

    Patterns for k-MST (per taxonomy Global_6):
      - FormatError: invalid format/types, invalid node ids, edge not in instance
      - Global6: tree structure violations (cycles, disconnected, wrong number of nodes)

    Constraints:
      1) solution is a list of edge dictionaries with "u" and "v" keys.
      2) each edge must exist in the instance (either as (u,v) or (v,u)).
      3) solution edges must form a tree (connected and acyclic).
      4) the tree must span exactly k nodes.
    """
    num_nodes: int = instance["num_nodes"]
    edges: List[Dict[str, Any]] = instance["edges"]
    k: int = instance["k"]
    node_id2idx = instance.get("node_id2idx", None)

    # Create edge set for validation (both directions for undirected)
    valid_edges = set()
    for edge in edges:
        u, v = edge["u"], edge["v"]
        valid_edges.add((u, v))
        valid_edges.add((v, u))

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of edges."

    # Special case: k=0 or k=1
    if k <= 1:
        if len(solution) == 0:
            return True, PATTERN_OK, f"Feasible k-MST solution (k={k}, empty tree)."
        else:
            return False, PATTERN_GLOBAL6, f"k={k} requires 0 edges, but solution has {len(solution)} edges."

    # Empty solution
    if len(solution) == 0:
        return False, PATTERN_GLOBAL6, f"Solution is empty but k={k} nodes need to be spanned."

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
            if u_raw not in node_id2idx or v_raw not in node_id2idx:
                return False, PATTERN_FORMAT, f"Solution edge {i} has u or v not in known node ids."
            u, v = node_id2idx[u_raw], node_id2idx[v_raw]
        else:
            u, v = u_raw, v_raw

        if not isinstance(u, int) or not isinstance(v, int):
            return False, PATTERN_FORMAT, f"Solution edge {i} has non-integer node indices."

        if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
            return False, PATTERN_FORMAT, (
                f"Solution edge {i} has node index out of range: (u={u}, v={v}), "
                f"valid range is 0..{num_nodes-1}."
            )

        # Check if edge exists in instance (either direction)
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

    # Check that solution spans exactly k nodes
    num_solution_nodes = len(nodes_in_solution)
    if num_solution_nodes != k:
        return False, PATTERN_GLOBAL6, (
            f"Solution spans {num_solution_nodes} nodes, but k={k} nodes are required."
        )

    # ---- Global6: Check that solution forms a tree ----
    # Property 1: A tree with n nodes has exactly n-1 edges
    num_solution_edges = len(solution_edges)
    expected_edges = num_solution_nodes - 1
    
    if num_solution_edges != expected_edges:
        return False, PATTERN_GLOBAL6, (
            f"Solution does not form a tree: {num_solution_nodes} nodes "
            f"but {num_solution_edges} edges (should be {expected_edges})."
        )

    # Property 2: Check connectivity via BFS
    start_node = next(iter(nodes_in_solution))
    visited = set()
    queue = [start_node]
    
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)
    
    if len(visited) != num_solution_nodes:
        return False, PATTERN_GLOBAL6, (
            f"Solution edges do not form a connected tree: "
            f"only {len(visited)} of {num_solution_nodes} nodes are reachable."
        )

    return True, PATTERN_OK, "Feasible k-MST solution."


def kmst_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    k-MST objective: minimize the total weight of edges in the k-node spanning tree.

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
            if u_raw not in node_id2idx or v_raw not in node_id2idx:
                return float("inf")
            u_idx, v_idx = node_id2idx[u_raw], node_id2idx[v_raw]
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

