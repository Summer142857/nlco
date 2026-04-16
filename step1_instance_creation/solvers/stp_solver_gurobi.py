from typing import List, Tuple, Iterable, Set, Optional
import gurobipy as gp
from gurobipy import GRB


def solve_stp_gurobi(
    num_nodes: int,
    edges: List[Tuple[int, int, float]],
    terminals: Iterable[int],
    time_limit: Optional[float] = 60.0,
    mip_gap: float = 0.0,
    verbose: bool = False,
    root: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    Solve Steiner Tree Problem (STP) to optimality using Gurobi.

    Parameters
    ----------
    num_nodes : int
        Nodes are assumed to be labeled 1..num_nodes.
    edges : list of (u, v, w)
        Undirected edges with positive costs (weights).
    terminals : iterable of int
        Terminal node ids subset of {1..num_nodes}. Must have size >= 2.
    time_limit : float | None
        Gurobi TimeLimit. If None, no limit.
    mip_gap : float
        MIPGap. Set 0 for proven optimal (subject to time limit).
    verbose : bool
        If True, show Gurobi log.
    root : int | None
        Root terminal for flow formulation. If None, choose the smallest terminal.

    Returns
    -------
    steiner_edges : list of (u, v)
        Chosen undirected edges.
    obj_value : float
        Optimal objective value.

    Notes
    -----
    - This is an SCF (single-commodity flow) formulation:
        Send 1 unit to each terminal (except root) from root through selected edges.
    - For guaranteed optimality, keep mip_gap=0 and ensure solver reaches OPTIMAL.
      If time_limit is hit, status may be TIME_LIMIT with a best incumbent.
    """
    T: List[int] = sorted(set(int(t) for t in terminals))
    if len(T) < 2:
        raise ValueError("STP requires at least 2 terminals.")
    if root is None:
        root = T[0]
    if root not in T:
        raise ValueError("Root must be a terminal.")

    # Demand: root supplies |T|-1, each other terminal demands 1.
    k = len(T) - 1  # total flow to send

    # Build arc list from undirected edges
    # We'll index edges by a canonical undirected key (min(u,v), max(u,v))
    undirected = []
    w = {}
    for (u, v, cost) in edges:
        u = int(u); v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        undirected.append((a, b))
        w[(a, b)] = float(cost)

    # Deduplicate edges if needed (keep minimum cost if duplicates exist)
    # This avoids double variables for same undirected edge.
    uniq = {}
    for (a, b) in undirected:
        if (a, b) not in uniq:
            uniq[(a, b)] = w[(a, b)]
        else:
            uniq[(a, b)] = min(uniq[(a, b)], w[(a, b)])
    undirected = list(uniq.keys())
    w = uniq

    arcs = []
    arc_to_undirected = {}
    for (a, b) in undirected:
        arcs.append((a, b))
        arcs.append((b, a))
        arc_to_undirected[(a, b)] = (a, b)
        arc_to_undirected[(b, a)] = (a, b)

    # Model
    m = gp.Model("STP_SCF")

    # Params
    m.Params.OutputFlag = 1 if verbose else 0

    m.Params.MIPGap = float(mip_gap)

    # Variables
    x = m.addVars(undirected, vtype=GRB.BINARY, name="x")     # select edge
    f = m.addVars(arcs, lb=0.0, ub=float(k), vtype=GRB.CONTINUOUS, name="f")  # flow

    # Objective
    m.setObjective(gp.quicksum(w[e] * x[e] for e in undirected), GRB.MINIMIZE)

    # Flow conservation
    # For each node v:
    #   in_flow - out_flow = b_v
    # where b_root = k, b_terminal(other) = -1, others = 0
    terminals_set: Set[int] = set(T)
    for v in range(1, num_nodes + 1):
        in_flow = gp.quicksum(f[u, v] for (u, vv) in arcs if vv == v)
        out_flow = gp.quicksum(f[v, u] for (vv, u) in arcs if vv == v)

        if v == root:
            b = k
        elif v in terminals_set:
            b = -1
        else:
            b = 0

        m.addConstr(in_flow - out_flow == b, name=f"flow_bal_{v}")

    # Capacity linking: f_uv <= k * x_e for e = {u,v}
    for (u, v) in arcs:
        e = arc_to_undirected[(u, v)]
        m.addConstr(f[u, v] <= float(k) * x[e], name=f"cap_{u}_{v}")

    # Optional: you can tighten with degree constraints on terminals, but not necessary.
    # Solve
    m.optimize()

    status = m.Status
    if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        # You can call m.computeIIS() for debugging, but keep solver clean here.
        raise RuntimeError("Gurobi: model infeasible (graph may not connect all terminals).")

    if status == GRB.OPTIMAL:
        pass
    elif status == GRB.TIME_LIMIT:
        # If time limit hit, but we may have an incumbent.
        if m.SolCount == 0:
            raise RuntimeError("Gurobi: time limit hit with no feasible solution.")
    else:
        # Other statuses: INTERRUPTED, NUMERIC, etc.
        if m.SolCount == 0:
            raise RuntimeError(f"Gurobi: ended with status {status} and no feasible solution.")

    # Extract solution edges
    chosen = []
    for (a, b) in undirected:
        if x[a, b].X > 0.5:
            chosen.append((a, b))

    obj_val = float(m.ObjVal)
    return chosen, obj_val
