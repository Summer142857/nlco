import random
from typing import Dict, Any, List, Tuple, Set, Union
from typing import Set, Tuple, Optional

import networkx as nx
from gurobipy import Model, GRB, quicksum


def solve_mis(
    G: nx.Graph,
    time_limit: Optional[float] = 60,
    mip_gap: Optional[float] = 0,
    verbose: bool = False,
) -> Tuple[Set[int], float]:
    """
    Exact MIS via Gurobi MIP.

    """
    model = Model("MIS")

    if not verbose:
        model.Params.OutputFlag = 0

    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # 1) x_v ∈ {0,1}
    x = {}
    for v in G.nodes():
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

    model.update()

    # 2) x_u + x_v <= 1
    for u, v in G.edges():
        model.addConstr(x[u] + x[v] <= 1, name=f"edge_{u}_{v}")

    # 3) maximize sum x_v
    model.setObjective(sum(x[v] for v in G.nodes()), GRB.MAXIMIZE)

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        print("Warning: MIS not optimal, skipping")
        return set(), float("nan")

    indep_set: Set[int] = set()
    for v in G.nodes():
        if x[v].X > 0.5:
            indep_set.add(v)

    obj_val = model.ObjVal
    return indep_set, float(obj_val)

def solve_mvc(G: nx.Graph) -> Tuple[Set[int], float]:

    mis_set, _ = solve_mis(G)
    vc_set = set(G.nodes) - mis_set
    return vc_set, float(len(vc_set))


def solve_mcp(
    G: nx.Graph,
    time_limit: Optional[float] = 60,
    mip_gap: Optional[float] = 0,
    verbose: bool = False,
) -> Tuple[Set[int], float]:

    model = Model("MaximumClique")

    if not verbose:
        model.Params.OutputFlag = 0

    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    nodes = list(G.nodes())
    x = {}
    for v in nodes:
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

    model.update()

    n = len(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            u = nodes[i]
            v = nodes[j]
            if not G.has_edge(u, v):
                model.addConstr(x[u] + x[v] <= 1, name=f"nonedge_{u}_{v}")

    model.setObjective(quicksum(x[v] for v in nodes), GRB.MAXIMIZE)

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Maximum clique not proven optimal. Status={model.Status}")

    clique_set: Set[int] = set()
    for v in nodes:
        if x[v].X > 0.5:
            clique_set.add(v)

    obj_val = model.ObjVal

    return clique_set, float(obj_val)

def solve_maxcut(
    G: nx.Graph,
    time_limit: Optional[float] = 60,
    mip_gap: Optional[float] = 0,
    verbose: bool = False,
) -> Tuple[Set[int], Set[int], float]:

    nodes = list(G.nodes())
    edges = list(G.edges())

    if not nodes:
        return set(), set(), 0.0

    model = Model("MaxCut")

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    #    y_v = 0 -> A, y_v = 1 -> B
    y = {}
    for v in nodes:
        y[v] = model.addVar(vtype=GRB.BINARY, name=f"y_{v}")

    x = {}
    for (u, v) in edges:
        x[(u, v)] = model.addVar(vtype=GRB.BINARY, name=f"x_{u}_{v}")

    model.update()

    for (u, v) in edges:
        # x >= y_u - y_v
        model.addConstr(x[(u, v)] >= y[u] - y[v], name=f"c1_{u}_{v}")
        # x >= y_v - y_u
        model.addConstr(x[(u, v)] >= y[v] - y[u], name=f"c2_{u}_{v}")
        # x <= y_u + y_v
        model.addConstr(x[(u, v)] <= y[u] + y[v], name=f"c3_{u}_{v}")
        # x <= 2 - (y_u + y_v)
        model.addConstr(x[(u, v)] <= 2 - (y[u] + y[v]), name=f"c4_{u}_{v}")

    model.setObjective(
        quicksum(x[(u, v)] for (u, v) in edges),
        GRB.MAXIMIZE
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"MaxCut not proven optimal. Status={model.Status}")

    A: Set[int] = set()
    B: Set[int] = set()
    for v in nodes:
        if y[v].X < 0.5:
            A.add(v)
        else:
            B.add(v)

    cut_value = model.ObjVal

    return A, B, float(cut_value)

def solve_gcp(
    G: nx.Graph,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Dict[int, int], int, float]:

    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}, 0, 0.0

    # 上界: 至多 n 种颜色 (0..K-1)
    K = n

    model = Model("GraphColoring")

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    x = {}
    for v in nodes:
        for c in range(K):
            x[(v, c)] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")

    y = {}
    for c in range(K):
        y[c] = model.addVar(vtype=GRB.BINARY, name=f"y_{c}")

    model.update()

    for v in nodes:
        model.addConstr(
            quicksum(x[(v, c)] for c in range(K)) == 1,
            name=f"assign_{v}",
        )

    for u, v in G.edges():
        for c in range(K):
            model.addConstr(
                x[(u, c)] + x[(v, c)] <= 1,
                name=f"edge_{u}_{v}_color_{c}",
            )

    for v in nodes:
        for c in range(K):
            model.addConstr(
                x[(v, c)] <= y[c],
                name=f"use_color_{v}_{c}",
            )

    model.setObjective(
        quicksum(y[c] for c in range(K)),
        GRB.MINIMIZE,
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"GCP not proven optimal. Status={model.Status}")

    coloring: Dict[int, int] = {}
    for v in nodes:
        for c in range(K):
            if x[(v, c)].X > 0.5:
                coloring[v] = c
                break

    used_colors = {c for c in range(K) if y[c].X > 0.5}
    num_colors = len(used_colors)
    obj_val = model.ObjVal

    return coloring, num_colors, float(obj_val)



def solve_mds(
    G: nx.Graph,
    time_limit: Optional[float] = None,
    mip_gap: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Set[int], float]:
    """
    Exact Minimum Dominating Set (MDS) solver using Gurobi.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return set(), 0.0

    model = Model("MinimumDominatingSet")

    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    x = {}
    for v in nodes:
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

    model.update()

    #    N[u] = {u} ∪ neighbors(u)
    for u in nodes:
        closed_neighborhood = [u] + list(G.neighbors(u))
        model.addConstr(
            quicksum(x[v] for v in closed_neighborhood) >= 1,
            name=f"dominate_{u}",
        )

    model.setObjective(
        quicksum(x[v] for v in nodes),
        GRB.MINIMIZE,
    )

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"MDS not proven optimal. Status={model.Status}")

    mds_set: Set[int] = set()
    for v in nodes:
        if x[v].X > 0.5:
            mds_set.add(v)

    obj_val = model.ObjVal

    return mds_set, float(obj_val)

