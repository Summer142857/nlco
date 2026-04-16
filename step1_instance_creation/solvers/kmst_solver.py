import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Dict, Hashable, Optional


def solve_kmst_scf(
    nodes: List[Hashable],
    edges: List[Tuple[Hashable, Hashable]],
    costs: Dict[Tuple[Hashable, Hashable], float],
    k: int,
    time_limit: Optional[float] = None,
    verbose: bool = True,
    super_root: Hashable = 0,
):
    """
    Unrooted k-MST via SCF with a super-root.

    Returns:
      (obj_value, selected_edges, model, chosen_root)
    where selected_edges excludes the super-root edge.
    """

    if k < 2:
        raise ValueError("k must be at least 2.")
    if k > len(nodes):
        raise ValueError(f"k = {k} larger than number of nodes {len(nodes)}.")
    if super_root in nodes:
        raise ValueError(f"super_root={super_root} already in nodes; choose a different id.")

    # -------------------------
    # Build undirected edge set E over original graph
    # -------------------------
    E = []
    edge_index = {}
    for (u, v) in edges:
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        if e not in edge_index:
            edge_index[e] = len(E)
            E.append(e)

    # Normalize costs to undirected keys
    c = {}
    for (u, v) in E:
        if (u, v) in costs:
            c[(u, v)] = float(costs[(u, v)])
        elif (v, u) in costs:
            c[(u, v)] = float(costs[(v, u)])
        else:
            raise ValueError(f"Cost not provided for edge {(u, v)}")

    # -------------------------
    # Add super-root star edges (super_root, v) with zero cost
    # We'll use z[v] to indicate which v is chosen as the "root attachment".
    # -------------------------
    all_nodes = [super_root] + list(nodes)
    star_edges = []
    for v in nodes:
        e = (super_root, v) if super_root < v else (v, super_root)
        star_edges.append(e)
        c[e] = 0.0  # zero cost

    E_all = E + star_edges

    # -------------------------
    # Model
    # -------------------------
    model = gp.Model("kMST_SCF_unrooted")
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Decision vars
    x = model.addVars(E_all, vtype=GRB.BINARY, name="x")   # edge selected
    y = model.addVars(all_nodes, vtype=GRB.BINARY, name="y")  # node selected

    # Choose exactly one attachment from super_root to a selected node
    z = model.addVars(nodes, vtype=GRB.BINARY, name="z")  # z[v]=1 if super_root attaches to v

    # Flow vars on directed arcs for each undirected edge in E_all
    f = {}
    for (u, v) in E_all:
        f[(u, v)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"f_{u}_{v}")
        f[(v, u)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"f_{v}_{u}")

    model.update()

    # Objective: sum of costs on selected edges
    model.setObjective(gp.quicksum(c[e] * x[e] for e in E_all), GRB.MINIMIZE)

    # -------------------------
    # Node selection
    # - Select exactly k real nodes
    # - Super root must be selected but not counted
    # -------------------------
    model.addConstr(gp.quicksum(y[v] for v in nodes) == k, name="select_k_real_nodes")
    model.addConstr(y[super_root] == 1, name="super_root_selected")

    # -------------------------
    # Exactly one attachment edge from super_root
    # -------------------------
    model.addConstr(gp.quicksum(z[v] for v in nodes) == 1, name="choose_one_root")
    for v in nodes:
        # if attach to v, then v must be selected
        model.addConstr(z[v] <= y[v], name=f"attach_implies_selected_{v}")

        # link z to the star edge selection
        e = (super_root, v) if super_root < v else (v, super_root)
        model.addConstr(x[e] == z[v], name=f"star_edge_equals_z_{v}")

    # -------------------------
    # Edge-node linking for all edges: edge implies endpoints selected
    # -------------------------
    for (u, v) in E_all:
        model.addConstr(x[(u, v)] <= y[u], name=f"edge_node_link1_{u}_{v}")
        model.addConstr(x[(u, v)] <= y[v], name=f"edge_node_link2_{u}_{v}")

    # -------------------------
    # Tree edge count:
    # With super_root + k real nodes => total nodes selected = k + 1
    # A tree over (k+1) nodes has (k) edges.
    # -------------------------
    model.addConstr(gp.quicksum(x[e] for e in E_all) == k, name="tree_edges_with_super_root")

    # -------------------------
    # Flow capacity: if edge not selected, no flow
    # Total flow sent from super_root = k
    # -------------------------
    for (u, v) in E_all:
        model.addConstr(f[(u, v)] + f[(v, u)] <= k * x[(u, v)], name=f"cap_{u}_{v}")

    # -------------------------
    # Flow conservation:
    # super_root supplies k units.
    # each selected real node consumes 1 unit.
    # unselected real nodes consume 0.
    # -------------------------
    for v in all_nodes:
        inflow = gp.quicksum(f[(u, v)] for (u, w) in E_all if w == v) + \
                 gp.quicksum(f[(u, v)] for (w, u) in E_all if w == v)
        outflow = gp.quicksum(f[(v, w)] for (u, w) in E_all if u == v) + \
                  gp.quicksum(f[(v, w)] for (w, u) in E_all if u == v)

        if v == super_root:
            model.addConstr(outflow - inflow == k, name="flow_balance_super_root")
        else:
            model.addConstr(inflow - outflow == y[v], name=f"flow_balance_{v}")

    model.optimize()

    if model.SolCount == 0:
        if verbose:
            print("[k-MST] No feasible solution.")
        return None, None, model, None

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        if verbose:
            print(f"[k-MST] Model ended with status {model.Status}")
        return None, None, model, None

    if verbose:
        print(f"[k-MST] Status={model.Status}, Obj={model.ObjVal:.4f}, "
              f"Bound={model.ObjBound:.4f}, Gap={model.MIPGap:.4e}")

    # Extract chosen root (the v with z[v]=1)
    chosen_root = None
    for v in nodes:
        if z[v].X > 0.5:
            chosen_root = v
            break

    # Extract selected edges, excluding the super-root star edge
    selected_edges = []
    for (u, v) in E:
        if x[(u, v)].X > 0.5:
            selected_edges.append((u, v))

    return model.ObjVal, selected_edges, model, chosen_root
