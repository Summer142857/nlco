import gurobipy as gp
from gurobipy import GRB


def solve_steiner_forest(nodes, edges, costs, terminal_sets, time_limit=None, verbose=True):

    E = []
    edge_index = {}
    for (u, v) in edges:
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        if e not in edge_index:
            edge_index[e] = len(E)
            E.append(e)

    c = {}
    for (u, v) in E:
        if (u, v) in costs:
            c[(u, v)] = costs[(u, v)]
        elif (v, u) in costs:
            c[(u, v)] = costs[(v, u)]
        else:
            raise ValueError(f"Cost not provided for edge {(u, v)}")

    model = gp.Model("SteinerForest_MCF")
    if not verbose:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    x = model.addVars(E, vtype=GRB.BINARY, name="x")

    K = range(len(terminal_sets))
    roots = {}
    for k in K:
        T_k = list(terminal_sets[k])
        if len(T_k) == 0:
            raise ValueError(f"Terminal set {k} is empty.")
        roots[k] = T_k[0]

    f = {}
    for k in K:
        T_k = list(terminal_sets[k])
        r_k = roots[k]
        for t in T_k:
            if t == r_k:
                continue
            for (u, v) in E:
                f[(k, t, u, v)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                               name=f"f_{k}_{t}_{u}_{v}")
                f[(k, t, v, u)] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                               name=f"f_{k}_{t}_{v}_{u}")

    model.update()

    model.setObjective(gp.quicksum(c[e] * x[e] for e in E), GRB.MINIMIZE)

    for k in K:
        T_k = list(terminal_sets[k])
        r_k = roots[k]
        for t in T_k:
            if t == r_k:
                continue

            for v in nodes:
                # outflow - inflow
                outflow = gp.LinExpr()
                inflow = gp.LinExpr()

                for (i, j) in E:
                    if i == v:
                        outflow += f[(k, t, i, j)]
                        inflow  += f[(k, t, j, i)]
                    elif j == v:
                        outflow += f[(k, t, j, i)]
                        inflow  += f[(k, t, i, j)]

                if v == r_k:
                    rhs = 1.0
                elif v == t:
                    rhs = -1.0
                else:
                    rhs = 0.0

                model.addConstr(outflow - inflow == rhs,
                                name=f"flow_bal_k{k}_t{t}_v{v}")

    for (u, v) in E:
        for k in K:
            T_k = list(terminal_sets[k])
            r_k = roots[k]
            for t in T_k:
                if t == r_k:
                    continue
                model.addConstr(
                    f[(k, t, u, v)] + f[(k, t, v, u)] <= x[(u, v)],
                    name=f"cap_k{k}_t{t}_{u}_{v}"
                )

    model.update()

    model.optimize()
    if model.SolCount == 0:
        print("[k-MST] No feasible solution.")
        return None, None, model
    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        print(f"Model ended with status {model.Status}")
        return None, None, model
    print(f"[k-MST] Status={model.Status}, Obj={model.ObjVal:.4f}, "
          f"Bound={model.ObjBound:.4f}, Gap={model.MIPGap:.4e}")
    selected_edges = []
    for e in E:
        if x[e].X > 0.5:
            selected_edges.append(e)

    obj_val = model.ObjVal

    return obj_val, selected_edges, model

