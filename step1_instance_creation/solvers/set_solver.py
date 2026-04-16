

import random
from typing import Dict, Any, List, Tuple, Set, Union
from typing import Set, Tuple, Optional
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("[WARN] gurobipy not found; SCP instances will be generated without solving.")

import networkx as nx
from gurobipy import Model, GRB, quicksum

# =====================
#  Gurobi solver for SCP
# =====================

def solve_scp_gurobi(instance: Dict[str, Any]):
    """
    Standard Set Covering MILP with unit costs:
      min  sum_j x_j
      s.t. for each row i: sum_{j in N(i)} x_j >= 1
      x_j ∈ {0,1}
    """


    m = instance["num_rows"]
    n = instance["num_cols"]
    rows = instance["rows"]

    model = gp.Model("scp_from_graph")
    model.Params.OutputFlag = 0  # silence

    # decision vars x[0..n-1], columns are 1..n
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    # objective: unit cost for each column → minimize number of selected columns
    obj_expr = gp.quicksum(x[j] for j in range(n))
    model.setObjective(obj_expr, GRB.MINIMIZE)

    # constraints: each row must be covered by at least one column
    for r in rows:
        i = r["i"]
        covering_cols = [j_idx - 1 for j_idx in r["columns"]]  # shift to 0-based
        if covering_cols:
            model.addConstr(
                gp.quicksum(x[j] for j in covering_cols) >= 1,
                name=f"cover_{i}",
            )

    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"[WARN] Gurobi status {model.status}, solution not optimal.")
        return None, None

    selected = [int(j + 1) for j in range(n) if x[j].X > 0.5]
    solution = [{"j": j} for j in selected]
    obj_val = model.objVal
    return solution, obj_val
