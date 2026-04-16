#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QSPP (Quadratic Shortest Path Problem) instance generator.

Generates Rostami-style QSPP instances (Grid1 / Grid2) and solves them with Gurobi.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# =========================
# CLI helpers
# =========================

def parse_node_range(node_str: str) -> Union[int, Tuple[int, int]]:
    """Parse '20' or '200-400' into int or (min,max)."""
    if "-" in node_str:
        a, b = map(int, node_str.split("-"))
        if a >= b:
            raise ValueError("Range must be MIN-MAX with MIN < MAX")
        if a < 1:
            raise ValueError("MIN must be >= 1")
        return (a, b)
    n = int(node_str)
    if n < 1:
        raise ValueError("n_nodes must be >= 1")
    return n


# =========================
# QSPP solver (Gurobi)
# =========================

def _reconstruct_st_path_bfs(
    source: int,
    target: int,
    selected_arcs: List[int],
    arc_endpoints: Dict[int, Tuple[int, int]],
    num_nodes: int,
) -> List[int]:
    """Find an s->t path inside selected arcs (robust even if cycles exist)."""
    adj = [[] for _ in range(num_nodes)]
    for j in selected_arcs:
        u, v = arc_endpoints[j]
        adj[u].append(v)

    from collections import deque
    q = deque([source])
    prev = {source: None}

    while q:
        u = q.popleft()
        if u == target:
            break
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)

    if target not in prev:
        raise RuntimeError("No s-t path found within selected arcs.")

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def solve_qspp(instance: Dict[str, Any]) -> Tuple[List[int], float]:
    """
    Solve QSPP:
      min x^T Q x + g^T x + c
      s.t. unit s->t flow (binary arc selection)
    """
    nodes = instance["nodes"]
    edges = instance["edges"]
    obj = instance["objective"]
    source = int(instance["source"])
    target = int(instance["target"])

    g = obj["linear"]
    Q_in = obj["quadratic"]
    c0 = float(obj.get("constant", 0.0))

    m = len(g)
    if len(edges) != m:
        raise ValueError("edges length must equal len(objective.linear)")

    arc_endpoints: Dict[int, Tuple[int, int]] = {}
    for e in edges:
        j = int(e["var_index"])
        arc_endpoints[j] = (int(e["from"]), int(e["to"]))

    model = gp.Model("QSPP")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2  # MIQP can be indefinite

    x = model.addVars(m, vtype=GRB.BINARY, name="x")

    Q = np.asarray(Q_in, dtype=float)
    if Q.shape != (m, m):
        raise ValueError(f"quadratic must be {m}x{m}, got {Q.shape}")
    Q = 0.5 * (Q + Q.T)

    quad_expr = gp.QuadExpr()
    for i in range(m):
        qii = float(Q[i, i])
        if qii != 0.0:
            quad_expr.add(qii * x[i])
        for j in range(i + 1, m):
            qij = float(Q[i, j])
            if qij != 0.0:
                quad_expr.add(2.0 * qij * x[i] * x[j])

    lin_expr = gp.LinExpr()
    for i, gi in enumerate(g):
        gi = float(gi)
        if gi != 0.0:
            lin_expr.add(gi * x[i])

    model.setObjective(quad_expr + lin_expr + c0, GRB.MINIMIZE)

    node_set = set(nodes)
    out_arcs = {v: [] for v in node_set}
    in_arcs = {v: [] for v in node_set}

    for j, (u, v) in arc_endpoints.items():
        if u not in node_set or v not in node_set:
            raise ValueError(f"Arc endpoint {u}->{v} not in node set.")
        out_arcs[u].append(j)
        in_arcs[v].append(j)

    for v in node_set:
        lhs = gp.LinExpr()
        for j in out_arcs[v]:
            lhs.add(x[j])
        for j in in_arcs[v]:
            lhs.add(-x[j])

        rhs = 1 if v == source else (-1 if v == target else 0)
        model.addConstr(lhs == rhs, name=f"flow_{v}")

    model.optimize()
    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Gurobi failed. Status={model.Status}")

    selected = [j for j in range(m) if x[j].X > 0.5]
    if not selected:
        raise RuntimeError("Empty solution (no arcs selected).")

    path = _reconstruct_st_path_bfs(
        source=source,
        target=target,
        selected_arcs=selected,
        arc_endpoints=arc_endpoints,
        num_nodes=len(nodes),
    )
    return path, float(model.ObjVal)


# =========================
# Rostami-style generators (Grid1 / Grid2)
# =========================

def _choose_factors_close_to_sqrt(N: int) -> Tuple[int, int]:
    """Return (a,b) with a*b=N minimizing |a-b|; fallback (1,N) if prime."""
    root = int(math.isqrt(N))
    for a in range(root, 0, -1):
        if N % a == 0:
            return a, N // a
    return 1, N


def build_rostami_grid1(
    k: int,
    seed: Optional[int],
    cost_low: int,
    cost_high: int,
) -> Dict[str, Any]:
    """
    Grid1: k x k nodes; arcs to RIGHT and UP.
    s = (0,0), t = (k-1,k-1).
    """
    if k < 2:
        raise ValueError("Grid1 requires k >= 2")
    rng = np.random.default_rng(seed)

    n = k * k
    nodes = list(range(n))

    def nid(r: int, c: int) -> int:
        return r * k + c

    s = nid(0, 0)
    t = nid(k - 1, k - 1)

    edges: List[Dict[str, Any]] = []
    for r in range(k):
        for c in range(k):
            u = nid(r, c)
            if c + 1 < k:      # right
                edges.append({"from": u, "to": nid(r, c + 1)})
            if r + 1 < k:      # up
                edges.append({"from": u, "to": nid(r + 1, c)})

    m = len(edges)
    for j, e in enumerate(edges):
        e["var_index"] = j

    linear = rng.integers(cost_low, cost_high + 1, size=m).astype(float)

    upper = rng.integers(cost_low, cost_high + 1, size=(m, m)).astype(float)
    Q = np.zeros((m, m), dtype=float)
    for i in range(m):
        Q[i, i] = upper[i, i]
        for j in range(i + 1, m):
            Q[i, j] = upper[i, j]
            Q[j, i] = upper[i, j]

    return {
        "name": f"Rostami_Grid1_k{k}_seed{seed}",
        "nodes": nodes,
        "edges": edges,
        "objective": {"constant": 0.0, "linear": linear.tolist(), "quadratic": Q.tolist()},
        "source": s,
        "target": t,
    }


def build_rostami_grid2(
    nr: int,
    nc: int,
    seed: Optional[int],
    cost_low: int,
    cost_high: int,
) -> Dict[str, Any]:
    """
    Grid2: nr x nc transshipment nodes + s + t.
      - s -> first column
      - last column -> t
      - internal arcs: RIGHT and DOWN
    """
    if nr < 1 or nc < 1:
        raise ValueError("Grid2 requires nr,nc >= 1")
    rng = np.random.default_rng(seed)

    n_trans = nr * nc
    s = n_trans
    t = n_trans + 1
    n = n_trans + 2

    nodes = list(range(n))

    def tid(r: int, c: int) -> int:
        return r * nc + c

    edges: List[Dict[str, Any]] = []

    for r in range(nr):  # s -> first column
        edges.append({"from": s, "to": tid(r, 0)})

    for r in range(nr):  # last col -> t
        edges.append({"from": tid(r, nc - 1), "to": t})

    for r in range(nr):
        for c in range(nc):
            u = tid(r, c)
            if c + 1 < nc:         # right
                edges.append({"from": u, "to": tid(r, c + 1)})
            if r + 1 < nr:         # down
                edges.append({"from": u, "to": tid(r + 1, c)})

    m = len(edges)
    for j, e in enumerate(edges):
        e["var_index"] = j

    linear = rng.integers(cost_low, cost_high + 1, size=m).astype(float)

    upper = rng.integers(cost_low, cost_high + 1, size=(m, m)).astype(float)
    Q = np.zeros((m, m), dtype=float)
    for i in range(m):
        Q[i, i] = upper[i, i]
        for j in range(i + 1, m):
            Q[i, j] = upper[i, j]
            Q[j, i] = upper[i, j]

    return {
        "nodes": nodes,
        "edges": edges,
        "objective": {"constant": 0.0, "linear": linear.tolist(), "quadratic": Q.tolist()},
        "source": s,
        "target": t,
    }


@dataclass
class RostamiGenConfig:
    family: str = "grid2"   # "grid1"|"grid2"|"random"
    shape: str = "square"   # for grid2: "square"|"long"|"wide"|"random"
    cost_low: int = 1
    cost_high: int = 10
    seed: Optional[int] = None


class RostamiQSPPInstanceExtractor:
    """Generate instances directly (no dataset)."""

    def __init__(self, cfg: RostamiGenConfig):
        self.cfg = cfg
        self._counter = 0
        self._choice_rng = random.Random(cfg.seed)

    def _next_seed(self) -> Optional[int]:
        if self.cfg.seed is None:
            return None
        self._counter += 1
        return int(self.cfg.seed) + self._counter

    def _choose_family(self) -> str:
        fam = self.cfg.family.lower().strip()
        if fam == "random":
            return self._choice_rng.choice(["grid1", "grid2"])
        return fam

    def _choose_grid2_shape(self) -> str:
        shape = self.cfg.shape.lower().strip()
        if shape == "random":
            return self._choice_rng.choice(["square", "long", "wide"])
        return shape

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], List[int], float]:
        fam = self._choose_family()

        if fam == "grid1":
            k = int(math.isqrt(n_nodes))
            if k * k != n_nodes:
                if self.cfg.family.lower().strip() == "random":
                    fam = "grid2"
                else:
                    raise ValueError(f"Grid1 needs n_nodes=k^2. Got {n_nodes}.")
            if fam == "grid1":
                if k < 2:
                    raise ValueError("Grid1 requires k >= 2 (n_nodes >= 4).")
                inst = build_rostami_grid1(
                    k=k,
                    seed=self._next_seed(),
                    cost_low=self.cfg.cost_low,
                    cost_high=self.cfg.cost_high,
                )

        if fam == "grid2":
            if n_nodes < 3:
                raise ValueError("Grid2 needs at least 3 nodes (nr*nc + 2).")
            N = n_nodes - 2

            shape = self._choose_grid2_shape()
            if shape == "square":
                nr, nc = _choose_factors_close_to_sqrt(N)
            elif shape == "long":
                nr, nc = (16, N // 16) if (N % 16 == 0) else (1, N)
            elif shape == "wide":
                nr, nc = (N // 16, 16) if (N % 16 == 0) else (N, 1)
            else:
                raise ValueError("shape must be one of: square, long, wide, random")

            inst = build_rostami_grid2(
                nr=nr,
                nc=nc,
                seed=self._next_seed(),
                cost_low=self.cfg.cost_low,
                cost_high=self.cfg.cost_high,
            )
        elif fam != "grid1":
            raise ValueError("family must be 'grid1', 'grid2', or 'random'")

        path, obj = solve_qspp(inst)
        return inst, path, float(obj)

    def get_problem_type(self) -> str:
        return "QSPP"


# =========================
# Batch generator + saving
# =========================

class InstanceGenerator:
    def __init__(self, extractor: RostamiQSPPInstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes: Union[int, Tuple[int, int]], n_instances: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        while len(out) < n_instances:
            try:
                if isinstance(n_nodes, tuple):
                    current_n = random.randint(n_nodes[0], n_nodes[1])
                else:
                    current_n = n_nodes

                inst, sol, obj = self.extractor.extract_instance(current_n)
                out.append({
                    "instance": inst,
                    "solution": sol,
                    "obj": float(obj),
                    "problem_type": self.extractor.get_problem_type(),
                })
                print(f"Generated {len(out)}/{n_instances} (n_nodes={len(inst['nodes'])}, obj={obj:.2f})")
            except Exception as e:
                print(f"Error: {e}")
                continue

        return out

    @staticmethod
    def save_to_json(instances: List[Dict[str, Any]], output_path: str) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(instances, f, indent=2)
        print(f"Saved {len(instances)} instances to {p}")
