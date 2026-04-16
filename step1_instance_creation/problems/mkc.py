"""
MkC (Maximum k-Coverage) instance extractor.
"""
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
from abc import abstractmethod

import networkx as nx
import matplotlib.pyplot as plt

from ..utils.base import InstanceExtractor, InstanceGenerator
from ..utils.graph_utils import load_all_mtx_graphs, RandomWalkSubgraphSampler

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("[WARN] gurobipy not found; instances will be generated without solving.")


def build_khop_set_system(
    G: nx.Graph,
    k_hop: int,
    rng: random.Random,
    keep_prob: float = 1.0,
    min_set_size: int = 2,
    max_set_size: int = None,
    remove_duplicate_sets: bool = True,
) -> Dict[str, Any] | None:
    num_elements = G.number_of_nodes()
    if num_elements == 0:
        print("[DEBUG][build_khop_set_system] Empty subgraph, drop instance.")
        return None

    node_list = sorted(G.nodes())
    node_to_idx = {v: v for v in node_list}

    set_list: List[List[int]] = []
    num_candidates = 0
    num_after_thin = 0
    num_after_size_filter = 0

    for u in node_list:
        layers = nx.single_source_shortest_path_length(G, u, cutoff=k_hop)
        ball_nodes = sorted(node_to_idx[v] for v, dist in layers.items() if dist <= k_hop)
        num_candidates += 1
        _raw_size = len(ball_nodes)

        if keep_prob < 1.0:
            ball_nodes = [x for x in ball_nodes if rng.random() < keep_prob]
            if not ball_nodes:
                ball_nodes = [node_to_idx[u]]
        num_after_thin += 1
        _thin_size = len(ball_nodes)

        if min_set_size is not None and len(ball_nodes) < min_set_size:
            continue
        if max_set_size is not None and len(ball_nodes) > max_set_size:
            continue

        num_after_size_filter += 1
        set_list.append(ball_nodes)

    if not set_list:
        print(
            "[DEBUG][build_khop_set_system] No sets kept.\n"
            f"  - num_elements = {num_elements}, k_hop = {k_hop}\n"
            f"  - num_candidates = {num_candidates}\n"
            f"  - keep_prob = {keep_prob}, "
            f"min_set_size = {min_set_size}, max_set_size = {max_set_size}\n"
            f"  -> after thinning & size filter, no valid sets."
        )
        return None

    before_dedup = len(set_list)

    if remove_duplicate_sets:
        unique_sets = []
        seen = set()
        for S in set_list:
            key = tuple(S)
            if key in seen:
                continue
            seen.add(key)
            unique_sets.append(S)
        set_list = unique_sets

    num_sets = len(set_list)
    if num_sets == 0:
        print(
            "[DEBUG][build_khop_set_system] All sets removed by dedup.\n"
            f"  - before_dedup = {before_dedup}, after_dedup = 0"
        )
        return None

    sets_structured = []
    element_to_sets = {i: [] for i in range(1, num_elements + 1)}

    for j, S in enumerate(set_list, start=1):
        sets_structured.append({"id": j, "elements": S})
        for e in S:
            element_to_sets[e].append(j)

    uncovered_elements = [i for i in range(1, num_elements + 1) if not element_to_sets[i]]
    if uncovered_elements:
        print(
            "[DEBUG][build_khop_set_system] Some elements are uncovered, drop instance.\n"
            f"  - num_elements = {num_elements}, num_sets = {num_sets}\n"
            f"  - num uncovered elements = {len(uncovered_elements)}\n"
            f"  - first few uncovered elements: {uncovered_elements[:10]}"
        )
        return None

    nnz = sum(len(S["elements"]) for S in sets_structured)
    density = nnz / (num_elements * num_sets)

    edges_list = [{"u": int(u), "v": int(v)} for u, v in G.edges()]

    set_system = {
        "num_elements": num_elements,
        "num_sets": num_sets,
        "density": float(density),
        "sets": sets_structured,
        "graph": {
            "num_nodes": num_elements,
            "edges": edges_list,
        },
    }

    return set_system


def draw_underlying_graph(inst_wrap: Dict[str, Any], save_path: str, title: str = ""):
    inst = inst_wrap["instance"]
    g_info = inst.get("graph", None)
    if g_info is None:
        return

    G = nx.Graph()
    for e in g_info["edges"]:
        G.add_edge(e["u"], e["v"])

    pos = nx.spring_layout(G, seed=42, k=0.4)
    plt.figure(figsize=(4, 4))
    nx.draw_networkx_edges(G, pos, edge_color="#555", width=1.0)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightgray",
        node_size=300,
        edgecolors="black",
        linewidths=0.5,
    )
    nx.draw_networkx_labels(G, pos, font_size=8)

    if title:
        plt.title(title, fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def draw_solution(inst_wrap: Dict[str, Any], save_path: str, title: str = ""):
    inst = inst_wrap["instance"]
    sol = inst_wrap.get("solution", None)
    g_info = inst.get("graph", None)
    problem_type = inst_wrap.get("problem_type")

    if g_info is None:
        return

    G = nx.Graph()
    for e in g_info["edges"]:
        G.add_edge(e["u"], e["v"])

    pos = nx.spring_layout(G, seed=42, k=0.4)

    node_colors = {v: "#D3D3D3" for v in G.nodes()}
    node_sizes = {v: 300 for v in G.nodes()}

    if sol is not None:
        chosen_ids = [c["id"] for c in sol]

        tableau_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        set_map = {S["id"]: S for S in inst.get("sets", [])}
        for idx, sid in enumerate(chosen_ids):
            S = set_map.get(sid)
            if S is None:
                continue
            elems = S["elements"]
            color = tableau_colors[idx % len(tableau_colors)]
            for v in elems:
                if v in node_colors:
                    node_colors[v] = color
                    node_sizes[v] = 400

    plt.figure(figsize=(4, 4))
    nx.draw_networkx_edges(G, pos, edge_color="#555", width=1.0)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[node_colors[v] for v in G.nodes()],
        node_size=[node_sizes[v] for v in G.nodes()],
        edgecolors="black",
        linewidths=0.5,
    )
    nx.draw_networkx_labels(G, pos, font_size=8)

    if title:
        plt.title(title, fontsize=10)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


class BaseGraphSetExtractor(InstanceExtractor):
    def __init__(
        self,
        dataset_dir: str,
        k_hop: int = 2,
        steps: int = 10_000,
        alpha: float = 0.15,
        rng_seed: int = 42,
        max_retry: int = 10,
        min_set_size: int = 2,
        max_set_size: int = None,
        remove_duplicate_sets: bool = True,
        keep_prob: float = 0.7,
    ):
        self.graphs, self.graph_files = load_all_mtx_graphs(dataset_dir)
        if not self.graphs:
            raise ValueError(f"No graphs loaded from {dataset_dir}")

        self.k_hop = k_hop
        self.steps = steps
        self.alpha = alpha
        self.rng = random.Random(rng_seed)
        self.max_retry = max_retry
        self.min_set_size = min_set_size
        self.max_set_size = max_set_size
        self.remove_duplicate_sets = remove_duplicate_sets
        self.keep_prob = keep_prob

    @abstractmethod
    def get_problem_type(self) -> str:
        ...

    @abstractmethod
    def _solve_on_set_system(self, inst_core: Dict[str, Any]):
        ...

    def _sample_connected_subgraph(self, G_full: nx.Graph, n_nodes: int) -> nx.Graph:
        if n_nodes < 2:
            raise ValueError("Need at least 2 nodes.")

        sampler = RandomWalkSubgraphSampler(
            steps=self.steps,
            alpha=self.alpha,
            seed=self.rng.randint(0, 10**9),
        )

        for _ in range(self.max_retry):
            start_node = self.rng.choice(list(G_full.nodes))
            subG = sampler.sample_subgraph(G_full, start_node, num_nodes=n_nodes)

            if subG.number_of_nodes() < n_nodes:
                print("[DEBUG] Failed: RWR subgraph too small")
                continue
            if not nx.is_connected(subG):
                print("[DEBUG] Failed: RWR subgraph not connected")
                continue

            old_nodes_sorted = sorted(subG.nodes())
            mapping = {old: new_id for new_id, old in enumerate(old_nodes_sorted, start=1)}
            G_renamed = nx.relabel_nodes(subG, mapping, copy=True)
            return G_renamed

        return None

    def extract_instance(self, n_nodes: int):
        idx = self.rng.randrange(len(self.graphs))
        G_full = self.graphs[idx]
        file_used = self.graph_files[idx]

        G_sub = self._sample_connected_subgraph(G_full, n_nodes=n_nodes)
        if G_sub is None:
            return None, None, None

        set_system = build_khop_set_system(
            G_sub,
            k_hop=self.k_hop,
            rng=self.rng,
            keep_prob=self.keep_prob,
            min_set_size=self.min_set_size,
            max_set_size=self.max_set_size,
            remove_duplicate_sets=self.remove_duplicate_sets,
        )
        if set_system is None:
            return None, None, None

        inst_core = {
            "source_file": Path(file_used).name,
            "k_hop": int(self.k_hop),
            **set_system,
        }

        sol, obj = self._solve_on_set_system(inst_core)

        return inst_core, sol, obj


class GraphToMaxCovExtractor(BaseGraphSetExtractor):
    def __init__(
        self,
        dataset_dir: str,
        k_hop: int = 2,
        steps: int = 10_000,
        alpha: float = 0.15,
        rng_seed: int = 42,
        max_retry: int = 10,
        min_set_size: int = 2,
        max_set_size: int = None,
        remove_duplicate_sets: bool = True,
        keep_prob: float = 0.7,
        budget_k: int | None = None,
        budget_ratio: float = 0.2,
    ):
        super().__init__(
            dataset_dir=dataset_dir,
            k_hop=k_hop,
            steps=steps,
            alpha=alpha,
            rng_seed=rng_seed,
            max_retry=max_retry,
            min_set_size=min_set_size,
            max_set_size=max_set_size,
            remove_duplicate_sets=remove_duplicate_sets,
            keep_prob=keep_prob,
        )
        self.budget_k = budget_k
        self.budget_ratio = budget_ratio

    def get_problem_type(self) -> str:
        return "MkC"

    def _compute_budget(self, num_sets: int) -> int:
        if self.budget_k is not None:
            return max(1, min(self.budget_k, num_sets))
        k = max(1, int(self.budget_ratio * num_sets))
        k = min(k, num_sets)
        return k

    def _solve_on_set_system(self, inst_core: Dict[str, Any]):
        if not HAS_GUROBI:
            print("[WARN] gurobipy not found; MkC solved as empty solution.")
            return [], 0.0

        M = inst_core["num_elements"]
        N = inst_core["num_sets"]
        sets = inst_core["sets"]

        k = self._compute_budget(N)
        inst_core["budget_k"] = k
        inst_core["budget_ratio"] = self.budget_ratio
        inst_core["budget_mode"] = "fixed" if self.budget_k is not None else "ratio"

        if N == 0 or k <= 0:
            return [], 0.0

        element_to_sets = {e: [] for e in range(1, M + 1)}
        for s in sets:
            sid = s["id"]
            for e in s["elements"]:
                element_to_sets[e].append(sid)

        model = gp.Model("maxcov_from_graph")
        model.Params.OutputFlag = 0

        x = model.addVars(N, vtype=GRB.BINARY, name="x")
        y = model.addVars(M, vtype=GRB.BINARY, name="y")

        for e in range(1, M + 1):
            incident = [sid - 1 for sid in element_to_sets[e]]
            idx_e = e - 1
            if incident:
                model.addConstr(
                    y[idx_e] <= gp.quicksum(x[j] for j in incident),
                    name=f"cover_link_{e}",
                )
            else:
                model.addConstr(y[idx_e] == 0, name=f"uncovered_{e}")

        model.addConstr(gp.quicksum(x[j] for j in range(N)) <= k, name="budget")

        model.setObjective(gp.quicksum(y[i] for i in range(M)), GRB.MAXIMIZE)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            print(f"[WARN] Gurobi MaxCov status {model.status}, fallback to empty.")
            return [], 0.0

        selected_sets = [int(j + 1) for j in range(N) if x[j].X > 0.5]
        covered_elements = sum(1 for i in range(M) if y[i].X > 0.5)
        solution = [{"id": sid} for sid in selected_sets]

        return solution, float(covered_elements)
