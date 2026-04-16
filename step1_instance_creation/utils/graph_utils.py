import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from collections import Counter
import networkx as nx
from scipy.io import mmread


def is_star_graph(G: nx.Graph) -> bool:

    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 2:
        return False

    if m != n - 1:
        return False

    degs = dict(G.degree())
    center_nodes = [v for v, d in degs.items() if d == n - 1]
    leaf_nodes = [v for v, d in degs.items() if d == 1]

    if len(center_nodes) == 1 and len(leaf_nodes) == n - 1:
        return True
    return False


def draw_graph_instance(inst: Dict[str, Any], save_path: str, title: str = ""):

    G = nx.Graph()
    for e in inst["edges"]:
        G.add_edge(e["u"], e["v"])

    pos = nx.spring_layout(G, seed=42, k=0.4)

    plt.figure(figsize=(6, 6))
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
        plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def draw_graph_solution(inst_wrap: Dict[str, Any], save_path: str, title: str = ""):
    inst = inst_wrap["instance"]
    sol = inst_wrap["solution"]
    problem = inst_wrap["problem_type"]

    G = nx.Graph()
    for e in inst["edges"]:
        G.add_edge(e["u"], e["v"])

    pos = nx.spring_layout(G, seed=42, k=0.4)

    node_colors = {v: "lightgray" for v in G.nodes()}
    node_sizes = {v: 300 for v in G.nodes()}

    # ---------- MIS ----------
    # solution: [v1, v2, ...]
    if problem == "MIS":
        S = set(sol)
        for v in S:
            if v in node_colors:
                node_colors[v] = "limegreen"

    # ---------- MVC ----------
    # solution: [v1, v2, ...]
    elif problem == "MVC":
        S = set(sol)
        for v in S:
            if v in node_colors:
                node_colors[v] = "gold"

    # ---------- MCP ----------
    # solution: [v1, v2, ...]
    elif problem == "MCP":
        S = set(sol)
        for v in S:
            if v in node_colors:
                node_colors[v] = "cornflowerblue"

    # ---------- MAXCUT ----------
    # solution: [[group1_vertices], [group2_vertices]]
    elif problem == "MAXCUT":
        groups = list(sol) if sol is not None else []
        A = set(groups[0]) if len(groups) > 0 else set()
        B = set(groups[1]) if len(groups) > 1 else set()

        for v in A:
            if v in node_colors:
                node_colors[v] = "lightcoral"
        for v in B:
            if v in node_colors:
                node_colors[v] = "skyblue"

    # ---------- GCP ----------
    # solution: [[vertices_with_color_0], [vertices_with_color_1], ...]
    elif problem == "GCP":
        cmap = list(mcolors.TABLEAU_COLORS.values())
        if sol:
            for color_idx, group in enumerate(sol):
                for v in group:
                    if v in node_colors:
                        node_colors[v] = cmap[color_idx % len(cmap)]

    # ---------- MDS ----------
    # solution: [v1, v2, ...]
    elif problem == "MDS":
        S = set(sol)
        for v in S:
            if v in node_colors:
                node_colors[v] = "orchid"

    # ---------- Draw ----------
    plt.figure(figsize=(6, 6))
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
        plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_mtx_graph(path: str) -> nx.Graph:
    mat = mmread(path).tocoo()
    G = nx.Graph()
    for u, v in zip(mat.row, mat.col):
        if u != v:
            G.add_edge(int(u), int(v))
    return G


def load_all_mtx_graphs(dir_path: str) -> Tuple[List[nx.Graph], List[str]]:

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory")

    all_mtx = sorted(p for p in dir_path.iterdir() if p.suffix == ".mtx")
    if not all_mtx:
        raise ValueError(f"No .mtx files found in {dir_path}")

    graphs: List[nx.Graph] = []
    files: List[str] = []

    print(f"[INFO] Found {len(all_mtx)} .mtx files in {dir_path}, loading once ...")
    for p in all_mtx:
        print(f"[INFO] Loading: {p}")
        try:
            G = load_mtx_graph(str(p))
        except Exception as exc:
            print(f"[WARN] Skipping invalid .mtx file {p}: {exc}")
            continue
        graphs.append(G)
        files.append(str(p))

    if not graphs:
        raise ValueError(f"No valid .mtx graphs could be loaded from {dir_path}")

    print(f"[INFO] Finished loading {len(graphs)} valid graphs.")
    return graphs, files


class RandomWalkSubgraphSampler:
    def __init__(self, steps: int = 10_000, alpha: float = 0.15, seed: int = None):
        self.steps = steps
        self.alpha = alpha
        self.rng = random.Random(seed)

    def random_walk_with_restart(self, G: nx.Graph, start_node: int) -> Counter:
        visit_count = Counter()
        current = start_node

        for _ in range(self.steps):
            visit_count[current] += 1

            if self.rng.random() < self.alpha:
                current = start_node
            else:
                neighbors = list(G.neighbors(current))
                if neighbors:
                    current = self.rng.choice(neighbors)
                else:
                    current = start_node

        return visit_count

    def sample_nodes(self, G: nx.Graph, start_node: int, num_nodes: int) -> Set[int]:
        visit_count = self.random_walk_with_restart(G, start_node)
        nodes_sorted = [n for n, _ in visit_count.most_common()]
        return set(nodes_sorted[:num_nodes])

    def sample_subgraph(self, G: nx.Graph, start_node: int, num_nodes: int) -> nx.Graph:
        nodes = self.sample_nodes(G, start_node, num_nodes)
        return G.subgraph(nodes).copy()
