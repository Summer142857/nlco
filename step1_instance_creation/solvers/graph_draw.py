import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, List
from pathlib import Path


def build_graph_from_instance(instance: Dict[str, Any]) -> nx.Graph:

    G = nx.Graph()
    num_nodes = instance["num_nodes"]
    terminals = set(instance["terminals"])

    for i in range(1, num_nodes + 1):
        G.add_node(i, is_terminal=(i in terminals))

    for e in instance["edges"]:
        u, v, w = int(e["u"]), int(e["v"]), float(e["w"])
        G.add_edge(u, v, weight=w)

    return G


def draw_instance_graph(
    instance: Dict[str, Any],
    solution: List[Dict[str, int]] = None,
    title: str = "STP Instance",
    save_path: str = None,
    show: bool = False,
):

    G = build_graph_from_instance(instance)

    pos = nx.spring_layout(G, seed=0)

    terminals = set(instance["terminals"])
    node_colors = ["red" if n in terminals else "lightblue" for n in G.nodes()]

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)

    sol_edge_set = set()
    if solution is not None:
        for e in solution:
            u, v = int(e["u"]), int(e["v"])
            if G.has_edge(u, v):
                sol_edge_set.add((u, v))
                sol_edge_set.add((v, u))  # 无向图两个方向都存一下

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=list({(u, v) for (u, v) in sol_edge_set if u < v}),
            width=3,
            alpha=0.9,
        )

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400)
    nx.draw_networkx_labels(G, pos, font_size=8)

    edge_labels = {(u, v): f'{d["weight"]:.0f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


def draw_solution_graph(
    instance: Dict[str, Any],
    solution: List[Dict[str, int]],
    title: str = "Steiner Tree Solution",
    save_path: str = None,
    show: bool = False,
):

    fullG = build_graph_from_instance(instance)
    terminals = set(instance["terminals"])

    sol_edges = []
    sol_nodes = set()
    for e in solution:
        u, v = int(e["u"]), int(e["v"])
        if fullG.has_edge(u, v):
            sol_edges.append((u, v))
            sol_nodes.add(u)
            sol_nodes.add(v)

    if not sol_edges:
        print("[Warn] Solution edges empty, nothing to draw.")
        return

    G_sol = nx.Graph()
    for u, v in sol_edges:
        w = fullG[u][v]["weight"]
        G_sol.add_edge(u, v, weight=w)

    pos_full = nx.spring_layout(fullG, seed=0)
    pos = {n: pos_full[n] for n in G_sol.nodes()}

    node_colors = []
    for n in G_sol.nodes():
        if n in terminals:
            node_colors.append("red")       # terminal
        else:
            node_colors.append("lightgreen")  # Steiner

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_edges(G_sol, pos, width=3, alpha=0.9)
    nx.draw_networkx_nodes(G_sol, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(G_sol, pos, font_size=8)

    edge_labels = {(u, v): f'{d["weight"]:.0f}' for u, v, d in G_sol.edges(data=True)}
    nx.draw_networkx_edge_labels(G_sol, pos, edge_labels=edge_labels, font_size=6)

    plt.title(title)
    plt.axis("off")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()
    else:
        plt.close()
