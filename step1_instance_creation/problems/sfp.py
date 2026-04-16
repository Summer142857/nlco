"""
SFP (Steiner Forest Problem) instance extractor.
"""
import json
import random
from pathlib import Path
from typing import Tuple, Set, Union, Optional, Dict, Any, List
from abc import abstractmethod
from collections import Counter

import networkx as nx

from ..utils.base import InstanceExtractor
from ..solvers.sfp_solver import solve_steiner_forest
from ..solvers.stp_solver import solve_stp_scipjack
from ..solvers.stp_solver_gurobi import solve_stp_gurobi
from ..solvers.graph_draw import draw_instance_graph, draw_solution_graph
from ..solvers.kmst_solver import solve_kmst_scf


def get_random_stp_file(dataset_path: str) -> str:
    dataset_path = Path(dataset_path)
    if dataset_path.is_file():
        return str(dataset_path)
    elif dataset_path.is_dir():
        files = list(dataset_path.glob("*.stp"))
        if not files:
            raise ValueError(f"No .stp files found in {dataset_path}")
        selected = random.choice(files)
        return str(selected)
    else:
        raise ValueError(f"Invalid dataset path: {dataset_path}")


def read_stp(path: str) -> Tuple[nx.Graph, Set[int]]:
    G = nx.Graph()
    terminals: Set[int] = set()
    section = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue

            if line.startswith("SECTION") | line.startswith("Section"):
                if "Graph" in line:
                    section = "graph"
                elif "Terminals" in line:
                    section = "terminals"
                else:
                    section = None
                continue

            if line.startswith("END"):
                section = None
                continue

            if section == "graph":
                if line.startswith("Nodes") or line.startswith("Edges"):
                    continue
                if line.startswith("E"):
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    _, u, v, w = parts[:4]
                    u, v, w = int(u), int(v), int(w)
                    G.add_edge(u, v, weight=w)

            elif section == "terminals":
                if line.startswith("Terminals"):
                    continue
                if line.startswith("T"):
                    _, t = line.split()
                    terminals.add(int(t))

    return G, terminals


def random_walk_with_restart(
    G: nx.Graph,
    start_node: int,
    steps: int = 10_000,
    alpha: float = 0.15,
    rng: random.Random = None,
) -> Counter:
    if rng is None:
        rng = random.Random()

    visit_count = Counter()
    current = start_node

    for _ in range(steps):
        visit_count[current] += 1
        if rng.random() < alpha:
            current = start_node
        else:
            neighbors = list(G.neighbors(current))
            if not neighbors:
                current = start_node
            else:
                current = rng.choice(neighbors)

    return visit_count


def multi_seed_rwr(
    G: nx.Graph,
    seed_nodes: List[int],
    steps_per_seed: int = 5_000,
    alpha: float = 0.15,
    rng: random.Random = None,
) -> Counter:
    if rng is None:
        rng = random.Random()

    total_counts = Counter()
    for s in seed_nodes:
        c = random_walk_with_restart(G, s, steps=steps_per_seed, alpha=alpha, rng=rng)
        total_counts.update(c)
    return total_counts


class InstanceGenerator:

    def __init__(self, extractor: InstanceExtractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes: Union[int, Tuple[int, int]], n_instances: int):
        results = []
        for _ in range(n_instances):
            if isinstance(n_nodes, tuple):
                n = random.randint(*n_nodes)
            else:
                n = n_nodes
            try:
                inst, sol, obj = self.extractor.extract_instance(n)
                results.append(
                    {
                        "instance": inst,
                        "solution": sol,
                        "obj": obj,
                        "problem_type": self.extractor.get_problem_type(),
                    }
                )
            except Exception as e:
                print(f"[Warn] Error generating instance: {e}")
                continue
        return results

    def save_to_json(self, instances, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"[Info] Saved {len(instances)} instances to {out_path}")

    def generate_and_save(
            self,
            n_nodes,
            n_instances: int,
            out_path: str,
            oversample_factor: float = 2,
            viz_dir: str = None,
    ):
        out_path = Path(out_path)

        target = n_instances
        total_to_try = int(target * oversample_factor)

        print(f"[Info] Target instances = {target}, oversample_factor = {oversample_factor}, "
              f"try to generate {total_to_try} raw instances.")

        raw = self.generate_instances(n_nodes, total_to_try)

        if len(raw) < target:
            print(f"[Warn] Only generated {len(raw)} valid instances (< {target}). "
                  f"Will save all of them.")
            data = raw
        else:
            data = raw[:target]

        if viz_dir is not None:
            viz_dir = Path(viz_dir)
            viz_dir.mkdir(parents=True, exist_ok=True)

            for idx, inst_wrap in enumerate(data):
                inst = inst_wrap["instance"]
                sol = inst_wrap["solution"]

                inst_png = viz_dir / f"instance_{idx:04d}.png"
                sol_png = viz_dir / f"solution_{idx:04d}.png"

                draw_instance_graph(
                    inst,
                    solution=sol,
                    title=f"Instance {idx}",
                    save_path=str(inst_png),
                )

                draw_solution_graph(
                    inst,
                    sol,
                    title=f"Solution {idx}",
                    save_path=str(sol_png),
                )

                inst_wrap["instance"]["viz_instance"] = str(inst_png)
                inst_wrap["instance"]["viz_solution"] = str(sol_png)

        self.save_to_json(data, out_path)
        return data


class STPSubgraphForestInstanceExtractor(InstanceExtractor):

    def __init__(
        self,
        dataset_path: str,
        steps_per_seed: int = 5_000,
        alpha: float = 0.15,
        num_seeds: int = 3,
        rng_seed: int = 42,
        t_ratio_choices: Optional[List[float]] = None,
        min_groups: int = 2,
        max_groups: int = 4,
    ):
        self.dataset_path = dataset_path
        self.steps_per_seed = steps_per_seed
        self.alpha = alpha
        self.num_seeds = num_seeds
        self.rng = random.Random(rng_seed)
        self.min_groups = min_groups
        self.max_groups = max_groups

        if t_ratio_choices is None:
            self.t_ratio_choices = [0.1, 0.2, 0.3]
        else:
            self.t_ratio_choices = t_ratio_choices

    def get_problem_type(self) -> str:
        return "SFP"

    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], Any, float]:
        stp_path = get_random_stp_file(self.dataset_path)
        G_full, T_full = read_stp(stp_path)

        if n_nodes < 2:
            raise ValueError("Steiner forest requires at least 2 nodes.")
        if n_nodes > G_full.number_of_nodes():
            raise ValueError(
                f"Requested {n_nodes} nodes, but only {G_full.number_of_nodes()} available in {stp_path}"
            )

        all_nodes = list(G_full.nodes)
        if T_full:
            seeds = self._sample_seeds_from_terminals(T_full)
        else:
            seeds = self._sample_seeds_from_nodes(all_nodes)

        visit_counts = multi_seed_rwr(
            G_full,
            seed_nodes=seeds,
            steps_per_seed=self.steps_per_seed,
            alpha=self.alpha,
            rng=self.rng,
        )

        ranked_nodes = [v for v, _ in visit_counts.most_common()]
        if len(ranked_nodes) < n_nodes:
            extra = [v for v in all_nodes if v not in visit_counts]
            self.rng.shuffle(extra)
            ranked_nodes.extend(extra)

        # Greedily select connected nodes: start from most visited, expand via neighbors.
        score = {v: rank for rank, v in enumerate(ranked_nodes)}
        start = ranked_nodes[0]
        selected = {start}
        frontier = set(G_full.neighbors(start))

        while len(selected) < n_nodes and frontier:
            best = min(frontier, key=lambda v: score.get(v, len(all_nodes)))
            selected.add(best)
            frontier.discard(best)
            for nb in G_full.neighbors(best):
                if nb not in selected:
                    frontier.add(nb)

        if len(selected) < n_nodes:
            raise RuntimeError(
                f"Could not find {n_nodes} connected nodes (only {len(selected)} reachable)."
            )

        V_sub = selected
        G_sub = G_full.subgraph(V_sub).copy()
        if not nx.is_connected(G_sub):
            print("[Info] Disconnected subgraph in SF extractor, skip this instance.")
            raise RuntimeError("Sampled subgraph is disconnected.")

        terminal_groups = self._choose_terminal_groups(V_sub)

        G_renamed, groups_renamed, node_mapping = self._relabel_groups_to_contiguous(
            G_sub, terminal_groups
        )
        T_renamed_flat = sorted({t for group in groups_renamed for t in group})

        edges_list = []
        solver_edges = []
        for u, v, data in G_renamed.edges(data=True):
            w = float(data.get("weight", 1.0))
            solver_edges.append((int(u), int(v), w))
            edges_list.append({"u": int(u), "v": int(v), "w": w})

        density = nx.density(G_renamed)

        instance_dict = {
            "problem_type": "SF",
            "num_nodes": G_renamed.number_of_nodes(),
            "num_edges": G_renamed.number_of_edges(),
            "edges": edges_list,
            "terminal_groups": [
                sorted(int(t) for t in group) for group in groups_renamed
            ],
            "terminals": T_renamed_flat,
            "num_groups": len(groups_renamed),
            "source_file": Path(stp_path).name,
            "density": float(density),
        }

        nodes = list(range(1, G_renamed.number_of_nodes() + 1))
        edges_uv = [(u, v) for (u, v, w) in solver_edges]
        costs = {(u, v): w for (u, v, w) in solver_edges}

        obj_val, chosen_edges, _ = solve_steiner_forest(
                nodes=nodes,
                edges=edges_uv,
                costs=costs,
                terminal_sets=groups_renamed,
                time_limit=60,
                verbose=False,
        )

        solution = [{"u": u, "v": v} for (u, v) in chosen_edges]
        obj = float(obj_val)

        return instance_dict, solution, obj

    def _sample_seeds_from_terminals(self, terminals: Set[int]) -> List[int]:
        T_list = list(terminals)
        if len(T_list) <= self.num_seeds:
            return T_list
        return self.rng.sample(T_list, self.num_seeds)

    def _sample_seeds_from_nodes(self, nodes: List[int]) -> List[int]:
        if len(nodes) <= self.num_seeds:
            return nodes
        return self.rng.sample(nodes, self.num_seeds)

    def _choose_terminal_groups(
            self, V_sub: Set[int]
    ) -> List[Set[int]]:
        n = len(V_sub)
        if n < 4:
            raise RuntimeError("Subgraph too small for a non-trivial Steiner forest.")

        r = self.rng.choice(self.t_ratio_choices)
        t = int(round(r * n))

        min_t_needed = 2 * self.min_groups
        if t < min_t_needed:
            t = min_t_needed
            if t > n:
                raise RuntimeError(
                    f"Cannot select enough terminals for forest: need at least {min_t_needed}, but n={n}."
                )
        if t > n:
            t = n

        V_list = list(V_sub)
        self.rng.shuffle(V_list)
        terminals = V_list[:t]

        k_max_by_terminals = t // 2
        if k_max_by_terminals < self.min_groups:
            raise RuntimeError(
                f"Not enough terminals {t} to form {self.min_groups} groups (each with >=2 terminals)."
            )

        k_upper = min(self.max_groups, k_max_by_terminals)
        k = self.rng.randint(self.min_groups, k_upper)

        groups: List[Set[int]] = [set() for _ in range(k)]
        idx = 0
        for g in range(k):
            groups[g].add(terminals[idx])
            groups[g].add(terminals[idx + 1])
            idx += 2

        for j in range(idx, t):
            g = self.rng.randrange(k)
            groups[g].add(terminals[j])

        return groups

    def _relabel_groups_to_contiguous(
        self, G_sub: nx.Graph, terminal_groups: List[Set[int]]
    ) -> Tuple[nx.Graph, List[Set[int]], Dict[int, int]]:
        old_nodes_sorted = sorted(G_sub.nodes())
        mapping = {old: new_id for new_id, old in enumerate(old_nodes_sorted, start=1)}

        G_new = nx.relabel_nodes(G_sub, mapping, copy=True)

        groups_new: List[Set[int]] = []
        for group in terminal_groups:
            groups_new.append({mapping[t] for t in group})

        return G_new, groups_new, mapping
