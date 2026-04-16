import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union
import networkx as nx
from .graph_utils import load_all_mtx_graphs, RandomWalkSubgraphSampler, is_star_graph, draw_graph_instance, draw_graph_solution


class InstanceExtractor(ABC):
    @abstractmethod
    def extract_instance(self, n_nodes: int) -> Tuple[Dict[str, Any], Any, float]:
        ...

    @abstractmethod
    def get_problem_type(self) -> str:
        ...


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

            inst, sol, obj = self.extractor.extract_instance(n)
            if inst is None:
                continue
            results.append(
                {
                    "instance": inst,
                    "solution": sol,
                    "obj": obj,
                    "problem_type": self.extractor.get_problem_type(),
                }
            )
        if len(results) < n_instances:
            print(f"[WARN] Only generated {len(results)} instances (target = {n_instances}).")
        return results

    def save_to_json(self, instances, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"[Info] Saved {len(instances)} instances to {out_path}")

    def generate_and_save(
        self,
        n_nodes: Union[int, Tuple[int, int]],
        n_instances: int,
        out_path: str,
        oversample_factor: float = 3.0,
        viz_dir: str = None,
        min_density: float = None,
        max_density: float = None,
    ):
        out_path = Path(out_path)
        target = n_instances
        total_to_try = int(target * oversample_factor)

        print(
            f"[Info] Target instances = {target}, oversample_factor = {oversample_factor}, "
            f"try to generate {total_to_try} raw instances."
        )

        raw = self.generate_instances(n_nodes, total_to_try)

        if (min_density is not None) or (max_density is not None):
            filtered = []
            for wrap in raw:
                inst = wrap["instance"]
                d = inst.get("density", None)
                if d is None:
                    continue
                if (min_density is not None) and (d < min_density):
                    continue
                if (max_density is not None) and (d > max_density):
                    continue
                filtered.append(wrap)

            print(
                f"[Info] Density filter: min={min_density}, max={max_density}, "
                f"kept {len(filtered)}/{len(raw)} instances."
            )
            raw = filtered
        # ==============================

        if len(raw) < target:
            print(
                f"[Warn] Only generated {len(raw)} valid instances (< {target}). "
                f"Will save all of them."
            )
            data = raw
        else:
            data = raw[:target]

        if viz_dir is not None:
            viz_path = Path(viz_dir)
            viz_path.mkdir(parents=True, exist_ok=True)

            for idx, inst_wrap in enumerate(data):
                inst = inst_wrap["instance"]

                inst_png = viz_path / f"instance_{idx:04d}.png"
                sol_png = viz_path / f"solution_{idx:04d}.png"

                draw_graph_instance(
                    inst,
                    save_path=str(inst_png),
                    title=f"{inst_wrap['problem_type']} Instance {idx}",
                )
                draw_graph_solution(
                    inst_wrap,
                    save_path=str(sol_png),
                    title=f"{inst_wrap['problem_type']} Solution {idx}",
                )

                inst["viz_instance"] = str(inst_png)
                inst["viz_solution"] = str(sol_png)
        # ===========================================

        V_list = []
        E_list = []
        D_list = []
        EC_list = []
        obj_list = []

        for wrap in data:
            inst = wrap["instance"]

            nV = inst.get("num_nodes")
            nE = inst.get("num_edges")

            if nV is not None:
                V_list.append(nV)
            if nE is not None:
                E_list.append(nE)

            # density
            if "density" in inst:
                D_list.append(inst["density"])

            # edge connectivity EC: k'(G) — only for graph instances
            if "edges" in inst:
                G_tmp = nx.Graph()
                for e in inst["edges"]:
                    G_tmp.add_edge(e["u"], e["v"])
                try:
                    ec_val = nx.edge_connectivity(G_tmp)
                except Exception as e:
                    print(f"[Warn] edge_connectivity failed for one instance: {e}")
                    ec_val = None

                if ec_val is not None:
                    EC_list.append(ec_val)
                    inst["edge_connectivity"] = int(ec_val)

            obj = wrap.get("obj", None)
            if obj is not None:
                obj_list.append(obj)

        stats: Dict[str, Any] = {
            "num_instances": len(data),
        }

        if V_list:
            stats["V"] = {
                "avg": float(sum(V_list) / len(V_list)),
                "max": int(max(V_list)),
                "min": int(min(V_list)),
            }

        if E_list:
            stats["E"] = {
                "avg": float(sum(E_list) / len(E_list)),
                "max": int(max(E_list)),
                "min": int(min(E_list)),
            }

        if D_list:
            stats["D"] = {
                "avg": float(sum(D_list) / len(D_list)),
                "min": float(min(D_list)),
                "max": float(max(D_list)),
            }

        if EC_list:
            stats["EC"] = {
                "avg": float(sum(EC_list) / len(EC_list)),
                "max": int(max(EC_list)),
                "min": float(min(EC_list)),
            }

        if obj_list:
            stats["obj"] = {
                "avg": float(sum(obj_list) / len(obj_list)),
                "min": float(min(obj_list)),
                "max": float(max(obj_list)),
            }

        self.save_to_json(data, out_path)
        print(stats)
        stats_path = out_path.with_name(out_path.stem + "_stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[Info] Saved stats to {stats_path}")

        return data


class SingleGraphSubgraphExtractor(InstanceExtractor, ABC):

    def __init__(
        self,
        dataset_dir: str,
        steps: int = 5_0000,
        alpha: float = 0.15,
        rng_seed: int = 42,
        max_retry: int = 10,
    ):
        self.dataset_dir = dataset_dir
        self.steps = steps
        self.alpha = alpha
        self.rng = random.Random(rng_seed)
        self.max_retry = max_retry

        self.graphs, self.graph_files = load_all_mtx_graphs(dataset_dir)
        if not self.graphs:
            raise ValueError(f"No graphs loaded from {dataset_dir}")

    @abstractmethod
    def get_problem_type(self) -> str:
        ...

    @abstractmethod
    def _solve_on_subgraph(
        self, G_sub: nx.Graph
    ) -> Tuple[Any, float]:
        ...

    def _sample_connected_subgraph(self, G_full: nx.Graph, n_nodes: int):
        if n_nodes < 2:
            raise ValueError("Need at least 2 nodes.")

        sampler = RandomWalkSubgraphSampler(
            steps=self.steps, alpha=self.alpha, seed=self.rng.randint(0, 10**9)
        )

        for _ in range(self.max_retry):
            start_node = self.rng.choice(list(G_full.nodes))
            subG = sampler.sample_subgraph(G_full, start_node, num_nodes=n_nodes)

            if subG.number_of_nodes() < n_nodes:
                continue
            if not nx.is_connected(subG):
                continue
            if is_star_graph(subG):
                continue
            # relabel -> 1..n
            old_nodes_sorted = sorted(subG.nodes())
            mapping = {
                old: new_id for new_id, old in enumerate(old_nodes_sorted, start=1)
            }
            G_renamed = nx.relabel_nodes(subG, mapping, copy=True)
            return G_renamed

        return None

    def extract_instance(self, n_nodes: int):
        idx = self.rng.randrange(len(self.graphs))
        G_full = self.graphs[idx]
        file_used = self.graph_files[idx]
        G_sub = self._sample_connected_subgraph(G_full, n_nodes=n_nodes)
        if G_sub is None:
            # sampling failed; return a marker for failure
            return None, None, None
        density = nx.density(G_sub)

        solution, obj = self._solve_on_subgraph(G_sub)

        edges_list: List[Dict[str, Any]] = []
        for u, v, data in G_sub.edges(data=True):
            edges_list.append({"u": int(u), "v": int(v)})

        instance_dict: Dict[str, Any] = {
            "problem_type": self.get_problem_type(),
            "num_nodes": G_sub.number_of_nodes(),
            "num_edges": G_sub.number_of_edges(),
            "edges": edges_list,
            "source_file": Path(file_used).name,
            "density": float(density),
        }

        return instance_dict, solution, float(obj)


class SimpleInstanceGenerator:
    def __init__(self, extractor):
        self.extractor = extractor

    def generate_instances(self, n_nodes, n_instances):
        # retry loop, same as assignment.py's InstanceGenerator
        results = []
        attempts = 0
        max_attempts = n_instances * 10
        while len(results) < n_instances and attempts < max_attempts:
            attempts += 1
            if isinstance(n_nodes, tuple):
                n = random.randint(*n_nodes)
            else:
                n = n_nodes
            try:
                inst, sol, obj = self.extractor.extract_instance(n)
                results.append({"instance": inst, "solution": sol, "obj": obj, "problem_type": self.extractor.get_problem_type()})
            except Exception as e:
                print(f"Error generating instance (attempt {attempts}): {e}")
                continue
        if len(results) < n_instances:
            print(f"Warning: requested {n_instances} instances but only generated {len(results)}")
        return results

    def save_to_json(self, instances, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(instances, f, indent=2)
        print(f"Saved {len(instances)} instances to {out_path}")

    def generate_and_save(self, n_nodes, n_instances, out_path):
        data = self.generate_instances(n_nodes, n_instances)
        self.save_to_json(data, out_path)
        return data
