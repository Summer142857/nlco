"""
GCP (Graph Coloring Problem) instance extractor.
"""
import random
from typing import Any, Dict, List, Tuple

import networkx as nx

from ..utils.base import SingleGraphSubgraphExtractor
from ..solvers.graph_solver import solve_gcp


class GCPSubgraphInstanceExtractor(SingleGraphSubgraphExtractor):
    def get_problem_type(self) -> str:
        return "GCP"

    def _solve_on_subgraph(self, G_sub: nx.Graph) -> Tuple[Any, float]:
        coloring, num_colors, obj_val = solve_gcp(
            G_sub,
            time_limit=60,
            mip_gap=0.0,
            verbose=False,
        )
        # coloring: dict[v] = color
        color_to_vertices: Dict[int, List[int]] = {}
        for v, c in coloring.items():
            c = int(c)
            color_to_vertices.setdefault(c, []).append(int(v))

        solution: List[List[int]] = []
        for c in sorted(color_to_vertices.keys()):
            group = sorted(color_to_vertices[c])
            solution.append(group)

        return solution, obj_val
