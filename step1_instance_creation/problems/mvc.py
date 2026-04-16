"""
MVC (Minimum Vertex Cover) instance extractor.
"""
import random
from typing import Any, Dict, List, Tuple

import networkx as nx

from ..utils.base import SingleGraphSubgraphExtractor
from ..solvers.graph_solver import solve_mvc


class MVCSubgraphInstanceExtractor(SingleGraphSubgraphExtractor):
    def get_problem_type(self) -> str:
        return "MVC"

    def _solve_on_subgraph(self, G_sub: nx.Graph) -> Tuple[Any, float]:
        vc_set, obj_val = solve_mvc(G_sub)
        solution = [int(v) for v in sorted(vc_set)]
        return solution, obj_val
