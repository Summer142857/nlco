"""
MaxCut (Maximum Cut) instance extractor.
"""
import random
from typing import Any, Dict, List, Tuple

import networkx as nx

from ..utils.base import SingleGraphSubgraphExtractor
from ..solvers.graph_solver import solve_maxcut


class MaxCutSubgraphInstanceExtractor(SingleGraphSubgraphExtractor):
    def get_problem_type(self) -> str:
        return "MAXCUT"

    def _solve_on_subgraph(self, G_sub: nx.Graph) -> Tuple[Any, float]:
        A, B, obj_val = solve_maxcut(
            G_sub,
            time_limit=60,
            mip_gap=0.0,
            verbose=False,
        )
        # solution = [[group1 vertices], [group2 vertices]]
        solution = [
            [int(v) for v in sorted(A)],
            [int(v) for v in sorted(B)],
        ]
        return solution, obj_val
