"""
MDS (Minimum Dominating Set) instance extractor.
"""
import random
from typing import Any, Dict, List, Tuple

import networkx as nx

from ..utils.base import SingleGraphSubgraphExtractor
from ..solvers.graph_solver import solve_mds


class MDSSubgraphInstanceExtractor(SingleGraphSubgraphExtractor):
    def get_problem_type(self) -> str:
        return "MDS"

    def _solve_on_subgraph(self, G_sub: nx.Graph) -> Tuple[Any, float]:
        mds_set, obj_val = solve_mds(
            G_sub,
            time_limit=60,
            mip_gap=0.0,
            verbose=False,
        )
        solution = [int(v) for v in sorted(mds_set)]
        return solution, obj_val
