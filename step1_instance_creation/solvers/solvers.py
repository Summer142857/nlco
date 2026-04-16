"""
Solvers for Combinatorial Optimization Problems

This module contains solver implementations for various optimization problems,
starting with TSP solvers.
"""

import os
import re
import tempfile
import uuid
from urllib.parse import urlparse
from subprocess import check_call, check_output

import numpy as np
import torch
import elkai
from scipy.spatial import cKDTree
import gurobipy as gp
from gurobipy import GRB
from typing import Any, Dict, List, Tuple


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters
    ----------
    point1 : np.ndarray
        Coordinates of the first point.
    point2 : np.ndarray
        Coordinates of the second point.

    Returns
    -------
    float
        Euclidean distance between the two points.
    """
    return np.linalg.norm(point1 - point2)


def compute_euclidean_distance_matrix(locations: np.ndarray) -> np.ndarray:

    num_nodes = locations.shape[0]
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_matrix[i, j] = euclidean_distance(locations[i], locations[j])
    return dist_matrix


def calculate_total_distance(tour: list[int], dist_matrix: np.ndarray) -> float:

    total_dist = 0.0
    for i in range(len(tour) - 1):
        from_node = tour[i]
        to_node = tour[i + 1]
        total_dist += dist_matrix[from_node][to_node]

    # Add the distance from the last node back to the starting node
    total_dist += dist_matrix[tour[-1]][tour[0]]
    return total_dist


def lkh(problem: torch.Tensor) -> tuple[list[int], float]:
    """
    Solve the TSP using the LKH (via elkai) solver.

    Parameters
    ----------
    problem : torch.Tensor
        A tensor of shape (N, 2) containing node coordinates.

    Returns
    -------
    tuple[list[int], float]
        A tuple where the first element is the best tour (list of node indices),
        and the second is the total distance of that tour.
    """
    if isinstance(problem, torch.Tensor):
        locations = problem.detach().cpu().numpy()
    else:
        locations = np.array(problem)

    dist_matrix = compute_euclidean_distance_matrix(locations)
    cities = elkai.DistanceMatrix(dist_matrix)
    tour = cities.solve_tsp()
    cost = calculate_total_distance(tour, dist_matrix)
    return tour, cost


def calculate_top_k_nearest_nodes(nodes: np.ndarray, k: int = 2) -> list[list[tuple[int, float]]]:
    """
    For each node, calculate its top k nearest neighbors using a k-d tree.

    Parameters
    ----------
    nodes : np.ndarray
        Coordinates of the nodes. Shape: (N, 2).
    k : int, optional
        Number of nearest neighbors to find for each node. Default is 2.

    Returns
    -------
    list[list[tuple[int, float]]]
        A list of length N, where each element is a list of k tuples (neighbor_index, distance).
    """
    kdtree = cKDTree(nodes)
    top_k_nearest_nodes = []
    for node in nodes:
        distances, indices = kdtree.query(node, k + 1)  # k+1 to include the node itself
        # Exclude the node itself (first index)
        distances, indices = distances[1:], indices[1:]
        neighbors = [(idx, dist) for idx, dist in zip(indices, distances)]
        top_k_nearest_nodes.append(neighbors)
    return top_k_nearest_nodes


def solve_bin_packing_gurobi(item_sizes: List[float], bin_capacity: float, 
                            time_limit: int = 300) -> Tuple[List[List[int]], int]:
    """
    Solve the Bin Packing Problem using Gurobi.
    
    Parameters
    ----------
    item_sizes : List[float]
        List of item sizes to be packed.
    bin_capacity : float
        Capacity of each bin.
    time_limit : int, optional
        Time limit for optimization in seconds. Default is 300.
    
    Returns
    -------
    Tuple[List[List[int]], int]
        A tuple where the first element is a list of bins (each bin contains item indices),
        and the second element is the total number of bins used.
    """
    n_items = len(item_sizes)
    
    # Create model
    model = gp.Model("bin_packing")
    model.setParam('OutputFlag', 0)  # Suppress output
    model.setParam('TimeLimit', time_limit)
    
    # Upper bound on number of bins (worst case: one item per bin)
    max_bins = n_items
    
    # Decision variables
    # x[i,j] = 1 if item i is assigned to bin j
    x = model.addVars(n_items, max_bins, vtype=GRB.BINARY, name="x")
    
    # y[j] = 1 if bin j is used
    y = model.addVars(max_bins, vtype=GRB.BINARY, name="y")
    
    # Objective: minimize number of bins
    model.setObjective(gp.quicksum(y[j] for j in range(max_bins)), GRB.MINIMIZE)
    
    # Constraints
    # Each item must be assigned to exactly one bin
    for i in range(n_items):
        model.addConstr(gp.quicksum(x[i, j] for j in range(max_bins)) == 1, f"assign_item_{i}")
    
    # Bin capacity constraints
    for j in range(max_bins):
        model.addConstr(
            gp.quicksum(item_sizes[i] * x[i, j] for i in range(n_items)) <= bin_capacity * y[j],
            f"capacity_bin_{j}"
        )
    
    # Symmetry breaking: if bin j is used, then bin j-1 must also be used
    for j in range(1, max_bins):
        model.addConstr(y[j] <= y[j-1], f"symmetry_{j}")
    
    # Solve
    model.optimize()
    
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        # Extract solution
        bins = []
        num_bins_used = 0
        
        for j in range(max_bins):
            if y[j].x > 0.5:  # Bin is used
                bin_items = []
                for i in range(n_items):
                    if x[i, j].x > 0.5:  # Item i is in bin j
                        bin_items.append(i)
                if bin_items:  # Only add non-empty bins
                    bins.append(bin_items)
                    num_bins_used += 1
        
        return bins, num_bins_used
    else:
        raise RuntimeError(f"Gurobi optimization failed with status: {model.status}")


def solve_bin_packing_first_fit_decreasing(item_sizes: List[float], bin_capacity: float) -> Tuple[List[List[int]], int]:
    """
    Solve the Bin Packing Problem using First Fit Decreasing heuristic.
    
    Parameters
    ----------
    item_sizes : List[float]
        List of item sizes to be packed.
    bin_capacity : float
        Capacity of each bin.
    
    Returns
    -------
    Tuple[List[List[int]], int]
        A tuple where the first element is a list of bins (each bin contains item indices),
        and the second element is the total number of bins used.
    """
    # Sort items by size in decreasing order, keeping track of original indices
    indexed_items = [(i, size) for i, size in enumerate(item_sizes)]
    indexed_items.sort(key=lambda x: x[1], reverse=True)
    
    bins = []
    bin_capacities = []
    
    for original_idx, item_size in indexed_items:
        # Try to fit in existing bins
        placed = False
        for bin_idx, remaining_capacity in enumerate(bin_capacities):
            if item_size <= remaining_capacity:
                bins[bin_idx].append(original_idx)
                bin_capacities[bin_idx] -= item_size
                placed = True
                break
        
        # If not placed, create new bin
        if not placed:
            bins.append([original_idx])
            bin_capacities.append(bin_capacity - item_size)
    
    return bins, len(bins)



# ============================================================
# Gurobi solver for PCTSP
# ============================================================

def solve_pctsp_gurobi(
    coords: np.ndarray,
    prizes: np.ndarray,
    penalties: np.ndarray,
    required_prize: float,
    depot: int = 0,
    time_limit: float = 60.0,
    threads: int = 8,
) -> Tuple[List[int], float]:
    """
    Solve PCTSP with a standard MILP formulation:
    """
    n = coords.shape[0]
    # Euclidean distance matrix
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(np.linalg.norm(coords[i] - coords[j]))

    model = gp.Model("PCTSP")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads

    # Decision variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")  # arc usage
    y = model.addVars(n, vtype=GRB.BINARY, name="y")     # visit node i

    # MTZ order variables (only meaningful for non-depot nodes)
    # u_i in [0, n-1], and if y_i=0 => u_i=0 by bound tying
    u = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=float(n - 1), name="u")

    # Depot must be visited
    model.addConstr(y[depot] == 1, name="visit_depot")

    # Degree constraints
    for i in range(n):
        if i == depot:
            # exactly 1 in / 1 out at depot
            model.addConstr(gp.quicksum(x[depot, j] for j in range(n) if j != depot) == 1, f"out_depot")
            model.addConstr(gp.quicksum(x[j, depot] for j in range(n) if j != depot) == 1, f"in_depot")
        else:
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == y[i], f"out_{i}")
            model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == y[i], f"in_{i}")

    # No self-loops
    for i in range(n):
        model.addConstr(x[i, i] == 0, f"no_loop_{i}")

    # Prize requirement
    model.addConstr(gp.quicksum(prizes[i] * y[i] for i in range(n)) >= float(required_prize), "prize_req")

    # MTZ subtour elimination (with optional nodes)
    # u_depot = 0
    model.addConstr(u[depot] == 0, "u_depot_zero")

    # For non-depot nodes, tie u_i to y_i (if y_i=0 => u_i=0; if y_i=1 => u_i>=1)
    for i in range(n):
        if i == depot:
            continue
        model.addConstr(u[i] <= (n - 1) * y[i], f"u_upper_{i}")
        model.addConstr(u[i] >= y[i], f"u_lower_{i}")

    # MTZ: u_i - u_j + n * x[i,j] <= n - 1 for i!=j, i,j != depot
    for i in range(n):
        if i == depot:
            continue
        for j in range(n):
            if j == depot or i == j:
                continue
            model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, f"mtz_{i}_{j}")

    # Objective: travel cost + penalties of unvisited nodes
    travel_cost = gp.quicksum(D[i, j] * x[i, j] for i in range(n) for j in range(n))
    unvisited_penalty = gp.quicksum((1 - y[i]) * float(penalties[i]) for i in range(n))
    model.setObjective(travel_cost + unvisited_penalty, GRB.MINIMIZE)

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # fallback: visit only depot
        return [depot, depot], 0.0

    # Build successor list
    succ = {i: None for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and x[i, j].X > 0.5:
                succ[i] = j

    # Reconstruct tour from depot
    tour = [depot]
    cur = depot
    visited_set = set([depot])
    while True:
        nxt = succ[cur]
        if nxt is None:
            break
        tour.append(nxt)
        if nxt == depot:
            break
        if nxt in visited_set:
            # safety stop in degenerate cases
            tour.append(depot)
            break
        visited_set.add(nxt)
        cur = nxt

    obj_val = model.objVal
    return tour, float(obj_val)




# ============================================================
# Gurobi solver for OP (Orienteering Problem)
# ============================================================

def solve_op_gurobi(
    coords: np.ndarray,
    prizes: np.ndarray,
    max_length: float,
    depot: int = 0,
    time_limit: float = 60.0,
    threads: int = 8,
) -> Tuple[List[int], float, float]:
    """
    Orienteering Problem (OP) MILP:

    """
    n = coords.shape[0]

    # Euclidean distances
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(np.linalg.norm(coords[i] - coords[j]))

    model = gp.Model("OP")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads

    # Variables
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")  # arc usage
    y = model.addVars(n, vtype=GRB.BINARY, name="y")     # visit node
    u = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=float(n - 1), name="u")  # MTZ order

    # Depot must be visited
    model.addConstr(y[depot] == 1, "visit_depot")

    # Degree constraints
    for i in range(n):
        if i == depot:
            model.addConstr(gp.quicksum(x[depot, j] for j in range(n) if j != depot) == 1, "out_depot")
            model.addConstr(gp.quicksum(x[j, depot] for j in range(n) if j != depot) == 1, "in_depot")
        else:
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == y[i], f"out_{i}")
            model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == y[i], f"in_{i}")

    # No self-loops
    for i in range(n):
        model.addConstr(x[i, i] == 0, f"no_loop_{i}")

    # MTZ tying for optional nodes
    model.addConstr(u[depot] == 0, "u_depot_zero")
    for i in range(n):
        if i == depot:
            continue
        model.addConstr(u[i] <= (n - 1) * y[i], f"u_upper_{i}")
        model.addConstr(u[i] >= y[i], f"u_lower_{i}")

    for i in range(n):
        if i == depot:
            continue
        for j in range(n):
            if j == depot or i == j:
                continue
            model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, f"mtz_{i}_{j}")

    # Length budget
    travel_cost = gp.quicksum(D[i, j] * x[i, j] for i in range(n) for j in range(n))
    model.addConstr(travel_cost <= float(max_length), "length_budget")

    # Objective: maximize collected prize
    model.setObjective(gp.quicksum(float(prizes[i]) * y[i] for i in range(n)), GRB.MAXIMIZE)

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # Fallback: stay at depot
        return [depot, depot], 0.0, 0.0

    # Successor reconstruction
    succ = {i: None for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and x[i, j].X > 0.5:
                succ[i] = j

    tour = [depot]
    cur = depot
    seen = {depot}
    while True:
        nxt = succ[cur]
        if nxt is None:
            break
        tour.append(nxt)
        if nxt == depot:
            break
        if nxt in seen:
            tour.append(depot)
            break
        seen.add(nxt)
        cur = nxt

    # Report
    travel_len = sum(np.linalg.norm(coords[tour[k]] - coords[tour[k+1]]) for k in range(len(tour)-1))
    collected = float(sum(prizes[i] for i in set(tour)))  # depot prize is 0
    return tour, collected, float(travel_len)



# ============================================================
# CVRP solver
# ============================================================

def solve_cvrp_gurobi(
    coords: np.ndarray,          # shape (n, 2) with depot at index 0
    demands: np.ndarray,         # length n, demands[0] = 0
    capacity: int,               # vehicle capacity (e.g., 30)
    depot: int = 0,
    time_limit: float = 60.0,
    threads: int = 8,
) -> Tuple[List[List[int]], float]:
    """
    Capacitated VRP MILP:

    """
    n = coords.shape[0]
    customers = [i for i in range(n) if i != depot]

    # Euclidean distances
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(np.linalg.norm(coords[i] - coords[j]))

    model = gp.Model("CVRP")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads

    # x[i,j] = 1 if arc i->j used (including depot arcs)
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    # MTZ load variables only for customers
    u = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=float(capacity), name="u")

    # No self-loops
    for i in range(n):
        model.addConstr(x[i, i] == 0, name=f"no_loop_{i}")

    # Each customer has exactly 1 in and 1 out
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1, name=f"out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1, name=f"in_{i}")

    # Depot flow balance: total out == total in (number of vehicles is free)
    model.addConstr(
        gp.quicksum(x[depot, j] for j in customers) ==
        gp.quicksum(x[j, depot] for j in customers),
        name="depot_balance"
    )

    # Capacity bounds for loads on customers
    # u[i] = load right after serving i; enforce demand[i] <= u[i] <= capacity if i is on a route
    for i in customers:
        model.addConstr(u[i] >= float(demands[i]), name=f"load_lb_{i}")
        model.addConstr(u[i] <= float(capacity), name=f"load_ub_{i}")
    model.addConstr(u[depot] == 0.0, name="load_depot_zero")

    # MTZ capacity constraints for CVRP (i,j are customers)
    # u[i] - u[j] + capacity * x[i,j] <= capacity - demand[j]
    # If i->j used, then u[j] >= u[i] + demand[j]
    for i in customers:
        for j in customers:
            if i == j:
                continue
            model.addConstr(
                u[i] - u[j] + capacity * x[i, j] <= capacity - float(demands[j]),
                name=f"mtz_cap_{i}_{j}"
            )

    # Also connect depot to first customer: if depot->i is used, then u[i] >= demand[i]
    # (This is already ensured by u[i] >= demand[i], but we ensure feasibility on arcs from depot)
    # Not strictly necessary beyond the bounds, so we keep model lean.

    # Objective: minimize total traveled distance
    model.setObjective(
        gp.quicksum(D[i, j] * x[i, j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE
    )

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        # Fallback: no route
        return [], 0.0

    # Build adjacency lists (can have multiple depot starts)
    succ = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and x[i, j].X > 0.5:
                succ[i].append(j)

    # Reconstruct routes starting from each depot outgoing arc
    routes: List[List[int]] = []
    used = set()  # visited customers
    for start in succ[depot]:
        if start in used:
            continue
        route = [depot, start]
        cur = start
        used.add(start)
        while True:
            nxts = succ[cur]
            if not nxts:
                # safety stop
                route.append(depot)
                break
            nxt = nxts[0]  # each customer has exactly one outgoing
            route.append(nxt)
            if nxt == depot:
                break
            if nxt in used:
                # safety: close route
                route.append(depot)
                break
            used.add(nxt)
            cur = nxt
        routes.append(route)

    # Compute total distance
    total_dist = 0.0
    for r in routes:
        for k in range(len(r) - 1):
            total_dist += float(np.linalg.norm(coords[r[k]] - coords[r[k + 1]]))

    return routes, float(total_dist)

def solve_top_gurobi(
    coords: np.ndarray,
    prizes: np.ndarray,
    max_length_per_vehicle: float,
    depot: int,
    n_vehicles: int,
    time_limit: float = 60.0,
    threads: int = 8,
):
    """
    Solve Team Orienteering Problem (TOP) with multiple vehicles.

    """
    n = coords.shape[0]
    V = range(n)
    customers = [i for i in V if i != depot]

    # distance matrix
    dist = np.zeros((n, n), dtype=float)
    for i in V:
        for j in V:
            if i != j:
                dist[i, j] = np.linalg.norm(coords[i] - coords[j])

    model = gp.Model("TOP")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads

    # x[i,j,k] = 1 if vehicle k uses arc i->j
    x = model.addVars(n, n, n_vehicles, vtype=GRB.BINARY, name="x")

    # y[i] = 1 if customer i is visited by any vehicle
    y = model.addVars(customers, vtype=GRB.BINARY, name="y")

    # MTZ variables for subtour elimination on customers for each vehicle
    u = model.addVars(customers, n_vehicles, lb=0.0, ub=n - 1,
                      vtype=GRB.CONTINUOUS, name="u")

    # objective: maximize total collected prize
    model.setObjective(gp.quicksum(prizes[i] * y[i] for i in customers),
                       GRB.MAXIMIZE)

    # no self loops
    for k in range(n_vehicles):
        for i in V:
            model.addConstr(x[i, i, k] == 0, name=f"no_self_{i}_{k}")

    # flow constraints
    for k in range(n_vehicles):
        # depot flow: in = out
        model.addConstr(
            gp.quicksum(x[depot, j, k] for j in V if j != depot)
            == gp.quicksum(x[i, depot, k] for i in V if i != depot),
            name=f"depot_flow_{k}"
        )
        # customers flow conservation
        for i in customers:
            model.addConstr(
                gp.quicksum(x[i, j, k] for j in V if j != i)
                == gp.quicksum(x[j, i, k] for j in V if j != i),
                name=f"flow_{i}_{k}"
            )

    # each customer visited at most once (by at most one vehicle)
    for i in customers:
        model.addConstr(
            gp.quicksum(x[i, j, k] for k in range(n_vehicles) for j in V if j != i)
            <= 1,
            name=f"visit_once_{i}"
        )

    # link y and x: y[i] = 1 iff any vehicle departs from i
    for i in customers:
        model.addConstr(
            y[i] == gp.quicksum(x[i, j, k] for k in range(n_vehicles) for j in V if j != i),
            name=f"y_link_{i}"
        )

    # per-vehicle route length constraints
    for k in range(n_vehicles):
        model.addConstr(
            gp.quicksum(dist[i, j] * x[i, j, k] for i in V for j in V if i != j)
            <= max_length_per_vehicle,
            name=f"length_{k}"
        )

    # MTZ subtour elimination on customers per vehicle
    for k in range(n_vehicles):
        for i in customers:
            for j in customers:
                if i == j:
                    continue
                model.addConstr(
                    u[i, k] - u[j, k] + 1 <= (n - 1) * (1 - x[i, j, k]),
                    name=f"mtz_{i}_{j}_{k}"
                )

    model.optimize()

    # reconstruct routes
    routes = []
    total_length = 0.0

    for k in range(n_vehicles):
        # build successor map for vehicle k
        succ = {i: None for i in V}
        for i in V:
            for j in V:
                if i != j and x[i, j, k].X > 0.5:
                    succ[i] = j

        # if vehicle k not used: route is depot -> depot
        if succ[depot] is None:
            routes.append([depot, depot])
            continue

        route = [depot]
        current = depot
        visited = set([depot])

        while True:
            nxt = succ.get(current, None)
            if nxt is None:
                break
            route.append(nxt)
            if nxt == depot:
                break
            if nxt in visited:
                # safety break against weird cycles
                route.append(depot)
                break
            visited.add(nxt)
            current = nxt

        if route[-1] != depot:
            route.append(depot)

        # compute length of this route
        length_k = 0.0
        for i in range(len(route) - 1):
            length_k += np.linalg.norm(coords[route[i]] - coords[route[i + 1]])
        total_length += length_k
        routes.append(route)

    total_prize = sum(prizes[i] for i in customers if y[i].X > 0.5)

    return routes, float(total_prize), float(total_length)




def solve_tsptw_gurobi(
    coords: np.ndarray,
    time_windows: np.ndarray,
    depot: int = 0,
    time_limit: float = 60.0,
    threads: int = 8,
):
    """
    Solve a single-vehicle Traveling Salesman Problem with Time Windows (TSPTW).

    """
    n = coords.shape[0]
    V = range(n)

    # distance matrix = travel times
    dist = np.zeros((n, n), dtype=float)
    for i in V:
        for j in V:
            if i != j:
                dist[i, j] = np.linalg.norm(coords[i] - coords[j])

    l = time_windows[:, 0]
    u = time_windows[:, 1]

    model = gp.Model("TSPTW")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads

    # x[i,j] = 1 if arc i->j is used
    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    # arrival time at node i
    t = model.addVars(n, vtype=GRB.CONTINUOUS, name="t")

    # objective: minimize total travel distance
    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in V for j in V if i != j),
        GRB.MINIMIZE,
    )

    # no self loops
    for i in V:
        model.addConstr(x[i, i] == 0, name=f"no_self_{i}")

    # each node has exactly one incoming and one outgoing arc
    for i in V:
        model.addConstr(
            gp.quicksum(x[i, j] for j in V if j != i) == 1,
            name=f"out_{i}",
        )
        model.addConstr(
            gp.quicksum(x[j, i] for j in V if j != i) == 1,
            name=f"in_{i}",
        )

    # time windows
    for i in V:
        if i == depot:
            # depot：只固定起始时间为 l[depot]，不加上界
            model.addConstr(t[i] == float(l[i]), name=f"depot_start")
        else:
            model.addConstr(t[i] >= float(l[i]), name=f"tw_lb_{i}")
            model.addConstr(t[i] <= float(u[i]), name=f"tw_ub_{i}")

    # big-M for time propagation
    M = float(np.max(u) + np.max(dist))

    for i in V:
        for j in V:
            if i == j:
                continue
            if j == depot:
                # 不对“回 depot”的弧施加时间传播约束
                continue
            # 如果走 i->j，则 t[j] >= t[i] + travel_ij
            model.addConstr(
                t[j] >= t[i] + dist[i, j] - M * (1 - x[i, j]),
                name=f"time_{i}_{j}",
            )

    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(f"TSPTW model has no feasible solution (status={model.Status})")

    # reconstruct tour from x
    succ = {i: None for i in V}
    for i in V:
        for j in V:
            if i != j and x[i, j].X > 0.5:
                succ[i] = j

    tour = [depot]
    current = depot
    visited = {depot}
    while True:
        nxt = succ[current]
        if nxt is None:
            break
        tour.append(nxt)
        if nxt == depot:
            break
        if nxt in visited:
            # safety
            tour.append(depot)
            break
        visited.add(nxt)
        current = nxt

    if tour[-1] != depot:
        tour.append(depot)

    total_dist = 0.0
    for i in range(len(tour) - 1):
        total_dist += np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]])

    return tour, float(total_dist)


# ============================================================
# LKH-3 utilities (TSPTW support)
# ============================================================


def get_lkh_executable(
    url: str = "http://webhotel4.ruc.dk/~keld/research/LKH-3/LKH-3.0.9.tgz",
) -> str:
    """Download (if needed) and compile LKH-3, returning the path to the LKH executable."""
    cwd = os.path.abspath("lkh")
    os.makedirs(cwd, exist_ok=True)

    tar_name = os.path.split(urlparse(url).path)[-1]
    file = os.path.join(cwd, tar_name)
    filedir = os.path.splitext(file)[0]
    version = os.path.splitext(tar_name)[0][4:]
    print(f">> LKH version: {version}")

    if not os.path.isdir(filedir):
        print(f"{filedir} not found, downloading and compiling")
        check_call(["wget", url], cwd=cwd)
        assert os.path.isfile(file), f"Download failed, {file} does not exist"
        check_call(["tar", "xvfz", file], cwd=cwd)
        assert os.path.isdir(filedir), f"Extracting failed, dir {filedir} does not exist"
        check_call(["make"], cwd=filedir)
        os.remove(file)

    executable = os.path.join(filedir, "LKH")
    assert os.path.isfile(executable), f"LKH executable not found at {executable}"
    return os.path.abspath(executable)


def write_lkh_par(filename: str, parameters: dict) -> None:
    """Write an LKH parameter (.par) file."""
    default_parameters = {
        "SPECIAL": None,
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 1,
        "SEED": 0,
    }
    with open(filename, "w") as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write(f"{k}\n")
            else:
                f.write(f"{k} = {v}\n")


def read_lkh_vrplib(filename: str, n: int) -> list:
    """Read an LKH tour file in VRPLIB format; returns list of node indices (0-based)."""
    tour = []
    dimension = None
    started = False

    with open(filename, "r") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("DIMENSION"):
                dimension = int(line.split()[-1])
                continue
            if line.startswith("TOUR_SECTION"):
                started = True
                continue
            if not started:
                continue
            try:
                v = int(line)
            except ValueError:
                continue
            if v == -1:
                break
            tour.append(v)

    if dimension is None:
        raise ValueError("Missing DIMENSION in TOUR file.")
    if len(tour) not in (dimension, dimension + 1):
        raise ValueError(f"Unexpected TOUR length {len(tour)} vs DIMENSION {dimension}.")
    if dimension != n:
        raise ValueError(f"TOUR DIMENSION {dimension} != expected n {n}")
    return tour


def write_tsptw_vrplib(filename, coords, time_windows, name="TSPTW_Instance", scale=1.0):
    """Write a TSPTW instance in VRPLIB format for LKH-3."""
    n = coords.shape[0]
    assert time_windows.shape[0] == n

    with open(filename, "w") as f:
        f.write("\n".join([
            f"NAME : {name}",
            "COMMENT : TSPTW Instance",
            "TYPE : TSPTW",
            f"DIMENSION : {n}",
            "EDGE_WEIGHT_TYPE : EUC_2D",
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i}\t{int(round(x * scale))}\t{int(round(y * scale))}\n")
        f.write("TIME_WINDOW_SECTION\n")
        for i, (e, l) in enumerate(time_windows, start=1):
            f.write(f"{i}\t{int(round(e * scale))}\t{int(round(l * scale))}\n")
        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")


def _check_tsptw_feasible(tour, coords, time_windows, speed=1.0):
    """Check if a tour satisfies all time window constraints."""
    t = 0.0
    for k in range(len(tour) - 1):
        i, j = tour[k], tour[k + 1]
        travel = float(np.linalg.norm(coords[i] - coords[j])) / float(speed)
        t += travel
        e, l = float(time_windows[j][0]), float(time_windows[j][1])
        if t < e:
            t = e
        if t > l + 1e-9:
            return False
    return True


def solve_tsptw_lkh(coords, time_windows, depot=0, time_limit=60.0, threads=8):
    """Solve TSPTW using LKH-3, falling back to Gurobi if LKH is unavailable."""
    n = coords.shape[0]
    if n <= 1:
        return [depot, depot], 0.0

    if depot != 0:
        perm = [depot] + [i for i in range(n) if i != depot]
        new_to_old = np.array(perm, dtype=int)
        coords_perm = coords[perm]
        tw_perm = time_windows[perm]
    else:
        coords_perm = coords
        tw_perm = time_windows
        new_to_old = np.arange(n, dtype=int)

    try:
        executable = get_lkh_executable()
    except Exception:
        return solve_tsptw_gurobi(coords, time_windows, depot=depot, time_limit=time_limit, threads=threads)

    with tempfile.TemporaryDirectory(prefix="lkh_tsptw_") as lkh_root:
        name = f"tsptw_{uuid.uuid4().hex}"
        problem_filename = os.path.join(lkh_root, f"{name}.vrp")
        tour_filename = os.path.join(lkh_root, f"{name}.tour")
        param_filename = os.path.join(lkh_root, f"{name}.par")
        log_filename = os.path.join(lkh_root, f"{name}.log")

        write_tsptw_vrplib(problem_filename, coords_perm, tw_perm, name=name)

        params = {
            "PROBLEM_FILE": problem_filename,
            "OUTPUT_TOUR_FILE": tour_filename,
            "RUNS": 1,
            "SEED": 1234,
            "MAX_TRIALS": 10000,
            "TIME_LIMIT": int(max(1, time_limit)),
        }
        write_lkh_par(param_filename, params)

        if os.path.exists(tour_filename):
            os.remove(tour_filename)

        try:
            with open(log_filename, "w") as f:
                check_call([executable, param_filename], stdout=f, stderr=f)
        except Exception:
            return solve_tsptw_gurobi(coords, time_windows, depot=depot, time_limit=time_limit, threads=threads)

        if not os.path.isfile(tour_filename):
            return solve_tsptw_gurobi(coords, time_windows, depot=depot, time_limit=time_limit, threads=threads)

        try:
            customer_seq = read_lkh_vrplib(tour_filename, n=coords_perm.shape[0])
        except Exception:
            return solve_tsptw_gurobi(coords, time_windows, depot=depot, time_limit=time_limit, threads=threads)

        if len(customer_seq) != n - 1:
            return solve_tsptw_gurobi(coords, time_windows, depot=depot, time_limit=time_limit, threads=threads)
        if len(set(customer_seq)) != len(customer_seq):
            return solve_tsptw_gurobi(coords, time_windows, depot=depot, time_limit=time_limit, threads=threads)
        if any((c < 0 or c >= n or c == 0) for c in customer_seq):
            return solve_tsptw_gurobi(coords, time_windows, depot=depot, time_limit=time_limit, threads=threads)

        tour_perm = [0] + list(customer_seq) + [0]
        tour = [int(new_to_old[i]) for i in tour_perm]

        if not _check_tsptw_feasible(tour_perm, coords_perm, tw_perm):
            return solve_tsptw_gurobi(coords, time_windows, depot=depot, time_limit=time_limit, threads=threads)

        total_dist = 0.0
        for i in range(len(tour) - 1):
            total_dist += float(np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]]))
        return tour, float(total_dist)


def solve_linear_ordering_problem_gurobi(
    weight_matrix: np.ndarray,
    time_limit: int = 300,
    mip_gap=None,
) -> Tuple[List[int], float]:
    """
    Solve the Linear Ordering Problem (LOP) with a MAXIMIZING objective using Gurobi.

    Given weights W (n x n), find a permutation π maximizing sum_{i<j} W[π_i, π_j].
    """
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError("weight_matrix must be a square 2D array")

    n = int(weight_matrix.shape[0])

    model = gp.Model("LOP_max")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    if mip_gap is not None:
        model.setParam("MIPGap", float(mip_gap))

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    for i in range(n):
        model.addConstr(x[i, i] == 0, name=f"self_{i}")

    for i in range(n):
        for j in range(i + 1, n):
            model.addConstr(x[i, j] + x[j, i] == 1, name=f"pair_{i}_{j}")

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                model.addConstr(x[i, j] + x[j, k] + x[k, i] <= 2, name=f"tri1_{i}_{j}_{k}")
                model.addConstr(x[i, k] + x[k, j] + x[j, i] <= 2, name=f"tri2_{i}_{j}_{k}")

    model.setObjective(
        gp.quicksum(weight_matrix[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j),
        GRB.MAXIMIZE,
    )

    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(f"No feasible solution found (status={model.status}).")

    pred_count = []
    for j in range(n):
        preds = sum(1 for i in range(n) if i != j and x[i, j].X > 0.5)
        pred_count.append((preds, j))

    pred_count.sort()
    ordering = [j for _, j in pred_count]

    objective_value = float(model.ObjVal)
    return ordering, objective_value


def solve_csp_gurobi(
    weights,
    demands,
    bin_capacity,
    time_limit=300
):
    """Solve the Cutting Stock Problem using Gurobi with pattern-based formulation."""
    import gurobipy as gp
    from gurobipy import GRB
    n_types = len(weights)
    patterns = []

    def generate_patterns(remaining_capacity, type_idx, current_pattern):
        if type_idx == n_types:
            if any(current_pattern):
                patterns.append(current_pattern[:])
            return
        max_qty = min(remaining_capacity // weights[type_idx], demands[type_idx])
        for qty in range(max_qty + 1):
            current_pattern[type_idx] = qty
            new_capacity = remaining_capacity - qty * weights[type_idx]
            generate_patterns(new_capacity, type_idx + 1, current_pattern)

    generate_patterns(bin_capacity, 0, [0] * n_types)

    for i in range(n_types):
        single_pattern = [0] * n_types
        single_pattern[i] = 1
        if single_pattern not in patterns:
            patterns.append(single_pattern)

    n_patterns = len(patterns)
    print(f"CSP: Generated {n_patterns} feasible cutting patterns for {n_types} item types")

    model = gp.Model("cutting_stock")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    x = model.addVars(n_patterns, vtype=GRB.INTEGER, lb=0, name="x")
    model.setObjective(gp.quicksum(x[p] for p in range(n_patterns)), GRB.MINIMIZE)

    for i in range(n_types):
        model.addConstr(
            gp.quicksum(patterns[p][i] * x[p] for p in range(n_patterns)) >= demands[i],
            f"demand_{i}"
        )

    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        bins = []
        num_bins_used = 0
        for p in range(n_patterns):
            times_used = int(round(x[p].X))
            if times_used > 0:
                for _ in range(times_used):
                    pattern_dict = {
                        item_type: patterns[p][item_type]
                        for item_type in range(n_types)
                        if patterns[p][item_type] > 0
                    }
                    if pattern_dict:
                        bins.append(pattern_dict)
                        num_bins_used += 1
        return bins, num_bins_used
    else:
        raise RuntimeError(f"Gurobi optimization failed with status: {model.status}")


def solve_cmp_gurobi(
    n,
    edges,
    time_limit=60.0,
    threads=8,
    output_flag=0,
):
    """Solve Cutwidth Minimization Problem (CMP) with Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    if n <= 0:
        raise ValueError("n must be positive")

    cleaned_edges = []
    for (u, v) in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge ({u},{v}) has endpoints outside 0..{n-1}")
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        cleaned_edges.append((a, b))
    cleaned_edges = list(dict.fromkeys(cleaned_edges))

    positions = range(n)
    cuts = range(n - 1)
    E = range(len(cleaned_edges))

    model = gp.Model("cutwidth")
    model.Params.OutputFlag = output_flag
    model.Params.Threads = threads
    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    L = model.addVars(n, n - 1, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="L")
    y = model.addVars(len(cleaned_edges), n - 1, vtype=GRB.BINARY, name="y")
    c = model.addVars(n - 1, lb=0.0, vtype=GRB.CONTINUOUS, name="c")
    C = model.addVar(lb=0.0, vtype=GRB.INTEGER, name="C")

    for v in range(n):
        model.addConstr(gp.quicksum(x[v, p] for p in positions) == 1, name=f"assign_v_{v}")
    for p in positions:
        model.addConstr(gp.quicksum(x[v, p] for v in range(n)) == 1, name=f"assign_p_{p}")

    first_id = gp.quicksum(v * x[v, 0] for v in range(n))
    last_id = gp.quicksum(v * x[v, n - 1] for v in range(n))
    model.addConstr(first_id <= last_id, name="break_reverse_sym")

    for v in range(n):
        for k in cuts:
            running = gp.quicksum(x[v, p] for p in range(k + 1))
            model.addConstr(L[v, k] == running, name=f"L_def_{v}_{k}")

    for e_idx, (u, v) in enumerate(cleaned_edges):
        for k in cuts:
            model.addConstr(y[e_idx, k] >= L[u, k] - L[v, k], name=f"y_ge1_{e_idx}_{k}")
            model.addConstr(y[e_idx, k] >= L[v, k] - L[u, k], name=f"y_ge2_{e_idx}_{k}")
            model.addConstr(y[e_idx, k] <= L[u, k] + L[v, k], name=f"y_le1_{e_idx}_{k}")
            model.addConstr(y[e_idx, k] <= 2 - (L[u, k] + L[v, k]), name=f"y_le2_{e_idx}_{k}")

    for k in cuts:
        model.addConstr(c[k] == gp.quicksum(y[e_idx, k] for e_idx in E), name=f"c_def_{k}")
        model.addConstr(C >= c[k], name=f"C_ge_c_{k}")

    model.setObjective(C, GRB.MINIMIZE)
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Gurobi failed to solve CMP; status = {model.Status}")
    if model.SolCount == 0:
        raise RuntimeError(f"No feasible solution found (status={model.Status}).")

    ordering = [None] * n
    for v in range(n):
        for p in positions:
            if x[v, p].X > 0.5:
                ordering[p] = v
                break

    if any(v is None for v in ordering):
        raise RuntimeError("Failed to extract a complete permutation from the solution.")

    return ordering, float(C.X)


def solve_pctsp_hgs(
    coords,
    prizes,
    penalties,
    required_prize,
    depot=0,
    time_limit=60.0,
    threads=8,
):
    """Solve Prize-Collecting TSP (PCTSP) using PyVRP."""
    import numpy as np
    from pyvrp import Model as _PyVRPModel
    from pyvrp.stop import MaxRuntime as _MaxRuntime

    n = coords.shape[0]
    if n <= 1:
        return [depot, depot], 0.0

    if depot != 0:
        perm = [depot] + [i for i in range(n) if i != depot]
        inv_perm = np.empty(n, dtype=int)
        for new_idx, old_idx in enumerate(perm):
            inv_perm[new_idx] = old_idx
        coords_perm = coords[perm]
        prizes_perm = prizes[perm]
        penalties_perm = penalties[perm]
    else:
        coords_perm = coords
        prizes_perm = prizes
        penalties_perm = penalties
        inv_perm = np.arange(n, dtype=int)

    m = _PyVRPModel()
    depot_loc = m.add_depot(x=float(coords_perm[0, 0]), y=float(coords_perm[0, 1]))
    m.add_vehicle_type(1, start_depot=depot_loc, end_depot=depot_loc)

    clients = []
    for idx in range(1, n):
        effective_prize = int(prizes_perm[idx] + penalties_perm[idx])
        client = m.add_client(
            x=float(coords_perm[idx, 0]),
            y=float(coords_perm[idx, 1]),
            prize=effective_prize,
            required=False,
        )
        clients.append(client)

    locations = [depot_loc] + clients
    for i, frm in enumerate(locations):
        for j, to in enumerate(locations):
            if i == j:
                distance = 0
            else:
                dist_val = float(np.linalg.norm(coords_perm[i] - coords_perm[j]))
                distance = int(round(dist_val))
            m.add_edge(frm, to, distance=distance)

    res = m.solve(stop=_MaxRuntime(time_limit), display=False)

    if res.best is None or not res.best.routes():
        return [int(inv_perm[0]), int(inv_perm[0])], 0.0

    tour_perm = [0]
    for route in res.best.routes():
        for trip in route.trips():
            tour_perm.extend(list(trip.visits()))
    tour_perm.append(0)

    tour = [int(inv_perm[idx]) for idx in tour_perm]
    visited_set = set(tour)

    travel_cost = 0.0
    for i in range(len(tour) - 1):
        travel_cost += float(np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]]))

    penalty_cost = sum(float(penalties[i]) for i in range(n) if i not in visited_set)
    return tour, float(travel_cost + penalty_cost)


import os as _os
import re as _re
from subprocess import check_output as _check_output

COMPASS_EXECUTABLE = _os.environ.get(
    "COMPASS_OP_SOLVER", "./op-solver/build/src/op-solver"
)


def write_oplib(filename, depot, loc, prize, max_length, name="problem"):
    """Write an Orienteering Problem instance in OPLIB format for COMPASS."""
    import numpy as np
    depot = np.asarray(depot, dtype=float)
    loc = np.asarray(loc, dtype=float)
    prize = np.asarray(prize, dtype=float)

    with open(filename, "w") as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name), ("TYPE", "OP"),
                ("DIMENSION", len(loc)),
                ("COST_LIMIT", int(max_length)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        all_nodes = np.vstack((depot[None, :], loc))
        for i, (x, y) in enumerate(all_nodes):
            f.write(f"{i + 1}\t{int(x)}\t{int(y)}\n")
        f.write("NODE_SCORE_SECTION\n")
        scores = np.concatenate(([0.0], prize))
        for i, s in enumerate(scores):
            f.write(f"{i + 1}\t{int(s)}\n")
        f.write("DEPOT_SECTION\n1\n-1\nEOF\n")


def solve_compass(coords, prizes, max_length, depot=0, executable=None):
    """Solve OP using the external COMPASS solver."""
    import numpy as np
    if executable is None:
        executable = COMPASS_EXECUTABLE
    n = coords.shape[0]
    if n <= 1:
        return 0.0, []
    if depot != 0:
        raise ValueError("solve_compass currently assumes depot index 0.")

    depot_coord = coords[0]
    loc = coords[1:]
    prize = prizes[1:]

    problem_filename = _os.path.join(".", "problem.oplib")
    write_oplib(problem_filename, depot_coord, loc, prize, max_length)

    output = _check_output([executable, "opt", "--op-exact", "0", problem_filename])
    text = output.decode("utf-8", errors="ignore")

    obj_match = _re.search(r"Objetive value:\s*([0-9]+\.[0-9]+)", text)
    objective_value = float(obj_match.group(1)) if obj_match else None

    cycle_match = _re.search(r"Cycle:\s*([0-9\s]+)", text)
    cycle = list(map(int, cycle_match.group(1).strip().split())) if cycle_match else None

    if objective_value is None or cycle is None:
        return 0.0, []
    return objective_value, cycle


def solve_op_compass(coords, prizes, max_length, depot=0, time_limit=60.0, threads=8):
    """Orienteering Problem (OP) solver using the external COMPASS solver."""
    import numpy as np
    coords = np.asarray(coords, dtype=float)
    prizes = np.asarray(prizes, dtype=float)
    n = coords.shape[0]

    if n <= 1:
        return [depot, depot], 0.0, 0.0

    obj_val, cycle = solve_compass(coords, prizes, max_length, depot=depot)
    if not cycle:
        return [depot, depot], 0.0, 0.0

    if 0 in cycle:
        visited_nodes = [node for node in cycle if node != 0]
    else:
        visited_nodes = [node - 1 for node in cycle if node != 1]

    tour = [depot] + visited_nodes + [depot]

    def _route_length(route):
        return sum(float(np.linalg.norm(coords[route[k]] - coords[route[k + 1]]))
                   for k in range(len(route) - 1))

    travel_len = _route_length(tour)
    tol = 1e-5
    while travel_len > float(max_length) + tol and len(tour) > 2:
        candidate = tour[-2]
        if candidate == depot:
            break
        tour.pop(-2)
        travel_len = _route_length(tour)

    if travel_len > float(max_length) + tol:
        tour = [depot, depot]
        travel_len = _route_length(tour)

    visited_set = set(tour) - {depot}
    collected = float(sum(prizes[i] for i in visited_set))
    return tour, collected, float(travel_len)


def solve_op_gurobi_mip(coords, prizes, max_length, depot=0, time_limit=60.0, threads=8):
    """Exact Orienteering Problem (OP) MILP solver with Gurobi."""
    import numpy as np
    import gurobipy as gp
    from gurobipy import GRB
    n = coords.shape[0]
    if n < 2:
        return [0, 0], 0.0, 0.0

    depot = 0
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(np.linalg.norm(coords[i] - coords[j]))

    model = gp.Model("OP_cycle")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    u = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=float(n - 1), name="u")

    model.addConstr(y[depot] == 1, "visit_depot")
    for i in range(n):
        if i == depot:
            model.addConstr(gp.quicksum(x[depot, j] for j in range(n) if j != depot) == 1, "out_depot")
            model.addConstr(gp.quicksum(x[j, depot] for j in range(n) if j != depot) == 1, "in_depot")
        else:
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == y[i], f"out_{i}")
            model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == y[i], f"in_{i}")

    for i in range(n):
        model.addConstr(x[i, i] == 0, f"no_loop_{i}")

    model.addConstr(u[depot] == 0, "u_depot_zero")
    for i in range(n):
        if i == depot:
            continue
        model.addConstr(u[i] <= (n - 1) * y[i], f"u_upper_{i}")
        model.addConstr(u[i] >= y[i], f"u_lower_{i}")

    for i in range(n):
        if i == depot:
            continue
        for j in range(n):
            if j == depot or i == j:
                continue
            model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, f"mtz_{i}_{j}")

    travel_cost = gp.quicksum(D[i, j] * x[i, j] for i in range(n) for j in range(n))
    model.addConstr(travel_cost <= float(max_length), "length_budget")
    model.setObjective(gp.quicksum(float(prizes[i]) * y[i] for i in range(n)), GRB.MAXIMIZE)

    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        return [depot, depot], 0.0, 0.0

    succ = {i: None for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and x[i, j].X > 0.5:
                succ[i] = j

    tour = [depot]
    cur = depot
    seen = {depot}
    while True:
        nxt = succ[cur]
        if nxt is None:
            break
        tour.append(nxt)
        if nxt == depot:
            break
        if nxt in seen:
            break
        seen.add(nxt)
        cur = nxt

    if tour[0] != depot:
        tour.insert(0, depot)
    if tour[-1] != depot:
        tour.append(depot)

    travel_len = sum(float(np.linalg.norm(coords[tour[k]] - coords[tour[k + 1]]))
                     for k in range(len(tour) - 1))
    visited_set = set(tour) - {depot}
    collected = float(sum(prizes[i] for i in visited_set))
    return tour, collected, float(travel_len)


def solve_pdp_gurobi(coords, demands, time_windows, pickup_delivery_pairs,
                     depot=0, time_limit=60.0, threads=8):
    """Solve single-vehicle Pickup and Delivery Problem with Time Windows using Gurobi."""
    import numpy as np
    import gurobipy as gp
    from gurobipy import GRB
    n = coords.shape[0]
    if time_windows.shape != (n, 2):
        raise ValueError("time_windows must have shape (n, 2)")

    start = depot
    end = n
    N = n + 1

    coords_ext = np.vstack([coords, coords[depot]])
    tw_ext = np.vstack([time_windows, time_windows[depot]])

    dist = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                dist[i, j] = float(np.linalg.norm(coords_ext[i] - coords_ext[j]))

    max_tw = float(np.max(tw_ext[:, 1]))
    max_d = float(np.max(dist))
    big_M_time = max_tw + max_d * N + 10.0

    model = gp.Model("PDPTW_no_capacity")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("Threads", threads)

    x = model.addVars(N, N, vtype=GRB.BINARY, name="x")
    u = model.addVars(N, lb=0, ub=N - 1, vtype=GRB.CONTINUOUS, name="u")
    t = model.addVars(N, lb=0, ub=big_M_time, vtype=GRB.CONTINUOUS, name="t")

    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j] for i in range(N) for j in range(N) if i != j),
        GRB.MINIMIZE,
    )

    for i in range(N):
        model.addConstr(x[i, i] == 0, name=f"no_self_{i}")

    model.addConstr(gp.quicksum(x[start, j] for j in range(N) if j != start) == 1, "start_out")
    model.addConstr(gp.quicksum(x[j, start] for j in range(N) if j != start) == 0, "start_in")
    model.addConstr(gp.quicksum(x[i, end] for i in range(N) if i != end) == 1, "end_in")
    model.addConstr(gp.quicksum(x[end, j] for j in range(N) if j != end) == 0, "end_out")

    for v in range(N):
        if v in (start, end):
            continue
        model.addConstr(gp.quicksum(x[v, j] for j in range(N) if j != v) == 1, name=f"out_{v}")
        model.addConstr(gp.quicksum(x[i, v] for i in range(N) if i != v) == 1, name=f"in_{v}")

    model.addConstr(u[start] == 0, "u_start")
    model.addConstr(u[end] == N - 1, "u_end")

    for i in range(N):
        for j in range(N):
            if i == j or j == start or i == end:
                continue
            model.addConstr(u[j] >= u[i] + 1 - N * (1 - x[i, j]), name=f"mtz_{i}_{j}")

    for i in range(N):
        model.addConstr(t[i] >= float(tw_ext[i, 0]), name=f"tw_lb_{i}")
        model.addConstr(t[i] <= float(tw_ext[i, 1]), name=f"tw_ub_{i}")

    model.addConstr(t[start] == float(tw_ext[start, 0]), "t_start_fixed")

    for i in range(N):
        for j in range(N):
            if i == j or i == end or j == start:
                continue
            model.addConstr(
                t[j] >= t[i] + dist[i, j] - big_M_time * (1 - x[i, j]),
                name=f"timeprop_{i}_{j}",
            )

    for pickup, delivery in pickup_delivery_pairs:
        if pickup == depot or delivery == depot:
            raise ValueError("pickup/delivery indices should not be the depot.")
        model.addConstr(u[pickup] + 1 <= u[delivery], name=f"prec_u_{pickup}_{delivery}")
        model.addConstr(t[pickup] <= t[delivery], name=f"prec_t_{pickup}_{delivery}")

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        raise RuntimeError("PDPTW model is infeasible.")
    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"Gurobi ended with status {model.status}")

    succ = {}
    for i in range(N):
        for j in range(N):
            if i != j and x[i, j].X > 0.5:
                succ[i] = j
                break

    route_ext = [start]
    cur = start
    safety = 0
    while cur != end and safety <= N + 5:
        cur = succ.get(cur, None)
        if cur is None:
            break
        route_ext.append(cur)
        safety += 1

    if not route_ext or route_ext[-1] != end:
        raise RuntimeError("Failed to extract a valid path from solution.")

    route = [v if v != end else depot for v in route_ext]
    total_dist = sum(dist[a, b] for a, b in zip(route_ext[:-1], route_ext[1:]))
    return [route], float(total_dist)


def solve_uflp_gurobi(opening_costs, connection_costs, time_limit=60.0, threads=8):
    """Solve the Uncapacitated Facility Location Problem using Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    n_facilities = len(opening_costs)
    n_customers = connection_costs.shape[1]

    model = gp.Model("UFLP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)

    x = model.addVars(n_facilities, vtype=GRB.BINARY, name="x")
    y = model.addVars(n_facilities, n_customers, vtype=GRB.BINARY, name="y")

    opening_cost = gp.quicksum(opening_costs[i] * x[i] for i in range(n_facilities))
    connection_cost = gp.quicksum(
        connection_costs[i, j] * y[i, j]
        for i in range(n_facilities) for j in range(n_customers)
    )
    model.setObjective(opening_cost + connection_cost, GRB.MINIMIZE)

    for j in range(n_customers):
        model.addConstr(gp.quicksum(y[i, j] for i in range(n_facilities)) == 1, f"serve_{j}")
    for i in range(n_facilities):
        for j in range(n_customers):
            model.addConstr(y[i, j] <= x[i], f"open_{i}_{j}")

    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        open_facilities = [i for i in range(n_facilities) if x[i].X > 0.5]
        assignments = [-1] * n_customers
        for j in range(n_customers):
            for i in range(n_facilities):
                if y[i, j].X > 0.5:
                    assignments[j] = i
                    break
        if -1 in assignments:
            raise RuntimeError("Invalid solution: some customers not assigned")
        return open_facilities, assignments, float(model.objVal)
    else:
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")


def solve_cflp_gurobi(opening_costs, capacities, connection_costs, demands,
                      time_limit=60.0, threads=8):
    """Solve the Capacitated Facility Location Problem using Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    n_facilities = len(opening_costs)
    n_customers = connection_costs.shape[1]

    model = gp.Model("CFLP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)

    x = model.addVars(n_facilities, vtype=GRB.BINARY, name="x")
    y = model.addVars(n_facilities, n_customers, vtype=GRB.BINARY, name="y")

    opening_cost = gp.quicksum(opening_costs[i] * x[i] for i in range(n_facilities))
    connection_cost = gp.quicksum(
        connection_costs[i, j] * y[i, j]
        for i in range(n_facilities) for j in range(n_customers)
    )
    model.setObjective(opening_cost + connection_cost, GRB.MINIMIZE)

    for j in range(n_customers):
        model.addConstr(gp.quicksum(y[i, j] for i in range(n_facilities)) == 1, f"serve_{j}")
    for i in range(n_facilities):
        model.addConstr(
            gp.quicksum(demands[j] * y[i, j] for j in range(n_customers)) <= capacities[i] * x[i],
            f"capacity_{i}"
        )
    for i in range(n_facilities):
        for j in range(n_customers):
            model.addConstr(y[i, j] <= x[i], f"open_{i}_{j}")

    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        open_facilities = [i for i in range(n_facilities) if x[i].X > 0.5]
        assignments = [-1] * n_customers
        for j in range(n_customers):
            for i in range(n_facilities):
                if y[i, j].X > 0.5:
                    assignments[j] = i
                    break
        if -1 in assignments:
            raise RuntimeError("Invalid solution: some customers not assigned")
        return open_facilities, assignments, float(model.objVal)
    else:
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")


def solve_pmed_gurobi(distance_matrix, p, time_limit=60.0, threads=8):
    """Solve the p-median problem using Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    n = distance_matrix.shape[0]
    if p > n:
        raise ValueError(f"Cannot select {p} facilities from {n} vertices")

    model = gp.Model("PMED")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")

    model.setObjective(
        gp.quicksum(distance_matrix[i, j] * y[i, j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE
    )
    model.addConstr(gp.quicksum(x[j] for j in range(n)) == p, "select_p")
    for i in range(n):
        model.addConstr(gp.quicksum(y[i, j] for j in range(n)) == 1, f"assign_{i}")
    for i in range(n):
        for j in range(n):
            model.addConstr(y[i, j] <= x[j], f"open_{i}_{j}")

    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        facilities = [j for j in range(n) if x[j].X > 0.5]
        assignments = [-1] * n
        for i in range(n):
            for j in range(n):
                if y[i, j].X > 0.5:
                    assignments[i] = j
                    break
        if -1 in assignments:
            raise RuntimeError("Invalid solution: some vertices not assigned")
        return facilities, assignments, float(model.objVal)
    else:
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")


def solve_mdp_gurobi(distance_matrix, m, time_limit=60.0, threads=8):
    """Solve the Maximum Diversity Problem using Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    n = distance_matrix.shape[0]
    if m > n:
        raise ValueError(f"Cannot select {m} elements from {n} elements")

    model = gp.Model("MDP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] != 0:
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")

    model.addConstr(gp.quicksum(x[i] for i in range(n)) == m, "select_m")
    for i in range(n):
        for j in range(i + 1, n):
            if distance_matrix[i, j] != 0:
                model.addConstr(y[i, j] <= x[i], f"lin1_{i}_{j}")
                model.addConstr(y[i, j] <= x[j], f"lin2_{i}_{j}")
                model.addConstr(y[i, j] >= x[i] + x[j] - 1, f"lin3_{i}_{j}")

    obj_expr = gp.quicksum(
        distance_matrix[i, j] * y[i, j]
        for i in range(n) for j in range(i + 1, n)
        if distance_matrix[i, j] != 0
    )
    model.setObjective(obj_expr, GRB.MAXIMIZE)
    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        selected = [i for i in range(n) if x[i].X > 0.5]
        obj_value = sum(
            distance_matrix[selected[i], selected[j]]
            for i in range(len(selected)) for j in range(i + 1, len(selected))
        )
        return selected, float(obj_value)
    else:
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")


def solve_pcenter_gurobi(distance_matrix, p, time_limit=60.0, threads=8):
    """Solve the p-center problem using Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    n = distance_matrix.shape[0]
    if p > n:
        raise ValueError(f"Cannot select {p} facilities from {n} vertices")

    model = gp.Model("PCENTER")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")
    z = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="z")

    model.setObjective(z, GRB.MINIMIZE)
    model.addConstr(gp.quicksum(x[j] for j in range(n)) == p, "select_p")
    for i in range(n):
        model.addConstr(gp.quicksum(y[i, j] for j in range(n)) == 1, f"assign_{i}")
    for i in range(n):
        for j in range(n):
            model.addConstr(y[i, j] <= x[j], f"open_{i}_{j}")
    for i in range(n):
        model.addConstr(
            gp.quicksum(distance_matrix[i, j] * y[i, j] for j in range(n)) <= z,
            f"max_dist_{i}"
        )

    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        facilities = [j for j in range(n) if x[j].X > 0.5]
        assignments = [-1] * n
        for i in range(n):
            for j in range(n):
                if y[i, j].X > 0.5:
                    assignments[i] = j
                    break
        if -1 in assignments:
            raise RuntimeError("Invalid solution: some vertices not assigned")
        max_dist = max(distance_matrix[i, assignments[i]] for i in range(n))
        return facilities, assignments, float(max_dist)
    else:
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")


def solve_mlp_gurobi(coords, depot=0, time_limit=60.0, threads=8):
    """Solve the Minimum Latency Problem (MLP) using Gurobi."""
    import numpy as np
    import gurobipy as gp
    from gurobipy import GRB
    n = coords.shape[0]
    V = range(n)
    if n <= 1:
        return [depot, depot], 0.0

    dist = np.zeros((n, n), dtype=float)
    for i in V:
        for j in V:
            if i != j:
                dist[i, j] = float(np.linalg.norm(coords[i] - coords[j]))

    model = gp.Model("MLP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    max_dist = float(np.sum(dist.max(axis=1)))
    t = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, ub=max_dist, name="t")

    model.setObjective(gp.quicksum(t[i] for i in V if i != depot), GRB.MINIMIZE)

    for i in V:
        model.addConstr(x[i, i] == 0, f"no_self_{i}")
    for i in V:
        model.addConstr(gp.quicksum(x[i, j] for j in V if j != i) == 1, f"out_{i}")
    for j in V:
        model.addConstr(gp.quicksum(x[i, j] for i in V if i != j) == 1, f"in_{j}")

    model.addConstr(t[depot] == 0, "depot_start")
    M = max_dist
    for i in V:
        for j in V:
            if i == j or j == depot:
                continue
            model.addConstr(t[j] >= t[i] + dist[i, j] - M * (1 - x[i, j]), f"time_{i}_{j}")

    model.optimize()

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError(f"MLP solver failed with status {model.status}")

    succ = {i: None for i in V}
    for i in V:
        for j in V:
            if i != j and x[i, j].X > 0.5:
                succ[i] = j
                break

    tour = [depot]
    current = depot
    visited = {depot}
    while True:
        nxt = succ[current]
        if nxt is None:
            break
        tour.append(nxt)
        if nxt == depot:
            break
        if nxt in visited:
            tour.append(depot)
            break
        visited.add(nxt)
        current = nxt

    if tour[-1] != depot:
        tour.append(depot)

    objective = sum(t[i].X for i in V if i != depot)
    return tour, float(objective)


def solve_2sp_gurobi(items, bin_width, max_height=None, time_limit=120.0, threads=8):
    """Solve 2D Strip Packing Problem using Gurobi."""
    import numpy as np
    import gurobipy as gp
    from gurobipy import GRB

    total_area = sum(w * h * d for w, h, d in items)
    if max_height is None:
        max_item_height = max(h for _, h, _ in items)
        estimated_height = int(np.ceil(total_area / bin_width)) + max_item_height
        max_height = estimated_height * 2

    expanded_items = []
    item_type_map = []
    for type_idx, (w, h, d) in enumerate(items):
        for _ in range(d):
            expanded_items.append((w, h))
            item_type_map.append(type_idx)

    n_items = len(expanded_items)
    if n_items == 0:
        return [], 0.0

    model = gp.Model("2SP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)
    model.setParam('MIPGap', 0.05)

    x = model.addVars(n_items, vtype=GRB.INTEGER, lb=0, ub=bin_width, name="x")
    y = model.addVars(n_items, vtype=GRB.INTEGER, lb=0, ub=max_height, name="y")
    H = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=max_height, name="H")

    left = {}; right = {}; below = {}; above = {}
    for i in range(n_items):
        for j in range(i + 1, n_items):
            left[i, j] = model.addVar(vtype=GRB.BINARY, name=f"left_{i}_{j}")
            right[i, j] = model.addVar(vtype=GRB.BINARY, name=f"right_{i}_{j}")
            below[i, j] = model.addVar(vtype=GRB.BINARY, name=f"below_{i}_{j}")
            above[i, j] = model.addVar(vtype=GRB.BINARY, name=f"above_{i}_{j}")

    model.setObjective(H, GRB.MINIMIZE)

    for i in range(n_items):
        w_i, h_i = expanded_items[i]
        model.addConstr(x[i] + w_i <= bin_width, f"width_{i}")
        model.addConstr(y[i] + h_i <= H, f"height_{i}")

    M = max(bin_width, max_height)
    for i in range(n_items):
        w_i, h_i = expanded_items[i]
        for j in range(i + 1, n_items):
            w_j, h_j = expanded_items[j]
            model.addConstr(left[i,j]+right[i,j]+below[i,j]+above[i,j] >= 1, f"no_overlap_{i}_{j}")
            model.addConstr(x[i]+w_i <= x[j]+M*(1-left[i,j]), f"left_{i}_{j}")
            model.addConstr(x[j]+w_j <= x[i]+M*(1-right[i,j]), f"right_{i}_{j}")
            model.addConstr(y[i]+h_i <= y[j]+M*(1-below[i,j]), f"below_{i}_{j}")
            model.addConstr(y[j]+h_j <= y[i]+M*(1-above[i,j]), f"above_{i}_{j}")

    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        packed_items = []
        for i in range(n_items):
            w_i, h_i = expanded_items[i]
            packed_items.append({
                'item_type': int(item_type_map[i]),
                'x': int(round(x[i].X)),
                'y': int(round(y[i].X)),
                'width': int(w_i),
                'height': int(h_i)
            })
        return packed_items, float(H.X)
    else:
        raise RuntimeError(f"2SP solver failed with status {model.status}")


def solve_qkp_gurobi(linear_coeffs, quadratic_coeffs, weights, capacity,
                     time_limit=60.0, threads=8, mip_gap=0.01):
    """Solve Quadratic Knapsack Problem (QKP) using Gurobi with linearization."""
    import gurobipy as gp
    from gurobipy import GRB
    n = len(linear_coeffs)

    model = gp.Model("QKP")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Threads = threads
    model.Params.MIPGap = mip_gap

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = {}
    for i in range(n):
        for j in range(i + 1, n):
            if quadratic_coeffs[i, j] != 0:
                y[i, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
                model.addConstr(y[i, j] <= x[i], f"lin1_{i}_{j}")
                model.addConstr(y[i, j] <= x[j], f"lin2_{i}_{j}")
                model.addConstr(y[i, j] >= x[i] + x[j] - 1, f"lin3_{i}_{j}")

    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, "capacity")

    linear_obj = gp.quicksum(linear_coeffs[i] * x[i] for i in range(n))
    quadratic_obj = gp.quicksum(
        quadratic_coeffs[i, j] * y[i, j]
        for i in range(n) for j in range(i + 1, n)
        if quadratic_coeffs[i, j] != 0
    )
    model.setObjective(linear_obj + quadratic_obj, GRB.MAXIMIZE)
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        return [], 0.0

    solution = [i for i in range(n) if x[i].X > 0.5]
    return solution, float(model.objVal)


def solve_gap_gurobi(resource_consumption, assignment_costs, capacities,
                     time_limit=60.0, threads=8):
    """Solve the Generalized Assignment Problem (GAP) using Gurobi."""
    import gurobipy as gp
    from gurobipy import GRB
    n_agents = resource_consumption.shape[0]
    n_tasks = resource_consumption.shape[1]

    model = gp.Model("GAP")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)
    model.setParam('Threads', threads)

    x = model.addVars(n_agents, n_tasks, vtype=GRB.BINARY, name="x")
    model.setObjective(
        gp.quicksum(
            assignment_costs[i, j] * x[i, j]
            for i in range(n_agents) for j in range(n_tasks)
        ),
        GRB.MINIMIZE
    )

    for j in range(n_tasks):
        model.addConstr(gp.quicksum(x[i, j] for i in range(n_agents)) == 1, f"task_{j}")
    for i in range(n_agents):
        model.addConstr(
            gp.quicksum(resource_consumption[i, j] * x[i, j] for j in range(n_tasks)) <= capacities[i],
            f"capacity_{i}"
        )

    model.optimize()

    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        assignments = [-1] * n_tasks
        for j in range(n_tasks):
            for i in range(n_agents):
                if x[i, j].X > 0.5:
                    assignments[j] = i
                    break
        if -1 in assignments:
            raise RuntimeError("Invalid solution: some tasks not assigned")
        return assignments, float(model.objVal)
    else:
        raise RuntimeError(f"Gurobi optimization failed with status {model.status}")
