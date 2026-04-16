from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL4 = "Global4"  # assignment constraints (permutation)


def _build_location_map(locations: List[Any], n: int) -> Dict[Any, int]:
    if len(locations) != n:
        raise ValueError(f"'locations' has {len(locations)} entries, expected {n}.")
    mapping: Dict[Any, int] = {}
    for idx, identifier in enumerate(locations):
        if identifier in mapping:
            raise ValueError(f"Duplicate identifier '{identifier}' in 'locations'.")
        mapping[identifier] = idx
    return mapping


def _map_location(
    value: Any, *, field: str, location_map: Dict[Any, int], n: int
) -> int:
    if location_map:
        if value not in location_map:
            raise ValueError(f"{field} value {value!r} not found in 'locations'.")
        return location_map[value]
    if not isinstance(value, int):
        raise ValueError(f"{field} value {value!r} must be integer-like.")
    if value < 0 or value >= n:
        raise ValueError(f"{field} index {value} out of range 0-{n-1}.")
    return value


def qap_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse QAP (Quadratic Assignment Problem) instance from `instance_variant`.

    Expected `instance_variant` format:
      {
        "num_facilities": int,
        "num_locations": int,
        "facilities": [str, ...],  # e.g., ['F1', 'F2', 'F3']
        "locations": [int, ...],  # e.g., [0, 1, 2]
        "distance_pairs": [
          {"from_id": int, "to_id": int, "distance": float},
          ...
        ],
        "flow_pairs": [
          {"from_id": str, "to_id": str, "flow": float},
          ...
        ]
      }

    Return:
      {
        "distance_matrix": n×n matrix of distances between locations,
        "flow_matrix": n×n matrix of flows between facilities,
        "n": problem size
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for QAP.")

    num_facilities = inst_var.get("num_facilities", None)
    num_locations = inst_var.get("num_locations", None)
    facilities = inst_var.get("facilities", None)
    locations = inst_var.get("locations", None)
    distance_pairs = inst_var.get("distance_pairs", None)
    flow_pairs = inst_var.get("flow_pairs", None)

    if num_facilities is None:
        raise ValueError("QAP instance missing 'num_facilities'.")
    if num_locations is None:
        raise ValueError("QAP instance missing 'num_locations'.")
    if distance_pairs is None:
        raise ValueError("QAP instance missing 'distance_pairs'.")
    if flow_pairs is None:
        raise ValueError("QAP instance missing 'flow_pairs'.")

    if not isinstance(num_facilities, int) or num_facilities <= 0:
        raise ValueError(f"Invalid num_facilities: {num_facilities}")
    if not isinstance(num_locations, int) or num_locations <= 0:
        raise ValueError(f"Invalid num_locations: {num_locations}")

    n = num_facilities
    if num_locations != n:
        raise ValueError(f"num_facilities ({n}) must equal num_locations ({num_locations}) for QAP.")

    # Build facility name to index mapping
    if facilities is not None and isinstance(facilities, list):
        if len(facilities) != n:
            raise ValueError(f"'facilities' has {len(facilities)} entries, expected {n}.")
        facility_name_to_idx = {name: idx for idx, name in enumerate(facilities)}
    else:
        facility_name_to_idx = {}

    # Build location identifier mapping
    if locations is not None:
        if not isinstance(locations, list):
            raise ValueError("'locations' must be a list.")
        location_map = _build_location_map(locations, n)
    else:
        location_map = {i: i for i in range(n)}
    location_set = set(location_map.keys())

    # Initialize matrices
    distance_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    flow_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]

    # Build distance matrix from distance_pairs
    if not isinstance(distance_pairs, list):
        raise ValueError("'distance_pairs' must be a list.")
    
    for pair in distance_pairs:
        if not isinstance(pair, dict):
            raise ValueError("Each distance_pair must be a dictionary.")
        
        from_id = pair.get("from_id", None)
        to_id = pair.get("to_id", None)
        distance = pair.get("distance", None)

        if from_id is None or to_id is None:
            raise ValueError("distance_pair missing 'from_id' or 'to_id'.")
        if distance is None:
            raise ValueError("distance_pair missing 'distance'.")

        from_idx = _map_location(from_id, field="distance_pair['from_id']", location_map=location_map, n=n)
        to_idx = _map_location(to_id, field="distance_pair['to_id']", location_map=location_map, n=n)

        try:
            dist_value = float(distance)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid distance value: {distance}")

        distance_matrix[from_idx][to_idx] = dist_value

    # Build flow matrix from flow_pairs
    if not isinstance(flow_pairs, list):
        raise ValueError("'flow_pairs' must be a list.")
    
    for pair in flow_pairs:
        if not isinstance(pair, dict):
            raise ValueError("Each flow_pair must be a dictionary.")
        
        from_id = pair.get("from_id", None)
        to_id = pair.get("to_id", None)
        flow = pair.get("flow", None)

        if from_id is None or to_id is None:
            raise ValueError("flow_pair missing 'from_id' or 'to_id'.")
        if flow is None:
            raise ValueError("flow_pair missing 'flow'.")

        # Map facility names to indices
        if isinstance(from_id, str):
            if from_id not in facility_name_to_idx:
                raise ValueError(f"flow_pair from_id '{from_id}' not found in facilities.")
            from_idx = facility_name_to_idx[from_id]
        else:
            # Try as integer
            if not isinstance(from_id, int) or from_id < 0 or from_id >= n:
                raise ValueError(f"Invalid flow_pair from_id: {from_id}")
            from_idx = from_id

        if isinstance(to_id, str):
            if to_id not in facility_name_to_idx:
                raise ValueError(f"flow_pair to_id '{to_id}' not found in facilities.")
            to_idx = facility_name_to_idx[to_id]
        else:
            # Try as integer
            if not isinstance(to_id, int) or to_id < 0 or to_id >= n:
                raise ValueError(f"Invalid flow_pair to_id: {to_id}")
            to_idx = to_id

        try:
            flow_value = float(flow)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid flow value: {flow}")

        flow_matrix[from_idx][to_idx] = flow_value

    return {
        "distance_matrix": distance_matrix,
        "flow_matrix": flow_matrix,
        "n": n,
        "location_id2idx": location_map,
    }


def _map_solution_location(value: Any, *, field: str, location_map: Dict[Any, int], n: int) -> Tuple[Optional[int], Optional[str]]:
    if location_map and value in location_map:
        return location_map[value], None
    if not isinstance(value, int):
        return None, f"{field} value {value!r} must be integer-like."
    if value < 0 or value >= n:
        return None, f"{field} index {value} out of range 0-{n-1}."
    return value, None


def qap_check_feasibility(
    solution: List[Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check QAP feasibility + unified violation pattern.

    Expected solution format:
      solution = [1, 2, 0]  # permutation where solution[i] = j means facility i is at location j

    Patterns for QAP (per taxonomy Global_4):
      - FormatError: invalid format/types, invalid indices
      - Global4: assignment violations (not a valid permutation)

    Constraints:
      1) solution is a list of n integers.
      2) solution must be a valid permutation of [0, 1, ..., n-1].
      3) each facility i is assigned to exactly one location solution[i] (Global4).
      4) each location j is assigned exactly one facility (Global4).
    """
    distance_matrix: List[List[float]] = instance["distance_matrix"]
    flow_matrix: List[List[float]] = instance["flow_matrix"]
    n: int = instance["n"]

    location_map = instance.get("location_id2idx", {})
    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list of location assignments."

    # Check length
    if len(solution) != n:
        return False, PATTERN_GLOBAL4, (
            f"Solution must have exactly {n} assignments, but has {len(solution)}."
        )

    # Validate all assignments
    assigned_locations = set()

    for i, location in enumerate(solution):
        mapped_loc, err = _map_solution_location(location, field=f"solution[{i}]", location_map=location_map, n=n)
        if err:
            return False, PATTERN_FORMAT, err
        location = mapped_loc

        # ---- Global4: Check that each location is used at most once ----
        if location in assigned_locations:
            return False, PATTERN_GLOBAL4, (
                f"Facility {i} assigned to location {location}, "
                f"but location {location} is already assigned to another facility."
            )

        assigned_locations.add(location)

    # ---- Global4: Check that all locations are used ----
    all_locations = set(range(n))
    missing_locations = all_locations - assigned_locations

    if missing_locations:
        return False, PATTERN_GLOBAL4, (
            f"Some locations are not assigned any facility: {sorted(missing_locations)}."
        )

    return True, PATTERN_OK, "Feasible QAP solution."


def qap_objective_model(
    solution: List[Any],
    instance: Dict[str, Any],
) -> float:
    """
    QAP objective: minimize the sum of flow×distance products.

    Objective = sum(flow[i][j] × distance[solution[i]][solution[j]] for all i, j)

    This represents the total cost of assigning facilities to locations, where:
    - flow[i][j] is the flow/interaction between facility i and facility j
    - distance[loc1][loc2] is the distance between location loc1 and location loc2
    - solution[i] is the location assigned to facility i

    Assumes solution is feasible. If not well-formed, return inf.
    """
    if not isinstance(solution, list):
        return float("inf")

    distance_matrix: List[List[float]] = instance["distance_matrix"]
    flow_matrix: List[List[float]] = instance["flow_matrix"]
    n: int = instance["n"]

    if len(solution) != n:
        return float("inf")
    location_map = instance.get("location_id2idx", {})

    mapped_solution: List[int] = []
    for idx, location in enumerate(solution):
        mapped_loc, err = _map_solution_location(location, field=f"solution[{idx}]", location_map=location_map, n=n)
        if err:
            return float("inf")
        mapped_solution.append(mapped_loc)

    # Compute total cost
    total_cost = 0.0
    for i in range(n):
        loc_i = mapped_solution[i]

        if not isinstance(loc_i, int) or loc_i < 0 or loc_i >= n:
            return float("inf")

        for j in range(n):
            loc_j = mapped_solution[j]

            if not isinstance(loc_j, int) or loc_j < 0 or loc_j >= n:
                return float("inf")

            # Cost contribution from assigning facility i to loc_i and facility j to loc_j
            total_cost += flow_matrix[i][j] * distance_matrix[loc_i][loc_j]

    return float(total_cost)


#instance =  {'distance_matrix': [[0.0, 2.0, 1.0, 2.0, 2.0, 3.0, 1.0, 5.0], [2.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0, 5.0], [1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0], [2.0, 2.0, 1.0, 0.0, 2.0, 3.0, 3.0, 3.0], [2.0, 2.0, 1.0, 2.0, 0.0, 1.0, 3.0, 3.0], [3.0, 3.0, 2.0, 3.0, 1.0, 0.0, 4.0, 2.0], [1.0, 3.0, 2.0, 3.0, 3.0, 4.0, 0.0, 6.0], [5.0, 5.0, 4.0, 3.0, 3.0, 2.0, 6.0, 0.0]], 'flow_matrix': [[0.0, 2.0, 6.0, 4.0, 4.0, 4.0, 2.0, 6.0], [2.0, 0.0, 5.0, 5.0, 3.0, 3.0, 9.0, 6.0], [6.0, 5.0, 0.0, 4.0, 3.0, 4.0, 5.0, 7.0], [4.0, 5.0, 4.0, 0.0, 8.0, 5.0, 5.0, 5.0], [4.0, 3.0, 3.0, 8.0, 0.0, 6.0, 8.0, 7.0], [4.0, 3.0, 4.0, 5.0, 6.0, 0.0, 1.0, 3.0], [2.0, 9.0, 5.0, 5.0, 8.0, 1.0, 0.0, 2.0], [6.0, 6.0, 7.0, 5.0, 7.0, 3.0, 2.0, 0.0]], 'n': 8, 'location_id2idx': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}}
#solution = [8, 1, 6, 4, 3, 2, 7, 5]
#print(qap_objective_model(solution, instance))