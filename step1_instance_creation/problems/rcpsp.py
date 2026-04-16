"""
RCPSP (Resource-Constrained Project Scheduling Problem) instance generation.

This module generates RCPSP instances by:
1. Loading existing .rcp files from the dataset
2. Modifying parameters (durations, resource demands, capacities) while preserving structure
3. Solving instances using OR-Tools CP-SAT
4. Saving instances in JSON format
"""

import time
import json
import random
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List, Optional

from ..solvers.scheduling_models.rcpsp_solver import solve_rcpsp


def parse_rcp_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse an RCPSP .rcp file.

    File format:
    - Line 1: number of tasks, number of resources (tab-separated)
    - Line 2: empty
    - Line 3: resource capacities (tab-separated)
    - Line 4: empty
    - Remaining lines: task data (tab-separated)
      For each task: duration, resource demands (one per resource),
                     number of successors, list of successor task IDs (1-indexed)

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the .rcp file

    Returns
    -------
    Dict[str, Any]
        Parsed instance data with format:
        {
            "nr_tasks": int,
            "nr_resources": int,
            "capacities": [int, ...],
            "tasks": [
                {
                    "duration": int,
                    "demands": [int, ...],
                    "successors": [int, ...]  # 0-indexed task IDs
                },
                ...
            ]
        }
    """
    file_path = Path(file_path)
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    header_parts = lines[0].split()
    if len(header_parts) < 2:
        raise ValueError(f"Invalid header format in {file_path}: expected 'num_tasks num_resources'")
    num_tasks = int(header_parts[0])
    num_resources = int(header_parts[1])

    capacities_line = lines[1]
    capacities = [int(x) for x in capacities_line.split()]
    if len(capacities) != num_resources:
        raise ValueError(f"Number of capacities ({len(capacities)}) doesn't match num_resources ({num_resources})")

    tasks = []
    task_lines = lines[2:]

    if len(task_lines) != num_tasks:
        raise ValueError(f"Number of task lines ({len(task_lines)}) doesn't match num_tasks ({num_tasks})")

    for task_id, line in enumerate(task_lines):
        parts = [int(x) for x in line.split()]

        if len(parts) < num_resources + 1:
            raise ValueError(f"Task {task_id}: insufficient data (expected at least {num_resources + 1} values)")

        duration = parts[0]
        demands = parts[1:1 + num_resources]

        if len(parts) < num_resources + 2:
            num_successors = 0
            successors_raw = []
        else:
            num_successors = parts[num_resources + 1]
            if num_successors > 0:
                successors_raw = parts[num_resources + 2:num_resources + 2 + num_successors]
            else:
                successors_raw = []

        successors = [s - 1 for s in successors_raw if 0 < s <= num_tasks]

        tasks.append({
            "duration": duration,
            "demands": demands,
            "successors": successors
        })

    return {
        "nr_tasks": num_tasks,
        "nr_resources": num_resources,
        "capacities": capacities,
        "tasks": tasks
    }


def modify_instance(
    instance: Dict[str, Any],
    duration_factor_range: Tuple[float, float] = (0.5, 2.0),
    demand_factor_range: Tuple[float, float] = (0.5, 2.0),
    capacity_factor_range: Tuple[float, float] = (0.8, 1.5),
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Modify an RCPSP instance by scaling durations, demands, and capacities.
    """
    if rng is None:
        rng = random

    modified = {
        "nr_tasks": instance["nr_tasks"],
        "nr_resources": instance["nr_resources"],
        "capacities": instance["capacities"].copy(),
        "tasks": []
    }

    for task in instance["tasks"]:
        duration_factor = rng.uniform(*duration_factor_range)
        new_duration = max(1, int(round(task["duration"] * duration_factor)))

        new_demands = []
        for demand in task["demands"]:
            demand_factor = rng.uniform(*demand_factor_range)
            new_demand = max(0, int(round(demand * demand_factor)))
            new_demands.append(new_demand)

        modified["tasks"].append({
            "duration": new_duration,
            "demands": new_demands,
            "successors": task["successors"].copy()
        })

    max_demands = [0] * modified["nr_resources"]
    for task in modified["tasks"]:
        for r in range(modified["nr_resources"]):
            max_demands[r] = max(max_demands[r], task["demands"][r])

    for r in range(modified["nr_resources"]):
        capacity_factor = rng.uniform(*capacity_factor_range)
        original_capacity = instance["capacities"][r]
        new_capacity = max(max_demands[r], int(round(original_capacity * capacity_factor)))
        modified["capacities"][r] = new_capacity

    return modified


def extend_instance_tasks(
    instance: Dict[str, Any],
    target_num_tasks: int,
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Extend an RCPSP instance by adding tasks to reach the target number.
    """
    if rng is None:
        rng = random

    current_num_tasks = instance["nr_tasks"]
    num_resources = instance["nr_resources"]
    capacities = instance["capacities"]

    if target_num_tasks <= current_num_tasks:
        return {
            "nr_tasks": instance["nr_tasks"],
            "nr_resources": instance["nr_resources"],
            "capacities": instance["capacities"].copy(),
            "tasks": [
                {
                    "duration": task["duration"],
                    "demands": task["demands"].copy(),
                    "successors": task["successors"].copy()
                }
                for task in instance["tasks"]
            ]
        }

    extended = {
        "nr_tasks": target_num_tasks,
        "nr_resources": num_resources,
        "capacities": capacities.copy(),
        "tasks": [
            {
                "duration": task["duration"],
                "demands": task["demands"].copy(),
                "successors": task["successors"].copy()
            }
            for task in instance["tasks"]
        ]
    }

    tasks_to_add = target_num_tasks - current_num_tasks

    existing_durations = [task["duration"] for task in instance["tasks"]]
    existing_demands = []
    for task in instance["tasks"]:
        existing_demands.append(task["demands"])

    avg_duration = sum(existing_durations) / len(existing_durations) if existing_durations else 1
    min_duration = min(existing_durations) if existing_durations else 1
    max_duration = max(existing_durations) if existing_durations else 3

    for i in range(tasks_to_add):
        duration = max(1, int(round(rng.uniform(min_duration, max_duration))))

        if existing_demands:
            base_demands = rng.choice(existing_demands)
            demands = []
            for r in range(num_resources):
                demand = max(0, int(round(base_demands[r] * rng.uniform(0.5, 1.5))))
                demands.append(demand)
        else:
            demands = [rng.randint(0, max(1, capacities[r] // 2)) for r in range(num_resources)]

        extended["tasks"].append({
            "duration": duration,
            "demands": demands,
            "successors": []
        })

    return extended


def reduce_instance_tasks(
    instance: Dict[str, Any],
    target_num_tasks: int,
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Reduce an RCPSP instance by removing tasks to reach the target number.
    """
    if rng is None:
        rng = random

    current_num_tasks = instance["nr_tasks"]

    if target_num_tasks >= current_num_tasks:
        return {
            "nr_tasks": instance["nr_tasks"],
            "nr_resources": instance["nr_resources"],
            "capacities": instance["capacities"].copy(),
            "tasks": [
                {
                    "duration": task["duration"],
                    "demands": task["demands"].copy(),
                    "successors": task["successors"].copy()
                }
                for task in instance["tasks"]
            ]
        }

    if target_num_tasks < 1:
        raise ValueError(f"target_num_tasks must be at least 1, got {target_num_tasks}")

    reduced = {
        "nr_tasks": target_num_tasks,
        "nr_resources": instance["nr_resources"],
        "capacities": instance["capacities"].copy(),
        "tasks": []
    }

    task_scores = []
    for task_id in range(current_num_tasks):
        task = instance["tasks"][task_id]
        num_successors = len(task["successors"])
        num_predecessors = sum(
            1 for t in instance["tasks"]
            if task_id in t["successors"]
        )
        score = num_predecessors + num_successors
        task_scores.append((score, task_id))

    task_scores.sort(reverse=True)
    tasks_to_keep = set(task_id for _, task_id in task_scores[:target_num_tasks])

    old_to_new = {}
    new_task_id = 0
    for old_task_id in sorted(tasks_to_keep):
        old_to_new[old_task_id] = new_task_id
        new_task_id += 1

    for old_task_id in sorted(tasks_to_keep):
        old_task = instance["tasks"][old_task_id]
        new_successors = [
            old_to_new[succ_id]
            for succ_id in old_task["successors"]
            if succ_id in tasks_to_keep
        ]

        reduced["tasks"].append({
            "duration": old_task["duration"],
            "demands": old_task["demands"].copy(),
            "successors": new_successors
        })

    return reduced


def reduce_instance_precedence(
    instance: Dict[str, Any],
    target_num_precedence: int,
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Reduce an RCPSP instance by removing precedence relations to reach the target number.
    """
    if rng is None:
        rng = random

    current_precedence = sum(len(task["successors"]) for task in instance["tasks"])

    if target_num_precedence >= current_precedence:
        return {
            "nr_tasks": instance["nr_tasks"],
            "nr_resources": instance["nr_resources"],
            "capacities": instance["capacities"].copy(),
            "tasks": [
                {
                    "duration": task["duration"],
                    "demands": task["demands"].copy(),
                    "successors": task["successors"].copy()
                }
                for task in instance["tasks"]
            ]
        }

    if target_num_precedence < 0:
        raise ValueError(f"target_num_precedence must be non-negative, got {target_num_precedence}")

    reduced = {
        "nr_tasks": instance["nr_tasks"],
        "nr_resources": instance["nr_resources"],
        "capacities": instance["capacities"].copy(),
        "tasks": []
    }

    all_precedence = []
    for task_id, task in enumerate(instance["tasks"]):
        for succ_id in task["successors"]:
            all_precedence.append((task_id, succ_id))

    if target_num_precedence == 0:
        precedence_set = set()
    else:
        precedence_to_keep = rng.sample(all_precedence, min(target_num_precedence, len(all_precedence)))
        precedence_set = set(precedence_to_keep)

    for task_id, task in enumerate(instance["tasks"]):
        new_successors = [
            succ_id
            for succ_id in task["successors"]
            if (task_id, succ_id) in precedence_set
        ]

        reduced["tasks"].append({
            "duration": task["duration"],
            "demands": task["demands"].copy(),
            "successors": new_successors
        })

    return reduced


def extend_instance_precedence(
    instance: Dict[str, Any],
    target_num_precedence: int,
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Extend an RCPSP instance by adding precedence relations to reach the target number.
    """
    if rng is None:
        rng = random

    num_tasks = instance["nr_tasks"]
    current_precedence = sum(len(task["successors"]) for task in instance["tasks"])

    if target_num_precedence <= current_precedence:
        return {
            "nr_tasks": instance["nr_tasks"],
            "nr_resources": instance["nr_resources"],
            "capacities": instance["capacities"].copy(),
            "tasks": [
                {
                    "duration": task["duration"],
                    "demands": task["demands"].copy(),
                    "successors": task["successors"].copy()
                }
                for task in instance["tasks"]
            ]
        }

    extended = {
        "nr_tasks": instance["nr_tasks"],
        "nr_resources": instance["nr_resources"],
        "capacities": instance["capacities"].copy(),
        "tasks": []
    }

    for task in instance["tasks"]:
        extended["tasks"].append({
            "duration": task["duration"],
            "demands": task["demands"].copy(),
            "successors": task["successors"].copy()
        })

    precedence_to_add = target_num_precedence - current_precedence

    existing_precedence = set()
    for task_id, task in enumerate(extended["tasks"]):
        for succ_id in task["successors"]:
            existing_precedence.add((task_id, succ_id))

    attempts = 0
    max_attempts = precedence_to_add * 10

    while len(existing_precedence) < target_num_precedence and attempts < max_attempts:
        attempts += 1

        task1 = rng.randint(0, num_tasks - 1)
        task2 = rng.randint(0, num_tasks - 1)

        if task1 == task2:
            continue

        def has_path(from_task: int, to_task: int, visited: set) -> bool:
            if from_task == to_task:
                return True
            if from_task in visited:
                return False
            visited.add(from_task)
            for succ in extended["tasks"][from_task]["successors"]:
                if has_path(succ, to_task, visited):
                    return True
            return False

        if has_path(task2, task1, set()):
            continue

        if (task1, task2) not in existing_precedence:
            extended["tasks"][task1]["successors"].append(task2)
            existing_precedence.add((task1, task2))

    if len(existing_precedence) < target_num_precedence:
        print(
            f"[Warn] Could not add all requested precedence relations. "
            f"Requested: {target_num_precedence}, Actual: {len(existing_precedence)}. "
            f"This may be due to cycle constraints or insufficient attempts."
        )

    return extended


def instance_to_json(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert RCPSP instance to JSON format (already in correct format, just return as-is).
    """
    return instance


def get_random_rcp_file(
    dataset_dir: Union[str, Path],
    num_tasks_range: Optional[Tuple[int, int]] = None,
    num_precedence_range: Optional[Tuple[int, int]] = None,
    rng: Optional[random.Random] = None
) -> Path:
    """
    Get a random .rcp file from the dataset directory, optionally filtered by criteria.
    """
    if rng is None:
        rng = random

    dataset_dir = Path(dataset_dir)
    all_rcp_files = list(dataset_dir.glob("*.rcp"))

    if not all_rcp_files:
        raise ValueError(f"No .rcp files found in {dataset_dir}")

    if num_tasks_range is None and num_precedence_range is None:
        return rng.choice(all_rcp_files)

    filtered_files = []
    for rcp_file in all_rcp_files:
        try:
            instance = parse_rcp_file(rcp_file)
            num_tasks = instance["nr_tasks"]

            num_precedence = sum(len(task["successors"]) for task in instance["tasks"])

            matches = True

            if num_tasks_range is not None:
                min_tasks, max_tasks = num_tasks_range
                if not (min_tasks <= num_tasks <= max_tasks):
                    matches = False

            if num_precedence_range is not None and matches:
                min_prec, max_prec = num_precedence_range
                if not (min_prec <= num_precedence <= max_prec):
                    matches = False

            if matches:
                filtered_files.append(rcp_file)
        except Exception:
            continue

    if not filtered_files:
        criteria_str = []
        if num_tasks_range:
            criteria_str.append(f"tasks={num_tasks_range[0]}-{num_tasks_range[1]}")
        if num_precedence_range:
            criteria_str.append(f"precedence={num_precedence_range[0]}-{num_precedence_range[1]}")
        raise ValueError(
            f"No .rcp files found matching criteria: {', '.join(criteria_str)}. "
            f"Total files checked: {len(all_rcp_files)}"
        )

    return rng.choice(filtered_files)


def generate_rcpsp_instance(
    dataset_dir: Union[str, Path],
    instance_idx: int,
    num_tasks: Optional[Union[int, Tuple[int, int]]] = None,
    num_precedence_relations: Optional[Union[int, Tuple[int, int]]] = None,
    duration_factor_range: Tuple[float, float] = (0.5, 2.0),
    demand_factor_range: Tuple[float, float] = (0.5, 2.0),
    capacity_factor_range: Tuple[float, float] = (0.8, 1.5),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a single RCPSP instance by loading and modifying a random base instance.
    """
    rng = random.Random(seed + instance_idx)

    num_tasks_range = None
    if num_tasks is not None:
        if isinstance(num_tasks, (tuple, list)) and len(num_tasks) == 2:
            num_tasks_range = tuple(num_tasks)
        elif isinstance(num_tasks, int):
            num_tasks_range = (num_tasks, num_tasks)
        else:
            raise ValueError(f"Invalid num_tasks format: {num_tasks}. Expected int or tuple (min, max)")

    num_precedence_range = None
    if num_precedence_relations is not None:
        if isinstance(num_precedence_relations, (tuple, list)) and len(num_precedence_relations) == 2:
            num_precedence_range = tuple(num_precedence_relations)
        elif isinstance(num_precedence_relations, int):
            num_precedence_range = (num_precedence_relations, num_precedence_relations)
        else:
            raise ValueError(
                f"Invalid num_precedence_relations format: {num_precedence_relations}. "
                f"Expected int or tuple (min, max)"
            )

    target_num_tasks = None
    if num_tasks_range and num_tasks_range[0] == num_tasks_range[1]:
        target_num_tasks = num_tasks_range[0]

    target_num_precedence = None
    if num_precedence_range and num_precedence_range[0] == num_precedence_range[1]:
        target_num_precedence = num_precedence_range[0]

    max_retries = 5
    base_file = None
    base_instance = None

    for retry in range(max_retries):
        try:
            base_file = get_random_rcp_file(
                dataset_dir,
                num_tasks_range=num_tasks_range,
                num_precedence_range=num_precedence_range,
                rng=rng
            )
            base_instance = parse_rcp_file(base_file)
            break
        except ValueError as e:
            if retry < max_retries - 1 and (target_num_tasks is not None or target_num_precedence is not None):
                if target_num_tasks is not None:
                    min_allowed = max(1, int(target_num_tasks * 0.5))
                    max_allowed = int(target_num_tasks * 2.0)
                    num_tasks_range = (min_allowed, max_allowed)
                if target_num_precedence is not None:
                    min_allowed = max(0, int(target_num_precedence * 0.5))
                    max_allowed = int(target_num_precedence * 2.0)
                    num_precedence_range = (min_allowed, max_allowed)
                continue
            else:
                if retry == max_retries - 1:
                    try:
                        base_file = get_random_rcp_file(dataset_dir, rng=rng)
                        base_instance = parse_rcp_file(base_file)
                        break
                    except Exception:
                        pass
                raise e

    if base_instance is None:
        raise ValueError(f"Failed to find matching base instance after {max_retries} retries")

    modified_instance = modify_instance(
        base_instance,
        duration_factor_range=duration_factor_range,
        demand_factor_range=demand_factor_range,
        capacity_factor_range=capacity_factor_range,
        rng=rng
    )

    if target_num_tasks is not None:
        current_tasks = modified_instance["nr_tasks"]
        if current_tasks < target_num_tasks:
            print(f"[Info] Extending instance from {current_tasks} to {target_num_tasks} tasks")
            modified_instance = extend_instance_tasks(
                modified_instance,
                target_num_tasks,
                rng=rng
            )
        elif current_tasks > target_num_tasks:
            print(f"[Info] Reducing instance from {current_tasks} to {target_num_tasks} tasks")
            modified_instance = reduce_instance_tasks(
                modified_instance,
                target_num_tasks,
                rng=rng
            )

    if target_num_precedence is not None:
        current_precedence = sum(len(task["successors"]) for task in modified_instance["tasks"])
        if current_precedence < target_num_precedence:
            print(f"[Info] Extending instance from {current_precedence} to {target_num_precedence} precedence relations")
            modified_instance = extend_instance_precedence(
                modified_instance,
                target_num_precedence,
                rng=rng
            )
        elif current_precedence > target_num_precedence:
            print(f"[Info] Reducing instance from {current_precedence} to {target_num_precedence} precedence relations")
            modified_instance = reduce_instance_precedence(
                modified_instance,
                target_num_precedence,
                rng=rng
            )

    return modified_instance


def solve_instance(
    instance_json: Dict[str, Any],
    time_limit: float = 60.0,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Solve an RCPSP instance using CP-SAT solver.
    """
    makespan, schedule = solve_rcpsp(
        instance_json,
        time_limit=time_limit
    )
    return makespan, schedule


def save_instances(
    instances: List[Dict[str, Any]],
    out_path: Union[str, Path],
) -> None:
    """
    Save instances to JSON file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(instances, f, indent=2)
    print(f"[Info] Saved {len(instances)} instances to {out_path}")


def generate_save_rcpsp(
    n_instances: int,
    out_path: str,
    num_tasks: Optional[Union[int, Tuple[int, int]]] = None,
    num_precedence_relations: Optional[Union[int, Tuple[int, int]]] = None,
    dataset_dir: Union[str, Path] = "step1_instance_creation/datasets/RCPSP",
    oversample_factor: float = 3.0,
    seed: int = 42,
    time_limit: float = 60.0,
    duration_factor_range: Tuple[float, float] = (0.5, 2.0),
    demand_factor_range: Tuple[float, float] = (0.5, 2.0),
    capacity_factor_range: Tuple[float, float] = (0.8, 1.5),
    size_category: Optional[str] = None,
):
    """
    Generate RCPSP instances and save them to JSON.
    """
    target = n_instances
    total_to_try = int(target * oversample_factor)

    print(
        f"[Info] Target instances = {target}, oversample_factor = {oversample_factor}, "
        f"try to generate {total_to_try} raw instances."
    )

    rng_sampling = random.Random(seed)

    if size_category is not None and num_tasks is None:
        task_ranges = {
            "S": (7, 22),
            "M": (22, 27),
            "L": (27, 51)
        }
        if size_category not in task_ranges:
            raise ValueError(f"Invalid size_category: {size_category}. Must be 'S', 'M', or 'L'")
        num_tasks = task_ranges[size_category]
        print(f"[Info] Using size_category '{size_category}' -> num_tasks range: {num_tasks}")

    if num_tasks is not None or num_precedence_relations is not None:
        criteria_parts = []
        if num_tasks is not None:
            if isinstance(num_tasks, (tuple, list)) and len(num_tasks) == 2:
                criteria_parts.append(f"tasks={num_tasks[0]}-{num_tasks[1]} (sampling per instance)")
            else:
                criteria_parts.append(f"tasks={num_tasks}")
        if num_precedence_relations is not None:
            if isinstance(num_precedence_relations, (tuple, list)) and len(num_precedence_relations) == 2:
                criteria_parts.append(f"precedence={num_precedence_relations[0]}-{num_precedence_relations[1]} (sampling per instance)")
            else:
                criteria_parts.append(f"precedence={num_precedence_relations}")
        print(f"[Info] Filtering base instances by: {', '.join(criteria_parts)}")

    raw_instances = []
    skipped_size_mismatch = 0

    for instance_idx in range(total_to_try):
        try:
            current_num_tasks = num_tasks
            if isinstance(num_tasks, (tuple, list)) and len(num_tasks) == 2:
                current_num_tasks = rng_sampling.randint(num_tasks[0], num_tasks[1])

            current_num_precedence = num_precedence_relations
            if isinstance(num_precedence_relations, (tuple, list)) and len(num_precedence_relations) == 2:
                current_num_precedence = rng_sampling.randint(
                    num_precedence_relations[0],
                    num_precedence_relations[1]
                )

            instance_json = generate_rcpsp_instance(
                dataset_dir,
                instance_idx,
                num_tasks=current_num_tasks,
                num_precedence_relations=current_num_precedence,
                duration_factor_range=duration_factor_range,
                demand_factor_range=demand_factor_range,
                capacity_factor_range=capacity_factor_range,
                seed=seed,
            )

            st = time.time()
            instance_num_tasks = instance_json["nr_tasks"]
            num_resources = instance_json["nr_resources"]
            print(f"[Info] Solving instance {instance_num_tasks}x{num_resources} - {instance_idx}...")
            makespan, schedule = solve_instance(instance_json, time_limit=time_limit)
            end = time.time()
            print(f"[Info] Solved instance {instance_num_tasks}x{num_resources} - {instance_idx} in {end - st:.2f} seconds")

            if makespan != makespan:  # Check for NaN
                print(f"[Warn] Solver failed for instance {instance_idx}, skipping...")
                continue

            instance_dict = {
                "instance": instance_json,
                "solution": schedule,
                "obj": float(makespan),
                "problem_type": "RCPSP",
            }

            raw_instances.append(instance_dict)

        except Exception as e:
            print(f"[Warn] Error generating instance {instance_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if skipped_size_mismatch > 0:
        print(f"[Info] Skipped {skipped_size_mismatch} instances due to size category mismatch")

    if len(raw_instances) < target:
        print(
            f"[Warn] Only generated {len(raw_instances)} valid instances (< {target}). "
            f"Will save all of them."
        )
        data = raw_instances
    else:
        data = raw_instances[:target]

    save_instances(data, out_path)

    return data
