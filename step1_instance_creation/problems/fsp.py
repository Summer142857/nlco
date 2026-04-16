"""
Flow Shop Scheduling Problem (FSP) instance generator.

Uses Taillard-style instance generation via utils/taillard.py.
"""
import json
import random
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List

from ..utils.taillard import (
    TaillardRNG,
    TaillardInstanceGenerator,
    _taillard_to_optimized_json,
    generate_fsp_instance,
    solve_instance,
    save_instances,
)


def generate_and_save_fsp(
    n_jobs: Union[int, Tuple[int, int]],
    n_machines: Union[int, Tuple[int, int]],
    n_instances: int,
    out_path: str,
    oversample_factor: float = 3.0,
    seed: int = 42,
    time_limit: float = 60.0,
) -> List[Dict[str, Any]]:
    """
    Generate Flow Shop Scheduling instances and save them to JSON.

    Parameters
    ----------
    n_jobs : Union[int, Tuple[int, int]]
        Number of jobs per instance (or range for random selection).
    n_machines : Union[int, Tuple[int, int]]
        Number of machines per instance (or range for random selection).
    n_instances : int
        Target number of instances to generate.
    out_path : str
        Output JSON file path.
    oversample_factor : float, optional
        Factor for oversampling to handle failed instances (default: 3.0).
    seed : int, optional
        Random seed for reproducibility (default: 42).
    time_limit : float, optional
        Solver time limit in seconds (default: 60.0).

    Returns
    -------
    List[Dict[str, Any]]
        List of generated instances with instance, solution, obj, and problem_type.
    """
    target = n_instances
    total_to_try = int(target * oversample_factor)

    print(
        f"[Info] Target instances = {target}, oversample_factor = {oversample_factor}, "
        f"try to generate {total_to_try} raw instances."
    )

    random.seed(seed)

    time_seed_base = seed
    machine_seed_base = seed + 10000

    raw_instances = []

    import time as _time

    for instance_idx in range(total_to_try):
        if isinstance(n_jobs, (tuple, list)) and len(n_jobs) == 2:
            current_n_jobs = random.randint(n_jobs[0], n_jobs[1])
        else:
            current_n_jobs = n_jobs

        if isinstance(n_machines, (tuple, list)) and len(n_machines) == 2:
            current_n_machines = random.randint(n_machines[0], n_machines[1])
        else:
            current_n_machines = n_machines

        instance_json = generate_fsp_instance(
            current_n_jobs,
            current_n_machines,
            instance_idx,
            time_seed_base,
            machine_seed_base,
        )

        st = _time.time()
        print(f"[Info] Solving FSP instance {current_n_jobs}x{current_n_machines} - {instance_idx}...")
        makespan, schedule = solve_instance(instance_json, time_limit=time_limit, problem_type="fsp")
        end = _time.time()
        print(f"[Info] Solved instance {current_n_jobs}x{current_n_machines} - {instance_idx} in {end - st:.2f} seconds")

        if makespan != makespan:  # Check for NaN
            print(f"[Warn] Solver failed for instance {instance_idx}, skipping...")
            continue

        instance_dict = {
            "instance": instance_json,
            "solution": schedule,
            "obj": float(makespan),
            "problem_type": "FSP",
        }

        raw_instances.append(instance_dict)

        if len(raw_instances) >= target:
            break

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
