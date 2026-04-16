"""
SMTWTP (Single Machine Total Weighted Tardiness Problem) instance generation.

Generates instances by:
1. Creating random job parameters (processing times, weights, due dates)
2. Solving with OR-Tools CP-SAT
3. Saving instances in JSON format
"""

import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
from dataclasses import dataclass, asdict

from ..solvers.scheduling_models.wt_solver import solve_wt


@dataclass
class WTInstance:
    id: int
    n: int
    p: List[int]  # Processing times
    w: List[int]  # Weights
    d: List[int]  # Due dates
    rdd: float
    tf: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_jobs": self.n,
            "processing_times": self.p,
            "weights": self.w,
            "due_dates": self.d,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WTInstance':
        return cls(
            id=data.get("id", 0),
            n=data["n_jobs"],
            p=data["processing_times"],
            w=data["weights"],
            d=data["due_dates"],
            rdd=data.get("rdd", 0.0),
            tf=data.get("tf", 0.0)
        )


def validate_solution(instance_dict: Dict[str, Any], solution: Dict[str, Any]) -> bool:
    """
    Deterministic feasibility check for SMTWTP solution.
    Checks:
    1. Consistency: end_time = start_time + processing_time
    2. Tardiness: tardiness = max(0, end_time - due_date)
    3. No overlap: Interval [start, end) must not overlap with others.

    Parameters:
    - instance_dict: Dict containing instance data
    - solution: Dict containing solution data (start_times, end_times, tardiness)

    Returns:
    - bool: True if valid, False otherwise
    """
    n = instance_dict['n_jobs']
    p = instance_dict['processing_times']
    d = instance_dict['due_dates']

    start_times = solution.get('start_times')
    end_times = solution.get('end_times')
    tardiness = solution.get('tardiness')

    if not start_times or not end_times or not tardiness:
        print("[Error] Solution missing required fields")
        return False

    if len(start_times) != n or len(end_times) != n:
        print(f"[Error] Solution size mismatch (n={n})")
        return False

    # 1. Consistency Check
    for i in range(n):
        if end_times[i] != start_times[i] + p[i]:
            print(f"[Error] Job {i}: duration mismatch. Start {start_times[i]} + P {p[i]} != End {end_times[i]}")
            return False

        expected_tardiness = max(0, end_times[i] - d[i])
        if abs(tardiness[i] - expected_tardiness) > 1e-6:
            print(f"[Error] Job {i}: tardiness mismatch. End {end_times[i]} - Due {d[i]} -> Exp {expected_tardiness} != Got {tardiness[i]}")
            return False

    # 2. No Overlap Check
    intervals = []
    for i in range(n):
        intervals.append((start_times[i], end_times[i], i))

    intervals.sort(key=lambda x: x[0])

    for i in range(n - 1):
        s1, e1, idx1 = intervals[i]
        s2, e2, idx2 = intervals[i+1]

        if e1 > s2:
            print(f"[Error] Overlap detected between Job {idx1} [{s1}, {e1}) and Job {idx2} [{s2}, {e2})")
            return False

    return True


def generate_wt_instance(
    n: int,
    instance_id: int,
    rdd: float,
    tf: float,
    seed: int = 42
) -> WTInstance:
    """
    Generates a single WT instance.
    """
    rng = random.Random(seed + instance_id)

    p = [rng.randint(1, 100) for _ in range(n)]

    w = [rng.randint(1, 10) for _ in range(n)]

    P = sum(p)

    lower_bound = int(P * (1.0 - tf - rdd / 2.0))
    upper_bound = int(P * (1.0 - tf + rdd / 2.0))

    d = []
    for _ in range(n):
        if lower_bound > upper_bound:
            val = lower_bound
        else:
            val = rng.randint(lower_bound, upper_bound)
        d.append(val)

    return WTInstance(
        id=instance_id,
        n=n,
        p=p,
        w=w,
        d=d,
        rdd=rdd,
        tf=tf
    )


def solve_instance_wrapper(instance: WTInstance, time_limit: float = 30.0) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """
    Wrapper to solve instance using the solver module.
    """
    return solve_wt(instance, time_limit_seconds=time_limit)


def save_instances(instances: List[Dict[str, Any]], out_path: Union[str, Path]) -> None:
    """
    Save instances to JSON file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(instances, f, indent=2)
    print(f"[Info] Saved {len(instances)} instances to {out_path}")


def generate_save_wt(
    n_jobs: Union[int, Tuple[int, int]],
    n_instances: int,
    out_path: str,
    oversample_factor: float = 3.0,
    seed: int = 42,
    time_limit: float = 30.0
):
    """
    Generates and solves SMTWTP instances, saving them to JSON.
    """
    print(f"[Info] Generating SMTWTP instances...")

    rdd_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    tf_values = [0.2, 0.4, 0.6, 0.8, 1.0]

    raw_instances = []
    instance_id = 1

    target = n_instances
    total_to_try = int(target * oversample_factor)

    rng = random.Random(seed)

    generated_count = 0
    attempts = 0

    while generated_count < target and attempts < total_to_try:
        for rdd in rdd_values:
            for tf in tf_values:
                if generated_count >= target:
                    break

                attempts += 1

                if isinstance(n_jobs, (tuple, list)) and len(n_jobs) == 2:
                    current_n = rng.randint(n_jobs[0], n_jobs[1])
                else:
                    current_n = n_jobs

                # 1. Generate Instance
                instance = generate_wt_instance(current_n, instance_id, rdd, tf, seed)

                # Check for invalid due dates (negative)
                if any(d < 0 for d in instance.d):
                    instance_id += 1
                    continue

                # 2. Solve Instance
                objective, solution = solve_instance_wrapper(instance, time_limit=time_limit)

                if objective is None:
                    pass
                else:
                    # 3. Validate
                    instance_dict = instance.to_dict()
                    is_valid = validate_solution(instance_dict, solution)

                    if not is_valid:
                        print(f"[Error] Solution validation failed for instance {instance_id}")
                    else:
                        # 4. Append to list
                        raw_instances.append({
                            "instance": instance_dict,
                            "solution": solution,
                            "obj": float(objective),
                            "problem_type": "SMTWT"
                        })
                        generated_count += 1
                        if generated_count % 10 == 0:
                            print(f"[Info] Generated {generated_count}/{target} instances")

                instance_id += 1

    # Save
    save_instances(raw_instances, out_path)
    return raw_instances
