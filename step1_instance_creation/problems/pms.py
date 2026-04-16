"""
PMS (Parallel Machine Scheduling) instance generation.

This module generates PMS instances by:
1. Loading existing .dat files from the dataset
2. Modifying parameters (processing times, release dates, deadlines) while preserving structure
3. Solving instances using OR-Tools CP-SAT (objective: minimize maximum tardiness T_max)
4. Saving instances in JSON format

Objective: Minimize maximum tardiness (T_max)
where tardiness = max(0, completion_time - deadline)
and T_max = max(tardiness over all jobs)
"""

import time
import json
import random
import hashlib
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List, Optional

from ..solvers.scheduling_models.pms_solver import solve_pms


def _parse_pms_header(file_path: Union[str, Path]) -> Tuple[int, int]:
    """Parse only the header (nr_jobs, nr_machines) from a PMS .dat file."""
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid header format in {file_path}: expected 'num_jobs num_machines'")
            return int(parts[0]), int(parts[1])
    raise ValueError(f"Invalid file format in {file_path}: expected at least header line")


def index_pms_dataset(dataset_dir: Union[str, Path]) -> Dict[Tuple[int, int], List[Path]]:
    """Index dataset .dat files by (nr_jobs, nr_machines) using header-only parsing."""
    dataset_dir = Path(dataset_dir)
    all_dat_files = list(dataset_dir.glob("**/*.dat"))
    if not all_dat_files:
        raise ValueError(f"No .dat files found in {dataset_dir}")

    index: Dict[Tuple[int, int], List[Path]] = {}
    for fp in all_dat_files:
        try:
            n, m = _parse_pms_header(fp)
        except Exception:
            continue
        index.setdefault((n, m), []).append(fp)
    if not index:
        raise ValueError(f"No parseable .dat files found in {dataset_dir}")
    return index


def instance_fingerprint(instance_json: Dict[str, Any]) -> str:
    """Stable fingerprint of the *instance* (ignores solution/obj/provenance)."""
    payload = {
        "nr_jobs": int(instance_json["nr_jobs"]),
        "nr_machines": int(instance_json["nr_machines"]),
        "jobs": [
            {
                "processing_time": int(j["processing_time"]),
                "release_date": int(j["release_date"]),
                "deadline": int(j["deadline"]),
            }
            for j in instance_json["jobs"]
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def parse_pms_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parse a PMS .dat file.

    File format:
    - Line 1: # n m (comment)
    - Line 2: n m (number of jobs, number of machines, space-separated)
    - Line 3: # job ptime rdate deadline (comment)
    - Remaining lines: job_id ptime rdate deadline (space-separated)

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the .dat file

    Returns
    -------
    Dict[str, Any]
        Parsed instance data with format:
        {
            "nr_jobs": int,
            "nr_machines": int,
            "jobs": [
                {
                    "processing_time": int,
                    "release_date": int,
                    "deadline": int
                },
                ...
            ]
        }
    """
    file_path = Path(file_path)
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith('#')]

    if len(lines) < 1:
        raise ValueError(f"Invalid file format in {file_path}: expected at least header line")

    header_parts = lines[0].split()
    if len(header_parts) < 2:
        raise ValueError(f"Invalid header format in {file_path}: expected 'num_jobs num_machines'")
    num_jobs = int(header_parts[0])
    num_machines = int(header_parts[1])

    jobs = []
    job_lines = lines[1:]

    if len(job_lines) != num_jobs:
        raise ValueError(f"Number of job lines ({len(job_lines)}) doesn't match num_jobs ({num_jobs})")

    for job_id, line in enumerate(job_lines):
        parts = [int(x) for x in line.split()]

        if len(parts) == 2:
            processing_time = parts[0]
            release_date = 0
            deadline = parts[1]
        elif len(parts) == 3:
            processing_time = parts[1]
            release_date = 0
            deadline = parts[2]
        elif len(parts) >= 4:
            processing_time = parts[1]
            release_date = parts[2]
            deadline = parts[3]
        else:
            raise ValueError(f"Job {job_id}: insufficient data (expected at least 2 values)")

        jobs.append({
            "processing_time": processing_time,
            "release_date": release_date,
            "deadline": deadline
        })

    return {
        "nr_jobs": num_jobs,
        "nr_machines": num_machines,
        "jobs": jobs
    }


def modify_instance(
    instance: Dict[str, Any],
    processing_time_factor_range: Tuple[float, float] = (0.5, 2.0),
    release_date_factor_range: Tuple[float, float] = (0.5, 2.0),
    deadline_factor_range: Tuple[float, float] = (0.8, 1.5),
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Modify a PMS instance by scaling processing times, release dates, and deadlines.
    """
    if rng is None:
        rng = random

    modified = {
        "nr_jobs": instance["nr_jobs"],
        "nr_machines": instance["nr_machines"],
        "jobs": []
    }

    for job in instance["jobs"]:
        pt_factor = rng.uniform(*processing_time_factor_range)
        new_processing_time = max(1, int(round(job["processing_time"] * pt_factor)))

        rd_factor = rng.uniform(*release_date_factor_range)
        new_release_date = max(0, int(round(job["release_date"] * rd_factor)))

        dl_factor = rng.uniform(*deadline_factor_range)
        original_deadline = job["deadline"]
        original_slack = original_deadline - job["release_date"] - job["processing_time"]

        new_deadline = max(
            new_release_date + new_processing_time,
            int(round(new_release_date + new_processing_time + original_slack * dl_factor))
        )

        modified["jobs"].append({
            "processing_time": new_processing_time,
            "release_date": new_release_date,
            "deadline": new_deadline
        })

    return modified


def extend_instance_jobs(
    instance: Dict[str, Any],
    target_num_jobs: int,
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Extend a PMS instance by adding jobs to reach the target number.
    """
    if rng is None:
        rng = random

    current_num_jobs = instance["nr_jobs"]
    num_machines = instance["nr_machines"]

    if target_num_jobs <= current_num_jobs:
        return {
            "nr_jobs": instance["nr_jobs"],
            "nr_machines": instance["nr_machines"],
            "jobs": [
                {
                    "processing_time": job["processing_time"],
                    "release_date": job["release_date"],
                    "deadline": job["deadline"]
                }
                for job in instance["jobs"]
            ]
        }

    extended = {
        "nr_jobs": target_num_jobs,
        "nr_machines": num_machines,
        "jobs": [
            {
                "processing_time": job["processing_time"],
                "release_date": job["release_date"],
                "deadline": job["deadline"]
            }
            for job in instance["jobs"]
        ]
    }

    jobs_to_add = target_num_jobs - current_num_jobs

    existing_processing_times = [job["processing_time"] for job in instance["jobs"]]
    existing_release_dates = [job["release_date"] for job in instance["jobs"]]
    existing_deadlines = [job["deadline"] for job in instance["jobs"]]

    min_pt = min(existing_processing_times) if existing_processing_times else 1
    max_pt = max(existing_processing_times) if existing_processing_times else 10
    avg_pt = sum(existing_processing_times) / len(existing_processing_times) if existing_processing_times else 5

    min_rd = min(existing_release_dates) if existing_release_dates else 0
    max_rd = max(existing_release_dates) if existing_release_dates else 100
    avg_rd = sum(existing_release_dates) / len(existing_release_dates) if existing_release_dates else 50

    max_deadline = max(existing_deadlines) if existing_deadlines else 200

    for i in range(jobs_to_add):
        processing_time = max(1, int(round(rng.uniform(min_pt, max_pt))))

        release_date = max(0, int(round(rng.uniform(min_rd, max_rd))))

        existing_slacks = [
            dl - rd - pt
            for dl, rd, pt in zip(existing_deadlines, existing_release_dates, existing_processing_times)
        ]
        avg_slack = sum(existing_slacks) / len(existing_slacks) if existing_slacks else processing_time
        min_slack = min(existing_slacks) if existing_slacks else processing_time

        slack = max(min_slack, int(round(rng.uniform(avg_slack * 0.5, avg_slack * 2.0))))
        deadline = release_date + processing_time + slack

        deadline = min(deadline, max_deadline * 2)

        extended["jobs"].append({
            "processing_time": processing_time,
            "release_date": release_date,
            "deadline": deadline
        })

    return extended


def reduce_instance_jobs(
    instance: Dict[str, Any],
    target_num_jobs: int,
    rng: Optional[random.Random] = None
) -> Dict[str, Any]:
    """
    Reduce a PMS instance by removing jobs to reach the target number.
    """
    if rng is None:
        rng = random

    current_num_jobs = instance["nr_jobs"]

    if target_num_jobs >= current_num_jobs:
        return {
            "nr_jobs": instance["nr_jobs"],
            "nr_machines": instance["nr_machines"],
            "jobs": [
                {
                    "processing_time": job["processing_time"],
                    "release_date": job["release_date"],
                    "deadline": job["deadline"]
                }
                for job in instance["jobs"]
            ]
        }

    if target_num_jobs < 1:
        raise ValueError(f"target_num_jobs must be at least 1, got {target_num_jobs}")

    jobs_to_keep = rng.sample(range(current_num_jobs), target_num_jobs)
    jobs_to_keep.sort()

    reduced = {
        "nr_jobs": target_num_jobs,
        "nr_machines": instance["nr_machines"],
        "jobs": [
            {
                "processing_time": instance["jobs"][job_id]["processing_time"],
                "release_date": instance["jobs"][job_id]["release_date"],
                "deadline": instance["jobs"][job_id]["deadline"]
            }
            for job_id in jobs_to_keep
        ]
    }

    return reduced


def instance_to_json(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PMS instance to JSON format (already in correct format, just return as-is).
    """
    return instance


def get_random_pms_file(
    dataset_dir: Union[str, Path],
    num_jobs_range: Optional[Tuple[int, int]] = None,
    num_machines_range: Optional[Tuple[int, int]] = None,
    dataset_index: Optional[Dict[Tuple[int, int], List[Path]]] = None,
    rng: Optional[random.Random] = None
) -> Path:
    """
    Get a random .dat file from the dataset directory, optionally filtered by criteria.
    """
    if rng is None:
        rng = random

    dataset_dir = Path(dataset_dir)

    if dataset_index is None:
        dataset_index = index_pms_dataset(dataset_dir)

    keys = list(dataset_index.keys())
    if not keys:
        raise ValueError(f"No .dat files found in {dataset_dir}")

    if num_jobs_range is None and num_machines_range is None:
        k = rng.choice(keys)
        return rng.choice(dataset_index[k])

    filtered_keys: List[Tuple[int, int]] = []
    for (n, m) in keys:
        matches = True
        if num_jobs_range is not None:
            min_jobs, max_jobs = num_jobs_range
            if not (min_jobs <= n <= max_jobs):
                matches = False
        if num_machines_range is not None and matches:
            min_machines, max_machines = num_machines_range
            if not (min_machines <= m <= max_machines):
                matches = False
        if matches:
            filtered_keys.append((n, m))

    if not filtered_keys:
        criteria_str = []
        if num_jobs_range:
            criteria_str.append(f"jobs={num_jobs_range[0]}-{num_jobs_range[1]}")
        if num_machines_range:
            criteria_str.append(f"machines={num_machines_range[0]}-{num_machines_range[1]}")
        raise ValueError(
            f"No .dat files found matching criteria: {', '.join(criteria_str)}. "
            f"Total header keys checked: {len(keys)}"
        )

    k = rng.choice(filtered_keys)
    return rng.choice(dataset_index[k])


def _choose_base_key(
    keys: List[Tuple[int, int]],
    target_num_jobs: Optional[int],
    target_num_machines: Optional[int],
    rng: random.Random,
) -> Tuple[int, int]:
    """Choose a (nr_jobs, nr_machines) key from dataset index, biased toward requested sizes."""
    if target_num_jobs is not None:
        candidates = [k for k in keys if k[0] >= target_num_jobs]
        if candidates:
            min_n = min(k[0] for k in candidates)
            candidates = [k for k in candidates if k[0] == min_n]
        else:
            max_n = max(k[0] for k in keys)
            candidates = [k for k in keys if k[0] == max_n]
    else:
        candidates = list(keys)

    if target_num_machines is None:
        return rng.choice(candidates)

    best_dist = min(abs(m - target_num_machines) for (_, m) in candidates)
    closest = [k for k in candidates if abs(k[1] - target_num_machines) == best_dist]
    return rng.choice(closest)


def generate_pms_instance(
    dataset_dir: Union[str, Path],
    instance_idx: int,
    num_jobs: Optional[Union[int, Tuple[int, int]]] = None,
    num_machines: Optional[Union[int, Tuple[int, int]]] = None,
    processing_time_factor_range: Tuple[float, float] = (0.5, 2.0),
    release_date_factor_range: Tuple[float, float] = (0.5, 2.0),
    deadline_factor_range: Tuple[float, float] = (0.8, 1.5),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a single PMS instance by loading and modifying a random base instance.
    """
    rng = random.Random(seed + instance_idx)

    num_jobs_range = None
    if num_jobs is not None:
        if isinstance(num_jobs, (tuple, list)) and len(num_jobs) == 2:
            num_jobs_range = tuple(num_jobs)
        elif isinstance(num_jobs, int):
            num_jobs_range = (num_jobs, num_jobs)
        else:
            raise ValueError(f"Invalid num_jobs format: {num_jobs}. Expected int or tuple (min, max)")

    num_machines_range = None
    if num_machines is not None:
        if isinstance(num_machines, (tuple, list)) and len(num_machines) == 2:
            num_machines_range = tuple(num_machines)
        elif isinstance(num_machines, int):
            num_machines_range = (num_machines, num_machines)
        else:
            raise ValueError(f"Invalid num_machines format: {num_machines}. Expected int or tuple (min, max)")

    target_num_jobs = None
    if num_jobs_range and num_jobs_range[0] == num_jobs_range[1]:
        target_num_jobs = num_jobs_range[0]

    target_num_machines = None
    if num_machines_range and num_machines_range[0] == num_machines_range[1]:
        target_num_machines = num_machines_range[0]

    dataset_dir = Path(dataset_dir)
    dataset_index = index_pms_dataset(dataset_dir)
    keys = list(dataset_index.keys())
    base_key = _choose_base_key(keys, target_num_jobs, target_num_machines, rng)
    base_file = rng.choice(dataset_index[base_key])
    base_instance = parse_pms_file(base_file)

    job_subset_indices: Optional[List[int]] = None

    working_instance = {
        "nr_jobs": base_instance["nr_jobs"],
        "nr_machines": base_instance["nr_machines"],
        "jobs": [dict(j) for j in base_instance["jobs"]],
    }

    if target_num_jobs is not None:
        current_jobs = working_instance["nr_jobs"]
        if target_num_jobs < current_jobs:
            job_subset_indices = sorted(rng.sample(range(current_jobs), target_num_jobs))
            working_instance = {
                "nr_jobs": target_num_jobs,
                "nr_machines": working_instance["nr_machines"],
                "jobs": [working_instance["jobs"][i] for i in job_subset_indices],
            }
        elif target_num_jobs > current_jobs:
            working_instance = extend_instance_jobs(working_instance, target_num_jobs, rng=rng)

    modified_instance = modify_instance(
        working_instance,
        processing_time_factor_range=processing_time_factor_range,
        release_date_factor_range=release_date_factor_range,
        deadline_factor_range=deadline_factor_range,
        rng=rng,
    )

    if target_num_machines is not None:
        modified_instance["nr_machines"] = int(target_num_machines)

    return modified_instance


def validate_schedule(instance_data: Dict[str, Any], schedule: List[Dict], max_tardiness: float) -> Tuple[bool, List[str]]:
    """
    Validate that a schedule is feasible and correct.
    """
    errors = []
    num_jobs = instance_data["nr_jobs"]
    num_machines = instance_data["nr_machines"]
    jobs = instance_data["jobs"]

    scheduled_jobs = set(s["job"] for s in schedule)
    if len(scheduled_jobs) != num_jobs:
        errors.append(f"Not all jobs scheduled: {len(scheduled_jobs)}/{num_jobs}")

    machine_intervals = {m: [] for m in range(num_machines)}
    actual_max_tardiness = 0

    for s in schedule:
        job_id = s["job"]
        machine_id = s["machine"]
        start = s["start"]
        end = s["end"]
        processing_time = s["processing_time"]

        if abs(end - start - processing_time) > 0.01:
            errors.append(f"Job {job_id}: end - start != processing_time")

        job_data = jobs[job_id]
        if start < job_data["release_date"]:
            errors.append(f"Job {job_id}: starts before release date")

        if 0 <= machine_id < num_machines:
            machine_intervals[machine_id].append((start, end, job_id))
        else:
            errors.append(f"Job {job_id}: invalid machine {machine_id}")

        tardiness = max(0, end - job_data["deadline"])
        actual_max_tardiness = max(actual_max_tardiness, tardiness)

    for machine_id, intervals in machine_intervals.items():
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        for i in range(len(sorted_intervals) - 1):
            curr_end = sorted_intervals[i][1]
            next_start = sorted_intervals[i+1][0]
            if curr_end > next_start:
                errors.append(f"Machine {machine_id}: overlap between jobs {sorted_intervals[i][2]} and {sorted_intervals[i+1][2]}")

    if abs(actual_max_tardiness - max_tardiness) > 0.01:
        errors.append(f"Max tardiness mismatch: reported={max_tardiness}, actual={actual_max_tardiness}")

    return len(errors) == 0, errors


def solve_instance(
    instance_json: Dict[str, Any],
    time_limit: float = 60.0,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Solve a PMS instance using CP-SAT solver and validate the schedule.
    """
    max_tardiness, schedule = solve_pms(
        instance_json,
        time_limit=time_limit
    )

    if schedule is not None and max_tardiness == max_tardiness:  # Check not NaN
        is_valid, errors = validate_schedule(instance_json, schedule, max_tardiness)
        if not is_valid:
            print(f"[Warning] Schedule validation failed:")
            for error in errors[:5]:
                print(f"  - {error}")

    return max_tardiness, schedule


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


def generate_save_pms(
    n_instances: int,
    out_path: str,
    num_jobs: Optional[Union[int, Tuple[int, int]]] = None,
    num_machines: Optional[Union[int, Tuple[int, int]]] = None,
    dataset_dir: Union[str, Path] = None,
    oversample_factor: float = 3.0,
    seed: int = 42,
    time_limit: float = 60.0,
    processing_time_factor_range: Tuple[float, float] = (0.5, 2.0),
    release_date_factor_range: Tuple[float, float] = (0.5, 2.0),
    deadline_factor_range: Tuple[float, float] = (0.8, 1.5),
    size_category: Optional[str] = None,
    max_attempt_factor: float = 20.0,
):
    """
    Generate PMS instances and save them to JSON.
    """
    if dataset_dir is None:
        dataset_dir = Path(__file__).parent.parent / "datasets" / "PMS"
    else:
        dataset_dir = Path(dataset_dir)

    if not dataset_dir.is_absolute():
        dataset_dir = Path.cwd() / dataset_dir

    dataset_dir = dataset_dir.resolve()

    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    target = n_instances
    total_to_try = int(target * max_attempt_factor)

    print(
        f"[Info] Target instances = {target}, oversample_factor = {oversample_factor}, "
        f"max_attempt_factor = {max_attempt_factor}, attempt up to {total_to_try} generations."
    )
    print(f"[Info] Using dataset directory: {dataset_dir}")

    rng_sampling = random.Random(seed)

    if size_category is not None and num_jobs is None:
        job_ranges = {
            "S": (5, 15),
            "M": (15, 30),
            "L": (30, 100)
        }
        machine_ranges = {
            "S": (2, 5),
            "M": (5, 10),
            "L": (10, 20)
        }
        if size_category not in job_ranges:
            raise ValueError(f"Invalid size_category: {size_category}. Must be 'S', 'M', or 'L'")
        num_jobs = job_ranges[size_category]
        num_machines = machine_ranges[size_category]
        print(f"[Info] Using size_category '{size_category}' -> num_jobs range: {num_jobs}, num_machines range: {num_machines}")

    if num_jobs is not None or num_machines is not None:
        criteria_parts = []
        if num_jobs is not None:
            if isinstance(num_jobs, (tuple, list)) and len(num_jobs) == 2:
                criteria_parts.append(f"jobs={num_jobs[0]}-{num_jobs[1]} (sampling per instance)")
            else:
                criteria_parts.append(f"jobs={num_jobs}")
        if num_machines is not None:
            if isinstance(num_machines, (tuple, list)) and len(num_machines) == 2:
                criteria_parts.append(f"machines={num_machines[0]}-{num_machines[1]} (sampling per instance)")
            else:
                criteria_parts.append(f"machines={num_machines}")
        print(f"[Info] Filtering base instances by: {', '.join(criteria_parts)}")

    raw_instances = []
    seen_fingerprints: set = set()

    for instance_idx in range(total_to_try):
        try:
            current_num_jobs = num_jobs
            if isinstance(num_jobs, (tuple, list)) and len(num_jobs) == 2:
                current_num_jobs = rng_sampling.randint(num_jobs[0], num_jobs[1])

            current_num_machines = num_machines
            if isinstance(num_machines, (tuple, list)) and len(num_machines) == 2:
                current_num_machines = rng_sampling.randint(num_machines[0], num_machines[1])

            instance_json = generate_pms_instance(
                dataset_dir,
                instance_idx,
                num_jobs=current_num_jobs,
                num_machines=current_num_machines,
                processing_time_factor_range=processing_time_factor_range,
                release_date_factor_range=release_date_factor_range,
                deadline_factor_range=deadline_factor_range,
                seed=seed,
            )

            fp = instance_fingerprint(instance_json)
            if fp in seen_fingerprints:
                continue

            st = time.time()
            instance_num_jobs = instance_json["nr_jobs"]
            instance_num_machines = instance_json["nr_machines"]
            print(f"[Info] Solving instance {instance_num_jobs} jobs x {instance_num_machines} machines - {instance_idx}...")
            max_tardiness, schedule = solve_instance(instance_json, time_limit=time_limit)
            end = time.time()
            print(f"[Info] Solved instance {instance_num_jobs} jobs x {instance_num_machines} machines - {instance_idx} in {end - st:.2f} seconds (T_max={max_tardiness})")

            if max_tardiness != max_tardiness:  # Check for NaN
                print(f"[Warn] Solver failed for instance {instance_idx}, skipping...")
                continue

            if schedule is None:
                print(f"[Warn] No schedule returned for instance {instance_idx}, skipping...")
                continue

            is_valid, errors = validate_schedule(instance_json, schedule, max_tardiness)
            if not is_valid:
                print(f"[Warn] Instance {instance_idx} has invalid schedule, skipping:")
                for error in errors[:3]:
                    print(f"  - {error}")
                continue

            instance_dict = {
                "instance": instance_json,
                "solution": schedule,
                "obj": float(max_tardiness),
                "problem_type": "PMS",
            }

            seen_fingerprints.add(fp)
            raw_instances.append(instance_dict)

            if len(raw_instances) >= target:
                break

        except Exception as e:
            print(f"[Warn] Error generating instance {instance_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

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
