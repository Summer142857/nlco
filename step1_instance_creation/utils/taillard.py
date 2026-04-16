import time
import json
import random
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List, Optional

from ..solvers.scheduling_models.standalone_cpsat_solver import solve_standalone_cpsat


# adapted from https://github.com/DominikRoB/Taillard-Instance-Tools/tree/development
class TaillardInstance:

    def __init__(self, type, seed, durations, machine_assignments=None):

        self.number_machines = 0
        self.number_jobs = 0

        self.type = type
        self.seed = seed
        self.durations = durations
        self.machine_assignments = machine_assignments
        self.lower_bound = self._calculate_lower_bound(type, durations,)

    def _calculate_lower_bound(self, type, durations):
        return None


class TaillardRNG:
    def __init__(self, seed=None):
        # Constants for the linear congruential generator
        self.a = 16807
        self.b = 127773
        self.c = 2836
        self.m = 2 ** 31 - 1

        # Initial seed
        self.X = seed if seed else int(time.time())

    def next(self):
        """ Denoted as U(0,1) """

        # Step 1: Modification of the seed
        k = self.X // self.b
        self.X = self.a * (self.X % self.b) - k * self.c

        # Check for negative value
        if self.X < 0:
            self.X = self.X + self.m

        # Step 2: New value of the seed
        return self.X / self.m

    def next_int(self, a, b):
        """Generate random integer between a and b (inclusive)
        Denotes as U[a, b]"""

        U = self.next()
        value = int(a + U * (b - a + 1))

        # Ensure value is within bounds (handle edge case when U is exactly 1.0)
        if value > b:
            value = b

        return value


class TaillardInstanceGenerator:
    def __init__(self, time_rng, machine_rng):

        self._time_rng = time_rng
        self._machine_rng = machine_rng

    def generate_flowshop(self, m, n):
        """
        Generate a flowshop instance with m machines and n jobs.
        :param m: Number of machines.
        :param n: Number of jobs.
        :return: A list of lists representing the processing times for each job on each machine.
        """

        d = self._generate_processing_times(m, n)

        return d

    def generate_jobshop(self, m, n):
        """
        Generate a job shop instance with m machines and n jobs.
        According to Taillard's method:
        1. Generate processing times d_ij uniformly in [1, 99]
        2. For each job j, create a random permutation of machines [0, m-1]
        :param m: Number of machines.
        :param n: Number of jobs.
        :return: A tuple of lists of lists representing the processing times and machine assignments for each job.
                 Returns (durations, machine_assignments) where:
                 - durations[j][i] = processing time for operation i of job j
                 - machine_assignments[j][i] = machine ID for operation i of job j
        """
        # Step 1: Generate processing times d_ij
        d = self._generate_processing_times(m, n)

        # Step 2: For each job, create a random permutation of machines
        # Using Fisher-Yates shuffle algorithm for uniform random permutation
        M = []
        for j in range(n):
            # Initialize with ordered machine sequence
            machine_seq = list(range(m))
            # Fisher-Yates shuffle to create random permutation
            for i in range(m - 1, 0, -1):
                # Pick random index from 0 to i (inclusive)
                i_rand = self._machine_rng.next_int(0, i)
                machine_seq[i], machine_seq[i_rand] = machine_seq[i_rand], machine_seq[i]
            M.append(machine_seq)

        return d, M

    def _generate_processing_times(self, m, n):
        return [[self._time_rng.next_int(1, 99) for _ in range(m)] for _ in range(n)]


def _taillard_to_optimized_json(
    durations: List[List[int]],
    machine_assignments: List[List[int]],
    problem_type: str = "jsp"
) -> Dict[str, Any]:
    """
    Convert Taillard format data to optimized JSON format for storage.
    Matches the Taillard file format structure.

    Parameters
    ----------
    durations : List[List[int]]
        Processing times for each operation of each job.
        durations[j][i] is the processing time for operation i of job j.
    machine_assignments : List[List[int]]
        Machine sequence for each job.
        machine_assignments[j][i] is the machine ID for operation i of job j.
    problem_type : str
        Problem type: "jsp" for Job Shop, "fsp" for Flow Shop, or "osp" for Open Shop.

    Returns
    -------
    Dict[str, Any]
        Optimized JSON structure matching Taillard format
    """
    num_jobs = len(durations)
    if num_jobs == 0:
        raise ValueError("durations list cannot be empty")

    if len(machine_assignments) == 0:
        raise ValueError("machine_assignments list cannot be empty")

    # Validate dimensions
    if len(machine_assignments) != num_jobs:
        raise ValueError(f"Number of jobs in durations ({num_jobs}) doesn't match "
                        f"number of jobs in machine_assignments ({len(machine_assignments)})")

    # Determine number of operations per job (should be consistent)
    num_operations_per_job = len(machine_assignments[0])
    if num_operations_per_job == 0:
        raise ValueError("Each job must have at least one operation")

    # Validate that all jobs have the same number of operations
    for j in range(num_jobs):
        if len(machine_assignments[j]) != num_operations_per_job:
            raise ValueError(f"Job {j} has {len(machine_assignments[j])} operations, "
                           f"expected {num_operations_per_job}")
        if len(durations[j]) != num_operations_per_job:
            raise ValueError(f"Job {j} has {len(durations[j])} durations, "
                           f"expected {num_operations_per_job}")

    # Determine number of machines from maximum machine ID used
    max_machine_id = max(max(machine_assignments[j]) for j in range(num_jobs))
    num_machines = max_machine_id + 1

    # Build optimized JSON structure
    # Both JSP, FSP, and OSP use the same format: array of [machine_id, processing_time] pairs per job
    # For FSP and OSP, machine_id is always 0, 1, 2, ..., m-1 (sequential, where m = num_machines)
    # For JSP, machine_id can be in any order per job
    jobs = []
    for job_id in range(num_jobs):
        job_operations = []
        for op_idx in range(num_operations_per_job):
            if problem_type == "fsp" or problem_type == "osp":
                # FSP and OSP: machine order is always 0, 1, 2, ..., m-1 (sequential)
                machine_id = op_idx
            else:
                # JSP: use the specified machine assignment
                machine_id = machine_assignments[job_id][op_idx]
            processing_time = durations[job_id][op_idx]
            job_operations.append([machine_id, processing_time])
        jobs.append(job_operations)

    instance = {
        "nr_machines": num_machines,
        "nr_jobs": num_jobs,
        "jobs": jobs  # Array of arrays: each job is [[machine_id, processing_time], ...]
    }

    return instance


def generate_fsp_instance(
    n_jobs: int,
    n_machines: int,
    instance_idx: int,
    time_seed_base: int,
    machine_seed_base: int,
) -> Dict[str, Any]:
    """
    Generate a single Flow Shop Scheduling instance in JSON format.

    Parameters
    ----------
    n_jobs : int
        Number of jobs
    n_machines : int
        Number of machines
    instance_idx : int
        Instance index (for seed generation)
    time_seed_base : int
        Base seed for time RNG
    machine_seed_base : int
        Base seed for machine RNG

    Returns
    -------
    Dict[str, Any]
        Instance JSON in optimized format
    """
    # Create RNGs with instance-specific seeds
    time_rng = TaillardRNG(seed=time_seed_base + instance_idx)
    machine_rng = TaillardRNG(seed=machine_seed_base + instance_idx)
    generator = TaillardInstanceGenerator(time_rng, machine_rng)

    # Flow Shop: all jobs follow the same machine sequence [0, 1, 2, ..., m-1]
    # durations[j][i] = processing time for job j on machine i (where i is also the operation index)
    durations = generator.generate_flowshop(n_machines, n_jobs)
    # Create fixed machine sequence for all jobs: each job visits machines in order [0, 1, 2, ..., m-1]
    machine_assignments = [list(range(n_machines)) for _ in range(n_jobs)]

    # Convert to optimized JSON format
    instance_json = _taillard_to_optimized_json(
        durations, machine_assignments, problem_type="fsp"
    )

    return instance_json


def generate_openshop_instance(
    n_jobs: int,
    n_machines: int,
    instance_idx: int,
    time_seed_base: int,
    machine_seed_base: int,
) -> Dict[str, Any]:
    """
    Generate a single Open Shop Scheduling instance in JSON format.

    Parameters
    ----------
    n_jobs : int
        Number of jobs
    n_machines : int
        Number of machines
    instance_idx : int
        Instance index (for seed generation)
    time_seed_base : int
        Base seed for time RNG
    machine_seed_base : int
        Base seed for machine RNG

    Returns
    -------
    Dict[str, Any]
        Instance JSON in optimized format
    """
    # Create RNGs with instance-specific seeds
    time_rng = TaillardRNG(seed=time_seed_base + instance_idx)
    machine_rng = TaillardRNG(seed=machine_seed_base + instance_idx)
    generator = TaillardInstanceGenerator(time_rng, machine_rng)

    # Open Shop: all jobs have operations on all machines [0, 1, 2, ..., m-1]
    # Similar to FSP in terms of instance structure, but no precedence constraints
    # durations[j][i] = processing time for job j on machine i (where i is also the operation index)
    durations = generator.generate_flowshop(n_machines, n_jobs)
    # Create fixed machine sequence for all jobs: each job visits machines in order [0, 1, 2, ..., m-1]
    machine_assignments = [list(range(n_machines)) for _ in range(n_jobs)]

    # Convert to optimized JSON format
    instance_json = _taillard_to_optimized_json(
        durations, machine_assignments, problem_type="osp"
    )

    return instance_json


def generate_jsp_instance(
    n_jobs: int,
    n_machines: int,
    instance_idx: int,
    time_seed_base: int,
    machine_seed_base: int,
) -> Dict[str, Any]:
    """
    Generate a single Job Shop Scheduling instance in JSON format.

    Parameters
    ----------
    n_jobs : int
        Number of jobs
    n_machines : int
        Number of machines
    instance_idx : int
        Instance index (for seed generation)
    time_seed_base : int
        Base seed for time RNG
    machine_seed_base : int
        Base seed for machine RNG

    Returns
    -------
    Dict[str, Any]
        Instance JSON in optimized format
    """
    # Create RNGs with instance-specific seeds
    time_rng = TaillardRNG(seed=time_seed_base + instance_idx)
    machine_rng = TaillardRNG(seed=machine_seed_base + instance_idx)
    generator = TaillardInstanceGenerator(time_rng, machine_rng)

    # Job Shop: each job has its own random permutation of machines
    # durations[j][i] = processing time for operation i of job j
    # machine_assignments[j][i] = machine for operation i of job j (random permutation)
    durations, machine_assignments = generator.generate_jobshop(n_machines, n_jobs)

    # Convert to optimized JSON format
    instance_json = _taillard_to_optimized_json(
        durations, machine_assignments, problem_type="jsp"
    )

    return instance_json


def solve_instance(
    instance_json: Dict[str, Any],
    time_limit: float = 60.0,
    problem_type: str = "jsp",
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Solve a scheduling instance using CP-SAT solver.

    Parameters
    ----------
    instance_json : Dict[str, Any]
        Instance in optimized JSON format
    time_limit : float, optional
        Solver time limit in seconds (default: 60.0)
    problem_type : str, optional
        Problem type: "jsp" for Job Shop, "fsp" for Flow Shop, or "osp" for Open Shop (default: "jsp")

    Returns
    -------
    Tuple[float, Optional[Dict[str, Any]]]
        A tuple containing:
        - makespan: float - the optimal/feasible makespan value (or NaN if no solution found)
        - schedule: Optional[Dict] - detailed schedule information
    """
    makespan, schedule = solve_standalone_cpsat(
        instance_json,
        time_limit=time_limit,
        problem_type=problem_type
    )
    return makespan, schedule


def save_instances(
    instances: List[Dict[str, Any]],
    out_path: Union[str, Path],
) -> None:
    """
    Save instances to JSON file.

    Parameters
    ----------
    instances : List[Dict[str, Any]]
        List of instances with instance, solution, obj, and problem_type
    out_path : Union[str, Path]
        Output JSON file path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(instances, f, indent=2)
    print(f"[Info] Saved {len(instances)} instances to {out_path}")
