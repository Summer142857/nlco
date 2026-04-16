"""
Standalone CP-SAT solver for Parallel Machine Scheduling (PMS).

This module provides a standalone implementation of CP-SAT solver for PMS
using OR-Tools. It reads instances in JSON format and solves them.

PMS involves:
- A set of jobs with processing times, release dates, and deadlines
- Multiple parallel machines
- Each job can be scheduled on any machine
- Goal: Minimize maximum tardiness (T_max)
  where tardiness = max(0, completion_time - deadline)
  and T_max = max(tardiness over all jobs)
"""

import collections
from typing import List, Tuple, Optional, Dict, Any

from ortools.sat.python import cp_model


def parse_instance_json(instance_dict: Dict[str, Any]) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Parse JSON instance in PMS format.
    
    Parameters
    ----------
    instance_dict : Dict[str, Any]
        Instance dictionary with format:
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
    
    Returns
    -------
    Tuple[int, int, List[Dict[str, Any]]]
        A tuple containing:
        - num_jobs: Number of jobs
        - num_machines: Number of machines
        - jobs: List of job dictionaries with processing_time, release_date, deadline
    
    Raises
    ------
    ValueError
        If the instance format is invalid
    """
    # Extract basic info
    if "nr_jobs" not in instance_dict:
        raise ValueError("Instance missing 'nr_jobs' field")
    if "nr_machines" not in instance_dict:
        raise ValueError("Instance missing 'nr_machines' field")
    if "jobs" not in instance_dict:
        raise ValueError("Instance missing 'jobs' field")
    
    num_jobs = instance_dict["nr_jobs"]
    num_machines = instance_dict["nr_machines"]
    jobs_raw = instance_dict["jobs"]
    
    # Validate dimensions
    if not isinstance(jobs_raw, list):
        raise ValueError(f"'jobs' must be a list, got {type(jobs_raw)}")
    if len(jobs_raw) != num_jobs:
        raise ValueError(f"Number of jobs in 'jobs' ({len(jobs_raw)}) doesn't match 'nr_jobs' ({num_jobs})")
    
    # Validate and convert jobs
    jobs = []
    for job_id, job_data in enumerate(jobs_raw):
        if not isinstance(job_data, dict):
            raise ValueError(f"Job {job_id} must be a dict, got {type(job_data)}")
        
        if "processing_time" not in job_data:
            raise ValueError(f"Job {job_id} missing 'processing_time' field")
        if "release_date" not in job_data:
            raise ValueError(f"Job {job_id} missing 'release_date' field")
        if "deadline" not in job_data:
            raise ValueError(f"Job {job_id} missing 'deadline' field")
        
        processing_time = job_data["processing_time"]
        release_date = job_data["release_date"]
        deadline = job_data["deadline"]
        
        # Validate types
        if not isinstance(processing_time, int):
            raise ValueError(f"Job {job_id}: processing_time must be int, got {type(processing_time)}")
        if not isinstance(release_date, int):
            raise ValueError(f"Job {job_id}: release_date must be int, got {type(release_date)}")
        if not isinstance(deadline, int):
            raise ValueError(f"Job {job_id}: deadline must be int, got {type(deadline)}")
        
        # Validate values
        if processing_time < 0:
            raise ValueError(f"Job {job_id}: processing_time must be non-negative, got {processing_time}")
        if release_date < 0:
            raise ValueError(f"Job {job_id}: release_date must be non-negative, got {release_date}")
        # Note: No validation for deadline >= release_date + processing_time
        # since we're minimizing tardiness, jobs can exceed their deadlines
        
        jobs.append({
            "processing_time": processing_time,
            "release_date": release_date,
            "deadline": deadline
        })
    
    return num_jobs, num_machines, jobs


def build_pms_model(
    num_jobs: int,
    num_machines: int,
    jobs: List[Dict[str, Any]]
) -> Tuple[cp_model.CpModel, Dict]:
    """
    Build CP-SAT model for PMS (minimizing maximum tardiness).
    
    Parameters
    ----------
    num_jobs : int
        Number of jobs
    num_machines : int
        Number of machines
    jobs : List[Dict[str, Any]]
        List of job dictionaries with processing_time, release_date, deadline
    
    Returns
    -------
    Tuple[cp_model.CpModel, Dict]
        A tuple containing:
        - model: The CP-SAT model
        - vars: Dictionary containing model variables (key: 'job_machine_intervals', 'machine_assignment', 'max_tardiness')
    """
    # Compute horizon dynamically as maximum deadline plus buffer for tardiness
    horizon = max(job["deadline"] for job in jobs) * 3 if jobs else 3000
    
    # Create the model
    model = cp_model.CpModel()
    
    # Named tuple to store information about created variables
    interval_type = collections.namedtuple("interval_type", "start end interval presence")
    
    # Create optional interval variables for each job-machine pair
    # Each job must be assigned to exactly one machine
    job_machine_intervals = {}
    machine_assignment = {}
    
    for job_id in range(num_jobs):
        job = jobs[job_id]
        processing_time = job["processing_time"]
        release_date = job["release_date"]
        deadline = job["deadline"]
        
        # Create machine assignment variables (one per machine for this job)
        # Exactly one must be True
        machine_vars = []
        job_intervals = {}
        
        for machine_id in range(num_machines):
            suffix = f"_job_{job_id}_machine_{machine_id}"
            
            # Presence literal: True if job is assigned to this machine
            presence_var = model.NewBoolVar("presence" + suffix)
            machine_vars.append(presence_var)
            
            # Machine-specific start and end variables
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            
            # Enforce release date
            model.Add(start_var >= release_date).OnlyEnforceIf(presence_var)
            
            # Create optional interval variable
            # NewOptionalIntervalVar automatically enforces: end = start + duration when present
            interval_var = model.NewOptionalIntervalVar(
                start_var, processing_time, end_var, presence_var, "interval" + suffix
            )
            
            job_intervals[machine_id] = interval_type(
                start=start_var,
                end=end_var,
                interval=interval_var,
                presence=presence_var
            )
        
        # Each job must be assigned to exactly one machine
        model.AddExactlyOne(machine_vars)
        machine_assignment[job_id] = machine_vars
        
        job_machine_intervals[job_id] = job_intervals
    
    # Add no-overlap constraints per machine
    for machine_id in range(num_machines):
        intervals_for_machine = []
        for job_id in range(num_jobs):
            intervals_for_machine.append(job_machine_intervals[job_id][machine_id].interval)
        model.AddNoOverlap(intervals_for_machine)
    
    # Objective: minimize maximum tardiness (T_max)
    # Tardiness = max(0, completion_time - deadline)
    # T_max = max(tardiness over all jobs)
    job_tardiness = []
    for job_id in range(num_jobs):
        job = jobs[job_id]
        processing_time = job["processing_time"]
        deadline = job["deadline"]
        
        # For each job, compute completion time based on which machine it's assigned to
        # We need to get the end time from the assigned machine's interval
        job_ends = []
        for machine_id in range(num_machines):
            end_var = job_machine_intervals[job_id][machine_id].end
            job_ends.append(end_var)
        
        # Create a variable for the actual job end time (will equal one of the machine-specific ends)
        job_end = model.NewIntVar(0, horizon, f"job_end_{job_id}")
        
        # Link job_end to the correct machine-specific end based on assignment
        for machine_id in range(num_machines):
            presence_var = job_machine_intervals[job_id][machine_id].presence
            end_var = job_machine_intervals[job_id][machine_id].end
            model.Add(job_end == end_var).OnlyEnforceIf(presence_var)
        
        # Tardiness = max(0, completion_time - deadline)
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{job_id}")
        model.Add(tardiness >= 0)
        model.Add(tardiness >= job_end - deadline)
        job_tardiness.append(tardiness)
    
    # Maximum tardiness (T_max = max of all tardiness)
    max_tardiness = model.NewIntVar(0, horizon, "max_tardiness")
    model.AddMaxEquality(max_tardiness, job_tardiness)
    model.Minimize(max_tardiness)
    
    return model, {
        "job_machine_intervals": job_machine_intervals,
        "machine_assignment": machine_assignment,
        "max_tardiness": max_tardiness
    }



def solve_pms(
    instance_data: Dict[str, Any],
    time_limit: float = 60.0
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Solve PMS instance using CP-SAT solver.
    
    The objective is to minimize maximum tardiness (T_max).
    Tardiness of a job = max(0, completion_time - deadline)
    T_max = max(tardiness over all jobs)
    
    Parameters
    ----------
    instance_data : Dict[str, Any]
        Instance dictionary with PMS format
    time_limit : float, optional
        Solver time limit in seconds. Default is 60.0.
    
    Returns
    -------
    Tuple[float, Optional[Dict[str, Any]]]
        A tuple containing:
        - max_tardiness: float - the optimal/feasible maximum tardiness value (or NaN if no solution found)
        - schedule: Optional[Dict] - detailed schedule information
          Format: List of job info dicts, each with "job", "machine", "start", "end", "processing_time"
    """
    try:
        # Parse instance
        num_jobs, num_machines, jobs = parse_instance_json(instance_data)
        
        # Build CP-SAT model
        model, vars = build_pms_model(num_jobs, num_machines, jobs)
        job_machine_intervals = vars['job_machine_intervals']
        max_tardiness_var = vars['max_tardiness']
        
        # Create solver and set time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        
        # Solve
        status = solver.Solve(model)
        
        # Check if solution found
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            max_tardiness = float(solver.ObjectiveValue())
            
            # Extract schedule
            schedule = []
            for job_id in range(num_jobs):
                job = jobs[job_id]
                processing_time = job["processing_time"]
                
                # Find which machine this job is assigned to and get its start/end times
                assigned_machine = None
                start_time = None
                end_time = None
                
                for machine_id in range(num_machines):
                    presence_var = job_machine_intervals[job_id][machine_id].presence
                    if solver.Value(presence_var):
                        assigned_machine = machine_id
                        start_time = solver.Value(job_machine_intervals[job_id][machine_id].start)
                        end_time = solver.Value(job_machine_intervals[job_id][machine_id].end)
                        break
                
                if assigned_machine is not None:
                    schedule.append({
                        "job": job_id,
                        "machine": assigned_machine,
                        "start": start_time,
                        "end": end_time,
                        "processing_time": processing_time,
                        "release_date": job["release_date"],
                        "deadline": job["deadline"]
                    })
                else:
                    # Job not assigned (should not happen if solution is valid)
                    schedule.append({
                        "job": job_id,
                        "machine": -1,
                        "start": 0,
                        "end": 0,
                        "processing_time": processing_time,
                        "release_date": job["release_date"],
                        "deadline": job["deadline"]
                    })
            
            return max_tardiness, schedule
        else:
            # No solution found
            return float("nan"), None
            
    except Exception as e:
        print(f"[Error] PMS CP-SAT solver failed: {e}")
        import traceback
        traceback.print_exc()
        return float("nan"), None

