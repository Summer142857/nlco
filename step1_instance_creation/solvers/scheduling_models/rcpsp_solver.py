"""
Standalone CP-SAT solver for Resource-Constrained Project Scheduling Problem (RCPSP).

This module provides a standalone implementation of CP-SAT solver for RCPSP
using OR-Tools. It reads instances in JSON format and solves them.

RCPSP involves:
- A set of tasks with durations
- Precedence constraints between tasks
- Resource constraints (each task requires certain amounts of resources)
- Goal: Minimize makespan (completion time of all tasks)
"""

import collections
from typing import List, Tuple, Optional, Dict, Any

from ortools.sat.python import cp_model


def parse_instance_json(instance_dict: Dict[str, Any]) -> Tuple[int, int, List[int], List[Dict[str, Any]]]:
    """
    Parse JSON instance in RCPSP format.
    
    Parameters
    ----------
    instance_dict : Dict[str, Any]
        Instance dictionary with format:
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
    
    Returns
    -------
    Tuple[int, int, List[int], List[Dict[str, Any]]]
        A tuple containing:
        - num_tasks: Number of tasks
        - num_resources: Number of resources
        - capacities: List of resource capacities
        - tasks: List of task dictionaries with duration, demands, successors
    
    Raises
    ------
    ValueError
        If the instance format is invalid
    """
    # Extract basic info
    if "nr_tasks" not in instance_dict:
        raise ValueError("Instance missing 'nr_tasks' field")
    if "nr_resources" not in instance_dict:
        raise ValueError("Instance missing 'nr_resources' field")
    if "capacities" not in instance_dict:
        raise ValueError("Instance missing 'capacities' field")
    if "tasks" not in instance_dict:
        raise ValueError("Instance missing 'tasks' field")
    
    num_tasks = instance_dict["nr_tasks"]
    num_resources = instance_dict["nr_resources"]
    capacities = instance_dict["capacities"]
    tasks_raw = instance_dict["tasks"]
    
    # Validate dimensions
    if not isinstance(capacities, list):
        raise ValueError(f"'capacities' must be a list, got {type(capacities)}")
    if len(capacities) != num_resources:
        raise ValueError(f"Number of capacities ({len(capacities)}) doesn't match 'nr_resources' ({num_resources})")
    
    if not isinstance(tasks_raw, list):
        raise ValueError(f"'tasks' must be a list, got {type(tasks_raw)}")
    if len(tasks_raw) != num_tasks:
        raise ValueError(f"Number of tasks in 'tasks' ({len(tasks_raw)}) doesn't match 'nr_tasks' ({num_tasks})")
    
    # Validate and convert tasks
    tasks = []
    for task_id, task_data in enumerate(tasks_raw):
        if not isinstance(task_data, dict):
            raise ValueError(f"Task {task_id} must be a dict, got {type(task_data)}")
        
        if "duration" not in task_data:
            raise ValueError(f"Task {task_id} missing 'duration' field")
        if "demands" not in task_data:
            raise ValueError(f"Task {task_id} missing 'demands' field")
        if "successors" not in task_data:
            raise ValueError(f"Task {task_id} missing 'successors' field")
        
        duration = task_data["duration"]
        demands = task_data["demands"]
        successors = task_data["successors"]
        
        # Validate types
        if not isinstance(duration, int):
            raise ValueError(f"Task {task_id}: duration must be int, got {type(duration)}")
        if not isinstance(demands, list):
            raise ValueError(f"Task {task_id}: demands must be a list, got {type(demands)}")
        if len(demands) != num_resources:
            raise ValueError(f"Task {task_id}: number of demands ({len(demands)}) doesn't match 'nr_resources' ({num_resources})")
        if not isinstance(successors, list):
            raise ValueError(f"Task {task_id}: successors must be a list, got {type(successors)}")
        
        # Validate duration is non-negative
        if duration < 0:
            raise ValueError(f"Task {task_id}: duration must be non-negative, got {duration}")
        
        # Validate demands are non-negative integers
        for r, demand in enumerate(demands):
            if not isinstance(demand, int):
                raise ValueError(f"Task {task_id}, resource {r}: demand must be int, got {type(demand)}")
            if demand < 0:
                raise ValueError(f"Task {task_id}, resource {r}: demand must be non-negative, got {demand}")
        
        # Validate successors are valid task IDs
        for succ_id in successors:
            if not isinstance(succ_id, int):
                raise ValueError(f"Task {task_id}: successor must be int, got {type(succ_id)}")
            if succ_id < 0 or succ_id >= num_tasks:
                raise ValueError(f"Task {task_id}: successor {succ_id} out of range [0, {num_tasks})")
            if succ_id == task_id:
                raise ValueError(f"Task {task_id}: cannot have itself as successor")
        
        tasks.append({
            "duration": duration,
            "demands": demands,
            "successors": successors
        })
    
    return num_tasks, num_resources, capacities, tasks


def build_rcpsp_model(
    num_tasks: int,
    num_resources: int,
    capacities: List[int],
    tasks: List[Dict[str, Any]]
) -> Tuple[cp_model.CpModel, Dict]:
    """
    Build CP-SAT model for RCPSP.
    
    Parameters
    ----------
    num_tasks : int
        Number of tasks
    num_resources : int
        Number of resources
    capacities : List[int]
        Resource capacities
    tasks : List[Dict[str, Any]]
        List of task dictionaries with duration, demands, successors
    
    Returns
    -------
    Tuple[cp_model.CpModel, Dict]
        A tuple containing:
        - model: The CP-SAT model
        - vars: Dictionary containing model variables (key: 'task_intervals', 'makespan')
    """
    # Compute horizon dynamically as sum of all durations
    horizon = sum(task["duration"] for task in tasks)
    
    # Create the model
    model = cp_model.CpModel()
    
    # Named tuple to store information about created variables
    task_type = collections.namedtuple("task_type", "start end interval")
    
    # Create interval variables for each task
    task_intervals = {}
    for task_id in range(num_tasks):
        duration = tasks[task_id]["duration"]
        suffix = f"_task_{task_id}"
        
        start_var = model.NewIntVar(0, horizon, "start" + suffix)
        end_var = model.NewIntVar(0, horizon, "end" + suffix)
        
        # Create interval variable
        # For zero-duration tasks: OR-Tools requires duration >= 1 for intervals
        # We use a separate interval_end_var for the interval, and keep end_var
        # separate for precedence constraints (end_var == start_var for zero-duration)
        if duration == 0:
            # Zero-duration task: create interval with duration 1 using separate end variable
            interval_end_var = model.NewIntVar(0, horizon, "interval_end" + suffix)
            interval_var = model.NewIntervalVar(
                start_var, 1, interval_end_var, "interval" + suffix
            )
            # For zero-duration tasks, end_var (used in precedence) equals start_var
            model.Add(end_var == start_var)
        else:
            # Normal task: create interval with actual duration
            interval_var = model.NewIntervalVar(
                start_var, duration, end_var, "interval" + suffix
            )
            # NewIntervalVar automatically enforces end_var = start_var + duration
        
        task_intervals[task_id] = task_type(
            start=start_var, end=end_var, interval=interval_var
        )
    
    # Add precedence constraints
    for task_id in range(num_tasks):
        for successor_id in tasks[task_id]["successors"]:
            # Task task_id must finish before task successor_id starts
            model.Add(
                task_intervals[task_id].end <= task_intervals[successor_id].start
            )
    
    # Add resource capacity constraints
    # For each resource, the sum of demands at any time cannot exceed capacity
    for resource_id in range(num_resources):
        capacity = capacities[resource_id]
        
        # Collect intervals and demands for this resource
        intervals = []
        demands_list = []
        
        for task_id in range(num_tasks):
            demand = tasks[task_id]["demands"][resource_id]
            duration = tasks[task_id]["duration"]
            # Only add tasks that use this resource AND have non-zero duration
            # Zero-duration tasks don't consume resources over time
            if demand > 0 and duration > 0:
                intervals.append(task_intervals[task_id].interval)
                demands_list.append(demand)
        
        # Add cumulative constraint: sum of demands at any time <= capacity
        if intervals:  # Only add if there are tasks using this resource
            model.AddCumulative(intervals, demands_list, capacity)
    
    # Makespan objective (minimize maximum completion time)
    makespan = model.NewIntVar(0, horizon, "makespan")
    all_end_times = [task_intervals[task_id].end for task_id in range(num_tasks)]
    model.AddMaxEquality(makespan, all_end_times)
    model.Minimize(makespan)
    
    return model, {"task_intervals": task_intervals, "makespan": makespan}


def solve_rcpsp(
    instance_data: Dict[str, Any],
    time_limit: float = 60.0
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Solve RCPSP instance using CP-SAT solver.
    
    Parameters
    ----------
    instance_data : Dict[str, Any]
        Instance dictionary with RCPSP format
    time_limit : float, optional
        Solver time limit in seconds. Default is 60.0.
    
    Returns
    -------
    Tuple[float, Optional[Dict[str, Any]]]
        A tuple containing:
        - makespan: float - the optimal/feasible makespan value (or NaN if no solution found)
        - schedule: Optional[Dict] - detailed schedule information
          Format: List of task info dicts, each with "task", "start", "end", "duration"
    """
    try:
        # Parse instance
        num_tasks, num_resources, capacities, tasks = parse_instance_json(instance_data)
        
        # Build CP-SAT model
        model, vars = build_rcpsp_model(num_tasks, num_resources, capacities, tasks)
        task_intervals = vars['task_intervals']
        
        # Create solver and set time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        
        # Solve
        status = solver.Solve(model)
        
        # Check if solution found
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            makespan = float(solver.ObjectiveValue())
            
            # Extract schedule
            schedule = []
            for task_id in range(num_tasks):
                start_time = solver.Value(task_intervals[task_id].start)
                end_time = solver.Value(task_intervals[task_id].end)
                duration = tasks[task_id]["duration"]
                
                schedule.append({
                    "task": task_id,
                    "start": start_time,
                    "end": end_time,
                    "duration": duration,
                    "demands": tasks[task_id]["demands"]
                })
            
            return makespan, schedule
        else:
            # No solution found
            return float("nan"), None
            
    except Exception as e:
        print(f"[Error] RCPSP CP-SAT solver failed: {e}")
        import traceback
        traceback.print_exc()
        return float("nan"), None

