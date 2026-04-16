import sys
import os

# Add parent directory to path to import wt
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ortools.sat.python import cp_model


def solve_wt(instance, time_limit_seconds: float = 30.0):
    """
    Solves the Single Machine Total Weighted Tardiness Problem using CP-SAT.
    """
    model = cp_model.CpModel()
    
    n = instance.n
    p = instance.p
    w = instance.w
    d = instance.d
    
    # Horizon: sum of all processing times (since single machine, no idle time needed)
    horizon = sum(p)
    
    # Variables
    start_vars = []
    end_vars = []
    interval_vars = []
    tardiness_vars = []
    
    for i in range(n):
        # Start time: 0 to horizon - p[i]
        start = model.NewIntVar(0, horizon - p[i], f'start_{i}')
        # End time: p[i] to horizon
        end = model.NewIntVar(p[i], horizon, f'end_{i}')
        # Interval constraint
        interval = model.NewIntervalVar(start, p[i], end, f'interval_{i}')
        
        start_vars.append(start)
        end_vars.append(end)
        interval_vars.append(interval)
        
        # Tardiness T_j = max(0, C_j - d_j)
        # T_j >= C_j - d_j  <==> T_j >= end - d_j
        # T_j >= 0
        
        # Upper bound for tardiness can be at most horizon (if d=0)
        t_var = model.NewIntVar(0, max(0, horizon), f'tardiness_{i}')
        
        # Constraint 1: T_j >= end_j - d_j
        model.Add(t_var >= end - d[i])
        
        tardiness_vars.append(t_var)
        
    # No Overlap constraint
    model.AddNoOverlap(interval_vars)
    
    # Objective: Minimize Sum(w_j * T_j)
    model.Minimize(sum(w[i] * tardiness_vars[i] for i in range(n)))
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.log_search_progress = False # Turn off for batch runs
    
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Extract solution
        start_times = [solver.Value(v) for v in start_vars]
        end_times = [solver.Value(v) for v in end_vars]
        tardiness_values = [solver.Value(v) for v in tardiness_vars]
        objective_value = solver.ObjectiveValue()
        
        return objective_value, {
            'start_times': start_times,
            'end_times': end_times,
            'tardiness': tardiness_values
        }
    else:
        return None, None
