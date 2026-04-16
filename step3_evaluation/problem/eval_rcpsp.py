from typing import Any, Dict, List, Tuple, Optional, Mapping
from typing import TYPE_CHECKING
import math

try:
    # When imported as a package from repo root
    from step3_evaluation.utils.eval_parsing import parse_json_field
except ModuleNotFoundError:  # pragma: no cover
    # When running from within `step3_evaluation/` (e.g., `python test_eval_rcpsp.py`)
    from utils.eval_parsing import parse_json_field

if TYPE_CHECKING:
    import pandas as pd  # pragma: no cover


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL7 = "Global7"  # temporal consistency (precedence / scheduling consistency)
PATTERN_GLOBAL2 = "Global2"  # budget/capacity (renewable resource capacities)


EPS = 1e-9


def _to_int_strict(x: Any, *, field: str) -> Tuple[Optional[int], Optional[str]]:
    """Convert x to int, only if it's integer-like (e.g., 1, '1', 1.0)."""
    if isinstance(x, bool):
        return None, f"'{field}' must be an integer, got bool."

    if isinstance(x, int):
        return x, None

    if isinstance(x, float):
        if not math.isfinite(x):
            return None, f"'{field}' must be finite, got {x}."
        if abs(x - round(x)) > EPS:
            return None, f"'{field}' must be an integer-like number, got {x}."
        return int(round(x)), None

    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None, f"'{field}' is empty."
        try:
            xf = float(s)
        except ValueError:
            return None, f"'{field}' must be an integer-like string, got {x!r}."
        if not math.isfinite(xf):
            return None, f"'{field}' must be finite, got {x!r}."
        if abs(xf - round(xf)) > EPS:
            return None, f"'{field}' must be an integer-like value, got {x!r}."
        return int(round(xf)), None

    return None, f"'{field}' must be an integer-like value, got {type(x).__name__}."


def _to_float_finite(x: Any, *, field: str) -> Tuple[Optional[float], Optional[str]]:
    """Convert x to float, must be finite."""
    if x is None:
        return None, f"Missing '{field}'."
    if isinstance(x, bool):
        return None, f"'{field}' must be a number, got bool."
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None, f"'{field}' must be numeric, got {x!r}."
    if not math.isfinite(xf):
        return None, f"'{field}' must be finite, got {x!r}."
    return xf, None


def _build_index_map(names: List[Any], n: int) -> Dict[Any, int]:
    if len(names) != n:
        raise ValueError(f"'names' list length {len(names)} does not match {n}.")
    mapping: Dict[Any, int] = {}
    for idx, identifier in enumerate(names):
        if identifier in mapping:
            raise ValueError(f"Duplicate identifier '{identifier}' in names list.")
        mapping[identifier] = idx
    return mapping


def _map_identifier(raw: Any, *, id_map: Dict[Any, int], n: int, field: str) -> Tuple[Optional[int], Optional[str]]:
    if id_map:
        if raw in id_map:
            return id_map[raw], None
        # Try string <-> int conversion
        if isinstance(raw, str):
            try:
                val = int(raw)
                if val in id_map:
                    return id_map[val], None
            except ValueError:
                pass
        elif isinstance(raw, int):
            val_s = str(raw)
            if val_s in id_map:
                return id_map[val_s], None

    return _to_int_strict(raw, field=field)


def rcpsp_parse_instance(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Parse RCPSP instance from `instance_variant`.

    Expected `instance_variant` format:
      {
        "nr_tasks": int,
        "nr_resources": int,
        "tasks": [str, ...],  # task names, e.g., ['T1', 'T2', ...]
        "resources": [int, ...],  # resource IDs, e.g., [1, 2, ...]
        "resource_capacities": [
          {"resource_id": int, "capacity": float},
          ...
        ],
        "task_records": [
          {"task_id": str, "duration": int, "demands": [float, ...], "successors": [str, ...]},
          ...
        ]
      }

    Return:
      {
        "nr_tasks": int,
        "nr_resources": int,
        "capacities": List[float],
        "tasks": List[Dict[str, Any]]
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for RCPSP.")

    nr_tasks_raw = inst_var.get("nr_tasks", None)
    nr_resources_raw = inst_var.get("nr_resources", None)
    task_names = inst_var.get("tasks", None)
    resource_ids = inst_var.get("resources", None)
    resource_capacities = inst_var.get("resource_capacities", None)
    task_records = inst_var.get("task_records", None)

    nt, err = _to_int_strict(nr_tasks_raw, field="nr_tasks")
    if err:
        raise ValueError(f"Invalid nr_tasks: {err}")
    nr, err = _to_int_strict(nr_resources_raw, field="nr_resources")
    if err:
        raise ValueError(f"Invalid nr_resources: {err}")

    if task_records is None:
        raise ValueError("RCPSP instance missing 'task_records'.")
    if not isinstance(task_records, list) or len(task_records) == 0:
        raise ValueError("RCPSP instance has empty/invalid 'task_records'.")

    if resource_capacities is None:
        raise ValueError("RCPSP instance missing 'resource_capacities'.")
    if not isinstance(resource_capacities, list) or len(resource_capacities) == 0:
        raise ValueError("RCPSP instance has empty/invalid 'resource_capacities'.")

    # Build task name to index mapping
    if task_names is not None:
        if not isinstance(task_names, list):
            raise ValueError("'tasks' must be a list if provided.")
        task_name_to_idx = _build_index_map(task_names, nt)
    else:
        task_name_to_idx = {}

    # Build resource ID to index mapping
    if resource_ids is not None:
        if not isinstance(resource_ids, list):
            raise ValueError("'resources' must be a list if provided.")
        resource_id_to_idx = _build_index_map(resource_ids, nr)
    else:
        resource_id_to_idx = {idx: idx for idx in range(nr)}

    # Build capacities list from resource_capacities
    capacities: List[float] = [0.0] * nr
    for cap_record in resource_capacities:
        if not isinstance(cap_record, dict):
            raise ValueError("Each resource_capacity must be a dictionary.")
        
        resource_id = cap_record.get("resource_id", None)
        capacity_raw = cap_record.get("capacity", None)

        if resource_id is None:
            raise ValueError("resource_capacity missing 'resource_id'.")
        if capacity_raw is None:
            raise ValueError("resource_capacity missing 'capacity'.")

        # Map resource_id to index
        if resource_id not in resource_id_to_idx:
            raise ValueError(f"resource_id {resource_id} not found in resources.")
        resource_idx = resource_id_to_idx[resource_id]

        capacity, err = _to_float_finite(capacity_raw, field="capacity")
        if err:
            raise ValueError(f"Invalid capacity: {err}")
        if capacity < -EPS:
            raise ValueError(f"Invalid capacity: capacity is negative ({capacity}).")

        if capacities[resource_idx] != 0.0:
            raise ValueError(f"Duplicate capacity definition for resource_id {resource_id}.")
        capacities[resource_idx] = float(capacity)

    # Verify all resources have capacities
    for r_idx in range(nr):
        if capacities[r_idx] == 0.0:
            raise ValueError(f"Missing capacity for resource at index {r_idx}.")

    # Convert task_records to tasks structure
    tasks: List[Optional[Dict[str, Any]]] = [None] * nt

    for record in task_records:
        if not isinstance(record, dict):
            raise ValueError("Each task_record must be a dictionary.")

        task_id = record.get("task_id", None)
        duration_raw = record.get("duration", None)
        demands_raw = record.get("demands", None)
        successors_raw = record.get("successors", None)

        if task_id is None:
            raise ValueError("task_record missing 'task_id'.")

        # Map task_id to task index
        task_idx, err = _map_identifier(
            task_id,
            id_map=task_name_to_idx,
            n=nt,
            field="task_id",
        )
        if err:
            raise ValueError(f"Invalid task_id: {err}")

        if task_idx < 0 or task_idx >= nt:
            raise ValueError(f"task_idx {task_idx} out of range [0, {nt}).")

        if tasks[task_idx] is not None:
            raise ValueError(f"Duplicate task_record for task {task_idx}.")

        # Parse duration
        duration, err = _to_float_finite(duration_raw, field="duration")
        if err:
            raise ValueError(f"Invalid duration: {err}")
        if duration < -EPS:
            raise ValueError(f"Invalid duration: duration is negative ({duration}).")

        # Parse demands
        if demands_raw is None:
            raise ValueError(f"task_record for task {task_idx} missing 'demands'.")
        if not isinstance(demands_raw, list):
            raise ValueError(f"task_record for task {task_idx}: 'demands' must be a list.")
        if len(demands_raw) != nr:
            raise ValueError(
                f"task_record for task {task_idx}: 'demands' has length {len(demands_raw)}, expected {nr}."
            )

        demands: List[float] = []
        for r_idx, demand_raw in enumerate(demands_raw):
            demand, err = _to_float_finite(demand_raw, field=f"demands[{r_idx}]")
            if err:
                raise ValueError(f"Invalid demand: {err}")
            if demand < -EPS:
                raise ValueError(f"Invalid demand: demand is negative ({demand}).")
            demands.append(float(demand))

        # Parse successors (convert string task IDs to integer indices)
        if successors_raw is None:
            successors_raw = []
        if not isinstance(successors_raw, list):
            raise ValueError(f"task_record for task {task_idx}: 'successors' must be a list.")

        successors: List[int] = []
        for succ_raw in successors_raw:
            succ_idx, err = _map_identifier(
                succ_raw,
                id_map=task_name_to_idx,
                n=nt,
                field="successor",
            )
            if err:
                raise ValueError(f"Invalid successor: {err}")
            
            if succ_idx < 0 or succ_idx >= nt:
                raise ValueError(f"Invalid successor index {succ_idx} for task {task_idx}.")
            if succ_idx == task_idx:
                raise ValueError(f"Task {task_idx} cannot be its own successor.")
            successors.append(succ_idx)

        tasks[task_idx] = {
            "duration": duration,
            "demands": demands,
            "successors": successors,
        }

    # Verify all tasks are present
    for task_idx in range(nt):
        if tasks[task_idx] is None:
            raise ValueError(f"Missing task_record for task {task_idx}.")

    return {
        "nr_tasks": nt,
        "nr_resources": nr,
        "capacities": capacities,
        "tasks": tasks,
        "task_name_to_idx": task_name_to_idx,
        "resource_id_to_idx": resource_id_to_idx,
    }



def rcpsp_check_feasibility(
    solution: List[Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check RCPSP feasibility + unified violation pattern.

    Patterns for RCPSP (per taxonomy Global_{7,2}):
      - FormatError: invalid format/types/ids, instance-schema mismatch, negative/NaN times, etc.
      - Global7: precedence violations
      - Global2: renewable resource capacity violations
    """
    nr_tasks: int = instance["nr_tasks"]
    nr_resources: int = instance["nr_resources"]
    capacities: List[float] = instance["capacities"]
    tasks: List[Dict[str, Any]] = instance["tasks"]
    task_name_to_idx: Dict[Any, int] = instance.get("task_name_to_idx", {})

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list."
    if len(solution) != nr_tasks:
        return False, PATTERN_FORMAT, f"Solution has {len(solution)} tasks, expected {nr_tasks}."

    # Normalize by task index
    sol_by_task: List[Optional[Dict[str, Any]]] = [None] * nr_tasks
    for entry in solution:
        if not isinstance(entry, dict):
            return False, PATTERN_FORMAT, "Each task schedule must be a dictionary."
        t_raw = entry.get("task", None)
        t_id, err = _map_identifier(
            t_raw,
            id_map=task_name_to_idx,
            n=nr_tasks,
            field="task",
        )
        if err:
            return False, PATTERN_FORMAT, f"Invalid task index: {err}"
        if t_id < 0 or t_id >= nr_tasks:
            return False, PATTERN_FORMAT, f"Invalid task index {t_id} (valid range: 0-{nr_tasks-1})."
        if sol_by_task[t_id] is not None:
            return False, PATTERN_FORMAT, f"Task {t_id} appears multiple times in solution."
        sol_by_task[t_id] = entry

    for t_id in range(nr_tasks):
        if sol_by_task[t_id] is None:
            return False, PATTERN_FORMAT, f"Task {t_id} is missing from solution."

    # Extract schedule (start/end) with validation
    starts: List[float] = [0.0] * nr_tasks
    ends: List[float] = [0.0] * nr_tasks

    for t_id in range(nr_tasks):
        inst_t = tasks[t_id]

        dur, err = _to_float_finite(inst_t.get("duration", None), field=f"task {t_id} duration")
        if err:
            return False, PATTERN_FORMAT, f"Instance error: {err}"
        if dur < -EPS:
            return False, PATTERN_FORMAT, f"Instance error: task {t_id} has negative duration ({dur})."

        dem_raw = inst_t.get("demands", None)
        if not isinstance(dem_raw, list) or len(dem_raw) != nr_resources:
            return False, PATTERN_FORMAT, (
                f"Instance error: task {t_id} 'demands' must be a list of length {nr_resources}."
            )

        succ_raw = inst_t.get("successors", [])
        if not isinstance(succ_raw, list):
            return False, PATTERN_FORMAT, f"Instance error: task {t_id} 'successors' must be a list."
        for s in succ_raw:
            sid, err = _to_int_strict(s, field=f"task {t_id} successor")
            if err:
                return False, PATTERN_FORMAT, f"Instance error: task {t_id} successor invalid: {err}"
            if sid < 0 or sid >= nr_tasks:
                return False, PATTERN_FORMAT, f"Instance error: task {t_id} has invalid successor {sid}."
            if sid == t_id:
                return False, PATTERN_FORMAT, f"Instance error: task {t_id} cannot be its own successor."

        sol_t = sol_by_task[t_id]  # not None

        # Validate optional duration field if present
        if "duration" in sol_t:
            sd, err = _to_float_finite(sol_t.get("duration", None), field=f"task {t_id} duration")
            if err:
                return False, PATTERN_FORMAT, f"Task {t_id}: {err}"
            if abs(sd - dur) > EPS:
                return False, PATTERN_FORMAT, f"Task {t_id}: duration mismatch (expected {dur}, got {sd})."

        # Validate optional demands field if present
        if "demands" in sol_t:
            sdem = sol_t.get("demands", None)
            if not isinstance(sdem, list) or len(sdem) != nr_resources:
                return False, PATTERN_FORMAT, f"Task {t_id}: 'demands' must be a list of length {nr_resources}."
            for r in range(nr_resources):
                exp, err = _to_float_finite(dem_raw[r], field=f"task {t_id} demands[{r}]")
                if err:
                    return False, PATTERN_FORMAT, f"Instance error: {err}"
                got, err = _to_float_finite(sdem[r], field=f"task {t_id} demands[{r}]")
                if err:
                    return False, PATTERN_FORMAT, f"Task {t_id}: {err}"
                if abs(got - exp) > EPS:
                    return False, PATTERN_FORMAT, (
                        f"Task {t_id}: demand mismatch for resource {r} (expected {exp}, got {got})."
                    )

        s, err = _to_float_finite(sol_t.get("start", None), field=f"task {t_id} start")
        if err:
            return False, PATTERN_FORMAT, f"Task {t_id}: {err}"
        if s < -EPS:
            return False, PATTERN_FORMAT, f"Task {t_id}: negative start time {s}."

        e_calc = s + dur
        if "end" in sol_t:
            e, err = _to_float_finite(sol_t.get("end", None), field=f"task {t_id} end")
            if err:
                return False, PATTERN_FORMAT, f"Task {t_id}: {err}"
            if abs(e - e_calc) > EPS:
                return False, PATTERN_FORMAT, (
                    f"Task {t_id}: end mismatch (expected start+duration={e_calc}, got {e})."
                )
            ends[t_id] = float(e)
        else:
            ends[t_id] = float(e_calc)

        starts[t_id] = float(s)

    # ---- Global7: precedence constraints ----
    for i in range(nr_tasks):
        succs = tasks[i].get("successors", [])
        for s in succs:
            j, err = _to_int_strict(s, field="successor")
            if err:
                return False, PATTERN_FORMAT, f"Instance error: task {i} has invalid successor: {err}"
            if starts[j] < ends[i] - EPS:
                return False, PATTERN_GLOBAL7, (
                    f"Precedence violation: task {j} starts at {starts[j]:.6f} "
                    f"before predecessor task {i} ends at {ends[i]:.6f}."
                )

    # ---- Global2: renewable resource capacity constraints (cumulative) ----
    # Sweep per resource with end-events processed before start-events at the same time.
    for r in range(nr_resources):
        cap = capacities[r]
        events: List[Tuple[float, int, float, int]] = []  # (time, kind, delta, task)
        for t_id in range(nr_tasks):
            dem_raw_r = tasks[t_id]["demands"][r]
            d, err = _to_float_finite(dem_raw_r, field=f"task {t_id} demands[{r}]")
            if err:
                return False, PATTERN_FORMAT, f"Instance error: {err}"
            if d < -EPS:
                return False, PATTERN_FORMAT, f"Instance error: task {t_id} has negative demand {d} for resource {r}."
            if d <= EPS:
                continue
            if d > cap + EPS:
                return False, PATTERN_GLOBAL2, (
                    f"Resource {r} violation: task {t_id} demand {d} exceeds capacity {cap}."
                )

            st, en = starts[t_id], ends[t_id]
            if en - st <= EPS:
                continue  # zero-length interval ignored

            events.append((en, 0, -d, t_id))  # end first
            events.append((st, 1, +d, t_id))  # start after

        events.sort(key=lambda x: (x[0], x[1]))
        usage = 0.0
        for time, _, delta, _t in events:
            usage += delta
            if usage > cap + EPS:
                return False, PATTERN_GLOBAL2, (
                    f"Resource {r} capacity exceeded at time {time:.6f}: "
                    f"usage {usage:.6f} > capacity {cap:.6f}."
                )
            if usage < -EPS:
                return False, PATTERN_FORMAT, (
                    f"Internal error in resource sweep for resource {r}: usage became negative ({usage})."
                )

    return True, PATTERN_OK, "Feasible RCPSP solution."


def rcpsp_objective_model(
    solution: List[Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    RCPSP objective: minimize makespan (max completion time).
    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    try:
        nr_tasks = int(instance["nr_tasks"])
        tasks = instance["tasks"]
    except Exception:
        return float("inf")
    task_name_to_idx: Dict[Any, int] = instance.get("task_name_to_idx", {})

    if not isinstance(solution, list) or len(solution) == 0:
        return float("inf")

    # Build quick lookups from solution by task id
    sol_by_task: Dict[int, Dict[str, Any]] = {}
    for entry in solution:
        if not isinstance(entry, dict):
            return float("inf")
        t_id, err = _map_identifier(
            entry.get("task", None),
            id_map=task_name_to_idx,
            n=nr_tasks,
            field="task",
        )
        if err or t_id is None:
            return float("inf")
        sol_by_task[t_id] = entry

    makespan = 0.0
    for t_id in range(nr_tasks):
        if t_id not in sol_by_task:
            return float("inf")
        dur, err = _to_float_finite(tasks[t_id].get("duration", None), field="duration")
        if err:
            return float("inf")
        s, err = _to_float_finite(sol_by_task[t_id].get("start", None), field="start")
        if err or s is None:
            return float("inf")
        completion = float(s + dur)
        if completion > makespan:
            makespan = completion

    return float(makespan)


