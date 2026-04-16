from typing import Any, Dict, List, Tuple, Optional
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


EPS = 1e-9
PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL7 = "Global7"  # temporal consistency (no-overlap scheduling)


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
            # allow "1" and also "1.0" but reject "1.2"
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


def _letter_to_index(label: str) -> Optional[int]:
    if not label:
        return None
    idx = 0
    for ch in label:
        if not ch.isalpha() or not ch.isupper():
            return None
        idx = idx * 26 + (ord(ch) - ord("A"))
    return idx


def _build_index_map(names: List[Any], n: int) -> Dict[Any, int]:
    if len(names) != n:
        raise ValueError(f"'names' list length {len(names)} does not match {n}.")
    mapping: Dict[Any, int] = {}
    for idx, identifier in enumerate(names):
        if identifier in mapping:
            raise ValueError(f"Duplicate identifier '{identifier}' in names list.")
        mapping[identifier] = idx
    return mapping


def _map_job_id(raw: Any, *, job_map: Dict[Any, int], nj: int) -> Tuple[Optional[int], Optional[str]]:
    if job_map:
        if raw in job_map:
            return job_map[raw], None
        # Try string <-> int conversion
        if isinstance(raw, str):
            try:
                val = int(raw)
                if val in job_map:
                    return job_map[val], None
            except ValueError:
                pass
        elif isinstance(raw, int):
            val_s = str(raw)
            if val_s in job_map:
                return job_map[val_s], None

    return _to_int_strict(raw, field="job")


def _map_machine(raw: Any, *, field: str, machine_map: Dict[Any, int], nm: int) -> Tuple[Optional[int], Optional[str]]:
    if machine_map:
        if raw in machine_map:
            return machine_map[raw], None
        # Try string <-> int conversion
        if isinstance(raw, str):
            try:
                val = int(raw)
                if val in machine_map:
                    return machine_map[val], None
            except ValueError:
                pass
        elif isinstance(raw, int):
            val_s = str(raw)
            if val_s in machine_map:
                return machine_map[val_s], None

    if isinstance(raw, str):
        idx = _letter_to_index(raw)
        if idx is not None:
            if idx < 0 or idx >= nm:
                return None, f"{field} machine {raw!r} index {idx} out of range 0-{nm-1}."
            return idx, None
    return _to_int_strict(raw, field=field)


def osp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse OSP instance from `instance_variant`.
    Return:
      {
        "nr_machines": int,
        "nr_jobs": int,
        "jobs": [[[machine_id, duration], ...], ...]
      }

    Expected instance format:
      instance.jobs[j][t] == [machine_id, duration]
    Typically, each job has exactly one operation per machine (len == nr_machines).
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for OSP.")

    nr_machines = inst_var.get("nr_machines", None)
    nr_jobs = inst_var.get("nr_jobs", None)
    records = inst_var.get("records", None)
    job_names = inst_var.get("jobs", None)
    machine_names = inst_var.get("machines", None) or inst_var.get("machine_names", None)

    if nr_machines is None:
        raise ValueError("OSP instance missing 'nr_machines'.")
    if nr_jobs is None:
        raise ValueError("OSP instance missing 'nr_jobs'.")
    if records is None:
        raise ValueError("OSP instance missing 'records'.")
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("OSP instance has empty/invalid 'records'.")

    nm, err = _to_int_strict(nr_machines, field="nr_machines")
    if err:
        raise ValueError(f"Invalid nr_machines: {err}")
    nj, err = _to_int_strict(nr_jobs, field="nr_jobs")
    if err:
        raise ValueError(f"Invalid nr_jobs: {err}")

    # Build job name to index mapping
    if job_names is not None:
        if not isinstance(job_names, list):
            raise ValueError("'jobs' must be a list if provided.")
        job_name_to_idx = _build_index_map(job_names, nj)
    else:
        job_name_to_idx = {}

    if machine_names is not None:
        if not isinstance(machine_names, list):
            raise ValueError("'machines' must be a list if provided.")
        machine_name_to_idx = _build_index_map(machine_names, nm)
    else:
        machine_name_to_idx = {}

    # Convert records to jobs structure: jobs[job_idx][task_idx] = [machine_id, duration]
    # Initialize jobs structure
    jobs = [[None for _ in range(nm)] for _ in range(nj)]

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be a dictionary.")

        job_id = record.get("job_id", None)
        task_id_raw = record.get("task_id", None)
        machine_id_raw = record.get("machine_id", None)
        duration_raw = record.get("duration", None)

        if job_id is None:
            raise ValueError("Record missing 'job_id'.")

        # Map job_id to job index
        job_idx, err = _map_job_id(job_id, job_map=job_name_to_idx, nj=nj)
        if err:
            raise ValueError(f"Invalid job_id: {err}")

        if job_idx < 0 or job_idx >= nj:
            raise ValueError(f"job_idx {job_idx} out of range [0, {nj}).")

        task_id, err = _to_int_strict(task_id_raw, field="task_id")
        if err:
            raise ValueError(f"Invalid task_id: {err}")

        if task_id < 0 or task_id >= nm:
            raise ValueError(f"task_id {task_id} out of range [0, {nm}).")

        machine_id, err = _map_machine(
            machine_id_raw,
            field="machine_id",
            machine_map=machine_name_to_idx,
            nm=nm,
        )
        if err:
            raise ValueError(f"Invalid machine_id: {err}")

        if machine_id < 0 or machine_id >= nm:
            raise ValueError(f"machine_id {machine_id} out of range [0, {nm}).")

        duration, err = _to_float_finite(duration_raw, field="duration")
        if err:
            raise ValueError(f"Invalid duration: {err}")

        if jobs[job_idx][task_id] is not None:
            raise ValueError(f"Duplicate record for job {job_idx}, task {task_id}.")

        jobs[job_idx][task_id] = [machine_id, duration]

    # Verify all tasks are present (each job should have exactly nr_machines tasks)
    for job_idx in range(nj):
        for task_idx in range(nm):
            if jobs[job_idx][task_idx] is None:
                raise ValueError(f"Missing record for job {job_idx}, task {task_idx}.")

    return {
        "nr_machines": nm,
        "nr_jobs": nj,
        "jobs": jobs,
        "job_name_to_idx": job_name_to_idx,
        "machine_name_to_idx": machine_name_to_idx,
    }



def _detect_job_index_shift(job_indices: List[int], nr_jobs: int) -> int:
    """
    Decide whether job indices are 0-based or 1-based.

    Returns:
        0   -> already 0-based (valid range [0, nr_jobs-1])
        -1  -> 1-based (valid range [1, nr_jobs]) -> shift down by 1

    Raises ValueError if it can't be interpreted cleanly.
    """
    if len(job_indices) == 0:
        return 0

    mn = min(job_indices)
    mx = max(job_indices)
    s = set(job_indices)

    # Clean 0-based
    if mn >= 0 and mx <= nr_jobs - 1:
        return 0

    # Clean 1-based (and not mixed with 0)
    if mn >= 1 and mx <= nr_jobs and 0 not in s:
        return -1

    # Ambiguous/mixed or out of bounds
    raise ValueError(
        f"Job indices do not match 0-based [0..{nr_jobs-1}] or 1-based [1..{nr_jobs}]. "
        f"Got min={mn}, max={mx}."
    )


def osp_check_feasibility(
    solution: List[Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check Open Shop Scheduling Problem (OSP) feasibility + unified violation pattern.
    """
    nr_machines: int = instance["nr_machines"]
    nr_jobs: int = instance["nr_jobs"]
    jobs: List[List[List[int]]] = instance["jobs"]
    job_name_to_idx: Dict[Any, int] = instance.get("job_name_to_idx", {})
    machine_name_to_idx: Dict[Any, int] = instance.get("machine_name_to_idx", {})

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list."

    if len(solution) != nr_jobs:
        return False, PATTERN_FORMAT, f"Solution has {len(solution)} jobs, expected {nr_jobs}."

    # ---- Normalize by job index (supports both 0-based and 1-based) ----
    # First pass: parse job indices, but do not range-check yet (because base may differ)
    parsed_jobs: List[Tuple[int, Dict[str, Any]]] = []
    numeric_indices: List[int] = []

    for job_schedule in solution:
        if not isinstance(job_schedule, dict):
            return False, PATTERN_FORMAT, "Each job schedule must be a dictionary."

        job_idx_raw = job_schedule.get("job", None)
        job_idx, err = _map_job_id(job_idx_raw, job_map=job_name_to_idx, nj=nr_jobs)
        if err:
            return False, PATTERN_FORMAT, f"Invalid job index: {err}"

        if not isinstance(job_idx, int):
            return False, PATTERN_FORMAT, f"Invalid job index {job_idx!r}."

        parsed_jobs.append((job_idx, job_schedule))

        # Only consider base-detection when we are NOT using name->idx mapping
        # (When job_name_to_idx is present and used, indices are already canonical.)
        if not job_name_to_idx:
            numeric_indices.append(job_idx)

    # Decide whether we should shift 1-based -> 0-based
    shift = 0
    if not job_name_to_idx:
        try:
            shift = _detect_job_index_shift(numeric_indices, nr_jobs)
        except ValueError as e:
            return False, PATTERN_FORMAT, str(e)

    # Second pass: apply shift, then validate range + uniqueness + coverage
    solution_by_job: List[Optional[Dict[str, Any]]] = [None] * nr_jobs

    for raw_idx, job_schedule in parsed_jobs:
        job_idx = raw_idx + shift

        if job_idx < 0 or job_idx >= nr_jobs:
            # Provide a helpful hint about detected base
            if shift == -1:
                valid = f"1-{nr_jobs} (1-based) or 0-{nr_jobs-1} (0-based)"
            else:
                valid = f"0-{nr_jobs-1} (0-based) or 1-{nr_jobs} (1-based)"
            return False, PATTERN_FORMAT, f"Invalid job index {raw_idx} (valid: {valid})."

        if solution_by_job[job_idx] is not None:
            return False, PATTERN_FORMAT, f"Job {job_idx} appears multiple times in solution."

        solution_by_job[job_idx] = job_schedule

    for job_idx in range(nr_jobs):
        if solution_by_job[job_idx] is None:
            return False, PATTERN_FORMAT, f"Job {job_idx} is missing from solution."

    # ---- existing logic below unchanged ----

    machine_intervals: List[List[Tuple[float, float, int, int]]] = [[] for _ in range(nr_machines)]
    job_intervals: List[List[Tuple[float, float, int, int]]] = [[] for _ in range(nr_jobs)]

    for job_idx in range(nr_jobs):
        job_schedule = solution_by_job[job_idx]  # not None
        tasks_raw = job_schedule.get("tasks", None)

        if not isinstance(tasks_raw, list):
            return False, PATTERN_FORMAT, f"Job {job_idx} 'tasks' must be a list."

        expected_tasks = jobs[job_idx]
        if not isinstance(expected_tasks, list) or len(expected_tasks) == 0:
            return False, PATTERN_FORMAT, f"Instance error: job {job_idx} has empty/invalid tasks list."

        n_tasks = len(expected_tasks)

        if n_tasks != nr_machines:
            return False, PATTERN_FORMAT, (
                f"Instance error: job {job_idx} has {n_tasks} operations, expected nr_machines={nr_machines}."
            )

        if len(tasks_raw) != n_tasks:
            return False, PATTERN_FORMAT, (
                f"Job {job_idx} has {len(tasks_raw)} tasks in solution, expected {n_tasks}."
            )

        task_map: Dict[int, Dict[str, Any]] = {}
        for entry in tasks_raw:
            if not isinstance(entry, dict):
                return False, PATTERN_FORMAT, f"Job {job_idx}: each task entry must be a dictionary."

            t_raw = entry.get("task", None)
            t_id, err = _to_int_strict(t_raw, field=f"job {job_idx} task")
            if err:
                return False, PATTERN_FORMAT, f"Job {job_idx}: invalid 'task' field: {err}"

            if t_id in task_map:
                return False, PATTERN_FORMAT, f"Job {job_idx}: duplicate task id {t_id}."
            task_map[t_id] = entry

        expected_ids = set(range(n_tasks))
        got_ids = set(task_map.keys())
        if got_ids != expected_ids:
            missing = sorted(expected_ids - got_ids)
            extra = sorted(got_ids - expected_ids)
            if missing:
                return False, PATTERN_FORMAT, f"Job {job_idx}: missing tasks {missing}."
            return False, PATTERN_FORMAT, f"Job {job_idx}: unexpected task ids {extra}."

        for task_idx in range(n_tasks):
            task_schedule = task_map[task_idx]

            expected_machine_raw, expected_duration_raw = expected_tasks[task_idx]

            exp_m, err = _to_int_strict(expected_machine_raw, field="expected_machine")
            if err:
                return False, PATTERN_FORMAT, f"Instance error: job {job_idx} task {task_idx} expected machine invalid: {err}"
            exp_d, err = _to_float_finite(expected_duration_raw, field="expected_duration")
            if err:
                return False, PATTERN_FORMAT, f"Instance error: job {job_idx} task {task_idx} expected duration invalid: {err}"

            m_raw = task_schedule.get("machine", None)
            m, err = _map_machine(
                m_raw,
                field=f"job {job_idx} task {task_idx} machine",
                machine_map=machine_name_to_idx,
                nm=nr_machines,
            )
            if err:
                return False, PATTERN_FORMAT, f"Job {job_idx}, task {task_idx}: {err}"

            if m < 0 or m >= nr_machines:
                return False, PATTERN_FORMAT, (
                    f"Job {job_idx}, task {task_idx}: invalid machine {m} (valid range: 0-{nr_machines-1})."
                )

            if m != exp_m:
                return False, PATTERN_FORMAT, (
                    f"Job {job_idx}, task {task_idx}: machine mismatch (expected {exp_m}, got {m})."
                )

            d_raw = task_schedule.get("duration", None)
            d, err = _to_float_finite(d_raw, field=f"job {job_idx} task {task_idx} duration")
            if err:
                return False, PATTERN_FORMAT, f"Job {job_idx}, task {task_idx}: {err}"

            if d < -EPS:
                return False, PATTERN_FORMAT, f"Job {job_idx}, task {task_idx}: negative duration {d}."

            if abs(d - exp_d) > EPS:
                return False, PATTERN_FORMAT, (
                    f"Job {job_idx}, task {task_idx}: duration mismatch (expected {exp_d}, got {d})."
                )

            s_raw = task_schedule.get("start", None)
            s, err = _to_float_finite(s_raw, field=f"job {job_idx} task {task_idx} start")
            if err:
                return False, PATTERN_FORMAT, f"Job {job_idx}, task {task_idx}: {err}"

            if s < -EPS:
                return False, PATTERN_FORMAT, f"Job {job_idx}, task {task_idx}: negative start time {s}."

            end = s + d
            machine_intervals[m].append((s, end, job_idx, task_idx))
            job_intervals[job_idx].append((s, end, job_idx, task_idx))

    for machine_id in range(nr_machines):
        intervals = machine_intervals[machine_id]
        intervals.sort(key=lambda x: x[0])
        for i in range(len(intervals) - 1):
            start1, end1, job1, task1 = intervals[i]
            start2, end2, job2, task2 = intervals[i + 1]
            if start2 < end1 - EPS:
                return False, PATTERN_GLOBAL7, (
                    f"Machine {machine_id} conflict: "
                    f"job {job1} task {task1} (ends at {end1:.6f}) "
                    f"overlaps with job {job2} task {task2} (starts at {start2:.6f})."
                )

    for job_id in range(nr_jobs):
        intervals = job_intervals[job_id]
        intervals.sort(key=lambda x: x[0])
        for i in range(len(intervals) - 1):
            start1, end1, job1, task1 = intervals[i]
            start2, end2, job2, task2 = intervals[i + 1]
            if start2 < end1 - EPS:
                return False, PATTERN_GLOBAL7, (
                    f"Job {job_id} conflict: "
                    f"task {task1} (ends at {end1:.6f}) "
                    f"overlaps with task {task2} (starts at {start2:.6f})."
                )

    return True, PATTERN_OK, "Feasible OSP solution."



def osp_objective_model(
    solution: List[Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    OSP objective: minimize makespan (max completion time).
    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    if not isinstance(solution, list) or len(solution) == 0:
        return float("inf")

    makespan = 0.0

    for job_schedule in solution:
        if not isinstance(job_schedule, dict):
            return float("inf")

        tasks = job_schedule.get("tasks", None)
        if not isinstance(tasks, list) or len(tasks) == 0:
            return float("inf")

        for task_schedule in tasks:
            if not isinstance(task_schedule, dict):
                return float("inf")

            s, err = _to_float_finite(task_schedule.get("start", None), field="start")
            if err:
                return float("inf")

            d, err = _to_float_finite(task_schedule.get("duration", None), field="duration")
            if err:
                return float("inf")

            if s < -EPS or d < -EPS:
                return float("inf")

            completion = s + d
            if completion > makespan:
                makespan = completion

    return float(makespan)
