from typing import Any, Dict, List, Tuple, Optional
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL7 = "Global7"  # temporal consistency (release constraints + no-overlap)

EPS = 1


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


def _resolve_id_in_list(raw: Any, names: List[Any]) -> Optional[int]:
    if raw in names:
        return names.index(raw)
    if isinstance(raw, str):
        try:
            val = int(raw)
            if val in names:
                return names.index(val)
        except ValueError:
            pass
    elif isinstance(raw, int):
        val_s = str(raw)
        if val_s in names:
            return names.index(val_s)
    return None


def pms_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Parallel Machine Scheduling (PMS) instance from `instance_variant`.
    Return:
      {
        "nr_machines": int,
        "nr_jobs": int,
        "jobs": [
          {"processing_time": ..., "release_date": ..., "deadline": ...},
          ...
        ]
      }

    Notes:
      - This matches the provided instance format.
      - Deadlines are used for the tardiness objective.
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for PMS.")

    nr_machines = inst_var.get("nr_machines", None)
    nr_jobs = inst_var.get("nr_jobs", None)
    job_records = inst_var.get("job_records", None)
    job_names = inst_var.get("jobs", None)
    machine_names = inst_var.get("machines", None)

    if nr_machines is None:
        raise ValueError("PMS instance missing 'nr_machines'.")
    if nr_jobs is None:
        raise ValueError("PMS instance missing 'nr_jobs'.")
    if job_records is None:
        raise ValueError("PMS instance missing 'job_records'.")
    if not isinstance(job_records, list) or len(job_records) == 0:
        raise ValueError("PMS instance has empty/invalid 'job_records'.")

    nm, err = _to_int_strict(nr_machines, field="nr_machines")
    if err:
        raise ValueError(f"Invalid nr_machines: {err}")
    nj, err = _to_int_strict(nr_jobs, field="nr_jobs")
    if err:
        raise ValueError(f"Invalid nr_jobs: {err}")

    # Build job name to index mapping
    if job_names is not None and isinstance(job_names, list):
        job_name_to_idx = {name: idx for idx, name in enumerate(job_names)}
    else:
        job_name_to_idx = {}

    # Convert job_records to jobs structure: jobs[job_idx] = {"processing_time": ..., "release_date": ..., "deadline": ...}
    jobs = [None] * nj

    for record in job_records:
        if not isinstance(record, dict):
            raise ValueError("Each job_record must be a dictionary.")

        job_id = record.get("job_id", None)
        processing_time_raw = record.get("processing_time", None)
        release_date_raw = record.get("release_date", None)
        deadline_raw = record.get("deadline", None)

        if job_id is None:
            raise ValueError("job_record missing 'job_id'.")

        # Map job_id to job index
        if isinstance(job_id, str) and job_id in job_name_to_idx:
            job_idx = job_name_to_idx[job_id]
        else:
            # Try to parse as integer
            job_idx, err = _to_int_strict(job_id, field="job_id")
            if err:
                raise ValueError(f"Invalid job_id: {err}")

        if job_idx < 0 or job_idx >= nj:
            raise ValueError(f"job_idx {job_idx} out of range [0, {nj}).")

        if jobs[job_idx] is not None:
            raise ValueError(f"Duplicate job_record for job {job_idx}.")

        processing_time, err = _to_float_finite(processing_time_raw, field="processing_time")
        if err:
            raise ValueError(f"Invalid processing_time: {err}")

        release_date, err = _to_float_finite(release_date_raw, field="release_date")
        if err:
            raise ValueError(f"Invalid release_date: {err}")

        deadline, err = _to_float_finite(deadline_raw, field="deadline")
        if err:
            raise ValueError(f"Invalid deadline: {err}")

        jobs[job_idx] = {
            "processing_time": processing_time,
            "release_date": release_date,
            "deadline": deadline,
        }

    # Verify all jobs are present
    for job_idx in range(nj):
        if jobs[job_idx] is None:
            raise ValueError(f"Missing job_record for job {job_idx}.")

    return {
        "nr_machines": nm,
        "nr_jobs": nj,
        "jobs": jobs,
        "machine_names": machine_names,
        "job_names": job_names,
    }



def pms_check_feasibility(
    solution: List[Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check PMS feasibility for identical parallel machines with release dates,
    plus unified violation patterns.

    Patterns (aligned with taxonomy: Global_7):
      - FormatError: malformed structure/types/ids, instance-field mismatches, etc.
      - Global7: temporal consistency violations (release constraint, no-overlap)
    """
    nr_machines: int = instance["nr_machines"]
    nr_jobs: int = instance["nr_jobs"]
    jobs: List[Dict[str, Any]] = instance["jobs"]

    if not isinstance(solution, list):
        return False, PATTERN_FORMAT, "Solution must be a list."

    if len(solution) != nr_jobs:
        return False, PATTERN_FORMAT, f"Solution has {len(solution)} jobs, expected {nr_jobs}."

    if not isinstance(jobs, list) or len(jobs) != nr_jobs:
        return False, PATTERN_FORMAT, "Instance error: 'jobs' must be a list of length nr_jobs."

    # Normalize by job index
    solution_by_job: List[Optional[Dict[str, Any]]] = [None] * nr_jobs
    for entry in solution:
        if not isinstance(entry, dict):
            return False, PATTERN_FORMAT, "Each solution entry must be a dictionary."

        job_raw = entry.get("job", None)
        job_names = instance.get("job_names", None)
        job_id = -1
        found_job = False

        if job_names is not None and isinstance(job_names, list):
            idx = _resolve_id_in_list(job_raw, job_names)
            if idx is not None:
                job_id = idx
                found_job = True

        if not found_job:
            job_id, err = _to_int_strict(job_raw, field="job")
            if err:
                return False, PATTERN_FORMAT, f"Invalid job index: {err}"

        if job_id < 0 or job_id >= nr_jobs:
            return False, PATTERN_FORMAT, f"Invalid job index {job_id} (valid range: 0-{nr_jobs-1})."

        if solution_by_job[job_id] is not None:
            return False, PATTERN_FORMAT, f"Job {job_id} appears multiple times in solution."

        solution_by_job[job_id] = entry

    for job_id in range(nr_jobs):
        if solution_by_job[job_id] is None:
            return False, PATTERN_FORMAT, f"Job {job_id} is missing from solution."

    # machine_intervals[m] = [(start, end, job_id), ...]
    machine_intervals: List[List[Tuple[float, float, int]]] = [[] for _ in range(nr_machines)]

    # Validate each job assignment
    for job_id in range(nr_jobs):
        sched = solution_by_job[job_id]  # not None

        inst = jobs[job_id]
        if not isinstance(inst, dict):
            return False, PATTERN_FORMAT, f"Instance error: jobs[{job_id}] must be a dict."

        p_inst, err = _to_float_finite(inst.get("processing_time", None), field=f"instance job {job_id} processing_time")
        if err:
            return False, PATTERN_FORMAT, f"Instance error: job {job_id}: {err}"
        r_inst, err = _to_float_finite(inst.get("release_date", None), field=f"instance job {job_id} release_date")
        if err:
            return False, PATTERN_FORMAT, f"Instance error: job {job_id}: {err}"
        d_inst, err = _to_float_finite(inst.get("deadline", None), field=f"instance job {job_id} deadline")
        if err:
            return False, PATTERN_FORMAT, f"Instance error: job {job_id}: {err}"

        if p_inst < -EPS:
            return False, PATTERN_FORMAT, f"Instance error: job {job_id}: negative processing_time {p_inst}."

        # Machine
        m_raw = sched.get("machine", None)
        machine_names = instance.get("machine_names", None)
        m = -1
        found = False

        # Try lookup in machine_names
        if machine_names is not None and isinstance(machine_names, list):
            idx = _resolve_id_in_list(m_raw, machine_names)
            if idx is not None:
                m = idx
                found = True

        # Fallback: try as integer index
        if not found:
            m_idx, err = _to_int_strict(m_raw, field=f"job {job_id} machine")
            if not err:
                if 0 <= m_idx < nr_machines:
                    m = m_idx
                    found = True

        if not found:
            return False, PATTERN_FORMAT, (
                f"Job {job_id}: invalid machine {m_raw} (valid range: 0-{nr_machines-1} or in {machine_names})."
            )

        # Start/End
        s, err = _to_float_finite(sched.get("start", None), field=f"job {job_id} start")
        if err:
            return False, PATTERN_FORMAT, f"Job {job_id}: {err}"
        e, err = _to_float_finite(sched.get("end", None), field=f"job {job_id} end")
        if err:
            return False, PATTERN_FORMAT, f"Job {job_id}: {err}"

        if s < -EPS:
            return False, PATTERN_FORMAT, f"Job {job_id}: negative start time {s}."
        if e < -EPS:
            return False, PATTERN_FORMAT, f"Job {job_id}: negative end time {e}."
        if e < s - EPS:
            return False, PATTERN_FORMAT, f"Job {job_id}: end time {e} is before start time {s}."

        # Processing time
        p_sol_raw = sched.get("processing_time", None)
        if p_sol_raw is not None:
            p_sol, err = _to_float_finite(p_sol_raw, field=f"job {job_id} processing_time")
            if err:
                return False, PATTERN_FORMAT, f"Job {job_id}: {err}"
            if p_sol < -EPS:
                return False, PATTERN_FORMAT, f"Job {job_id}: negative processing_time {p_sol}."

            if abs((e - s) - p_sol) > EPS:
                return False, PATTERN_FORMAT, (
                    f"Job {job_id}: inconsistent timing: end-start={e - s:.6f} but processing_time={p_sol:.6f}."
                )
            if abs(p_sol - p_inst) > EPS:
                return False, PATTERN_FORMAT, (
                    f"Job {job_id}: processing_time mismatch (expected {p_inst}, got {p_sol})."
                )
        else:
            # If not provided, just check end-start against instance
            if abs((e - s) - p_inst) > EPS:
                return False, PATTERN_FORMAT, (
                    f"Job {job_id}: duration mismatch: end-start={e - s:.6f} but instance processing_time={p_inst:.6f}."
                )

        # Release date / deadline match instance (optional in solution)
        r_sol_raw = sched.get("release_date", None)
        if r_sol_raw is not None:
            r_sol, err = _to_float_finite(r_sol_raw, field=f"job {job_id} release_date")
            if err:
                return False, PATTERN_FORMAT, f"Job {job_id}: {err}"
            if abs(r_sol - r_inst) > EPS:
                return False, PATTERN_FORMAT, (
                    f"Job {job_id}: release_date mismatch (expected {r_inst}, got {r_sol})."
                )

        d_sol_raw = sched.get("deadline", None)
        if d_sol_raw is not None:
            d_sol, err = _to_float_finite(d_sol_raw, field=f"job {job_id} deadline")
            if err:
                return False, PATTERN_FORMAT, f"Job {job_id}: {err}"
            if abs(d_sol - d_inst) > EPS:
                return False, PATTERN_FORMAT, (
                    f"Job {job_id}: deadline mismatch (expected {d_inst}, got {d_sol})."
                )

        # ---- Global7: release constraint ----
        if s < r_inst - EPS:
            return False, PATTERN_GLOBAL7, (
                f"Job {job_id}: starts at {s:.6f} before release_date {r_inst:.6f}."
            )

        machine_intervals[m].append((s, e, job_id))

    # ---- Global7: no overlap on the same machine ----
    for machine_id in range(nr_machines):
        intervals = machine_intervals[machine_id]
        intervals.sort(key=lambda x: x[0])

        for i in range(len(intervals) - 1):
            start1, end1, job1 = intervals[i]
            start2, end2, job2 = intervals[i + 1]

            if start2 < end1 - EPS:
                return False, PATTERN_GLOBAL7, (
                    f"Machine {machine_id} conflict: "
                    f"job {job1} (ends at {end1:.6f}) overlaps with job {job2} (starts at {start2:.6f})."
                )

    return True, PATTERN_OK, "Feasible PMS solution."


def pms_objective_model(
    solution: List[Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    PMS objective: minimize MAXIMUM TARDINESS, max_j max(0, C_j - d_j),
    where C_j is completion time (end) and d_j is deadline.

    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    if not isinstance(solution, list) or len(solution) == 0:
        return float("inf")

    nr_jobs = instance.get("nr_jobs", None)
    jobs = instance.get("jobs", None)

    if not isinstance(nr_jobs, int) or not isinstance(jobs, list) or len(jobs) == 0:
        return float("inf")

    max_tardiness = 0.0

    for entry in solution:
        if not isinstance(entry, dict):
            return float("inf")

        job_raw = entry.get("job", None)
        job_names = instance.get("job_names", None)
        job_id = -1
        found_job = False

        if job_names is not None and isinstance(job_names, list):
            idx = _resolve_id_in_list(job_raw, job_names)
            if idx is not None:
                job_id = idx
                found_job = True

        if not found_job:
            job_id, err = _to_int_strict(job_raw, field="job")
            if err or job_id is None:
                return float("inf")

        if job_id < 0 or job_id >= len(jobs):
            return float("inf")

        end, err = _to_float_finite(entry.get("end", None), field="end")
        if err:
            return float("inf")

        deadline, err = _to_float_finite(jobs[job_id].get("deadline", None), field="deadline")
        if err:
            return float("inf")

        t = end - deadline
        if t > 0.0:
            if t > max_tardiness:
                max_tardiness = t

    return float(max_tardiness)
