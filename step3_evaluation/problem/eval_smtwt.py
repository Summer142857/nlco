from typing import Any, Dict, List, Tuple, Optional, Mapping, Set
from typing import TYPE_CHECKING
import math

try:
    # When imported as a package from repo root
    from step3_evaluation.utils.eval_parsing import parse_json_field
except ModuleNotFoundError:  # pragma: no cover
    # When running from within `step3_evaluation/` (e.g., `python test_eval_smtwt.py`)
    from utils.eval_parsing import parse_json_field

if TYPE_CHECKING:
    import pandas as pd  # pragma: no cover

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL7 = "Global7"  # temporal consistency (noOverlap + timing consistency)


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


def smtwt_parse_instance(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Parse Single-Machine Total Weighted Tardiness (SMTWT) instance from `instance_variant`.

    Expected `instance_variant` format:
      {
        "n_jobs": int,
        "jobs": [int, ...],  # job IDs, e.g., [0, 1, 2, 3, 4]
        "job_records": [
          {"job_id": int, "processing_time": float, "weight": float, "due_date": float},
          ...
        ]
      }

    Return:
      {
        "n_jobs": int,
        "processing_times": [p_j],
        "weights": [w_j],
        "due_dates": [d_j]
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for SMTWT.")

    n_jobs_raw = inst_var.get("n_jobs", None)
    job_records = inst_var.get("job_records", None)
    job_ids_raw = inst_var.get("jobs", None)

    if n_jobs_raw is None:
        raise ValueError("SMTWT instance missing 'n_jobs'.")
    if job_records is None:
        raise ValueError("SMTWT instance missing 'job_records'.")
    if not isinstance(job_records, list) or len(job_records) == 0:
        raise ValueError("SMTWT instance has empty/invalid 'job_records'.")

    n_jobs, err = _to_int_strict(n_jobs_raw, field="n_jobs")
    if err:
        raise ValueError(f"Invalid n_jobs: {err}")

    # Build job ID to index mapping (if jobs list provided)
    job_ids: Optional[List[Any]]
    job_id_to_idx: Dict[Any, int]
    job_ids_list_valid = isinstance(job_ids_raw, list)
    if job_ids_raw is None:
        job_ids = None
    else:
        job_ids = job_ids_raw if job_ids_list_valid else None

    if job_ids is not None:
        if len(job_ids) != n_jobs:
            raise ValueError(f"'jobs' has {len(job_ids)} entries, expected {n_jobs}.")
        if len(set(job_ids)) != n_jobs:
            raise ValueError("Duplicate entries found in 'jobs'.")
        job_id_to_idx = {job_id: idx for idx, job_id in enumerate(job_ids)}
    else:
        job_id_to_idx = {}
        assigned_idxs: Set[int] = set()
        def _allocate_next_idx() -> int:
            for candidate in range(n_jobs):
                if candidate not in assigned_idxs:
                    assigned_idxs.add(candidate)
                    return candidate
            raise ValueError(f"Too many unique job_ids; expected {n_jobs}.")

    job_ids_by_idx: List[Optional[Any]] = [None] * n_jobs

    # Initialize lists with None to track which jobs are present
    processing_times: List[Optional[float]] = [None] * n_jobs
    weights: List[Optional[float]] = [None] * n_jobs
    due_dates: List[Optional[float]] = [None] * n_jobs

    # Convert job_records to separate lists
    for record in job_records:
        if not isinstance(record, dict):
            raise ValueError("Each job_record must be a dictionary.")

        job_id = record.get("job_id", None)
        processing_time_raw = record.get("processing_time", None)
        weight_raw = record.get("weight", None)
        due_date_raw = record.get("due_date", None)

        if job_id is None:
            raise ValueError("job_record missing 'job_id'.")
        if processing_time_raw is None:
            raise ValueError("job_record missing 'processing_time'.")
        if weight_raw is None:
            raise ValueError("job_record missing 'weight'.")
        if due_date_raw is None:
            raise ValueError("job_record missing 'due_date'.")

        if job_id in job_id_to_idx:
            job_idx = job_id_to_idx[job_id]
        else:
            if job_ids is not None:
                raise ValueError(f"job_id {job_id!r} not declared in 'jobs'.")
            candidate_idx: Optional[int] = None
            idx_candidate, idx_err = _to_int_strict(job_id, field="job_id")
            if idx_err is None:
                if idx_candidate < 0 or idx_candidate >= n_jobs:
                    raise ValueError(f"job_idx {idx_candidate} out of range [0, {n_jobs}).")
                if idx_candidate in assigned_idxs:
                    raise ValueError(f"Duplicate job_idx derived from job_id {job_id!r}.")
                candidate_idx = idx_candidate
                assigned_idxs.add(candidate_idx)
            if candidate_idx is None:
                candidate_idx = _allocate_next_idx()
            job_idx = candidate_idx
            job_id_to_idx[job_id] = job_idx

        if job_idx < 0 or job_idx >= n_jobs:
            raise ValueError(f"job_idx {job_idx} out of range [0, {n_jobs}).")

        if processing_times[job_idx] is not None:
            raise ValueError(f"Duplicate job_record for job {job_idx}.")

        processing_time, err = _to_float_finite(processing_time_raw, field="processing_time")
        if err:
            raise ValueError(f"Invalid processing_time: {err}")
        if processing_time < -EPS:
            raise ValueError(f"Invalid processing_time: negative value ({processing_time}).")

        weight, err = _to_float_finite(weight_raw, field="weight")
        if err:
            raise ValueError(f"Invalid weight: {err}")
        if weight < -EPS:
            raise ValueError(f"Invalid weight: negative value ({weight}).")

        due_date, err = _to_float_finite(due_date_raw, field="due_date")
        if err:
            raise ValueError(f"Invalid due_date: {err}")

        processing_times[job_idx] = float(processing_time)
        weights[job_idx] = float(weight)
        due_dates[job_idx] = float(due_date)
        job_ids_by_idx[job_idx] = job_id

    # Verify all jobs are present and convert to List[float]
    for job_idx in range(n_jobs):
        if processing_times[job_idx] is None:
            raise ValueError(f"Missing job_record for job {job_idx}.")
        if job_ids_by_idx[job_idx] is None:
            raise ValueError(f"Missing job_id for job {job_idx}.")

    # allow referencing jobs by their zero-based index as well
    for idx in range(n_jobs):
        job_id_to_idx.setdefault(idx, idx)

    return {
        "n_jobs": n_jobs,
        "processing_times": [float(p) for p in processing_times],
        "weights": [float(w) for w in weights],
        "due_dates": [float(d) for d in due_dates],
        "job_ids": [job_ids_by_idx[idx] for idx in range(n_jobs)],  # type: ignore[arg-type]
        "job_id_to_idx": job_id_to_idx,
    }


def _build_schedule_from_order(
    order: List[Any],
    instance: Dict[str, Any],
) -> Tuple[Optional[List[float]], Optional[List[float]], Optional[List[float]], Optional[str]]:
    """Interpret a permutation-order solution as start/end/tardiness per job."""
    if not isinstance(order, list):
        return None, None, None, "Solution order must be a list."

    n_jobs = instance.get("n_jobs")
    if not isinstance(n_jobs, int):
        return None, None, None, "Instance missing 'n_jobs'."

    p = instance.get("processing_times")
    d = instance.get("due_dates")
    if not isinstance(p, list) or not isinstance(d, list):
        return None, None, None, "Instance missing 'processing_times' or 'due_dates'."
    if len(p) != n_jobs or len(d) != n_jobs:
        return None, None, None, "Instance list lengths do not match 'n_jobs'."

    job_id_to_idx = instance.get("job_id_to_idx")
    if not isinstance(job_id_to_idx, dict):
        return None, None, None, "Instance missing job identifier mapping."

    if len(order) != n_jobs:
        return None, None, None, (
            f"Solution order has len={len(order)}, expected {n_jobs}."
        )

    seen: Set[int] = set()
    order_indices: List[int] = []
    for entry in order:
        job_idx = job_id_to_idx.get(entry)
        if job_idx is None:
            return None, None, None, f"Unknown job {entry!r} in solution order."
        if not isinstance(job_idx, int):
            return None, None, None, f"Mapped job index for {entry!r} is invalid."
        if job_idx in seen:
            return None, None, None, f"Job {entry!r} appears multiple times in order."
        seen.add(job_idx)
        order_indices.append(job_idx)

    current = 0.0
    starts = [0.0] * n_jobs
    ends = [0.0] * n_jobs
    for job_idx in order_indices:
        pj = float(p[job_idx])
        starts[job_idx] = current
        current += pj
        ends[job_idx] = current

    tardiness = [max(0.0, ends[idx] - float(d[idx])) for idx in range(n_jobs)]
    return starts, ends, tardiness, None



def smtwt_check_feasibility(
    solution: Any,
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check SMTWT feasibility for a single machine + unified violation pattern.

    Patterns (aligned with taxonomy: Global_7):
      - FormatError: invalid structure/types/lengths, instance-field errors
      - Global7: temporal consistency violations (timing, no-overlap, tardiness definition)
    """
    n_jobs: int = instance["n_jobs"]
    p: List[Any] = instance["processing_times"]
    d: List[Any] = instance["due_dates"]

    if isinstance(solution, dict):
        starts = solution.get("start_times", None)
        ends = solution.get("end_times", None)
        tard = solution.get("tardiness", None)

        if not isinstance(starts, list):
            return False, PATTERN_FORMAT, "Solution 'start_times' must be a list."
        if not isinstance(ends, list):
            return False, PATTERN_FORMAT, "Solution 'end_times' must be a list."

        if len(starts) != n_jobs or len(ends) != n_jobs:
            return False, PATTERN_FORMAT, (
                f"Solution has len(start_times)={len(starts)} and len(end_times)={len(ends)}, "
                f"expected n_jobs={n_jobs}."
            )

        if tard is not None:
            if not isinstance(tard, list):
                return False, PATTERN_FORMAT, "Solution 'tardiness' must be a list if provided."
            if len(tard) != n_jobs:
                return False, PATTERN_FORMAT, f"Solution has len(tardiness)={len(tard)}, expected n_jobs={n_jobs}."

    elif isinstance(solution, list):
        starts, ends, tard, err_msg = _build_schedule_from_order(solution, instance)
        if err_msg is not None:
            return False, PATTERN_FORMAT, err_msg

    else:
        return False, PATTERN_FORMAT, "Solution must be a dictionary or permutation list."

    # Validate all jobs + collect intervals
    intervals: List[Tuple[float, float, int]] = []
    for j in range(n_jobs):
        s, err = _to_float_finite(starts[j], field=f"start_times[{j}]")
        if err:
            return False, PATTERN_FORMAT, err
        e, err = _to_float_finite(ends[j], field=f"end_times[{j}]")
        if err:
            return False, PATTERN_FORMAT, err

        # ---- Global7: time consistency ----
        if s < -EPS:
            return False, PATTERN_GLOBAL7, f"Job {j}: negative start time {s}."
        if e < s - EPS:
            return False, PATTERN_GLOBAL7, f"Job {j}: end time {e} is before start time {s}."

        pj, err = _to_float_finite(p[j], field=f"processing_times[{j}]")
        if err:
            return False, PATTERN_FORMAT, f"Instance error: {err}"
        if pj < -EPS:
            return False, PATTERN_FORMAT, f"Instance error: job {j}: negative processing_time {pj}."

        if abs((e - s) - pj) > EPS:
            return False, PATTERN_GLOBAL7, (
                f"Job {j}: inconsistent timing: end-start={e - s:.6f} but processing_time={pj:.6f}."
            )

        dj, err = _to_float_finite(d[j], field=f"due_dates[{j}]")
        if err:
            return False, PATTERN_FORMAT, f"Instance error: {err}"

        if tard is not None:
            Tj, err = _to_float_finite(tard[j], field=f"tardiness[{j}]")
            if err:
                return False, PATTERN_FORMAT, err
            if Tj < -EPS:
                return False, PATTERN_GLOBAL7, f"Job {j}: negative tardiness {Tj}."
            expected_T = max(0.0, e - dj)
            if abs(Tj - expected_T) > EPS:
                return False, PATTERN_GLOBAL7, (
                    f"Job {j}: tardiness mismatch (expected {expected_T:.6f}, got {Tj:.6f})."
                )

        intervals.append((float(s), float(e), j))

    # ---- Global7: no-overlap on the single machine ----
    intervals.sort(key=lambda x: x[0])
    for i in range(len(intervals) - 1):
        s1, e1, j1 = intervals[i]
        s2, e2, j2 = intervals[i + 1]
        if s2 < e1 - EPS:
            return False, PATTERN_GLOBAL7, (
                f"Single-machine conflict: job {j1} (ends at {e1:.6f}) "
                f"overlaps with job {j2} (starts at {s2:.6f})."
            )

    return True, PATTERN_OK, "Feasible SMTWT solution."


def smtwt_objective_model(
    solution: Any,
    instance: Dict[str, Any],
) -> float:
    """
    SMTWT objective: minimize TOTAL WEIGHTED TARDINESS:
      sum_j w_j * max(0, C_j - d_j)
    where C_j is completion time (end_time).

    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    ends: Optional[List[Any]] = None
    if isinstance(solution, dict):
        ends = solution.get("end_times", None)
        if not isinstance(ends, list) or len(ends) == 0:
            return float("inf")
    elif isinstance(solution, list):
        _, ends, _, err_msg = _build_schedule_from_order(solution, instance)
        if err_msg is not None or ends is None:
            return float("inf")
    else:
        return float("inf")

    n_jobs = instance.get("n_jobs", None)
    p = instance.get("processing_times", None)
    w = instance.get("weights", None)
    d = instance.get("due_dates", None)

    if not isinstance(n_jobs, int) or not isinstance(w, list) or not isinstance(d, list):
        return float("inf")
    if len(ends) != n_jobs or len(w) != n_jobs or len(d) != n_jobs:
        return float("inf")

    obj = 0.0
    for j in range(n_jobs):
        end, err = _to_float_finite(ends[j], field="end_times")
        if err:
            return float("inf")
        wj, err = _to_float_finite(w[j], field="weights")
        if err:
            return float("inf")
        dj, err = _to_float_finite(d[j], field="due_dates")
        if err:
            return float("inf")
        T = end - dj
        if T > 0.0:
            obj += wj * T

    return float(obj)
