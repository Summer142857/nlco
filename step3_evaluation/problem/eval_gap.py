from typing import Any, Dict, List, Tuple, Optional, Union
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


EPS = 1e-9


# =============================
# Unified feasibility patterns
# GAP taxonomy: Global_{4,2}
#   - Global4: assignment structure (each task assigned exactly once)
#   - Global2: capacity / budget
# =============================
PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL2 = "Global2"
PATTERN_GLOBAL4 = "Global4"


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


def gap_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Generalized Assignment Problem (GAP) instance from `instance_variant`.

    Supports:
      (A) nested format: {"instance": {...}}
      (B) flat format (your dataframe column): {...}

    Expected fields (flat or under "instance"):
      {
        "problem_type": "GAP" (optional),
        "num_agents": m,
        "num_tasks": n,
        "agents": ["A1", ...],             # agent ids (strings)
        "tasks": [1,2,...],                # task ids (ints or strings)
        "capacity_pairs": [{"agent_id": "A1", "capacity": ...}, ...],
        "resource_pairs": [{"agent_id":"A1","task_id":1,"consumption":...}, ...],
        "cost_pairs": [{"agent_id":"A1","task_id":1,"cost":...}, ...]
      }

    Return (normalized):
      {
        "num_agents": int,
        "num_tasks": int,
        "agents": [str] length m,
        "tasks": [Any] length n,
        "capacities": [float] length m,
        "consumption": [[float]] m x n,
        "costs": [[float]] m x n
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for GAP.")

    instance = inst_var.get("instance", None)
    if instance is None:
        instance = inst_var
    if not isinstance(instance, dict) or not instance:
        raise ValueError("GAP instance_variant has no usable instance dictionary.")

    m_raw = instance.get("num_agents", None)
    n_raw = instance.get("num_tasks", None)
    agents = instance.get("agents", None)
    tasks = instance.get("tasks", None)
    cap_pairs = instance.get("capacity_pairs", None)
    res_pairs = instance.get("resource_pairs", None)
    cost_pairs = instance.get("cost_pairs", None)

    if m_raw is None:
        raise ValueError("GAP instance missing 'num_agents'.")
    if n_raw is None:
        raise ValueError("GAP instance missing 'num_tasks'.")
    if agents is None:
        raise ValueError("GAP instance missing 'agents'.")
    if tasks is None:
        raise ValueError("GAP instance missing 'tasks'.")
    if cap_pairs is None:
        raise ValueError("GAP instance missing 'capacity_pairs'.")
    if res_pairs is None:
        raise ValueError("GAP instance missing 'resource_pairs'.")
    if cost_pairs is None:
        raise ValueError("GAP instance missing 'cost_pairs'.")

    m, err = _to_int_strict(m_raw, field="num_agents")
    if err:
        raise ValueError(f"Invalid num_agents: {err}")
    n, err = _to_int_strict(n_raw, field="num_tasks")
    if err:
        raise ValueError(f"Invalid num_tasks: {err}")
    if m <= 0:
        raise ValueError(f"num_agents must be positive, got {m}.")
    if n <= 0:
        raise ValueError(f"num_tasks must be positive, got {n}.")

    if not isinstance(agents, list) or len(agents) != m:
        raise ValueError(f"'agents' must be a list of length num_agents={m}.")
    if not isinstance(tasks, list) or len(tasks) != n:
        raise ValueError(f"'tasks' must be a list of length num_tasks={n}.")

    agent_ids: List[str] = []
    seen_agents = set()
    for i, a in enumerate(agents):
        if not isinstance(a, str) or a.strip() == "":
            raise ValueError(f"agents[{i}] must be a non-empty string.")
        a = a.strip()
        if a in seen_agents:
            raise ValueError(f"Duplicate agent id {a!r}.")
        seen_agents.add(a)
        agent_ids.append(a)

    task_ids: List[Any] = []
    seen_tasks = set()
    for j, t in enumerate(tasks):
        try:
            hash(t)
        except Exception:
            raise ValueError(f"tasks[{j}] is unhashable: {t!r}.")
        if t in seen_tasks:
            raise ValueError(f"Duplicate task id {t!r}.")
        seen_tasks.add(t)
        task_ids.append(t)

    agent_to_idx = {a: i for i, a in enumerate(agent_ids)}
    task_to_idx = {t: j for j, t in enumerate(task_ids)}

    if not isinstance(cap_pairs, list) or len(cap_pairs) == 0:
        raise ValueError("'capacity_pairs' must be a non-empty list.")
    if not isinstance(res_pairs, list) or len(res_pairs) == 0:
        raise ValueError("'resource_pairs' must be a non-empty list.")
    if not isinstance(cost_pairs, list) or len(cost_pairs) == 0:
        raise ValueError("'cost_pairs' must be a non-empty list.")

    capacities: List[Optional[float]] = [None] * m
    for k, rec in enumerate(cap_pairs):
        if not isinstance(rec, dict):
            raise ValueError(f"capacity_pairs[{k}] must be a dict.")
        aid = rec.get("agent_id", None)
        if not isinstance(aid, str) or aid.strip() == "":
            raise ValueError(f"capacity_pairs[{k}].agent_id must be a non-empty string.")
        aid = aid.strip()
        if aid not in agent_to_idx:
            raise ValueError(f"capacity_pairs[{k}].agent_id={aid!r} is not in agents list.")
        cap, err = _to_float_finite(rec.get("capacity", None), field=f"capacity_pairs[{k}].capacity")
        if err:
            raise ValueError(err)
        if cap < -EPS:
            raise ValueError(f"capacity_pairs[{k}].capacity is negative: {cap}.")
        i = agent_to_idx[aid]
        if capacities[i] is not None:
            raise ValueError(f"Duplicate capacity entry for agent {aid!r}.")
        capacities[i] = float(cap)

    missing_caps = [agent_ids[i] for i, v in enumerate(capacities) if v is None]
    if missing_caps:
        raise ValueError(f"Missing capacities for agents {missing_caps}.")

    consumption: List[List[Optional[float]]] = [[None] * n for _ in range(m)]
    for k, rec in enumerate(res_pairs):
        if not isinstance(rec, dict):
            raise ValueError(f"resource_pairs[{k}] must be a dict.")
        aid = rec.get("agent_id", None)
        tid = rec.get("task_id", None)
        if not isinstance(aid, str) or aid.strip() == "":
            raise ValueError(f"resource_pairs[{k}].agent_id must be a non-empty string.")
        aid = aid.strip()
        if aid not in agent_to_idx:
            raise ValueError(f"resource_pairs[{k}].agent_id={aid!r} is not in agents list.")
        if tid not in task_to_idx:
            raise ValueError(f"resource_pairs[{k}].task_id={tid!r} is not in tasks list.")
        c, err = _to_float_finite(rec.get("consumption", None), field=f"resource_pairs[{k}].consumption")
        if err:
            raise ValueError(err)
        if c < -EPS:
            raise ValueError(f"resource_pairs[{k}].consumption is negative: {c}.")
        i = agent_to_idx[aid]
        j = task_to_idx[tid]
        if consumption[i][j] is not None:
            raise ValueError(f"Duplicate consumption entry for agent={aid!r}, task={tid!r}.")
        consumption[i][j] = float(c)

    missing_res: List[Tuple[str, Any]] = []
    for i in range(m):
        for j in range(n):
            if consumption[i][j] is None:
                missing_res.append((agent_ids[i], task_ids[j]))
    if missing_res:
        sample = missing_res[:10]
        raise ValueError(f"Missing consumption entries for {len(missing_res)} (agent,task) pairs. Sample: {sample}")

    costs: List[List[Optional[float]]] = [[None] * n for _ in range(m)]
    for k, rec in enumerate(cost_pairs):
        if not isinstance(rec, dict):
            raise ValueError(f"cost_pairs[{k}] must be a dict.")
        aid = rec.get("agent_id", None)
        tid = rec.get("task_id", None)
        if not isinstance(aid, str) or aid.strip() == "":
            raise ValueError(f"cost_pairs[{k}].agent_id must be a non-empty string.")
        aid = aid.strip()
        if aid not in agent_to_idx:
            raise ValueError(f"cost_pairs[{k}].agent_id={aid!r} is not in agents list.")
        if tid not in task_to_idx:
            raise ValueError(f"cost_pairs[{k}].task_id={tid!r} is not in tasks list.")
        c, err = _to_float_finite(rec.get("cost", None), field=f"cost_pairs[{k}].cost")
        if err:
            raise ValueError(err)
        i = agent_to_idx[aid]
        j = task_to_idx[tid]
        if costs[i][j] is not None:
            raise ValueError(f"Duplicate cost entry for agent={aid!r}, task={tid!r}.")
        costs[i][j] = float(c)

    missing_cost: List[Tuple[str, Any]] = []
    for i in range(m):
        for j in range(n):
            if costs[i][j] is None:
                missing_cost.append((agent_ids[i], task_ids[j]))
    if missing_cost:
        sample = missing_cost[:10]
        raise ValueError(f"Missing cost entries for {len(missing_cost)} (agent,task) pairs. Sample: {sample}")

    cap_out = [float(x) for x in capacities]  # type: ignore[arg-type]
    cons_out = [[float(x) for x in rowc] for rowc in consumption]  # type: ignore[arg-type]
    cost_out = [[float(x) for x in rowc] for rowc in costs]  # type: ignore[arg-type]

    return {
        "num_agents": m,
        "num_tasks": n,
        "agents": agent_ids,
        "tasks": task_ids,
        "capacities": cap_out,
        "consumption": cons_out,
        "costs": cost_out,
    }


def _is_intlike(x: Any) -> bool:
    """Return True if x can be parsed as an integer-like value."""
    v, err = _to_int_strict(x, field="_is_intlike")
    return err is None and v is not None


def _extract_assignments(solution: Any) -> Tuple[Optional[Any], str, str]:
    """
    Accept solution in several common formats:

    1) List aligned with instance['tasks'] order:
         ["A1","A2","A1",...]  OR  [0,1,0,...]  (indices can be 0-based or 1-based)
       or list of dicts:
         [{"task": 1, "agent": "A2"}, ...]

    2) Dict:
         {"assignments": [...]}  (same as list above)
         {"assignments": {task_id: agent_id, ...}}
         {"task_to_agent": {...}}
         {"agent_tasks": {"A1":[1,3], "A2":[2,4,5]}}

    Returns a raw object that will be normalized later.
    """
    if isinstance(solution, list):
        return solution, PATTERN_OK, ""

    if isinstance(solution, dict):
        for key in ("assignments", "task_to_agent", "task_assignments"):
            if key in solution:
                return solution[key], PATTERN_OK, ""
        if "agent_tasks" in solution:
            return ("agent_tasks", solution["agent_tasks"]), PATTERN_OK, ""
        return (
            None,
            PATTERN_FORMAT,
            "Solution dict must contain 'assignments'/'task_to_agent'/'task_assignments' or 'agent_tasks'.",
        )
    return None, PATTERN_FORMAT, "Solution must be a list or a dict."


def _normalize_assignment(raw: Any, instance: Dict[str, Any]) -> Tuple[Optional[List[int]], str, str]:
    """
    Normalize assignment to a list agent_idx_of_task[t] of length n_tasks,
    aligned with instance['tasks'] order.
    """
    m: int = instance["num_agents"]
    n: int = instance["num_tasks"]
    agents: List[str] = instance["agents"]
    tasks: List[Any] = instance["tasks"]
    agent_to_idx = {a: i for i, a in enumerate(agents)}
    task_to_pos = {t: j for j, t in enumerate(tasks)}

    def coerce_task_id(x: Any) -> Tuple[Optional[Any], Optional[str]]:
        """Try to map x to an existing task id in instance['tasks']."""
        if x in task_to_pos:
            return x, None
        # Allow int-like strings/floats when tasks are numeric ids
        if isinstance(x, (str, float)):
            xi, err = _to_int_strict(x, field="task")
            if err is None and xi in task_to_pos:
                return xi, None
        return None, f"Unknown task id {x!r}."

    def coerce_agent(x: Any, *, field: str) -> Tuple[Optional[int], Optional[str], str]:
        """Parse agent as either an id string or an integer index (0/1-based)."""
        if isinstance(x, str):
            s = x.strip()
            if s in agent_to_idx:
                return agent_to_idx[s], None, "id"
            # If it's numeric-like, allow treating it as an index.
            ai_raw, err = _to_int_strict(s, field=field)
            if err is None and ai_raw is not None:
                ai = ai_raw - 1 if (1 <= ai_raw <= m and ai_raw != 0) else ai_raw
                if ai < 0 or ai >= m:
                    return None, f"Invalid agent index {ai_raw} ({field}).", "index"
                return ai, None, "index"
            return None, f"Unknown agent id {x!r} ({field}).", "id"

        ai_raw, err = _to_int_strict(x, field=field)
        if err:
            return None, err, "index"
        assert ai_raw is not None
        ai = ai_raw - 1 if (1 <= ai_raw <= m and ai_raw != 0) else ai_raw
        if ai < 0 or ai >= m:
            return None, f"Invalid agent index {ai_raw} ({field}).", "index"
        return ai, None, "index"

    # Case: ("agent_tasks", mapping)
    if isinstance(raw, tuple) and len(raw) == 2 and raw[0] == "agent_tasks":
        mapping = raw[1]
        if not isinstance(mapping, dict):
            return None, PATTERN_FORMAT, "'agent_tasks' must be a dict {agent_id: [task_ids]}."
        out: List[Optional[int]] = [None] * n
        for aid, tlist in mapping.items():
            if not isinstance(aid, str) or aid.strip() == "":
                return None, PATTERN_FORMAT, "agent_tasks keys must be non-empty strings."
            aid = aid.strip()
            if aid not in agent_to_idx:
                return None, PATTERN_GLOBAL4, f"Unknown agent id {aid!r} in agent_tasks."
            if not isinstance(tlist, list):
                return None, PATTERN_FORMAT, f"agent_tasks[{aid!r}] must be a list of task ids."
            ai = agent_to_idx[aid]
            for x in tlist:
                tid, terr = coerce_task_id(x)
                if terr:
                    return None, PATTERN_GLOBAL4, f"agent_tasks[{aid!r}]: {terr}"
                assert tid is not None
                j = task_to_pos[tid]
                if out[j] is not None:
                    return None, PATTERN_GLOBAL4, f"Task {tid!r} assigned multiple times."
                out[j] = ai
        missing = [tasks[j] for j, v in enumerate(out) if v is None]
        if missing:
            return None, PATTERN_GLOBAL4, f"Missing assignments for tasks {missing}."
        return [int(v) for v in out], PATTERN_OK, ""  # type: ignore[arg-type]

    # Case: dict mapping task->agent
    if isinstance(raw, dict):
        out: List[Optional[int]] = [None] * n
        for k, v in raw.items():
            task_id, terr = coerce_task_id(k)
            if terr:
                return None, PATTERN_GLOBAL4, f"Assignment mapping: {terr}"
            assert task_id is not None
            j = task_to_pos[task_id]

            ai, aerr, _kind = coerce_agent(v, field=f"agent for task {task_id!r}")
            if aerr:
                # type errors are format; unknown ids/indices are assignment
                pat = PATTERN_FORMAT if "must" in aerr.lower() else PATTERN_GLOBAL4
                return None, pat, aerr
            assert ai is not None

            if out[j] is not None:
                return None, PATTERN_GLOBAL4, f"Task {task_id!r} assigned multiple times."
            out[j] = ai

        missing = [tasks[j] for j, v in enumerate(out) if v is None]
        if missing:
            return None, PATTERN_GLOBAL4, f"Missing assignments for tasks {missing}."
        return [int(v) for v in out], PATTERN_OK, ""  # type: ignore[arg-type]

    # Case: list of dicts [{"task":..,"agent":..}, ...]
    if isinstance(raw, list) and len(raw) > 0 and all(isinstance(x, dict) for x in raw):
        out: List[Optional[int]] = [None] * n
        for idx, rec in enumerate(raw):
            task_id = rec.get("task", None)
            agent_val = rec.get("agent", None)
            if task_id is None:
                return None, PATTERN_FORMAT, f"Entry {idx}: missing 'task'."

            tid, terr = coerce_task_id(task_id)
            if terr:
                return None, PATTERN_GLOBAL4, f"Entry {idx}: {terr}"
            assert tid is not None
            j = task_to_pos[tid]

            if agent_val is None:
                return None, PATTERN_FORMAT, f"Entry {idx}: missing 'agent'."

            ai, aerr, _kind = coerce_agent(agent_val, field=f"entry {idx} agent")
            if aerr:
                pat = PATTERN_FORMAT if "must" in aerr.lower() else PATTERN_GLOBAL4
                return None, pat, f"Entry {idx}: {aerr}"
            assert ai is not None

            if out[j] is not None:
                return None, PATTERN_GLOBAL4, f"Task {tid!r} assigned multiple times."
            out[j] = ai

        missing = [tasks[j] for j, v in enumerate(out) if v is None]
        if missing:
            return None, PATTERN_GLOBAL4, f"Missing assignments for tasks {missing}."
        return [int(v) for v in out], PATTERN_OK, ""  # type: ignore[arg-type]

    # Case: list aligned with tasks
    if isinstance(raw, list):
        if len(raw) != n:
            return None, PATTERN_GLOBAL4, f"Assignments list has length {len(raw)}, expected {n}."
        out: List[int] = []
        # Detect ids vs indices robustly.
        # Prefer id-mode only when every entry is a known agent id.
        all_known_ids = all(isinstance(x, str) and x.strip() in agent_to_idx for x in raw)
        all_intlike = all(_is_intlike(x) for x in raw)

        if all_known_ids:
            for j, x in enumerate(raw):
                aid = str(x).strip()
                out.append(agent_to_idx[aid])
            return out, PATTERN_OK, ""

        if all_intlike:
            ints: List[int] = []
            for j, x in enumerate(raw):
                v, err = _to_int_strict(x, field=f"assignments[{j}]")
                if err:
                    return None, PATTERN_FORMAT, err
                assert v is not None
                ints.append(v)
            is_one_based = (all(1 <= v <= m for v in ints) and all(v != 0 for v in ints))
            for v in ints:
                ai = v - 1 if is_one_based else v
                if ai < 0 or ai >= m:
                    base = "1-based" if is_one_based else "0-based"
                    return None, PATTERN_GLOBAL4, f"Agent index {v} ({base}) out of range."
                out.append(ai)
            return out, PATTERN_OK, ""

        # Mixed/ambiguous list: try per-entry coercion.
        for j, x in enumerate(raw):
            ai, aerr, _kind = coerce_agent(x, field=f"assignments[{j}]")
            if aerr:
                pat = PATTERN_FORMAT if "must" in aerr.lower() else PATTERN_GLOBAL4
                return None, pat, aerr
            assert ai is not None
            out.append(ai)
        return out, PATTERN_OK, ""

    return None, PATTERN_FORMAT, "Unsupported solution format."


def gap_check_feasibility(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check GAP feasibility.

    Constraints:
      1) each task is assigned to exactly one agent.
      2) for each agent i: sum(consumption[i][task]) <= capacity[i].

    Returns:
      (feasible, pattern, message)
    where `pattern` is one of: OK, FormatError, Global4 (assignment), Global2 (capacity).
    """
    raw, pat, msg = _extract_assignments(solution)
    if pat != PATTERN_OK or raw is None:
        return False, pat, msg

    assign, pat, msg = _normalize_assignment(raw, instance)
    if pat != PATTERN_OK or assign is None:
        return False, pat, msg

    m: int = instance["num_agents"]
    n: int = instance["num_tasks"]
    caps: List[float] = instance["capacities"]
    cons: List[List[float]] = instance["consumption"]

    loads = [0.0] * m
    for t in range(n):
        a = assign[t]
        loads[a] += float(cons[a][t])

    for i in range(m):
        if loads[i] > float(caps[i]) + EPS:
            return False, PATTERN_GLOBAL2, (
                f"Capacity violated for agent {instance['agents'][i]!r}: "
                f"load {loads[i]:.6f} > cap {caps[i]:.6f}."
            )

    return True, PATTERN_OK, "Feasible GAP solution."


def gap_objective_model(
    solution: Union[List[Any], Dict[str, Any]],
    instance: Dict[str, Any],
) -> float:
    """
    GAP objective (cost-minimization): minimize total assignment cost.
    Returns total cost (NOT negated).

    Assumes solution is feasible; if not well-formed or infeasible, return +inf (never raise).
    """
    try:
        raw, pat, _msg = _extract_assignments(solution)
        if pat != PATTERN_OK or raw is None:
            return float("inf")

        assign, pat, _msg = _normalize_assignment(raw, instance)
        if pat != PATTERN_OK or assign is None:
            return float("inf")

        m: int = int(instance.get("num_agents", -1))
        n: int = int(instance.get("num_tasks", -1))
        if m <= 0 or n <= 0:
            return float("inf")

        caps = instance.get("capacities", None)
        cons = instance.get("consumption", None)
        costs = instance.get("costs", None)
        if not isinstance(caps, list) or not isinstance(cons, list) or not isinstance(costs, list):
            return float("inf")
        if len(caps) != m or len(cons) != m or len(costs) != m:
            return float("inf")
        for i in range(m):
            if not isinstance(cons[i], list) or not isinstance(costs[i], list):
                return float("inf")
            if len(cons[i]) != n or len(costs[i]) != n:
                return float("inf")

        loads = [0.0] * m
        total_cost = 0.0
        for t in range(n):
            a = assign[t]
            loads[a] += float(cons[a][t])
            total_cost += float(costs[a][t])

        for i in range(m):
            if loads[i] > float(caps[i]) + EPS:
                return float("inf")

        return float(total_cost)
    except Exception:
        return float("inf")
