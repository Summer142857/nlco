from typing import Any, Dict, List, Tuple, Optional
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field


PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL4 = "Global4"  # assignment/linking (customer -> open facility)
PATTERN_GLOBAL2 = "Global2"  # budget/capacity (facility capacities)

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


def _resolve_facility_index(
    x: Any,
    *,
    n_fac: int,
    facility_id_to_index: Dict[str, int],
    field: str,
) -> Tuple[Optional[int], Optional[str]]:
    """Interpret x as a facility index or identifier, validating against the instance."""
    # 1. Try as integer (index or numeric ID)
    idx, err_int = _to_int_strict(x, field=field)
    if err_int is None:
        # a) Check if it matches a facility ID (e.g. "1" -> index 0)
        if str(idx) in facility_id_to_index:
            return facility_id_to_index[str(idx)], None
            
        # b) Check if F{idx} matches a facility ID (common benchmark format)
        f_prefixed = f"F{idx}"
        if f_prefixed in facility_id_to_index:
            return facility_id_to_index[f_prefixed], None
        
        # c) Check if it is a valid 0-based index
        if 0 <= idx < n_fac:
            return idx, None
            
        return None, f"{field}={idx} is out of range 0-{n_fac-1} and not a known facility ID."

    # 2. Try as string ID
    if isinstance(x, str):
        fid = x.strip()
        if fid == "":
            return None, f"{field} is empty."
        
        # Direct string match
        mapped = facility_id_to_index.get(fid)
        if mapped is not None:
            return mapped, None
            
        # Try converting string to int to match int-like string IDs
        try:
            f = float(fid)
            if f.is_integer():
                i = int(f)
                # Check if the integer version (as string) exists
                if str(i) in facility_id_to_index:
                    return facility_id_to_index[str(i)], None
        except ValueError:
            pass
            
        return None, f"{field}: unknown facility id {fid!r}."

    return None, err_int


def cflp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Capacitated Facility Location Problem (CFLP) instance from `instance_variant`.

    Expected instance format:
      {
        "opening_costs": [f_i],
        "capacities": [cap_i],
        "connection_costs": [[c_{i,0}, c_{i,1}, ...], ...],
        "demands": [dem_j],
        "objective": float (optional)
      }

    Return:
      {
        "opening_costs": [...],
        "capacities": [...],
        "connection_costs": [...],
        "demands": [...],
        "n_facilities": int,
        "n_customers": int
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for CFLP.")

    num_fac_raw = inst_var.get("num_facilities", None)
    num_cust_raw = inst_var.get("num_customers", None)
    facilities = inst_var.get("facilities", None)
    customers = inst_var.get("customers", None)
    conn_list = inst_var.get("connection_costs", None)

    n_fac, err = _to_int_strict(num_fac_raw, field="num_facilities")
    if err:
        raise ValueError(f"Invalid num_facilities: {err}")
    n_cust, err = _to_int_strict(num_cust_raw, field="num_customers")
    if err:
        raise ValueError(f"Invalid num_customers: {err}")

    if not isinstance(facilities, list) or len(facilities) != n_fac:
        raise ValueError(f"'facilities' must be a list of length num_facilities={n_fac}.")
    if not isinstance(customers, list) or len(customers) != n_cust:
        raise ValueError(f"'customers' must be a list of length num_customers={n_cust}.")
    if not isinstance(conn_list, list):
        raise ValueError("'connection_costs' must be a list (of dicts).")

    # Facility id -> index
    facility_ids: List[str] = []
    fac_index: Dict[str, int] = {}
    opening_costs: List[float] = [0.0] * n_fac
    capacities: List[float] = [0.0] * n_fac

    for i, f in enumerate(facilities):
        if not isinstance(f, dict):
            raise ValueError(f"facilities[{i}] must be a dict.")
        fid = f.get("id", None)
        
        # Allow int IDs by converting to string
        if isinstance(fid, int):
            fid = str(fid)
        elif not isinstance(fid, str) or fid.strip() == "":
            raise ValueError(f"facilities[{i}].id must be a non-empty string or int.")
            
        fid = fid.strip()
        if fid in fac_index:
            raise ValueError(f"Duplicate facility id: {fid!r}.")
        fac_index[fid] = i
        facility_ids.append(fid)

        oc, err = _to_float_finite(f.get("opening_cost", None), field=f"facilities[{i}].opening_cost")
        if err:
            raise ValueError(err)
        if oc < -EPS:
            raise ValueError(f"facilities[{i}].opening_cost is negative: {oc}.")
        opening_costs[i] = float(oc)

        cap, err = _to_float_finite(f.get("capacity", None), field=f"facilities[{i}].capacity")
        if err:
            raise ValueError(err)
        if cap < -EPS:
            raise ValueError(f"facilities[{i}].capacity is negative: {cap}.")
        capacities[i] = float(cap)

    # Customer id -> index
    customer_ids: List[Any] = []
    cust_index: Dict[Any, int] = {}
    demands: List[float] = [0.0] * n_cust

    for j, c in enumerate(customers):
        if not isinstance(c, dict):
            raise ValueError(f"customers[{j}] must be a dict.")
        cid = c.get("id", None)
        if cid is None:
            raise ValueError(f"customers[{j}].id is missing.")
            
        # Allow int IDs by converting to string
        if isinstance(cid, int):
            cid = str(cid)
            
        if cid in cust_index:
            raise ValueError(f"Duplicate customer id: {cid!r}.")
        cust_index[cid] = j
        customer_ids.append(cid)

        dem, err = _to_float_finite(c.get("demand", None), field=f"customers[{j}].demand")
        if err:
            raise ValueError(err)
        if dem < -EPS:
            raise ValueError(f"customers[{j}].demand is negative: {dem}.")
        demands[j] = float(dem)

    # Build full connection matrix initialized to None (detect missing pairs)
    conn_matrix: List[List[Optional[float]]] = [[None for _ in range(n_cust)] for _ in range(n_fac)]

    for k, rec in enumerate(conn_list):
        if not isinstance(rec, dict):
            raise ValueError(f"connection_costs[{k}] must be a dict.")
        fid = rec.get("facility", None)
        cid = rec.get("customer", None)
        cost_raw = rec.get("cost", None)

        # Allow int IDs by converting to string
        if isinstance(fid, int):
            fid = str(fid)
        if isinstance(cid, int):
            cid = str(cid)

        if not isinstance(fid, str) or fid.strip() == "":
            raise ValueError(f"connection_costs[{k}].facility must be a non-empty string or int.")
        fid = fid.strip()
        if fid not in fac_index:
            raise ValueError(f"connection_costs[{k}]: unknown facility id {fid!r}.")
        if cid not in cust_index:
            raise ValueError(f"connection_costs[{k}]: unknown customer id {cid!r}.")

        cost, err = _to_float_finite(cost_raw, field=f"connection_costs[{k}].cost")
        if err:
            raise ValueError(err)
        if cost < -EPS:
            raise ValueError(f"connection_costs[{k}].cost is negative: {cost}.")

        i = fac_index[fid]
        j = cust_index[cid]
        if conn_matrix[i][j] is not None:
            raise ValueError(f"Duplicate connection cost for facility {fid!r} and customer {cid!r}.")
        conn_matrix[i][j] = float(cost)

    # Ensure complete matrix
    for i in range(n_fac):
        for j in range(n_cust):
            if conn_matrix[i][j] is None:
                raise ValueError(
                    f"Missing connection cost for facility {facility_ids[i]!r} and customer {customer_ids[j]!r}."
                )

    connection_costs: List[List[float]] = [[float(conn_matrix[i][j]) for j in range(n_cust)] for i in range(n_fac)]

    return {
        "opening_costs": opening_costs,
        "capacities": capacities,
        "connection_costs": connection_costs,
        "demands": demands,
        "n_facilities": n_fac,
        "n_customers": n_cust,
        "facility_ids": facility_ids,
        "customer_ids": customer_ids,
        "facility_id_to_index": dict(fac_index),
    }



def cflp_check_feasibility(
    solution: Dict[str, Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check CFLP feasibility + unified violation pattern.

    Expected solution format:
      {
        "open_facilities" or "selected": [ ... ],  # facility indices or IDs (open ones)
        "assignments": [ ... ]  # facility index or ID for each customer
      }

    Instance must provide:
      - n_facilities: int
      - n_customers: int
      - capacities: List[number] length n_facilities  (capacity of each facility)
      - demands:    List[number] length n_customers   (demand of each customer)

    Constraints:
      1) open_facilities is a list of unique facility indices within [0, n_facilities-1].
      2) assignments is a list of length n_customers.
      3) each assignment is an integer-like facility index within range.
      4) each customer must be assigned to an OPEN facility.                 -> Global4
      5) for each open facility f: sum(demand[j] for j assigned to f) <= cap[f]. -> Global2
      6) if n_customers > 0, then open_facilities must be non-empty (else cannot assign). -> Global4
    """
    n_fac: int = instance["n_facilities"]
    n_cust: int = instance["n_customers"]

    capacities_raw = instance.get("capacities", None)
    demands_raw = instance.get("demands", None)

    if not isinstance(capacities_raw, list) or len(capacities_raw) != n_fac:
        return False, PATTERN_FORMAT, f"Instance error: 'capacities' must be a list of length n_facilities={n_fac}."
    if not isinstance(demands_raw, list) or len(demands_raw) != n_cust:
        return False, PATTERN_FORMAT, f"Instance error: 'demands' must be a list of length n_customers={n_cust}."

    # Parse capacities/demands to floats (and validate non-negative)
    capacities: List[float] = []
    for i in range(n_fac):
        cap_i, err = _to_float_finite(capacities_raw[i], field=f"capacities[{i}]")
        if err:
            return False, PATTERN_FORMAT, f"Instance error: {err}"
        if cap_i < -EPS:
            return False, PATTERN_FORMAT, f"Instance error: capacities[{i}] is negative ({cap_i})."
        capacities.append(float(cap_i))

    demands: List[float] = []
    for j in range(n_cust):
        dem_j, err = _to_float_finite(demands_raw[j], field=f"demands[{j}]")
        if err:
            return False, PATTERN_FORMAT, f"Instance error: {err}"
        if dem_j < -EPS:
            return False, PATTERN_FORMAT, f"Instance error: demands[{j}] is negative ({dem_j})."
        demands.append(float(dem_j))

    # ---- Solution structure ----
    if not isinstance(solution, dict):
        return False, PATTERN_FORMAT, "Solution must be a dictionary."

    assignments = solution.get("assignments", None)
    open_fac_field = None
    if "open_facilities" in solution:
        open_fac = solution["open_facilities"]
        open_fac_field = "open_facilities"
    elif "selected" in solution:
        open_fac = solution["selected"]
        open_fac_field = "selected"
    else:
        return False, PATTERN_FORMAT, "Solution must include 'open_facilities' or 'selected'."

    if not isinstance(open_fac, list):
        return False, PATTERN_FORMAT, f"Solution '{open_fac_field}' must be a list."
    if not isinstance(assignments, list):
        return False, PATTERN_FORMAT, "Solution 'assignments' must be a list."

    if len(assignments) != n_cust:
        return False, PATTERN_FORMAT, f"Solution has {len(assignments)} assignments, expected n_customers={n_cust}."

    if n_cust > 0 and len(open_fac) == 0:
        return False, PATTERN_GLOBAL4, "Solution has customers but no open facilities."

    facility_id_to_index: Dict[str, int] = instance.get("facility_id_to_index", {})
    facility_ids: List[str] = instance.get("facility_ids", [])
    # ---- Parse open facilities ----
    open_set = set()
    for k, x in enumerate(open_fac):
        field_name = f"{open_fac_field}[{k}]"
        f, err = _resolve_facility_index(
            x,
            n_fac=n_fac,
            facility_id_to_index=facility_id_to_index,
            field=field_name,
        )
        if err:
            return False, PATTERN_FORMAT, err
        
        fid_label = facility_ids[f]
        if f in open_set:
            return False, PATTERN_FORMAT, f"Facility {fid_label!r} appears multiple times in {open_fac_field}."
        open_set.add(f)

    # ---- Global4: assignments must point to an open facility ----
    loads: List[float] = [0.0] * n_fac
    for j, x in enumerate(assignments):
        a, err = _resolve_facility_index(
            x,
            n_fac=n_fac,
            facility_id_to_index=facility_id_to_index,
            field=f"assignments[{j}]",
        )
        if err:
            return False, PATTERN_FORMAT, err

        if a not in open_set:
            fid_label = facility_ids[a]
            return False, PATTERN_GLOBAL4, (
                f"Customer {j} assigned to facility {fid_label} (index {a}), but it is not open."
            )
        loads[a] += demands[j]

    # ---- Global2: capacity constraints on open facilities ----
    for f in open_set:
        if loads[f] > capacities[f] + EPS:
            return False, PATTERN_GLOBAL2, (
                f"Facility {f} capacity exceeded: load={loads[f]:.6f} > capacity={capacities[f]:.6f}."
            )

    return True, PATTERN_OK, "Feasible CFLP solution."


def cflp_objective_model(
    solution: Dict[str, Any],
    instance: Dict[str, Any],
) -> float:
    """
    CFLP objective: minimize
        sum_{i in open} opening_costs[i] + sum_{j} connection_costs[a_j][j].

    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    if not isinstance(solution, dict):
        return float("inf")

    opening_costs = instance.get("opening_costs", None)
    connection_costs = instance.get("connection_costs", None)
    capacities = instance.get("capacities", None)
    demands = instance.get("demands", None)
    n_fac = instance.get("n_facilities", None)
    n_cust = instance.get("n_customers", None)

    if not isinstance(opening_costs, list) or not isinstance(connection_costs, list):
        return float("inf")
    if not isinstance(capacities, list) or not isinstance(demands, list):
        return float("inf")
    if not isinstance(n_fac, int) or not isinstance(n_cust, int):
        return float("inf")

    open_fac = solution.get("open_facilities", None)
    assignments = solution.get("assignments", None)
    facility_id_to_index: Dict[str, int] = instance.get("facility_id_to_index", {})
    open_fac_field = None
    if "open_facilities" in solution:
        open_fac = solution["open_facilities"]
        open_fac_field = "open_facilities"
    elif "selected" in solution:
        open_fac = solution["selected"]
        open_fac_field = "selected"
    else:
        return float("inf")
    if not isinstance(open_fac, list) or not isinstance(assignments, list):
        return float("inf")
    if len(assignments) != n_cust:
        return float("inf")

    # Build open set; if duplicates/out-of-range, return inf.
    open_set = set()
    open_cost = 0.0
    used = {}
    field_base = open_fac_field or "open_facilities"
    for k, x in enumerate(open_fac):
        i, err = _resolve_facility_index(
            x,
            n_fac=n_fac,
            facility_id_to_index=facility_id_to_index,
            field=f"{field_base}[{k}]",
        )
        if err or i is None:
            return float("inf")
        if i < 0 or i >= n_fac or i in open_set:
            return float("inf")
        open_set.add(i)
        used[i] = 0.0
        oc, err = _to_float_finite(opening_costs[i], field="opening_costs")
        if err:
            return float("inf")
        open_cost += oc

    # Connection cost + capacity tracking; any invalid assignment => inf
    conn_cost = 0.0
    for j, x in enumerate(assignments):
        a, err = _resolve_facility_index(
            x,
            n_fac=n_fac,
            facility_id_to_index=facility_id_to_index,
            field=f"assignments[{j}]",
        )
        if err or a is None:
            return float("inf")
        if a < 0 or a >= n_fac:
            return float("inf")
        if a not in open_set:
            return float("inf")

        dem_j, err = _to_float_finite(demands[j], field="demands")
        if err:
            return float("inf")
        used[a] += dem_j

        row = connection_costs[a]
        if not isinstance(row, list) or j >= len(row):
            return float("inf")
        cij, err = _to_float_finite(row[j], field="connection_costs")
        if err:
            return float("inf")
        conn_cost += cij

    # Enforce capacities; if violated => inf
    for i in open_set:
        cap_i, err = _to_float_finite(capacities[i], field="capacities")
        if err:
            return float("inf")
        if used[i] > cap_i + EPS:
            return float("inf")

    return float(open_cost + conn_cost)