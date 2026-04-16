from typing import Any, Dict, List, Tuple, Optional
import math
import pandas as pd

from step3_evaluation.utils.eval_parsing import parse_json_field

PATTERN_OK = "OK"
PATTERN_FORMAT = "FormatError"
PATTERN_GLOBAL4 = "Global4"  # assignment (customers -> open facilities)

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
    """Interpret x as a facility index or identifier."""
    # 1. Try as integer (index or numeric ID)
    idx, err_int = _to_int_strict(x, field=field)
    if err_int is None:
        # a) Check if it matches a facility ID (e.g. "1" -> index 0)
        if str(idx) in facility_id_to_index:
            return facility_id_to_index[str(idx)], None
        
        # b) Check if it is a valid 0-based index
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
            
        # Try converting string to int to match int-like string IDs (e.g. "1.0" -> "1")
        # Or if facility_id_to_index has keys like "1", and x is "1.0" (unlikely for ID but possible)
        # But more importantly, if facility_id_to_index has keys like "1", and x is "1" (handled above)
        
        # What if facility_id_to_index has keys like "1" (from int 1 in source but converted to str),
        # and x is "1" (str). Handled.
        
        # What if facility_id_to_index has keys like "1", and x is "01" (str)? No, IDs are exact.
        
        return None, f"{field}: unknown facility id {fid!r}."

    return None, err_int


def uflp_parse_instance(row: pd.Series) -> Dict[str, Any]:
    """
    Parse Uncapacitated Facility Location Problem (UFLP) instance from `instance_variant`.

    Expected instance format:
      {
        "num_facilities": int,
        "num_customers": int,
        "facilities": [
          {"id": str, "opening_cost": float},
          ...
        ],
        "customers": [
          {"id": str},
          ...
        ],
        "connection_costs": [
          {"facility": str, "customer": str, "cost": float},
          ...
        ],
        "objective": float (optional)
      }

    Return:
      {
        "opening_costs": [...],
        "connection_costs": [...],
        "n_facilities": int,
        "n_customers": int
      }
    """
    inst_var = parse_json_field(row["instance_variant"])
    if inst_var is None:
        raise ValueError("Failed to parse 'instance_variant' for UFLP.")

    num_facilities_raw = inst_var.get("num_facilities", None)
    num_customers_raw = inst_var.get("num_customers", None)
    facilities = inst_var.get("facilities", None)
    customers = inst_var.get("customers", None)
    connection_costs_records = inst_var.get("connection_costs", None)

    if num_facilities_raw is None:
        raise ValueError("UFLP instance missing 'num_facilities'.")
    if num_customers_raw is None:
        raise ValueError("UFLP instance missing 'num_customers'.")
    if facilities is None:
        raise ValueError("UFLP instance missing 'facilities'.")
    if customers is None:
        raise ValueError("UFLP instance missing 'customers'.")
    if connection_costs_records is None:
        raise ValueError("UFLP instance missing 'connection_costs'.")

    n_fac, err = _to_int_strict(num_facilities_raw, field="num_facilities")
    if err:
        raise ValueError(f"Invalid num_facilities: {err}")
    n_cust, err = _to_int_strict(num_customers_raw, field="num_customers")
    if err:
        raise ValueError(f"Invalid num_customers: {err}")

    if not isinstance(facilities, list) or len(facilities) == 0:
        raise ValueError("'facilities' must be a non-empty list.")
    if not isinstance(customers, list) or len(customers) == 0:
        raise ValueError("'customers' must be a non-empty list.")
    if not isinstance(connection_costs_records, list) or len(connection_costs_records) == 0:
        raise ValueError("'connection_costs' must be a non-empty list.")

    if len(facilities) != n_fac:
        raise ValueError(f"'facilities' has {len(facilities)} entries, expected {n_fac}.")
    if len(customers) != n_cust:
        raise ValueError(f"'customers' has {len(customers)} entries, expected {n_cust}.")

    # Build facility name to index mapping
    facility_name_to_idx: Dict[str, int] = {}
    facility_ids: List[str] = []
    opening_costs: List[float] = [0.0] * n_fac

    for idx, facility in enumerate(facilities):
        if not isinstance(facility, dict):
            raise ValueError(f"Facility {idx} must be a dictionary.")
        
        facility_id = facility.get("id", None)
        opening_cost_raw = facility.get("opening_cost", None)

        if facility_id is None:
            raise ValueError(f"Facility {idx} missing 'id'.")
        if opening_cost_raw is None:
            raise ValueError(f"Facility {idx} missing 'opening_cost'.")

        # Allow int IDs by converting to string
        if isinstance(facility_id, int):
            facility_id = str(facility_id)
        elif not isinstance(facility_id, str):
            raise ValueError(f"Facility {idx} 'id' must be a string or int, got {type(facility_id).__name__}.")
            
        if facility_id in facility_name_to_idx:
            raise ValueError(f"Duplicate facility id '{facility_id}'.")
        
        facility_name_to_idx[facility_id] = idx
        facility_ids.append(facility_id)

        opening_cost, err = _to_float_finite(opening_cost_raw, field="opening_cost")
        if err:
            raise ValueError(f"Invalid opening_cost: {err}")
        if opening_cost < -EPS:
            raise ValueError(f"Invalid opening_cost: negative value ({opening_cost}).")

        opening_costs[idx] = float(opening_cost)

    # Build customer name to index mapping
    customer_name_to_idx: Dict[str, int] = {}
    for idx, customer in enumerate(customers):
        if not isinstance(customer, dict):
            raise ValueError(f"Customer {idx} must be a dictionary.")
        
        customer_id = customer.get("id", None)
        if customer_id is None:
            raise ValueError(f"Customer {idx} missing 'id'.")
            
        # Allow int IDs by converting to string
        if isinstance(customer_id, int):
            customer_id = str(customer_id)
            
        if customer_id in customer_name_to_idx:
            raise ValueError(f"Duplicate customer id '{customer_id}'.")
        
        customer_name_to_idx[customer_id] = idx

    # Build connection costs matrix from records
    connection_costs: List[List[float]] = [[0.0] * n_cust for _ in range(n_fac)]
    connection_costs_seen: set = set()

    for record in connection_costs_records:
        if not isinstance(record, dict):
            raise ValueError("Each connection_cost must be a dictionary.")
        
        facility_id = record.get("facility", None)
        customer_id = record.get("customer", None)
        cost_raw = record.get("cost", None)

        if facility_id is None:
            raise ValueError("connection_cost missing 'facility'.")
        if customer_id is None:
            raise ValueError("connection_cost missing 'customer'.")
        if cost_raw is None:
            raise ValueError("connection_cost missing 'cost'.")

        # Allow int IDs by converting to string
        if isinstance(facility_id, int):
            facility_id = str(facility_id)
        if isinstance(customer_id, int):
            customer_id = str(customer_id)

        if facility_id not in facility_name_to_idx:
            raise ValueError(f"connection_cost has unknown facility '{facility_id}'.")
        if customer_id not in customer_name_to_idx:
            raise ValueError(f"connection_cost has unknown customer '{customer_id}'.")

        facility_idx = facility_name_to_idx[facility_id]
        customer_idx = customer_name_to_idx[customer_id]

        if (facility_idx, customer_idx) in connection_costs_seen:
            raise ValueError(f"Duplicate connection_cost for facility '{facility_id}' and customer '{customer_id}'.")

        cost, err = _to_float_finite(cost_raw, field="cost")
        if err:
            raise ValueError(f"Invalid cost: {err}")
        if cost < -EPS:
            raise ValueError(f"Invalid cost: negative value ({cost}).")

        connection_costs[facility_idx][customer_idx] = float(cost)
        connection_costs_seen.add((facility_idx, customer_idx))

    # Verify all facility-customer pairs are present
    for facility_idx in range(n_fac):
        for customer_idx in range(n_cust):
            if (facility_idx, customer_idx) not in connection_costs_seen:
                raise ValueError(
                    f"Missing connection_cost for facility at index {facility_idx} "
                    f"and customer at index {customer_idx}."
                )

    return {
        "opening_costs": opening_costs,
        "connection_costs": connection_costs,
        "n_facilities": n_fac,
        "n_customers": n_cust,
        "facility_ids": facility_ids,
        "facility_id_to_index": dict(facility_name_to_idx),
    }


def uflp_check_feasibility(
    solution: Dict[str, Any],
    instance: Dict[str, Any],
) -> Tuple[bool, str, str]:
    """
    Check UFLP feasibility + unified violation pattern.

    Patterns for UFLP (per taxonomy Global_4):
      - FormatError: wrong structure/types/lengths/out-of-range ids
      - Global4: assignment/linking violations (customer assigned to closed facility, no open facility when customers exist)
    """
    n_fac: int = instance["n_facilities"]
    n_cust: int = instance["n_customers"]

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
    open_set = set()
    for k, x in enumerate(open_fac):
        field_name = f"{open_fac_field}[{k}]"
        i, err = _resolve_facility_index(
            x,
            n_fac=n_fac,
            facility_id_to_index=facility_id_to_index,
            field=field_name,
        )
        if err:
            return False, PATTERN_FORMAT, err
        if i is None:
            return False, PATTERN_FORMAT, f"{field_name} resolved to None."
            
        fid_label = facility_ids[i]
        if i in open_set:
            return False, PATTERN_FORMAT, f"Facility {fid_label!r} appears multiple times in {open_fac_field}."
        open_set.add(i)

    for j, x in enumerate(assignments):
        a, err = _resolve_facility_index(
            x,
            n_fac=n_fac,
            facility_id_to_index=facility_id_to_index,
            field=f"assignments[{j}]",
        )
        if err:
            return False, PATTERN_FORMAT, err
        if a is None:
            return False, PATTERN_FORMAT, f"assignments[{j}] resolved to None."

        if a not in open_set:
            fid_label = facility_ids[a]
            return False, PATTERN_GLOBAL4, (
                f"Customer {j} assigned to facility {fid_label} (index {a}), but facility {fid_label} is not open."
            )

    return True, PATTERN_OK, "Feasible UFLP solution."


def uflp_objective_model(
    solution: Dict[str, Any],
    instance: Dict[str, Any],
) -> float:
    """
    UFLP objective: minimize
        sum_{i in open} opening_costs[i] + sum_{j} connection_costs[a_j][j].

    Assumes solution is feasible; if not well-formed, return inf (never raise).
    """
    if not isinstance(solution, dict):
        return float("inf")

    opening_costs = instance.get("opening_costs", None)
    connection_costs = instance.get("connection_costs", None)
    n_fac = instance.get("n_facilities", None)
    n_cust = instance.get("n_customers", None)

    if not isinstance(opening_costs, list) or not isinstance(connection_costs, list):
        return float("inf")
    if not isinstance(n_fac, int) or not isinstance(n_cust, int):
        return float("inf")

    assignments = solution.get("assignments", None)
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

    facility_id_to_index: Dict[str, int] = instance.get("facility_id_to_index", {})
    facility_ids: List[str] = instance.get("facility_ids", [])
    open_set = set()
    open_cost = 0.0
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
        oc, err = _to_float_finite(opening_costs[i], field="opening_costs")
        if err:
            return float("inf")
        open_cost += oc

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
        if a < 0 or a >= n_fac or a not in open_set:
            return float("inf")
        row = connection_costs[a]
        if not isinstance(row, list) or j >= len(row):
            return float("inf")
        cij, err = _to_float_finite(row[j], field="connection_costs")
        if err:
            return float("inf")
        conn_cost += cij

    return float(open_cost + conn_cost)
