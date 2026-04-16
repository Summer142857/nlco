import json
import random
from typing import Dict, Any, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for facility location & clustering problems:

- PCENTER : p-center problem on a distance matrix
- PMED    : p-median problem on a distance matrix
- CFLP    : capacitated facility location
- UFLP    : uncapacitated facility location

PCENTER/PMED input supports ONLY ONE render mode now:
  - pair : one (from,to,distance) per line
    NOTE: diagonal pairs (i==j) are omitted.
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _build_labels(num: int, index_base) -> List[Any]:
    return make_labels(num, index_base)


def _remap_solution_facility(
    solution: Any,
    problem_type: str,
    node_id_map: Optional[Dict[int, Any]] = None,
    facility_id_map: Optional[Dict[int, Any]] = None,
) -> Any:
    if solution is None:
        return None

    pt = problem_type.upper()
    sol_out: Dict[str, Any] = {}

    if pt in ["PCENTER", "PMED"]:
        if node_id_map is None or not isinstance(solution, dict):
            return solution
        facs = solution.get("selected") or solution.get("facilities")
        assigns = solution.get("assignments")
        if facs is not None:
            sol_out["selected"] = [node_id_map.get(int(i), i) for i in facs]
        if assigns is not None:
            sol_out["assignments"] = [node_id_map.get(int(a), a) for a in assigns]
        return sol_out

    if pt in ["CFLP", "UFLP"]:
        if facility_id_map is None or not isinstance(solution, dict):
            return solution
        open_f = solution.get("selected") or solution.get("open_facilities")
        assigns = solution.get("assignments")
        if open_f is not None:
            sol_out["selected"] = [facility_id_map.get(int(i), i) for i in open_f]
        if assigns is not None:
            sol_out["assignments"] = [facility_id_map.get(int(a), a) for a in assigns]
        return sol_out

    return solution


def _build_labeled_facility_instance(
    inst_core: Dict[str, Any],
    problem_type: str,
    node_id_map: Optional[Dict[int, Any]],
    facility_id_map: Optional[Dict[int, Any]],
    customer_id_map: Optional[Dict[int, Any]],
) -> Dict[str, Any]:
    pt = problem_type.upper()
    inst_variant: Dict[str, Any] = {"problem_type": problem_type}

    if pt in ["PCENTER", "PMED"]:
        dist_mat = inst_core.get("distance_matrix", [])
        k_val = inst_core.get("p")
        num_nodes = len(dist_mat)

        inst_variant["num_nodes"] = num_nodes
        if k_val is not None:
            inst_variant["num_open"] = k_val

        # Keep a structured "sites" view for debug / downstream usage
        sites: List[Dict[str, Any]] = []
        for i in range(num_nodes):
            label_i = node_id_map.get(i, i) if node_id_map else i
            dist_dict = {}
            for j, d_ij in enumerate(dist_mat[i]):
                label_j = node_id_map.get(j, j) if node_id_map else j
                dist_dict[label_j] = d_ij
            sites.append({"id": label_i, "distances": dist_dict})
        inst_variant["sites"] = sites

    elif pt in ["CFLP", "UFLP"]:
        opening_costs = inst_core.get("opening_costs", [])
        connection_costs = inst_core.get("connection_costs", [])
        capacities = inst_core.get("capacities")
        demands = inst_core.get("demands")

        num_fac = len(opening_costs)
        num_cust = len(connection_costs[0]) if connection_costs else 0

        inst_variant["num_facilities"] = num_fac
        inst_variant["num_customers"] = num_cust

        facilities: List[Dict[str, Any]] = []
        for i in range(num_fac):
            fid = facility_id_map.get(i, i) if facility_id_map else i
            rec = {"id": fid, "opening_cost": opening_costs[i]}
            if capacities is not None and i < len(capacities):
                rec["capacity"] = capacities[i]
            facilities.append(rec)
        inst_variant["facilities"] = facilities

        customers: List[Dict[str, Any]] = []
        for j in range(num_cust):
            cid = customer_id_map.get(j, j) if customer_id_map else j
            rec = {"id": cid}
            if demands is not None and j < len(demands):
                rec["demand"] = demands[j]
            customers.append(rec)
        inst_variant["customers"] = customers

        conn_triplets: List[Dict[str, Any]] = []
        for i in range(num_fac):
            fid = facility_id_map.get(i, i) if facility_id_map else i
            for j, cost_ij in enumerate(connection_costs[i]):
                cid = customer_id_map.get(j, j) if customer_id_map else j
                conn_triplets.append({"facility": fid, "customer": cid, "cost": cost_ij})
        inst_variant["connection_costs"] = conn_triplets

    if "objective" in inst_core:
        inst_variant["objective"] = inst_core["objective"]

    return inst_variant


def facility_render_input(
    fmt: str,
    inst_core: Dict[str, Any],
    node_id_map: Optional[Dict[int, Any]],
    facility_id_map: Optional[Dict[int, Any]],
    customer_id_map: Optional[Dict[int, Any]],
    nl_style: Dict[str, Any] | None,
    problem_type: str,
    scenario_hint: Dict[str, Any] | None = None,
) -> str:
    pt = problem_type.upper()

    # ==================================================================
    # PCENTER / PMED (PAIR MODE ONLY, NO DIAGONAL)
    # ==================================================================
    if pt in ["PCENTER", "PMED"]:
        # defaults
        num_nodes_key = "num_nodes"
        num_open_key = "num_open"   # do not expose "p" in user-facing inputs
        global_nodes_key = "nodes"

        from_id_key = "from_id"
        to_id_key = "to_id"
        distance_key = "distance"


        # renaming via scenario_hint
        if scenario_hint and isinstance(scenario_hint, dict):
            for field in scenario_hint.get("global_fields", []):
                orig, new = field.get("name"), field.get("new_name")
                if not new:
                    continue
                if orig == "num_nodes":
                    num_nodes_key = new
                elif orig in ["num_open", "p"]:
                    num_open_key = new
                elif orig == "nodes":
                    global_nodes_key = new

            for field in scenario_hint.get("pair_item_fields", []):
                orig, new = field.get("name"), field.get("new_name")
                if not new:
                    continue
                if orig == "from_id":
                    from_id_key = new
                elif orig == "to_id":
                    to_id_key = new
                elif orig == "distance":
                    distance_key = new

        dist_mat = inst_core.get("distance_matrix", [])
        k_val = inst_core.get("p")
        num_nodes = len(dist_mat)

        node_ids = [node_id_map.get(i, i) if node_id_map else i for i in range(num_nodes)]
        nodes_str = ", ".join(str(x) for x in node_ids)

        # build pair records WITHOUT diagonal
        pair_records: List[Dict[str, Any]] = []
        for i in range(num_nodes):
            li = node_id_map.get(i, i) if node_id_map else i
            for j in range(num_nodes):
                if i == j:
                    continue  # omit diagonal
                lj = node_id_map.get(j, j) if node_id_map else j
                pair_records.append(
                    {
                        from_id_key: li,
                        to_id_key: lj,
                        distance_key: dist_mat[i][j],
                    }
                )

        if fmt == "json":
            obj: Dict[str, Any] = {num_nodes_key: num_nodes}
            if k_val is not None:
                obj[num_open_key] = k_val
            obj[global_nodes_key] = node_ids
            obj["data"] = pair_records
            return json.dumps(obj, ensure_ascii=False, indent=2)

        if fmt == "csv":
            lines: List[str] = []
            lines.append(f"# {num_nodes_key}={num_nodes}")
            if k_val is not None:
                lines.append(f"# {num_open_key}={k_val}")
            lines.append(f"# {global_nodes_key}={nodes_str}")
            lines.append(f"{from_id_key},{to_id_key},{distance_key}")
            for rec in pair_records:
                lines.append(f"{rec[from_id_key]},{rec[to_id_key]},{rec[distance_key]}")
            return "\n".join(lines)

        if fmt == "markdown_table":
            if isinstance(nl_style, dict):
                header_tmpl = nl_style.get("header", "") or ""
                footer_tmpl = nl_style.get("footer", "") or ""
                pair_line_tmpl = nl_style.get("pair_item_fields_line_template", "") or ""

                parts: List[str] = []
                base_vars = {
                    "num_nodes": num_nodes,
                    "num_open": k_val if k_val is not None else "",
                    "nodes": nodes_str,
                    num_nodes_key: num_nodes,
                    num_open_key: k_val if k_val is not None else "",
                    global_nodes_key: nodes_str,
                }

                if header_tmpl:
                    parts.append(_format_safe(header_tmpl, **base_vars))

                if pair_line_tmpl:
                    for rec in pair_records:
                        vars_ = {
                            **base_vars,
                            "from_id": rec.get(from_id_key, ""),
                            "to_id": rec.get(to_id_key, ""),
                            "distance": rec.get(distance_key, ""),
                            from_id_key: rec.get(from_id_key, ""),
                            to_id_key: rec.get(to_id_key, ""),
                            distance_key: rec.get(distance_key, ""),
                        }
                        parts.append(_format_safe(pair_line_tmpl, **vars_))

                if footer_tmpl:
                    parts.append(_format_safe(footer_tmpl, **base_vars))

                return "\n".join(parts)

            # fallback table
            lines: List[str] = []
            lines.append(f"There are {num_nodes} candidate locations.")
            lines.append(f"Locations: {nodes_str}")
            if k_val is not None:
                lines.append(f"You must open exactly {k_val} facilities.")
            lines.append("")
            lines.append(f"| {from_id_key} | {to_id_key} | {distance_key} |")
            lines.append("|---|---|---|")
            for rec in pair_records:
                lines.append(f"| {rec[from_id_key]} | {rec[to_id_key]} | {rec[distance_key]} |")
            return "\n".join(lines)

        if fmt == "nl" and isinstance(nl_style, dict):
            header_tmpl = nl_style.get("header", "") or ""
            footer_tmpl = nl_style.get("footer", "") or ""
            pair_line_tmpl = nl_style.get("pair_item_fields_line_template", "") or ""

            parts: List[str] = []
            base_vars = {
                "num_nodes": num_nodes,
                "num_open": k_val if k_val is not None else "",
                "nodes": nodes_str,
                num_nodes_key: num_nodes,
                num_open_key: k_val if k_val is not None else "",
                global_nodes_key: nodes_str,
            }

            if header_tmpl:
                parts.append(_format_safe(header_tmpl, **base_vars))

            if pair_line_tmpl:
                for rec in pair_records:
                    vars_ = {
                        **base_vars,
                        "from_id": rec.get(from_id_key, ""),
                        "to_id": rec.get(to_id_key, ""),
                        "distance": rec.get(distance_key, ""),
                        from_id_key: rec.get(from_id_key, ""),
                        to_id_key: rec.get(to_id_key, ""),
                        distance_key: rec.get(distance_key, ""),
                    }
                    parts.append(_format_safe(pair_line_tmpl, **vars_))

            if footer_tmpl:
                parts.append(_format_safe(footer_tmpl, **base_vars))

            return "\n".join(parts)

        return ""

    # ==================================================================
    # CFLP / UFLP (unchanged logic)
    # ==================================================================
    num_fac_key = "num_facilities"
    num_cust_key = "num_customers"
    facility_id_key = "facility_id"
    customer_id_key = "customer_id"
    opening_cost_key = "opening_cost"
    capacity_key = "capacity"
    demand_key = "demand"
    cost_key = "cost"

    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig, new = field.get("name"), field.get("new_name")
            if not new:
                continue
            if orig == "num_facilities":
                num_fac_key = new
            elif orig == "num_customers":
                num_cust_key = new

        for group in ["facility_item_fields", "customer_item_fields", "connection_item_fields"]:
            for field in scenario_hint.get(group, []):
                orig, new = field.get("name"), field.get("new_name")
                if not new:
                    continue
                if orig == "facility_id":
                    facility_id_key = new
                elif orig == "customer_id":
                    customer_id_key = new
                elif orig == "opening_cost":
                    opening_cost_key = new
                elif orig == "capacity":
                    capacity_key = new
                elif orig == "demand":
                    demand_key = new
                elif orig == "cost":
                    cost_key = new

    opening_costs = inst_core.get("opening_costs", [])
    connection_costs = inst_core.get("connection_costs", [])
    capacities = inst_core.get("capacities")
    demands = inst_core.get("demands")

    num_fac = len(opening_costs)
    num_cust = len(connection_costs[0]) if connection_costs else 0

    facility_records: List[Dict[str, Any]] = []
    for i in range(num_fac):
        fid = facility_id_map.get(i, i) if facility_id_map else i
        rec = {facility_id_key: fid, opening_cost_key: opening_costs[i]}
        if capacities is not None and i < len(capacities):
            rec[capacity_key] = capacities[i]
        facility_records.append(rec)

    customer_records: List[Dict[str, Any]] = []
    for j in range(num_cust):
        cid = customer_id_map.get(j, j) if customer_id_map else j
        rec = {customer_id_key: cid}
        if demands is not None and j < len(demands):
            rec[demand_key] = demands[j]
        customer_records.append(rec)

    conn_records: List[Dict[str, Any]] = []
    for i in range(num_fac):
        fid = facility_id_map.get(i, i) if facility_id_map else i
        for j, c_ij in enumerate(connection_costs[i]):
            cid = customer_id_map.get(j, j) if customer_id_map else j
            conn_records.append({facility_id_key: fid, customer_id_key: cid, cost_key: c_ij})

    def _group_key_from_id_key(id_key: str) -> str:
        base = id_key[:-3] if id_key.endswith("_id") else id_key
        return base + "s"

    def _group_key_from_cost_key(cost_key_: str) -> str:
        if cost_key_ == "cost":
            return "connections"
        base = cost_key_[:-5] if cost_key_.endswith("_cost") else cost_key_
        return base + "s"

    def _group_title_from_id_key(id_key: str) -> str:
        base = id_key[:-3] if id_key.endswith("_id") else id_key
        base = base.replace("_", " ")
        if not base.endswith("s"):
            base = base + "s"
        return base.title()

    def _title_from_cost_key(cost_key_: str) -> str:
        return cost_key_.replace("_", " ").title()

    facility_group_key = _group_key_from_id_key(facility_id_key)
    customer_group_key = _group_key_from_id_key(customer_id_key)
    connection_group_key = _group_key_from_cost_key(cost_key)

    facility_group_title = _group_title_from_id_key(facility_id_key)
    customer_group_title = _group_title_from_id_key(customer_id_key)
    connection_title = _title_from_cost_key(cost_key)

    if fmt == "json":
        obj: Dict[str, Any] = {
            num_fac_key: num_fac,
            num_cust_key: num_cust,
            facility_group_key: facility_records,
            customer_group_key: customer_records,
            connection_group_key: conn_records,
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_fac_key}={num_fac}")
        lines.append(f"# {num_cust_key}={num_cust}")

        fac_cols = [facility_id_key, opening_cost_key]
        if capacities is not None:
            fac_cols.append(capacity_key)
        lines.append(f"# {facility_group_key.upper()}")
        lines.append(",".join(fac_cols))
        for rec in facility_records:
            lines.append(",".join(str(rec.get(c, "")) for c in fac_cols))

        cust_cols = [customer_id_key]
        if demands is not None:
            cust_cols.append(demand_key)
        lines.append(f"# {customer_group_key.upper()}")
        lines.append(",".join(cust_cols))
        for rec in customer_records:
            lines.append(",".join(str(rec.get(c, "")) for c in cust_cols))

        conn_cols = [facility_id_key, customer_id_key, cost_key]
        lines.append(f"# {connection_group_key.upper()}")
        lines.append(",".join(conn_cols))
        for rec in conn_records:
            lines.append(",".join(str(rec.get(c, "")) for c in conn_cols))

        return "\n".join(lines)

    if fmt == "markdown_table":
        lines: List[str] = []
        lines.append(
            f"There are {num_fac} {facility_group_title.lower()} and {num_cust} {customer_group_title.lower()} in this instance."
        )
        lines.append("")

        fac_cols = [facility_id_key, opening_cost_key]
        if capacities is not None:
            fac_cols.append(capacity_key)
        lines.append(f"**{facility_group_title}**")
        lines.append("| " + " | ".join(fac_cols) + " |")
        lines.append("|" + "|".join("---" for _ in fac_cols) + "|")
        for rec in facility_records:
            lines.append("| " + " | ".join(str(rec.get(c, "")) for c in fac_cols) + " |")
        lines.append("")

        cust_cols = [customer_id_key]
        if demands is not None:
            cust_cols.append(demand_key)
        lines.append(f"**{customer_group_title}**")
        lines.append("| " + " | ".join(cust_cols) + " |")
        lines.append("|" + "|".join("---" for _ in cust_cols) + "|")
        for rec in customer_records:
            lines.append("| " + " | ".join(str(rec.get(c, "")) for c in cust_cols) + " |")
        lines.append("")

        conn_cols = [facility_id_key, customer_id_key, cost_key]
        lines.append(f"**{connection_title}**")
        lines.append("| " + " | ".join(conn_cols) + " |")
        lines.append("|" + "|".join("---" for _ in conn_cols) + "|")
        for rec in conn_records:
            lines.append("| " + " | ".join(str(rec.get(c, "")) for c in conn_cols) + " |")

        return "\n".join(lines)

    if fmt == "nl" and isinstance(nl_style, dict):
        header_tmpl = nl_style.get("header", "") or ""
        footer_tmpl = nl_style.get("footer", "") or ""
        fac_tmpl = nl_style.get("facility_item_fields_line_template", "") or ""
        cust_tmpl = nl_style.get("customer_item_fields_line_template", "") or ""
        conn_tmpl = nl_style.get("connection_item_fields_line_template", "") or nl_style.get("line_template", "") or ""

        parts: List[str] = []
        header_vars = {
            "num_facilities": num_fac,
            "num_customers": num_cust,
            "opening_costs": ", ".join(str(c) for c in opening_costs) if opening_costs else "",
            "capacities": ", ".join(str(c) for c in (capacities or [])) if capacities else "",
            "demands": ", ".join(str(d) for d in (demands or [])) if demands else "",
        }
        if header_tmpl:
            parts.append(_format_safe(header_tmpl, **header_vars))

        def _mk_vars(rec: Dict[str, Any]) -> Dict[str, Any]:
            v = dict(header_vars)

            if facility_id_key in rec:
                v["facility_id"] = rec.get(facility_id_key, "")
                v[facility_id_key] = rec.get(facility_id_key, "")
            if opening_cost_key in rec:
                v["opening_cost"] = rec.get(opening_cost_key, "")
                v[opening_cost_key] = rec.get(opening_cost_key, "")
            if capacity_key in rec:
                v["capacity"] = rec.get(capacity_key, "")
                v[capacity_key] = rec.get(capacity_key, "")

            if customer_id_key in rec:
                v["customer_id"] = rec.get(customer_id_key, "")
                v[customer_id_key] = rec.get(customer_id_key, "")
            if demand_key in rec:
                v["demand"] = rec.get(demand_key, "")
                v[demand_key] = rec.get(demand_key, "")

            if cost_key in rec:
                v["cost"] = rec.get(cost_key, "")
                v[cost_key] = rec.get(cost_key, "")

            return v

        if fac_tmpl:
            for rec in facility_records:
                parts.append(_format_safe(fac_tmpl, **_mk_vars(rec)))

        if cust_tmpl:
            for rec in customer_records:
                parts.append(_format_safe(cust_tmpl, **_mk_vars(rec)))

        if conn_tmpl:
            for rec in conn_records:
                parts.append(_format_safe(conn_tmpl, **_mk_vars(rec)))

        if footer_tmpl:
            parts.append(_format_safe(footer_tmpl, **header_vars))

        return "\n".join(parts)

    return ""


def contextualize_instance_facility(
    inst: Dict[str, Any],
    task_name: str,
    contexts: List[Dict[str, Any]],
    precomputed_templates: Dict[int, str],
    nl_styles: Dict[tuple, Dict[str, Any]],
    scenario_hints: Dict[int, Dict[str, Any]],
    instance_idx: int = 0,
    difficulty_tier: str | None = None,
) -> List[Dict[str, Any]]:
    rng = random.Random()
    results: List[Dict[str, Any]] = []

    inst_core = inst.get("instance")
    solution = inst.get("solution")
    obj = inst.get("obj")
    problem_type = inst.get("problem_type", task_name)

    if inst_core is None:
        return results

    pt = problem_type.upper()

    node_id_map = None
    facility_id_map = None
    customer_id_map = None

    # validate sizes
    if pt in ["PCENTER", "PMED"]:
        dist_mat = inst_core.get("distance_matrix", [])
        num_nodes = len(dist_mat)
        if num_nodes == 0:
            return results
    else:
        opening_costs = inst_core.get("opening_costs", [])
        connection_costs = inst_core.get("connection_costs", [])
        num_fac = len(opening_costs)
        num_cust = len(connection_costs[0]) if connection_costs else 0
        if num_fac == 0 or num_cust == 0:
            return results

    # choose context
    K = len(contexts) if contexts else 1
    valid_context_indices = sorted(idx for idx in precomputed_templates if idx != 0)
    if not valid_context_indices:
        valid_context_indices = sorted(precomputed_templates)

    K = len(valid_context_indices)
    if K == 0:
        return results

    context_index = valid_context_indices[instance_idx % K]
    template = precomputed_templates[context_index]
    nl_style = nl_styles.get((task_name.upper(), context_index))
    scenario_hint = scenario_hints.get(context_index)

    fmt = rng.choice(FORMATS)
    index_base = rng.choice(INDEX_BASES)

    # label maps
    if pt in ["PCENTER", "PMED"]:
        labels = _build_labels(num_nodes, index_base)
        node_id_map = {i: labels[i] for i in range(num_nodes)}
    else:
        cust_labels = _build_labels(num_cust, index_base)
        facility_id_map = {i: f"F{i + 1}" for i in range(num_fac)}
        customer_id_map = {j: cust_labels[j] for j in range(num_cust)}

    solution_variant = _remap_solution_facility(
        solution=solution,
        problem_type=problem_type,
        node_id_map=node_id_map,
        facility_id_map=facility_id_map,
    )

    input_text = facility_render_input(
        fmt=fmt,
        inst_core=inst_core,
        node_id_map=node_id_map,
        facility_id_map=facility_id_map,
        customer_id_map=customer_id_map,
        nl_style=nl_style,
        problem_type=problem_type,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_facility_instance(
        inst_core=inst_core,
        problem_type=problem_type,
        node_id_map=node_id_map,
        facility_id_map=facility_id_map,
        customer_id_map=customer_id_map,
    )

    record = build_output_record(
        task_id=task_name,
        difficulty_tier=difficulty_tier or "UNKNOWN",
        example_index=context_index,
        prompt=full_instruction,
        surface_format=fmt,
        indexing_scheme=index_base,
        instance_canonical_json=inst_core,
        reference_solution_canonical_json=solution,
        reference_objective_value=obj,
        instance_surface_json=instance_variant,
        reference_solution_surface_json=solution_variant,
    )

    results.append(record)
    return results


def _get_problem_specs(problem_type: str) -> Dict[str, str]:
    pt = problem_type.upper()

    if pt == "PCENTER":
        desc = (
            "p-Center Problem (PCENTER): You are given a set of candidate locations and the travel distance between every "
            "pair of locations. You must choose a fixed number of locations to open as service points. Every location must be "
            "assigned to exactly one opened service point. The objective is to make the worst-case assigned travel distance "
            "as small as possible (i.e., minimize the maximum distance from any location to its assigned service point)."
        )
        baseline_template = (
            "You are given a center-based facility location optimization problem.\n\n"
            "Problem description:\n"
            "There is a set of locations. Each location represents a demand point, and any location "
            "may also be opened as a service center.\n"
            "The travel distance between every ordered pair of locations is given (as pairs).\n"
            "Every location must be served by exactly one opened service center.\n\n"
            "Decision variables:\n"
            "- Select a fixed number of locations to open as service centers (the required number is specified in the input).\n"
            "- Assign every location to exactly one opened service center.\n\n"
            "Constraints:\n"
            "- Each location must be assigned to exactly one service center.\n"
            "- A location can only be assigned to a service center that is opened.\n\n"
            "Objective:\n"
            "Minimize the worst-case service distance, defined as the maximum travel distance "
            "between any location and the service center it is assigned to.\n\n"
            "You will receive the problem instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to choose which locations to open as service centers and how to assign "
            "all locations to them, so that the maximum assignment distance is minimized.\n\n"
            "If multiple optimal solutions exist, you may return any one of them.\n\n"
            "Reply using exactly the following JSON structure (no extra keys, no explanations):\n"
            "```json\n"
            "{\n"
            '  "solution": {\n'
            '    "selected": [<location_to_open>, <location_to_open>, ...],\n'
            '    "assignments": [<chosen_open_location>, <chosen_open_location>, ...]\n'
            "  }\n"
            "}\n"
            "```\n"
            "Here, `selected` is the list of locations chosen to host service centers.\n"
            "The `assignments` list has the same length and order as the locations described in the input; "
            "the i-th entry specifies which opened location serves location i.\n"
            "All identifiers must match exactly those used in the input, and every assignment must refer to one of the "
            "locations listed in `selected`.\n"
            "**Do not include explanations or any extra keys.**"
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": {\n'
            '    "selected": [<location_to_open>, <location_to_open>, ...],\n'
            '    "assignments": [<chosen_open_location>, <chosen_open_location>, ...]\n'
            "  }\n"
            "}\n"
            "```\n"
        )

    elif pt == "PMED":
        desc = (
            "p-Median Problem (PMED): You are given a set of candidate locations and the travel distance between every pair "
            "of locations. You must choose a fixed number of locations to open as service points. Every location must be assigned "
            "to exactly one opened service point. The objective is to minimize the total travel distance across all assignments "
            "(i.e., sum of distances from each location to its assigned service point)."
        )
        baseline_template = (
            "You are given a median-based facility location optimization problem.\n\n"
            "Problem description:\n"
            "There is a set of locations, each of which may be opened as a service facility.\n"
            "The travel distance between every ordered pair of locations is given (as pairs).\n"
            "Every location must be assigned to exactly one opened facility.\n\n"
            "Decision variables:\n"
            "- Select a fixed number of locations to open as facilities (the required number is specified in the input).\n"
            "- Assign every location to exactly one opened facility.\n\n"
            "Constraints:\n"
            "- Each location must be assigned to exactly one facility.\n"
            "- Assignments can only be made to facilities that are opened.\n\n"
            "Objective:\n"
            "Minimize the total service cost, defined as the sum of travel distances "
            "from each location to the facility that serves it.\n\n"
            "You will receive the problem instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to choose facility locations and assignments that minimize "
            "the total travel distance over all locations.\n\n"
            "Reply using exactly the following JSON structure (no extra keys, no explanations):\n"
            "```json\n"
            "{\n"
            '  "solution": {\n'
            '    "selected": [<location_to_open>, <location_to_open>, ...],\n'
            '    "assignments": [<chosen_open_location>, <chosen_open_location>, ...]\n'
            "  }\n"
            "}\n"
            "```\n"
            "Here, `selected` is the list of locations chosen to host service centers.\n"
            "The `assignments` list has the same length and order as the locations described in the input; "
            "the i-th entry specifies which opened location serves location i.\n"
            "All identifiers must match exactly those used in the input, and every assignment must refer to one of the "
            "locations listed in `selected`.\n"
            "**Do not include explanations or any extra keys.**"
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": {\n'
            '    "selected": [<location_to_open>, <location_to_open>, ...],\n'
            '    "assignments": [<chosen_open_location>, <chosen_open_location>, ...]\n'
            "  }\n"
            "}\n"
            "```\n"
        )


    elif pt == "CFLP":
        desc = (
            "Capacitated Facility Location Problem (CFLP): You are given candidate facilities with opening costs and capacity limits, "
            "and customers with demands. Serving a customer from a facility incurs a connection cost. You must decide which facilities to open "
            "and assign each customer to exactly one opened facility, without exceeding any facility's capacity. The objective is to minimize "
            "the total cost (opening costs + connection costs)."
        )
        baseline_template = (
            "You are given a capacitated facility location optimization problem.\n\n"
            "Problem description:\n"
            "There is a set of candidate facilities, each with a fixed opening cost and a limited service capacity.\n"
            "There is also a set of customers, each with a known demand.\n"
            "Serving a customer from a facility incurs a connection cost.\n\n"
            "Decision variables:\n"
            "- Decide which facilities to open.\n"
            "- Assign each customer to exactly one opened facility.\n\n"
            "Constraints:\n"
            "- Each customer must be assigned to exactly one opened facility.\n"
            "- A customer can only be assigned to a facility that is opened.\n"
            "- For every facility, the total demand of assigned customers must not exceed its capacity.\n\n"
            "Objective:\n"
            "Minimize the total cost, defined as the sum of facility opening costs "
            "and customer-to-facility connection costs.\n\n"
            "You will receive the problem instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": {\n'
            '    "selected": [<location_to_open>, <location_to_open>, ...],\n'
            '    "assignments": [<chosen_open_location>, <chosen_open_location>, ...]\n'
            "  }\n"
            "}\n"
            "```\n"
            "**Do not include explanations or any extra keys.**"
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": {\n'
            '    "selected": [<location_to_open>, <location_to_open>, ...],\n'
            '    "assignments": [<chosen_open_location>, <chosen_open_location>, ...]\n'
            "  }\n"
            "}\n"
            "```\n"
        )

    else:  # UFLP
        desc = (
            "Uncapacitated Facility Location Problem (UFLP): You are given candidate facilities with opening costs and customers. "
            "Serving a customer from a facility incurs a connection cost. You must decide which facilities to open and assign each customer "
            "to exactly one opened facility. There are no capacity limits. The objective is to minimize the total cost (opening costs + connection costs)."
        )
        baseline_template = (
            "You are given an uncapacitated facility location optimization problem.\n\n"
            "Problem description:\n"
            "There is a set of candidate facilities, each with a fixed opening cost,\n"
            "and a set of customers. Serving a customer from a facility incurs a connection cost.\n"
            "Facilities have no capacity limits.\n\n"
            "Decision variables:\n"
            "- Decide which facilities to open.\n"
            "- Assign each customer to exactly one opened facility.\n\n"
            "Constraints:\n"
            "- Each customer must be assigned to exactly one opened facility.\n"
            "- Assignments can only be made to facilities that are opened.\n"
            "- There are no capacity constraints on facilities.\n\n"
            "Objective:\n"
            "Minimize the total cost, defined as the sum of facility opening costs "
            "and customer-to-facility connection costs.\n\n"
            "You will receive the problem instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": {\n'
            '    "selected": [<location_to_open>, <location_to_open>, ...],\n'
            '    "assignments": [<chosen_open_location>, <chosen_open_location>, ...]\n'
            "  }\n"
            "}\n"
            "```\n"
            "**Do not include explanations or any extra keys.**"
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": {\n'
            '    "selected": [<location_to_open>, <location_to_open>, ...],\n'
            '    "assignments": [<chosen_open_location>, <chosen_open_location>, ...]\n'
            "  }\n"
            "}\n"
            "```\n"
        )

    return {"task_description": desc, "baseline_template": baseline_template, "output_format": output_format}


def _get_hint(problem_type: str) -> Dict[str, Any]:
    pt = problem_type.upper()

    if pt in ["PCENTER", "PMED"]:
        global_fields = [
            {"name": "num_nodes", "description": "total number of candidate locations"},
            {"name": "num_open", "description": "how many service points must be opened"},
            {"name": "nodes", "description": "list of all location identifiers in the instance"},
        ]

        pair_item_fields = [
            {"name": "from_id", "description": "source location identifier"},
            {"name": "to_id", "description": "target location identifier"},
            {"name": "distance", "description": "distance from source to target"},
        ]

        allowed = [
            "num_nodes", "num_open", "nodes",
            "from_id", "to_id", "distance",
        ]

        return {
            "global_fields": global_fields,
            "pair_item_fields": pair_item_fields,
            "allowed_placeholders": allowed,
        }

    if pt == "CFLP":
        global_fields = [
            {"name": "num_facilities", "description": "total number of candidate facilities"},
            {"name": "num_customers", "description": "total number of customers"},
        ]
        facility_item_fields = [
            {"name": "facility_id", "description": "identifier of a facility"},
            {"name": "opening_cost", "description": "opening cost of this facility"},
            {"name": "capacity", "description": "capacity of this facility"},
        ]
        customer_item_fields = [
            {"name": "customer_id", "description": "identifier of a customer"},
            {"name": "demand", "description": "demand of this customer"},
        ]
        connection_item_fields = [
            {"name": "facility_id", "description": "identifier of a facility"},
            {"name": "customer_id", "description": "identifier of a customer"},
            {"name": "cost", "description": "service cost from facility to customer"},
        ]
        allowed = [
            "num_facilities", "num_customers", "opening_costs", "capacities", "demands",
            "facility_id", "opening_cost", "capacity", "customer_id", "demand", "cost",
        ]
        return {
            "global_fields": global_fields,
            "facility_item_fields": facility_item_fields,
            "customer_item_fields": customer_item_fields,
            "connection_item_fields": connection_item_fields,
            "allowed_placeholders": allowed,
        }

    # UFLP
    global_fields = [
        {"name": "num_facilities", "description": "total number of candidate facilities"},
        {"name": "num_customers", "description": "total number of customers"},
        {"name": "opening_costs", "description": "opening costs of all facilities"},
    ]
    facility_item_fields = [
        {"name": "facility_id", "description": "identifier of a facility"},
        {"name": "opening_cost", "description": "opening cost of this facility"},
    ]
    connection_item_fields = [
        {"name": "facility_id", "description": "identifier of a facility"},
        {"name": "customer_id", "description": "identifier of a customer"},
        {"name": "cost", "description": "connection cost from facility to customer"},
    ]
    allowed = [
        "num_facilities", "num_customers", "opening_costs",
        "facility_id", "opening_cost", "customer_id", "cost",
    ]
    return {
        "global_fields": global_fields,
        "facility_item_fields": facility_item_fields,
        "connection_item_fields": connection_item_fields,
        "allowed_placeholders": allowed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Contextualize facility-location instances (PCENTER, PMED, CFLP, UFLP) into NL/JSON/CSV/Markdown."
    )
    parser.add_argument("--problem_type", type=str, default="PCENTER", choices=["PCENTER", "PMED", "CFLP", "UFLP"])
    parser.add_argument("--instance_dir", type=str, default=None)
    parser.add_argument("--output_root_dir", type=str, default=None)
    parser.add_argument("--k_per_call", type=int, default=20)
    parser.add_argument("--n_target", type=int, default=50)

    args = parser.parse_args()
    problem_type = args.problem_type
    task_name = problem_type

    specs = _get_problem_specs(problem_type)
    desc = specs["task_description"]
    baseline_template = specs["baseline_template"]
    output_format = specs["output_format"]

    instance_dir = args.instance_dir or f"./step1_instance_creation/generated_data/{problem_type}"
    output_root_dir = args.output_root_dir or f"step2_contextualization/dataset/{problem_type}"

    llm = OpenAILlmService()
    prompter = PromptLoader()
    hint = _get_hint(problem_type)

    generate_dataset_for_task(
        task_name=task_name,
        task_description=desc,
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_instance_facility,
        llm=llm,
        prompter=prompter,
        hint=hint,
        baseline_template=baseline_template,
        output_format=output_format,
        k_per_call=args.k_per_call,
        n_target=args.n_target,
    )


if __name__ == "__main__":
    main()


"""
python -m step2_contextualization.contextualize_facility \
  --problem_type PCENTER \
  --instance_dir ./step1_instance_creation/generated_data/PCENTER \
  --output_root_dir step2_contextualization/dataset/PCENTER \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_facility \
  --problem_type PMED \
  --instance_dir ./step1_instance_creation/generated_data/PMED \
  --output_root_dir step2_contextualization/dataset/PMED \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_facility \
  --problem_type CFLP \
  --instance_dir ./step1_instance_creation/generated_data/CFLP \
  --output_root_dir step2_contextualization/dataset/CFLP \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_facility \
  --problem_type UFLP \
  --instance_dir ./step1_instance_creation/generated_data/UFLP \
  --output_root_dir step2_contextualization/dataset/UFLP \
  --k_per_call 20 \
  --n_target 50
"""
