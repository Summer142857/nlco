import json
import random
from typing import Dict, Any, List, Tuple

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader

"""
Contextualizer for routing problems:

- TSP    : Traveling Salesman Problem
- CVRP   : Capacitated Vehicle Routing Problem
- OP     : Orienteering / Prize-Collecting TSP (single route, max length)
- TOP    : Team Orienteering Problem (multiple routes)
- TSPTW  : TSP with Time Windows
- MLP    : Minimum Latency Problem
- PCTSP  : Prize-Collecting TSP
"""

FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]

def _make_sentence_from_desc(desc: str, value: Any) -> str:
    """
    Use scenario_hint.global_fields[i]['description'] to generate a natural sentence.

    Examples:
    - desc: "total number of customers"  -> "There are 10 customers."
    - desc: "identifier of the depot"   -> "The identifier of the depot is D0."
    - desc: "vehicle capacity"          -> "The vehicle capacity is 30."
    """
    if desc is None:
        desc = ""
    desc = desc.strip()
    if not desc:
        return f"The value is {value}."

    lower = desc.lower()

    # Pattern 1: "total number of X"
    prefix = "total number of "
    if isinstance(value, int) and lower.startswith(prefix):
        tail = desc[len(prefix):].strip()
        # e.g. "customers" -> "There are 10 customers."
        return f"There are {value} {tail}."

    # Pattern 2: "identifier of ..."
    id_prefix = "identifier of"
    if lower.startswith(id_prefix):
        # e.g. "identifier of the depot" -> "The identifier of the depot is D0."
        return f"The {desc} is {value}."

    # Generic fallback
    return f"The {desc} is {value}."

def _humanize_label(raw: str) -> str:
    if not isinstance(raw, str):
        return str(raw)
    label = raw
    if label.startswith("total_"):
        label = label[len("total_") :]
    label = label.replace("_", " ")
    return label



def _build_node_label_map(num_nodes: int, index_base) -> Dict[int, Any]:
    """
    Build mapping from internal node index (0..num_nodes-1) to human-friendly labels.
    The raw instances (solution, depot, etc.) use 0-based indices.
    """
    labels = make_labels(num_nodes, index_base)
    return {i: labels[i] for i in range(num_nodes)}


def _remap_solution_routing(
    solution: Any,
    problem_type: str,
    node_id_map: Dict[int, Any],
) -> Any:
    """
    Remap solution node indices to labels via node_id_map.

    - TSP / OP / TSPTW / MLP / PCTSP: single route [0, i1, ..., 0]
    - CVRP / TOP: list of routes [[0,...,0], [0,...,0], ...]
    """
    if solution is None:
        return None

    pt = problem_type.upper()

    def remap_route(route: List[int]) -> List[Any]:
        return [node_id_map.get(int(v), v) for v in route]

    # single-route type
    if pt in ["TSP", "OP", "TSPTW", "MLP", "PCTSP"]:
        if isinstance(solution, list) and solution and not isinstance(solution[0], list):
            return remap_route(solution)
        if isinstance(solution, list) and solution and isinstance(solution[0], list):
            return [remap_route(r) for r in solution]
        return solution

    # multi-route type
    if pt in ["CVRP", "TOP"]:
        if isinstance(solution, list) and solution and isinstance(solution[0], list):
            return [remap_route(r) for r in solution]
        if isinstance(solution, list) and solution and not isinstance(solution[0], list):
            return [remap_route(solution)]
        return solution

    return solution


def _build_labeled_routing_instance(
    inst_core: Any,
    node_id_map: Dict[int, Any],
    problem_type: str,
) -> Dict[str, Any]:
    """
    Build a 'labeled' variant of the routing instance where node ids
    are human-readable labels rather than numeric indices.
    """
    pt = problem_type.upper()

    if isinstance(inst_core, list):
        coordinates = inst_core
        extra = {}
    elif isinstance(inst_core, dict):
        coordinates = inst_core.get("coordinates", [])
        extra = inst_core
    else:
        coordinates = []
        extra = {}

    num_nodes = len(coordinates)
    nodes_variant: List[Dict[str, Any]] = []

    demands = extra.get("demands")
    prizes = extra.get("prizes")
    time_windows = extra.get("time_windows")
    penalties = extra.get("penalties")  # for PCTSP

    for i in range(num_nodes):
        x, y = coordinates[i]
        label = node_id_map.get(i, i)
        node_rec: Dict[str, Any] = {
            "id": label,
            "x": x,
            "y": y,
        }
        if demands is not None and i < len(demands):
            node_rec["demand"] = demands[i]
        if prizes is not None and i < len(prizes):
            node_rec["prize"] = prizes[i]
        if penalties is not None and i < len(penalties):
            node_rec["penalty"] = penalties[i]
        if time_windows is not None and i < len(time_windows):
            tw = time_windows[i]
            node_rec["tw_start"] = tw[0]
            node_rec["tw_end"] = tw[1]
        nodes_variant.append(node_rec)

    inst_variant: Dict[str, Any] = {
        "problem_type": problem_type,
        "num_nodes": num_nodes,
        "nodes": nodes_variant,
    }

    depot = extra.get("depot", 0)
    if depot is not None:
        inst_variant["depot"] = node_id_map.get(int(depot), depot)

    # problem-specific globals
    if pt == "CVRP":
        if "capacity" in extra:
            inst_variant["capacity"] = extra["capacity"]
        if "num_vehicles" in extra:
            inst_variant["num_vehicles"] = extra["num_vehicles"]
    elif pt == "OP":
        if "max_length" in extra:
            inst_variant["max_length"] = extra["max_length"]
    elif pt == "TOP":
        if "max_length_per_vehicle" in extra:
            inst_variant["max_length_per_vehicle"] = extra["max_length_per_vehicle"]
        if "n_vehicles" in extra:
            inst_variant["n_vehicles"] = extra["n_vehicles"]
    elif pt == "PCTSP":
        # extra PCTSP-specific fields
        if "required_prize" in extra:
            inst_variant["required_prize"] = extra["required_prize"]
        if "route_prize" in extra:
            inst_variant["route_prize"] = extra["route_prize"]
        if "unvisited_penalty" in extra:
            inst_variant["unvisited_penalty"] = extra["unvisited_penalty"]
    elif pt == "TSPTW":
        pass
    elif pt == "MLP":
        pass

    # generic objective-related keys
    for key in [
        "total_distance",
        "route_length",
        "total_route_length",
        "collected_prize",
        "objective",
    ]:
        if key in extra:
            inst_variant[key] = extra[key]

    return inst_variant




def routing_render_nl_with_style(
    inst_core: Any,
    node_id_map: Dict[int, Any],
    nl_style: Dict[str, Any],
    problem_type: str,
) -> str:
    line_t = nl_style.get("line_template")
    header_t = nl_style.get("header")
    footer_t = nl_style.get("footer")

    out: List[str] = []

    if isinstance(inst_core, list):
        coordinates = inst_core
        extra = {}
    else:
        coordinates = inst_core.get("coordinates", [])
        extra = inst_core

    num_nodes = len(coordinates)
    pt = problem_type.upper()

    base_ctx: Dict[str, Any] = {
        "num_nodes": num_nodes,
        "problem_type": problem_type,
    }

    depot = extra.get("depot", 0)
    if depot is not None:
        base_ctx["depot"] = node_id_map.get(int(depot), depot)

    if pt == "CVRP":
        base_ctx["capacity"] = extra.get("capacity")
    elif pt == "OP":
        base_ctx["max_length"] = extra.get("max_length")
    elif pt == "TOP":
        base_ctx["max_length_per_vehicle"] = extra.get("max_length_per_vehicle")
        base_ctx["n_vehicles"] = extra.get("n_vehicles")
    elif pt == "PCTSP":
        base_ctx["required_prize"] = extra.get("required_prize")
    elif pt == "TSPTW":
        pass
    elif pt == "MLP":
        pass

    demands = extra.get("demands")
    prizes = extra.get("prizes")
    time_windows = extra.get("time_windows")
    penalties = extra.get("penalties")

    # header
    if header_t:
        out.append(_format_safe(header_t, **base_ctx))

    for i in range(num_nodes):
        x, y = coordinates[i]
        label = node_id_map.get(i, i)
        ctx = {
            **base_ctx,
            "node_id": label,
            "x": x,
            "y": y,
            "demand": demands[i] if demands is not None and i < len(demands) else None,
            "prize": prizes[i] if prizes is not None and i < len(prizes) else None,
            "penalty": penalties[i] if penalties is not None and i < len(penalties) else None,
            "tw_start": time_windows[i][0] if time_windows is not None and i < len(time_windows) else None,
            "tw_end": time_windows[i][1] if time_windows is not None and i < len(time_windows) else None,
        }
        if line_t:
            out.append(_format_safe(line_t, **ctx))

    if footer_t:
        out.append(_format_safe(footer_t, **base_ctx))

    return "\n".join(out)


def routing_render_input(
    fmt: str,
    inst_core: Any,
    node_id_map: Dict[int, Any],
    nl_style: Dict[str, Any] | None,
    problem_type: str,
    scenario_hint: Dict[str, Any] | None = None,
) -> str:
    if isinstance(inst_core, list):
        coordinates = inst_core
        extra = {}
    else:
        coordinates = inst_core.get("coordinates", [])
        extra = inst_core

    num_nodes = len(coordinates)
    pt = problem_type.upper()

    # default field names
    num_nodes_key = "num_nodes"
    depot_key = "depot"
    capacity_key = "capacity"
    max_length_key = "max_length"
    max_length_per_vehicle_key = "max_length_per_vehicle"
    n_vehicles_key = "n_vehicles"
    required_prize_key = "required_prize"

    node_id_key = "node_id"
    x_key = "x"
    y_key = "y"
    demand_key = "demand"
    prize_key = "prize"
    penalty_key = "penalty"
    tw_start_key = "tw_start"
    tw_end_key = "tw_end"

    # apply scenario_hint renaming (like graph version)
    if scenario_hint and isinstance(scenario_hint, dict):
        for field in scenario_hint.get("global_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "num_nodes":
                num_nodes_key = new
            elif orig == "depot":
                depot_key = new
            elif orig == "capacity":
                capacity_key = new
            elif orig == "max_length":
                max_length_key = new
            elif orig == "max_length_per_vehicle":
                max_length_per_vehicle_key = new
            elif orig == "n_vehicles":
                n_vehicles_key = new
            elif orig == "required_prize":
                required_prize_key = new

        for field in scenario_hint.get("item_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "node_id":
                node_id_key = new
            elif orig == "x":
                x_key = new
            elif orig == "y":
                y_key = new
            elif orig == "demand":
                demand_key = new
            elif orig == "prize":
                prize_key = new
            elif orig == "penalty":
                penalty_key = new
            elif orig == "tw_start":
                tw_start_key = new
            elif orig == "tw_end":
                tw_end_key = new

    # build per-node records
    demands = extra.get("demands")
    prizes = extra.get("prizes")
    time_windows = extra.get("time_windows")
    penalties = extra.get("penalties")

    node_records: List[Dict[str, Any]] = []
    for i in range(num_nodes):
        x, y = coordinates[i]
        label = node_id_map.get(i, i)
        rec: Dict[str, Any] = {
            node_id_key: label,
            x_key: x,
            y_key: y,
        }
        if demands is not None and i < len(demands):
            rec[demand_key] = demands[i]
        if prizes is not None and i < len(prizes):
            rec[prize_key] = prizes[i]
        if penalties is not None and i < len(penalties):
            rec[penalty_key] = penalties[i]
        if time_windows is not None and i < len(time_windows):
            rec[tw_start_key] = time_windows[i][0]
            rec[tw_end_key] = time_windows[i][1]
        node_records.append(rec)

    # ---------------- JSON ----------------
    if fmt == "json":
        obj: Dict[str, Any] = {
            num_nodes_key: num_nodes,
            "nodes": node_records,
        }

        depot = extra.get("depot", 0)
        if depot is not None:
            obj[depot_key] = node_id_map.get(int(depot), depot)

        if pt == "CVRP":
            if "capacity" in extra:
                obj[capacity_key] = extra["capacity"]
        elif pt == "OP":
            if "max_length" in extra:
                obj[max_length_key] = extra["max_length"]
        elif pt == "TOP":
            if "max_length_per_vehicle" in extra:
                obj[max_length_per_vehicle_key] = extra["max_length_per_vehicle"]
            if "n_vehicles" in extra:
                obj[n_vehicles_key] = extra["n_vehicles"]
        elif pt == "TSPTW":
            # time windows already embedded per node
            pass
        elif pt == "PCTSP":
            if "required_prize" in extra:
                obj[required_prize_key] = extra["required_prize"]
        elif pt == "MLP":
            pass

        return json.dumps(obj, ensure_ascii=False, indent=2)

    # ---------------- CSV ----------------
    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_nodes_key}={num_nodes}")
        depot = extra.get("depot", 0)
        if depot is not None:
            lines.append(f"# {depot_key}={node_id_map.get(int(depot), depot)}")
        if pt == "CVRP":
            if "capacity" in extra:
                lines.append(f"# {capacity_key}={extra['capacity']}")
        elif pt == "OP":
            if "max_length" in extra:
                lines.append(f"# {max_length_key}={extra['max_length']}")
        elif pt == "TOP":
            if "max_length_per_vehicle" in extra:
                lines.append(
                    f"# {max_length_per_vehicle_key}={extra['max_length_per_vehicle']}"
                )
            if "n_vehicles" in extra:
                lines.append(f"# {n_vehicles_key}={extra['n_vehicles']}")
        elif pt == "TSPTW":
            lines.append("# time windows are listed per node in the table")
        elif pt == "PCTSP":
            if "required_prize" in extra:
                lines.append(f"# {required_prize_key}={extra['required_prize']}")

        # header row
        cols = [node_id_key, x_key, y_key]
        has_demand = any(demands) if demands is not None else False
        has_prize = any(prizes) if prizes is not None else False
        has_penalty = any(penalties) if penalties is not None else False
        has_tw = time_windows is not None

        if has_demand:
            cols.append(demand_key)
        if has_prize:
            cols.append(prize_key)
        if has_penalty:
            cols.append(penalty_key)
        if has_tw:
            cols.extend([tw_start_key, tw_end_key])

        lines.append(",".join(cols))

        for rec in node_records:
            row_vals = [rec.get(col, "") for col in cols]
            lines.append(",".join(str(v) for v in row_vals))

        return "\n".join(lines)

    # -------------- Markdown table --------------
    if fmt == "markdown_table":
        ctx = {
            "num_nodes": num_nodes,
            "problem_type": problem_type,
            "depot": node_id_map.get(int(extra.get("depot", 0)), extra.get("depot", 0)),

            # PCTSP fields
            "required_prize": extra.get("required_prize"),
            "route_prize": extra.get("route_prize"),
            "unvisited_penalty": extra.get("unvisited_penalty"),
            "objective": extra.get("objective"),
            "route_length": extra.get("route_length"),

            # OP / TOP / CVRP etc.
            "capacity": extra.get("capacity"),
            "max_length": extra.get("max_length"),
            "max_length_per_vehicle": extra.get("max_length_per_vehicle"),
            "n_vehicles": extra.get("n_vehicles"),
        }

        # ---- 1) NL header ----
        nl_header = ""
        if nl_style and nl_style.get("header"):
            nl_header = _format_safe(nl_style["header"], **ctx)

        lines = []
        if nl_header:
            lines.append(nl_header)
            lines.append("")

        # ---- 2) Markdown Table ----
        cols = [node_id_key, x_key, y_key]
        if demands: cols.append(demand_key)
        if prizes: cols.append(prize_key)
        if penalties: cols.append(penalty_key)
        if time_windows: cols.extend([tw_start_key, tw_end_key])

        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")

        for rec in node_records:
            row_vals = [rec.get(col, "") for col in cols]
            lines.append("| " + " | ".join(str(v) for v in row_vals) + " |")

        # ---- 3) NL footer ----
        nl_footer = ""
        if nl_style and nl_style.get("footer"):
            nl_footer = _format_safe(nl_style["footer"], **ctx)

        if nl_footer:
            lines.append("")
            lines.append(nl_footer)

        return "\n".join(lines)

    # -------------- NL --------------
    if fmt == "nl" and isinstance(nl_style, dict):
        return routing_render_nl_with_style(
            inst_core=inst_core,
            node_id_map=node_id_map,
            nl_style=nl_style,
            problem_type=problem_type,
        )

    # fallback
    return ""




def contextualize_instance_routing(
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

    # coordinates & num_nodes
    if isinstance(inst_core, list):
        coordinates = inst_core
    elif isinstance(inst_core, dict):
        coordinates = inst_core.get("coordinates", [])
    else:
        coordinates = []

    num_nodes = len(coordinates)
    if num_nodes == 0:
        return results

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

    # node index -> label
    node_id_map = _build_node_label_map(num_nodes, index_base)

    solution_variant = _remap_solution_routing(
        solution=solution,
        problem_type=problem_type,
        node_id_map=node_id_map,
    )

    input_text = routing_render_input(
        fmt=fmt,
        inst_core=inst_core,
        node_id_map=node_id_map,
        nl_style=nl_style,
        problem_type=problem_type,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_routing_instance(
        inst_core=inst_core,
        node_id_map=node_id_map,
        problem_type=problem_type,
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


# ----------------------------------------------------------------------
# 4. Problem specs
# ----------------------------------------------------------------------


def _get_problem_specs(problem_type: str) -> Dict[str, str]:
    pt = problem_type.upper()

    if pt == "TSP":
        desc = (
            "Traveling Salesman Problem (TSP): Given a set of locations with coordinates, "
            "the goal is to find a minimum-length tour that starts and ends at the depot and "
            "visits every other location exactly once."
        )
        baseline_template = (
            "You are given a Traveling Salesman Problem (TSP).\n"
            "The input describes a set of locations, each with coordinates. "
            "One location is the depot, where the tour must start and end.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to propose a single tour that starts at the depot, visits every other location "
            "exactly once, and returns to the depot, aiming to minimize total distance.\n\n"
            "Return your decision in **exactly** this JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
            "Here, `solution` is the sequence of node identifiers in the order you visit them, "
            "starting and ending at the depot. Use **exactly** the same node identifiers that appear in the input. "
            "Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
        )

    elif pt == "CVRP":
        desc = (
            "Capacitated Vehicle Routing Problem (CVRP): Given a depot, customer locations with demands, "
            "and a vehicle capacity, the goal is to choose a set of vehicle routes such that every customer's "
            "demand is served exactly once. Each route must start and end at the depot, and the total demand "
            "on any route must not exceed the vehicle capacity, while minimizing the total travel distance."
        )

        baseline_template = (
            "You are given a Capacitated Vehicle Routing Problem (CVRP).\n"
            "The input describes a depot, a set of customer locations with demands, and a vehicle capacity. "
            "Each route must start and end at the depot.\n"
            "**Each customer must be visited exactly once and must appear in exactly one route.**\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to propose a set of routes "
            "starting and ending at the depot. The sum of customer demands on any route must not exceed the "
            "vehicle capacity.\n\n"
            "The objective is to minimize the total travel distance across all routes.\n\n"
            "Return your decision in **exactly** this JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [[depot_id, location_id, ... , depot_id], [depot_id, location_id, ... , depot_id], ...]\n'
            "}\n"
            "```\n"
            "Here, each inner list is a single vehicle's route. Use exactly the same node identifiers as in the input. "
            "Do not include explanations or extra keys."
        )

        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [[depot_id, location_id, ... , depot_id], [depot_id, location_id, ... , depot_id], ...]\n'
            "}\n"
            "```\n"
        )

    elif pt == "OP":
        desc = (
            "Orienteering Problem (OP) / Prize-Collecting TSP: Given a depot, locations with associated prizes, "
            "and a maximum route length, the goal is to choose a single tour that starts and ends at the depot, "
            "does not exceed the length limit, and maximizes the total collected prize. "
            "Each non-depot location may be visited at most once."
        )
        baseline_template = (
            "You are given an Orienteering Problem (OP) / Prize-Collecting TSP instance.\n"
            "The input describes a depot, a set of locations with coordinates and associated prizes, and a maximum "
            "allowed tour length.\n"
            "**Each non-depot location may be visited at most once.**\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to propose a single tour that starts and ends at the depot, "
            "does not exceed the maximum route length, and maximizes the sum of prizes collected from the visited nodes.\n\n"
            "Return your decision in **exactly** this JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
            "Here, `solution` is the sequence of node identifiers in visiting order, starting and ending at the depot. "
            "Use exactly the same node identifiers as in the input. Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
        )

    elif pt == "TOP":
        desc = (
            "Team Orienteering Problem (TOP): Given a depot, locations with prizes, a number of vehicles, "
            "and a maximum route length per vehicle, the goal is to choose multiple routes (one per vehicle) "
            "starting and ending at the depot, each not exceeding the length limit, so as to maximize the total "
            "collected prize over all visited locations. Each non-depot location may be visited at most once "
            "across all routes."
        )
        baseline_template = (
            "You are given a Team Orienteering Problem (TOP) instance.\n"
            "The input describes a depot, a set of locations with coordinates and prizes, a number of vehicles, "
            "and a maximum route length per vehicle.\n"
            "**Each non-depot location may be visited at most once across all routes.**\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to propose a set of routes (one per vehicle), where each route starts and ends at the depot, "
            "does not exceed the maximum length, and together they maximize the total collected prize.\n\n"
            "Return your decision in **exactly** this JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [[depot_id, location_id, ... , depot_id], [depot_id, location_id, ... , depot_id], ...]\n'
            "}\n"
            "```\n"
            "Each inner list represents a single vehicle's route. Use exactly the same node identifiers as in the input. "
            "Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [[depot_id, location_id, ... , depot_id], [depot_id, location_id, ... , depot_id], ...]\n'
            "}\n"
            "```\n"
        )

    elif pt == "TSPTW":
        desc = (
            "Traveling Salesman Problem with Time Windows (TSPTW): Given a depot and locations each with a time window, "
            "the goal is to find a tour that starts and ends at the depot, visits every location exactly once, "
            "and respects the time windows while minimizing total travel distance (waiting before a time window opens is allowed)."
        )
        baseline_template = (
            "You are given a Traveling Salesman Problem with Time Windows (TSPTW).\n"
            "The input describes a depot and a set of locations, each with coordinates and a time window [start, end]. "
            "The tour must start and end at the depot, visit each location exactly once, and may wait at nodes until "
            "their time window opens.\n\n"
            "The objective is to minimize total travel distance (waiting before a time window opens is allowed)."
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to propose a single tour that starts and ends at the depot, visits every location exactly once, "
            "and respects all time windows.\n\n"
            "Return your decision in **exactly** this JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
            "Here, `solution` is the visiting order of node identifiers, starting and ending at the depot. "
            "Use exactly the same node identifiers as in the input. Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
        )

    elif pt == "MLP":
        desc = (
            "Minimum Latency Problem (MLP): Given a depot and a set of locations with coordinates, "
            "the goal is to find a tour that starts at the depot and visits all locations (typically returning "
            "to the depot), minimizing the sum of arrival times (latencies) at all locations."
        )
        baseline_template = (
            "You are given a Minimum Latency Problem (MLP).\n"
            "The input describes a depot and a set of locations, each with coordinates. "
            "A single vehicle starts at the depot and must visit every location. "
            "The quality of a tour is measured by the sum of arrival times (latencies) at all locations.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to propose a single tour that starts at the depot, visits every location "
            "(and typically returns to the depot), aiming to minimize the sum of arrival times.\n\n"
            "Return your decision in **exactly** this JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
            "Here, `solution` is the sequence of node identifiers in visiting order, starting at the depot "
            "and usually returning to it. Use **exactly** the same node identifiers that appear in the input. "
            "Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
        )

    else:
        desc = (
            "Prize-Collecting Traveling Salesman Problem (PCTSP): Given a depot and locations each with a prize "
            "and a penalty for not visiting, plus a required minimum total prize, the goal is to choose a single tour "
            "that starts and ends at the depot, collects at least the required prize, and minimizes an objective that "
            "combines travel cost with penalties for unvisited locations."
        )
        baseline_template = (
            "You are given a Prize-Collecting Traveling Salesman Problem (PCTSP) instance.\n"
            "The input describes a depot and a set of locations, each with coordinates, a prize for visiting, "
            "and a penalty if it is not visited. There is also a required minimum total prize that must be collected.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Your task is to propose a single tour that starts and ends at the depot, collects at least the required "
            "total prize, and aims to minimize the overall objective (which combines tour length with penalties for "
            "unvisited locations).\n\n"
            "Return your decision in **exactly** this JSON format:\n"
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
            "Here, `solution` is the visiting order of node identifiers, starting and ending at the depot. "
            "Use exactly the same node identifiers as in the input. Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": [depot_id, location_id, ..., depot_id]\n'
            "}\n"
            "```\n"
        )

    return {
        "task_description": desc,
        "baseline_template": baseline_template,
        "output_format": output_format,
    }


# ----------------------------------------------------------------------
# 5. Hints for NL template generation
# ----------------------------------------------------------------------


def _get_hint(problem_type: str) -> Dict[str, Any]:
    pt = problem_type.upper()

    if pt == "TSP":
        global_fields = [
            {"name": "num_nodes", "description": "total number of locations"},
            {"name": "depot", "description": "identifier of the depot node"},
        ]
        item_fields = [
            {"name": "node_id", "description": "identifier of a location"},
            {"name": "x", "description": "x coordinate (or longitude-like)"},
            {"name": "y", "description": "y coordinate (or latitude-like)"},
        ]
        allowed = ["num_nodes", "depot", "node_id", "x", "y"]

    elif pt == "CVRP":
        global_fields = [
            {"name": "num_nodes", "description": "total number of locations (including depot)"},
            {"name": "depot", "description": "identifier of the depot node"},
            {"name": "capacity", "description": "vehicle capacity"},
        ]
        item_fields = [
            {"name": "node_id", "description": "identifier of a location"},
            {"name": "x", "description": "x coordinate"},
            {"name": "y", "description": "y coordinate"},
            {"name": "demand", "description": "demand at this location"},
        ]
        allowed = ["num_nodes", "depot", "capacity", "node_id", "x", "y", "demand"]

    elif pt == "OP":
        global_fields = [
            {"name": "num_nodes", "description": "total number of locations (including depot)"},
            {"name": "depot", "description": "identifier of the depot node"},
            {"name": "max_length", "description": "maximum allowed tour length"},
        ]
        item_fields = [
            {"name": "node_id", "description": "identifier of a location"},
            {"name": "x", "description": "x coordinate"},
            {"name": "y", "description": "y coordinate"},
            {"name": "prize", "description": "prize collected by visiting this location"},
        ]
        allowed = ["num_nodes", "depot", "max_length", "node_id", "x", "y", "prize"]

    elif pt == "TOP":
        global_fields = [
            {"name": "num_nodes", "description": "total number of locations (including depot)"},
            {"name": "depot", "description": "identifier of the depot node"},
            {"name": "max_length_per_vehicle", "description": "maximum route length each vehicle may travel"},
            {"name": "n_vehicles", "description": "number of vehicles"},
        ]
        item_fields = [
            {"name": "node_id", "description": "identifier of a location"},
            {"name": "x", "description": "x coordinate"},
            {"name": "y", "description": "y coordinate"},
            {"name": "prize", "description": "prize collected by visiting this location"},
        ]
        allowed = [
            "num_nodes",
            "depot",
            "max_length_per_vehicle",
            "n_vehicles",
            "node_id",
            "x",
            "y",
            "prize",
        ]

    elif pt == "TSPTW":
        global_fields = [
            {"name": "num_nodes", "description": "total number of locations (including depot)"},
            {"name": "depot", "description": "identifier of the depot node"},
        ]
        item_fields = [
            {"name": "node_id", "description": "identifier of a location"},
            {"name": "x", "description": "x coordinate"},
            {"name": "y", "description": "y coordinate"},
            {"name": "tw_start", "description": "earliest allowed arrival time"},
            {"name": "tw_end", "description": "latest allowed arrival time"},
        ]
        allowed = ["num_nodes", "depot", "node_id", "x", "y", "tw_start", "tw_end"]

    elif pt == "MLP":
        global_fields = [
            {"name": "num_nodes", "description": "total number of locations"},
            {"name": "depot", "description": "identifier of the depot node"},
        ]
        item_fields = [
            {"name": "node_id", "description": "identifier of a location"},
            {"name": "x", "description": "x coordinate"},
            {"name": "y", "description": "y coordinate"},
        ]
        allowed = ["num_nodes", "depot", "node_id", "x", "y"]

    else:  # PCTSP
        global_fields = [
            {"name": "num_nodes", "description": "total number of locations (including depot)"},
            {"name": "depot", "description": "identifier of the depot node"},
            {"name": "required_prize", "description": "minimum total prize that must be collected"},
        ]
        item_fields = [
            {"name": "node_id", "description": "identifier of a location"},
            {"name": "x", "description": "x coordinate"},
            {"name": "y", "description": "y coordinate"},
            {"name": "prize", "description": "prize collected by visiting this location"},
            {"name": "penalty", "description": "penalty if this location is not visited"},
        ]
        allowed = [
            "num_nodes",
            "depot",
            "required_prize",
            "node_id",
            "x",
            "y",
            "prize",
            "penalty",
        ]

    return {
        "global_fields": global_fields,
        "item_fields": item_fields,
        "allowed_placeholders": allowed,
    }


# ----------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------

def run_routing_step2(
    problem_type: str,
    instance_dir: str,
    output_root_dir: str,
    k_per_call: int = 20,
    n_target: int = 50,
    scale: str | None = None,
    load_cached_contexts: bool = True,
    contexts_audit_path: str | None = None,
) -> str:

    problem_type = problem_type.upper()
    task_name = problem_type

    specs = _get_problem_specs(problem_type)
    desc = specs["task_description"]
    baseline_template = specs["baseline_template"]
    output_format = specs["output_format"]

    llm = OpenAILlmService()
    prompter = PromptLoader()
    hint = _get_hint(problem_type)

    generate_dataset_for_task(
        task_name=task_name,
        task_description=desc,
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_instance_routing,
        llm=llm,
        prompter=prompter,
        hint=hint,
        baseline_template=baseline_template,
        output_format=output_format,
        k_per_call=int(k_per_call),
        n_target=int(n_target),
        scale=scale,
        load_cached_contexts=load_cached_contexts,
        contexts_audit_path=contexts_audit_path,
    )
    return str(output_root_dir)



def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Contextualize routing instances (TSP, CVRP, OP, TOP, TSPTW, MLP, PCTSP) into NL/JSON/CSV/Markdown."
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="TSP",
        choices=["TSP", "CVRP", "OP", "TOP", "TSPTW", "MLP", "PCTSP"],
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        default=None,
        help="Directory containing raw JSON instances (default: ./generated_data/<problem_type>).",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=None,
        help="Root directory to save contextualized data (default: step2_contextualization/dataset/<problem_type>).",
    )
    parser.add_argument(
        "--k_per_call",
        type=int,
        default=20,
        help="How many contextualized examples to generate per LLM call.",
    )
    parser.add_argument(
        "--n_target",
        type=int,
        default=50,
        help="Total number of contextualized examples to generate (per split).",
    )

    args = parser.parse_args()
    problem_type = args.problem_type
    task_name = problem_type

    specs = _get_problem_specs(problem_type)
    desc = specs["task_description"]
    baseline_template = specs["baseline_template"]
    output_format = specs["output_format"]

    if args.instance_dir is not None:
        instance_dir = args.instance_dir
    else:
        instance_dir = f"./generated_data/{problem_type}"

    if args.output_root_dir is not None:
        output_root_dir = args.output_root_dir
    else:
        output_root_dir = f"step2_contextualization/dataset/{problem_type}"

    llm = OpenAILlmService()
    prompter = PromptLoader()

    hint = _get_hint(problem_type)

    generate_dataset_for_task(
        task_name=task_name,
        task_description=desc,
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_instance_routing,
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




'''
python -m step2_contextualization.contextualize_routing \
  --problem_type TSPTW \
  --instance_dir ./step1_instance_creation/generated_data/TSPTW \
  --output_root_dir step2_contextualization/dataset/TSPTW \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_routing \
  --problem_type CVRP \
  --instance_dir ./step1_instance_creation/generated_data/CVRP \
  --output_root_dir step2_contextualization/dataset/CVRP \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_routing \
  --problem_type OP \
  --instance_dir ./step1_instance_creation/generated_data/OP \
  --output_root_dir step2_contextualization/dataset/OP \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_routing \
  --problem_type TSP \
  --instance_dir ./step1_instance_creation/generated_data/TSP \
  --output_root_dir step2_contextualization/dataset/TSP \
  --k_per_call 20 \
  --n_target 50

python -m step2_contextualization.contextualize_routing \
  --problem_type TOP \
  --instance_dir ./step1_instance_creation/generated_data/TOP \
  --output_root_dir step2_contextualization/dataset/TOP \
  --k_per_call 20 \
  --n_target 50
python -m step2_contextualization.contextualize_routing \
  --problem_type PCTSP \
  --instance_dir ./step1_instance_creation/generated_data/PCTSP \
  --output_root_dir step2_contextualization/dataset/PCTSP \
  --k_per_call 20 \
  --n_target 50
  
python -m step2_contextualization.contextualize_routing \
  --problem_type MLP \
  --instance_dir ./step1_instance_creation/generated_data/MLP \
  --output_root_dir step2_contextualization/dataset/MLP \
  --k_per_call 20 \
  --n_target 50
'''
