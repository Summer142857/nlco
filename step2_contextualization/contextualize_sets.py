import json
import random
from typing import Dict, Any, List, Optional

from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
from step2_contextualization.utils.dataset_schema import build_output_record
from step2_contextualization.utils.nl_template_utils import _format_safe, make_labels
from step2_contextualization.utils.llm_service import OpenAILlmService
from step2_contextualization.utils.prompt_loader import PromptLoader


FORMATS = ["nl", "csv", "json", "markdown_table"]
INDEX_BASES = [0, 1, "names"]


def _remap_solution_sets(
    solution: list | None,
    problem_type: str,
    element_labels: list,
    set_labels: list,
) -> list:
    if not solution:
        return []

    elem_map = {i + 1: element_labels[i] for i in range(len(element_labels))}
    set_map = {j + 1: set_labels[j] for j in range(len(set_labels))}

    pt = problem_type.upper()

    if pt == "HSP":
        return [elem_map.get(int(s), s) for s in solution]
    else:
        return [set_map.get(int(s), s) for s in solution]



def _build_labeled_set_system(
    inst_core: Dict[str, Any],
    element_labels: List[Any],
    set_labels: List[Any],
) -> Dict[str, Any]:
    num_elements = inst_core.get("num_elements")
    num_sets = inst_core.get("num_sets")
    sets = inst_core.get("sets", [])
    budget_k = inst_core.get("budget_k", None)

    elem_map = {i + 1: element_labels[i] for i in range(len(element_labels))}
    set_map = {j + 1: set_labels[j] for j in range(len(set_labels))}

    sets_variant = []
    for S in sets:
        sid = S["id"]
        elems = S["elements"]
        cost = S.get("cost")
        S_out = {
            "id": set_map.get(int(sid), sid),
            "elements": [elem_map.get(int(e), e) for e in elems],
        }
        if cost is not None:
            S_out["cost"] = cost
        sets_variant.append(S_out)

    inst_variant = {
        "num_elements": num_elements,
        "num_sets": num_sets,
        "sets": sets_variant,
    }
    if budget_k is not None:
        inst_variant["budget_k"] = budget_k

    return inst_variant


def set_render_nl_with_style(
    inst_core: Dict[str, Any],
    element_labels: List[Any],
    set_labels: List[Any],
    nl_style: Dict[str, Any],
    problem_type: str,
) -> str:
    line_t = nl_style.get("line_template")
    header_t = nl_style.get("header")
    footer_t = nl_style.get("footer")

    out: List[str] = []

    num_elements = inst_core.get("num_elements")
    num_sets = inst_core.get("num_sets")
    budget_k = inst_core.get("budget_k", None)

    elem_map = {i + 1: element_labels[i] for i in range(len(element_labels))}
    set_map = {j + 1: set_labels[j] for j in range(len(set_labels))}

    base_ctx = {
        "num_elements": num_elements,
        "num_sets": num_sets,
        "budget_k": budget_k,
        "problem_type": problem_type,
    }

    if header_t:
        out.append(_format_safe(header_t, **base_ctx))

    sets = inst_core.get("sets", [])
    for S in sets:
        sid = S["id"]
        elems = S["elements"]
        cost = S.get("cost")
        set_label = set_map.get(int(sid), sid)
        elem_labels_line = [elem_map.get(int(e), e) for e in elems]
        ctx = {
            **base_ctx,
            "set_id": set_label,
            "elements": " ".join(str(x) for x in elem_labels_line),
            "cost": cost,
        }
        if line_t:
            out.append(_format_safe(line_t, **ctx))

    if footer_t:
        out.append(_format_safe(footer_t, **base_ctx))

    return "\n".join(out)


def set_render_input(
    fmt: str,
    inst_core: Dict[str, Any],
    element_labels: List[Any],
    set_labels: List[Any],
    nl_style: Dict[str, Any] | None,
    problem_type: str,
    scenario_hint: Dict[str, Any] | None = None,
) -> str:
    num_elements = inst_core.get("num_elements")
    num_sets = inst_core.get("num_sets")
    budget_k = inst_core.get("budget_k", None)
    sets = inst_core.get("sets", [])

    # -----------------------------
    # default keys (global + item)
    # -----------------------------
    num_elements_key = "num_elements"
    num_sets_key = "num_sets"
    budget_k_key = "budget_k"
    sets_key = "sets"

    # item-level keys
    item_id_key = "id"          # set identifier field in each row/object
    elements_key = "elements"   # list of element labels
    cost_key = "cost"           # optional

    # -----------------------------
    # apply scenario_hint renaming
    # -----------------------------
    if scenario_hint and isinstance(scenario_hint, dict):
        # global fields
        for field in scenario_hint.get("global_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue
            if orig == "num_elements":
                num_elements_key = new
            elif orig == "num_sets":
                num_sets_key = new
            elif orig == "budget_k":
                budget_k_key = new
            elif orig == "sets":
                sets_key = new

        # item fields (support both set_id and id as "identifier")
        for field in scenario_hint.get("item_fields", []):
            orig = field.get("name")
            new = field.get("new_name")
            if not new:
                continue

            if orig in ("set_id", "id"):
                item_id_key = new
            elif orig == "elements":
                elements_key = new
            elif orig == "cost":
                cost_key = new

    # -----------------------------
    # label remap
    # -----------------------------
    elem_map = {i + 1: element_labels[i] for i in range(len(element_labels))}
    set_map = {j + 1: set_labels[j] for j in range(len(set_labels))}

    labeled_sets: List[Dict[str, Any]] = []
    for S in sets:
        sid = S["id"]
        elems = S["elements"]
        cost = S.get("cost")

        S_out = {
            item_id_key: set_map.get(int(sid), sid),
            elements_key: [elem_map.get(int(e), e) for e in elems],
        }
        if cost is not None:
            S_out[cost_key] = cost
        labeled_sets.append(S_out)

    # ============= JSON =============
    if fmt == "json":
        obj: Dict[str, Any] = {
            num_elements_key: num_elements,
            num_sets_key: num_sets,
            sets_key: labeled_sets,
        }
        if budget_k is not None:
            obj[budget_k_key] = budget_k
        return json.dumps(obj, ensure_ascii=False, indent=2)

    # ============= CSV =============
    if fmt == "csv":
        lines: List[str] = []
        lines.append(f"# {num_elements_key}={num_elements}")
        lines.append(f"# {num_sets_key}={num_sets}")
        if budget_k is not None:
            lines.append(f"# {budget_k_key}={budget_k}")

        has_cost = any(cost_key in S for S in labeled_sets)
        if has_cost:
            lines.append(f"{item_id_key},{cost_key},{elements_key}")
        else:
            lines.append(f"{item_id_key},{elements_key}")

        for S in labeled_sets:
            sid = S[item_id_key]
            elems_str = " ".join(str(e) for e in S[elements_key])
            if has_cost:
                cost = S.get(cost_key, "")
                lines.append(f"{sid},{cost},{elems_str}")
            else:
                lines.append(f"{sid},{elems_str}")
        return "\n".join(lines)

    # ============= Markdown table =============
    if fmt == "markdown_table":
        lines: List[str] = []

        lines.append(f"- **{num_elements_key}**: {num_elements}")
        lines.append(f"- **{num_sets_key}**: {num_sets}")
        if budget_k is not None:
            lines.append(f"- **{budget_k_key}**: {budget_k}")
        lines.append("")

        has_cost = any(cost_key in S for S in labeled_sets)

        if has_cost:
            lines.append(f"| {item_id_key} | {cost_key} | {elements_key} |")
            lines.append("|---|---|---|")
        else:
            lines.append(f"| {item_id_key} | {elements_key} |")
            lines.append("|---|---|")

        for S in labeled_sets:
            sid = S.get(item_id_key, "")
            elems_str = " ".join(str(e) for e in S.get(elements_key, []))
            if has_cost:
                lines.append(f"| {sid} | {S.get(cost_key,'')} | {elems_str} |")
            else:
                lines.append(f"| {sid} | {elems_str} |")

        return "\n".join(lines)

    # ============= NL =============
    if fmt == "nl" and isinstance(nl_style, dict):
        return set_render_nl_with_style(
            inst_core=inst_core,
            element_labels=element_labels,
            set_labels=set_labels,
            nl_style=nl_style,
            problem_type=problem_type,
        )

    return ""


def contextualize_instance_sets(
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
    solution = [x['id'] for x in solution]
    obj = inst.get("obj")
    problem_type = inst.get("problem_type", task_name)

    if inst_core is None:
        return results

    num_elements = inst_core.get("num_elements")
    num_sets = inst_core.get("num_sets")

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

    element_labels = make_labels(num_elements, index_base)
    set_labels = [f"S{j + 1}" for j in range(num_sets)]

    solution_variant = _remap_solution_sets(
        solution=solution,
        problem_type=problem_type,
        element_labels=element_labels,
        set_labels=set_labels,
    )

    input_text = set_render_input(
        fmt=fmt,
        inst_core=inst_core,
        element_labels=element_labels,
        set_labels=set_labels,
        nl_style=nl_style,
        problem_type=problem_type,
        scenario_hint=scenario_hint,
    )

    full_instruction = template.replace("{{INSTANCE_INPUT}}", input_text)

    instance_variant = _build_labeled_set_system(
        inst_core=inst_core,
        element_labels=element_labels,
        set_labels=set_labels,
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


from typing import Dict


def _get_problem_specs(problem_type: str) -> Dict[str, str]:
    pt = problem_type.upper()

    if pt == "SCP":
        desc = (
            "Set Cover Problem (SCP): We are given a universe of elements and a collection of sets. "
            "Each set is a subset of the universe. The goal is to choose a minimum number of sets "
            "such that every element is contained in at least one chosen set."
        )
        baseline_template = (
            "You are given a Set Cover Problem (SCP).\n\n"
            "You have a list of elements that must all be covered. You also have many candidate sets, and each set covers "
            "some of the elements. By choosing a set, you cover all elements inside it.\n\n"
            "Your goal:\n"
            "Pick as few sets as possible while still ensuring that every element in the universe is covered by at least one "
            "chosen set.\n\n"
            "Practical interpretation:\n"
            "Think of elements as requirements, and sets as bundles that satisfy some requirements. You want the smallest "
            "collection of bundles that collectively satisfies everything.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  "solution": ["set_id", ...]\n'
            "}\n"
            "```\n"
            "\"solution\" is the list of set IDs you choose. Use exactly the same set identifiers that appear in the input.\n"
            "Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": ["set_id", ...]\n'
            "}\n"
            "```\n"
        )


    elif pt == "MKC":

        desc = (
            "Maximum Coverage Problem (MkC): We are given a universe of elements and a collection of sets. "
            "There is a limit on how many sets may be selected. The objective is to choose a subset of sets, "
            "within this limit, so that the total number of distinct covered elements is maximized."
        )

        baseline_template = (
            "You are given a Maximum Coverage Problem (MkC).\n\n"
            "You have a list of elements you would like to cover, and many candidate sets, each covering some elements. "
            "However, there is a strict limit on how many sets you are allowed to choose.\n\n"
            "Your goal:\n"
            "Choose a subset of sets, not exceeding the allowed number specified in the input, "
            "so that the number of *distinct* elements covered (the union of chosen sets) is as large as possible.\n\n"
            "Notes:\n"
            "- You are not required to use the full allowance; selecting fewer sets is permitted.\n"
            "- Overlaps do not count multiple times: an element covered by several chosen sets is still counted only once.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  \"solution\": [\"set_id\", ...]\n'
            "}\n"
            "```\n"
            "\"solution\" is the list of set IDs you decide to pick. "
            "Do not select more sets than the allowed limit given in the input. "
            "Use exactly the same set identifiers that appear in the input.\n"
            "Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  \"solution\": [\"set_id\", ...]\n'
            "}\n"
            "```\n"
        )


    elif pt == "SP":
        desc = (
            "Set Packing Problem (SP): We are given a universe of elements and a collection of sets. "
            "Each set is a subset of the universe. The goal is to select as many sets as possible such that "
            "no element appears in more than one selected set (pairwise disjointness)."
        )
        baseline_template = (
            "You are given a Set Packing Problem (SP).\n\n"
            "You have many candidate sets, each containing some elements. You want to pick several sets, but with a strict rule: "
            "no element is allowed to appear in two different chosen sets.\n\n"
            "Your goal:\n"
            "Select a collection of pairwise disjoint sets (no overlaps) and make this collection as large as possible "
            "(maximize how many sets you managed to pick).\n\n"
            "Practical interpretation:\n"
            "Each element is a resource that can be used at most once. Each set is a package of resources. "
            "You want to accept as many packages as possible without reusing any resource.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  "solution": ["set_id", ...]\n'
            "}\n"
            "```\n"
            "\"solution\" is the list of set IDs you select. Use exactly the same set identifiers that appear in the input.\n"
            "Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": ["set_id", ...]\n'
            "}\n"
            "```\n"
        )

    elif pt == "HSP":
        desc = (
            "Hitting Set Problem (HSP): We are given a collection of sets defined over a common universe of elements. "
            "Each set represents a constraint that must be 'hit'. The task is to choose a minimum-size subset of "
            "elements from the universe such that every set contains at least one chosen element. "
            "The objective is to minimize the number of selected elements. "
            "Note that in HSP we select elements (not sets), and a solution is valid if it intersects every given set."
        )

        baseline_template = (
            "You are given a Hitting Set Problem (HSP).\n\n"
            "You have many sets, and you want to pick individual elements (not sets) so that every set is \"hit\"—meaning "
            "each set contains at least one element you picked.\n\n"
            "Your goal:\n"
            "Pick as few elements as possible while ensuring that every given set contains at least one selected element.\n\n"
            "Practical interpretation:\n"
            "Think of each set as a constraint group. Selecting an element satisfies (hits) all sets that contain it. "
            "You want the smallest group of elements that collectively touches all sets.\n\n"
            "You will receive the instance in the following format:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (structure-wise):\n"
            "```json\n"
            "{\n"
            '  "solution": ["element_id", ...]\n'
            "}\n"
            "```\n"
            "\"solution\" is the list of element IDs you choose. Use exactly the same element identifiers that appear in the input.\n"
            "Do not include explanations or extra keys."
        )
        output_format = (
            "```json\n"
            "{\n"
            '  "solution": ["element_id", ...]\n'
            "}\n"
            "```\n"
        )

    else:  # SPP
        desc = (
            "Set Partitioning Problem (SPP): We are given a universe of elements and a collection of candidate sets. "
            "Each set covers some elements and may have an associated cost. "
            "We must select a subset of sets such that every element is covered by exactly one chosen set (a partition), "
            "while minimizing the total cost of the chosen sets."
        )

        baseline_template = (
            "You are given a Set Partitioning Problem (SPP).\n\n"
            "You have a universe of elements and many candidate sets. Each set covers some elements and may have a cost.\n\n"
            "Constraints (must all hold):\n"
            "- Exact cover / partition: every element must appear in exactly one chosen set.\n"
            "  (equivalently: cover all elements, and chosen sets must be disjoint w.r.t. elements.)\n\n"
            "Objective:\n"
            "- Minimize the total cost of the chosen sets (sum of costs for all selected set IDs).\n"
            "  If the instance provides a cost for each set, use it.\n\n"
            "You will receive the instance below:\n"
            "{{INSTANCE_INPUT}}\n\n"
            "Return your decision in **exactly** this JSON format (no extra keys, no explanations):\n"
            "```json\n"
            "{\n"
            '  "solution": ["set_id", ...]\n'
            "}\n"
            "```\n"
            "\"solution\" is the list of set IDs you choose.\n"
            "Use exactly the same set identifiers that appear in the input."
        )

        output_format = (
            "```json\n"
            "{\n"
            '  "solution": ["set_id", ...]\n'
            "}\n"
            "```\n"
        )

    return {
        "task_description": desc,
        "baseline_template": baseline_template,
        "output_format": output_format,
    }



def _get_hint(problem_type: str) -> Dict[str, Any]:
    pt = problem_type.upper()

    if pt == "SCP":
        global_fields = [
            {"name": "num_elements", "description": "total number of elements that must be covered"},
            {"name": "num_sets", "description": "total number of available sets"},
        ]
        item_fields = [
            {"name": "set_id", "description": "identifier of a set"},
            {"name": "elements", "description": "space-separated element identifiers in this set"},
        ]
        allowed = ["num_elements", "num_sets", "set_id", "elements"]

    elif pt == "MKC":
        global_fields = [
            {"name": "num_elements", "description": "total number of elements"},
            {"name": "num_sets", "description": "total number of available sets"},
            {"name": "budget_k", "description": "maximum number of sets allowed in the solution"},
        ]
        item_fields = [
            {"name": "set_id", "description": "identifier of a set"},
            {"name": "elements", "description": "space-separated element identifiers belonging to this set"},
        ]
        allowed = ["num_elements", "num_sets", "budget_k", "set_id", "elements"]

    elif pt == "SP":
        global_fields = [
            {"name": "num_elements", "description": "total number of elements"},
            {"name": "num_sets", "description": "total number of sets available"},
        ]
        item_fields = [
            {"name": "set_id", "description": "identifier of a set"},
            {"name": "elements", "description": "space-separated elements in the set"},
        ]
        allowed = ["num_elements", "num_sets", "set_id", "elements"]

    elif pt == "HSP":
        global_fields = [
            {"name": "num_elements", "description": "total number of elements in the universe"},
            {"name": "num_sets", "description": "total number of sets that must each be hit"},
        ]
        item_fields = [
            {"name": "set_id", "description": "identifier of a set"},
            {"name": "elements", "description": "space-separated elements in this set"},
        ]
        allowed = ["num_elements", "num_sets", "set_id", "elements"]

    else:  # SPP
        global_fields = [
            {"name": "num_elements", "description": "total number of elements to partition"},
            {"name": "num_sets", "description": "total candidate sets that may be used for partition"},
        ]
        item_fields = [
            {"name": "set_id", "description": "identifier for a candidate set"},
            {"name": "elements", "description": "element identifiers belonging to this set"},
            {"name": "cost", "description": "numerical cost associated with selecting this set"},
        ]
        allowed = ["num_elements", "num_sets", "set_id", "elements", "cost"]

    return {
        "global_fields": global_fields,
        "item_fields": item_fields,
        "allowed_placeholders": allowed,
    }

def run_set_step2(
    *,
    problem_type: str,
    instance_dir: str,
    output_root_dir: str,
    k_per_call: int = 20,
    n_target: int = 50,
    scale: str | None = None,
    load_cached_contexts: bool = True,
    contexts_audit_path: str | None = None,
) -> str:

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
        contextualize_fn=contextualize_instance_sets,
        llm=llm,
        prompter=prompter,
        hint=hint,
        baseline_template=baseline_template,
        output_format=output_format,
        k_per_call=k_per_call,
        n_target=n_target,
        scale=scale,
        load_cached_contexts=load_cached_contexts,
        contexts_audit_path=contexts_audit_path,
    )

    return output_root_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Contextualize set-system instances (SCP, MkC, SP, HSP, SPP) into NL/JSON/CSV/Markdown."
    )
    parser.add_argument(
        "--problem_type",
        type=str,
        default="SCP",
        choices=["SCP", "MkC", "SP", "HSP", "SPP"],
    )
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

    instance_dir = args.instance_dir or f"../step1_instance_creation/generated_data/{problem_type}"
    output_root_dir = args.output_root_dir or f"dataset/{problem_type}"

    llm = OpenAILlmService()
    prompter = PromptLoader()
    hint = _get_hint(problem_type)

    generate_dataset_for_task(
        task_name=task_name,
        task_description=desc,
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_instance_sets,
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
python -m step2_contextualization.contextualize_sets \
  --problem_type SCP \
  --instance_dir ./step1_instance_creation/generated_data/SCP \
  --output_root_dir step2_contextualization/dataset/SCP


python -m step2_contextualization.contextualize_sets \
  --problem_type MkC \
  --instance_dir ./step1_instance_creation/generated_data/MkC \
  --output_root_dir step2_contextualization/dataset/MkC

python -m step2_contextualization.contextualize_sets \
  --problem_type SP \
  --instance_dir ./step1_instance_creation/generated_data/SP \
  --output_root_dir step2_contextualization/dataset/SP

python -m step2_contextualization.contextualize_sets \
  --problem_type SPP \
  --instance_dir ./step1_instance_creation/generated_data/SPP \
  --output_root_dir step2_contextualization/dataset/SPP
python -m step2_contextualization.contextualize_sets \
  --problem_type HSP \
  --instance_dir ./step1_instance_creation/generated_data/HSP \
  --output_root_dir step2_contextualization/dataset/HSP

'''
