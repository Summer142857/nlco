from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict


MODULE_BY_PROBLEM: Dict[str, str] = {
    "2SP": "contextualize_packing",
    "AP3": "contextualize_ap3",
    "BPP": "contextualize_packing",
    "CFLP": "contextualize_facility",
    "CMP": "contextualize_cmp",
    "CSP": "contextualize_packing",
    "CVRP": "contextualize_routing",
    "FSP": "contextualize_shop_scheduling",
    "GAP": "contextualize_gap",
    "GCP": "contextualize_graphs",
    "HSP": "contextualize_sets",
    "JSP": "contextualize_shop_scheduling",
    "KMST": "contextualize_trees",
    "KP": "contextualize_packing",
    "LOP": "contextualize_lop",
    "MAXCUT": "contextualize_graphs",
    "MCP": "contextualize_graphs",
    "MDP": "contextualize_mdp",
    "MDS": "contextualize_graphs",
    "MIS": "contextualize_graphs",
    "MLP": "contextualize_routing",
    "MVC": "contextualize_graphs",
    "MkC": "contextualize_sets",
    "OP": "contextualize_routing",
    "OSP": "contextualize_shop_scheduling",
    "PCENTER": "contextualize_facility",
    "PCTSP": "contextualize_routing",
    "PDP": "contextualize_pdp",
    "PMED": "contextualize_facility",
    "PMS": "contextualize_pms",
    "QAP": "contextualize_qap",
    "QKP": "contextualize_qkp",
    "QSPP": "contextualize_qspp",
    "RCPSP": "contextualize_rcpsp",
    "SCP": "contextualize_sets",
    "SFP": "contextualize_trees",
    "SMTWT": "contextualize_smtwt",
    "SP": "contextualize_sets",
    "SPP": "contextualize_sets",
    "STP": "contextualize_trees",
    "TSP": "contextualize_routing",
    "TSPTW": "contextualize_routing",
    "UFLP": "contextualize_facility",
}


RUN_FN_BY_MODULE: Dict[str, str] = {
    "contextualize_graphs": "run_graph_step2",
    "contextualize_sets": "run_set_step2",
    "contextualize_trees": "run_tree_step2",
    "contextualize_routing": "run_routing_step2",
    "contextualize_packing": "run_packing_step2",
}


def _resolve_contextualize_fn(mod: Any) -> Callable:
    for name in dir(mod):
        if name.startswith("contextualize_instance_"):
            fn = getattr(mod, name)
            if callable(fn):
                return fn
    raise AttributeError(f"{mod.__name__} missing contextualize_instance_* function")


def run_problem_step2(problem_type: str, cfg: Dict[str, Any]) -> str:
    problem_type = str(problem_type)
    step2 = cfg.get("step2", {})
    scale = cfg.get("scale")

    instance_dir = step2["instance_dir"]
    output_root_dir = step2["output_root_dir"]
    k_per_call = int(step2.get("k_per_call", 20))
    n_target = int(step2.get("n_target", 50))
    load_cached_contexts = bool(step2.get("load_cached_contexts", True))
    contexts_audit_path = step2.get("contexts_audit_path")

    module_short = MODULE_BY_PROBLEM.get(problem_type)
    if not module_short:
        raise KeyError(f"Unsupported step2 problem type: {problem_type}")

    mod = import_module(f"step2_contextualization.{module_short}")
    run_fn_name = RUN_FN_BY_MODULE.get(module_short)
    if run_fn_name and hasattr(mod, run_fn_name):
        run_fn = getattr(mod, run_fn_name)
        return run_fn(
            problem_type=problem_type,
            instance_dir=instance_dir,
            output_root_dir=output_root_dir,
            k_per_call=k_per_call,
            n_target=n_target,
            scale=scale,
            load_cached_contexts=load_cached_contexts,
            contexts_audit_path=contexts_audit_path,
        )

    specs = mod._get_problem_specs(problem_type)  # type: ignore[attr-defined]
    hint = mod._get_hint(problem_type)  # type: ignore[attr-defined]
    contextualize_fn = _resolve_contextualize_fn(mod)

    from step2_contextualization.utils.dataset_generator import generate_dataset_for_task
    from step2_contextualization.utils.llm_service import OpenAILlmService
    from step2_contextualization.utils.prompt_loader import PromptLoader

    llm = OpenAILlmService()
    prompter = PromptLoader()

    generate_dataset_for_task(
        task_name=problem_type,
        task_description=specs["task_description"],
        instance_dir=instance_dir,
        output_root_dir=output_root_dir,
        contextualize_fn=contextualize_fn,
        llm=llm,
        prompter=prompter,
        hint=hint,
        baseline_template=specs["baseline_template"],
        output_format=specs["output_format"],
        k_per_call=k_per_call,
        n_target=n_target,
        scale=scale,
        load_cached_contexts=load_cached_contexts,
        contexts_audit_path=contexts_audit_path,
    )
    return str(output_root_dir)
