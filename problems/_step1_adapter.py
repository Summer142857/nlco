from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from step1_instance_creation.registry import get_problem_entry


SizeRange = Union[int, Tuple[int, int]]
REGISTRY_PROBLEM_ALIAS = {
    "SMTWT": "SMTWTP",
}

COMMON_STEP1_KEYS = {
    "size",
    "output",
    "dataset_root",
    "dataset_dir",
    "dataset_path",
    "n_instances",
    "seed",
    "params",
    "size_range",
}

STEP1_SIZE_KEYS_BY_PROBLEM = {
    "MIS": {"n_nodes"},
    "MVC": {"n_nodes"},
    "MCP": {"n_nodes"},
    "MAXCUT": {"n_nodes"},
    "GCP": {"n_nodes"},
    "MDS": {"n_nodes"},
    "SCP": {"n_nodes"},
    "MkC": {"n_nodes"},
    "SP": {"n_nodes"},
    "HSP": {"n_nodes"},
    "STP": {"n_nodes"},
    "KMST": {"n_nodes"},
    "SFP": {"n_nodes"},
    "QAP": {"n_nodes"},
    "AP3": {"n_nodes"},
    "TSP": {"n_nodes"},
    "BPP": {"n_nodes"},
    "PCTSP": {"n_nodes"},
    "OP": {"n_nodes"},
    "CVRP": {"n_nodes"},
    "LOP": {"n_nodes"},
    "TSPTW": {"n_nodes"},
    "MLP": {"n_nodes"},
    "CMP": {"n_nodes"},
    "UFLP": {"n_nodes"},
    "CFLP": {"n_nodes"},
    "QSPP": {"n_nodes"},
    "GAP": {"n_tasks"},
    "RCPSP": {"n_tasks"},
    "JSP": {"n_jobs"},
    "FSP": {"n_jobs"},
    "OSP": {"n_jobs"},
    "PMS": {"n_jobs"},
    "SMTWT": {"n_jobs"},
    "KP": {"n_items"},
    "2SP": {"n_items"},
    "QKP": {"n_items"},
    "SPP": {"items_range"},
    "PDP": {"n_requests"},
    "CSP": {"n_types"},
    "PMED": {"n_vertices"},
    "PCENTER": {"n_vertices"},
    "MDP": {"n_vertices"},
}

STEP1_PARAM_KEYS_BY_PROBLEM = {
    "MIS": {"steps", "alpha", "rng_seed", "max_retry", "oversample_factor", "min_density", "max_density"},
    "MVC": {"steps", "alpha", "rng_seed", "max_retry", "oversample_factor", "min_density", "max_density"},
    "MCP": {"steps", "alpha", "rng_seed", "max_retry", "oversample_factor", "min_density", "max_density"},
    "MAXCUT": {"steps", "alpha", "rng_seed", "max_retry", "oversample_factor", "min_density", "max_density"},
    "GCP": {"steps", "alpha", "rng_seed", "max_retry", "oversample_factor", "min_density", "max_density"},
    "MDS": {"steps", "alpha", "rng_seed", "max_retry", "oversample_factor", "min_density", "max_density"},
    "SCP": {
        "k_hop", "steps", "alpha", "rng_seed", "max_retry",
        "min_set_size", "max_set_size", "remove_duplicate_sets", "keep_prob",
        "oversample_factor", "min_density", "max_density",
    },
    "SP": {
        "k_hop", "steps", "alpha", "rng_seed", "max_retry",
        "min_set_size", "max_set_size", "remove_duplicate_sets", "keep_prob",
        "oversample_factor", "min_density", "max_density",
    },
    "HSP": {
        "k_hop", "steps", "alpha", "rng_seed", "max_retry",
        "min_set_size", "max_set_size", "remove_duplicate_sets", "keep_prob",
        "oversample_factor", "min_density", "max_density",
    },
    "MkC": {
        "k_hop", "steps", "alpha", "rng_seed", "max_retry",
        "min_set_size", "max_set_size", "remove_duplicate_sets", "keep_prob",
        "budget_k", "budget_ratio", "oversample_factor", "min_density", "max_density",
    },
    "STP": {"steps_per_seed", "alpha", "num_seeds", "rng_seed", "t_ratio_choices", "oversample_factor", "viz_dir"},
    "KMST": {"steps_per_seed", "alpha", "num_seeds", "rng_seed", "k_ratio_choices", "oversample_factor", "viz_dir"},
    "SFP": {
        "steps_per_seed", "alpha", "num_seeds", "rng_seed", "t_ratio_choices",
        "min_groups", "max_groups", "oversample_factor", "viz_dir",
    },
    "QAP": {"oversample_factor"},
    "GAP": {"oversample_factor"},
    "AP3": {"oversample_factor"},
    "TSP": {"oversample_factor"},
    "BPP": {"oversample_factor"},
    "PCTSP": {"oversample_factor"},
    "OP": {"oversample_factor"},
    "CVRP": {"oversample_factor"},
    "PDP": {"oversample_factor"},
    "CMP": {"oversample_factor"},
    "UFLP": {"oversample_factor"},
    "CFLP": {"oversample_factor"},
    "PMED": {"oversample_factor"},
    "PCENTER": {"oversample_factor"},
    "MDP": {"oversample_factor"},
    "CSP": {"oversample_factor"},
    "QKP": {"oversample_factor"},
    "KP": {"alpha", "alphas", "R", "time_limit", "threads"},
    "SPP": set(),
    "QSPP": set(),
    "LOP": set(),
    "TSPTW": set(),
    "MLP": set(),
    "JSP": {"n_machines"},
    "FSP": {"n_machines"},
    "OSP": {"n_machines"},
    "PMS": {"n_machines", "size_category"},
    "RCPSP": set(),
    "SMTWT": set(),
    "2SP": set(),
}


def _parse_size_range(value: Any) -> Optional[SizeRange]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip().strip('"').strip("'")
        if "-" in s:
            a, b = s.split("-", 1)
            try:
                return int(a.strip()), int(b.strip())
            except Exception:
                return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _infer_dataset_root(step1: Dict[str, Any], default: str = ".") -> str:
    if step1.get("dataset_root"):
        return str(step1["dataset_root"])

    dataset_ref = step1.get("dataset_dir") or step1.get("dataset_path")
    if not dataset_ref:
        return default

    p = Path(str(dataset_ref))
    parts = list(p.parts)
    if "datasets" in parts:
        i = parts.index("datasets")
        if i > 0:
            root = Path(*parts[:i])
            if str(root):
                return str(root)
    return default


def _pick_size_range(step1: Dict[str, Any]) -> Optional[SizeRange]:
    candidate_keys = (
        "size_range",
        "n_nodes",
        "n_items",
        "items_range",
        "n_jobs",
        "n_tasks",
        "n_requests",
        "n_types",
        "n_vertices",
    )
    for key in candidate_keys:
        if key in step1:
            parsed = _parse_size_range(step1.get(key))
            if parsed is not None:
                return parsed
    return None


def _validate_step1_config(problem_type: str, step1: Dict[str, Any]) -> None:
    allowed_top_keys = set(COMMON_STEP1_KEYS)
    allowed_top_keys.update(STEP1_SIZE_KEYS_BY_PROBLEM.get(problem_type, set()))
    unknown_top_keys = sorted(k for k in step1.keys() if k not in allowed_top_keys)
    if unknown_top_keys:
        raise ValueError(
            f"Unsupported step1 keys for {problem_type}: {unknown_top_keys}. "
            f"Allowed keys: {sorted(allowed_top_keys)}"
        )

    params = step1.get("params")
    if params is None:
        return
    if not isinstance(params, dict):
        raise ValueError(f"step1.params for {problem_type} must be a mapping, got {type(params).__name__}")

    allowed_param_keys = STEP1_PARAM_KEYS_BY_PROBLEM.get(problem_type)
    if allowed_param_keys is None:
        return
    unknown_param_keys = sorted(k for k in params.keys() if k not in allowed_param_keys)
    if unknown_param_keys:
        raise ValueError(
            f"Unsupported step1.params keys for {problem_type}: {unknown_param_keys}. "
            f"Allowed params: {sorted(allowed_param_keys)}"
        )


def generate_step1_for_problem(problem_type: str, cfg: Dict[str, Any]) -> str:
    step1 = cfg.get("step1", {})
    _validate_step1_config(problem_type, step1)
    size = str(step1.get("size", cfg.get("scale", "S"))).upper()

    output = step1.get("output") or f"step1_instance_creation/generated_data/{problem_type}/{problem_type}_{size}.json"
    dataset_root = _infer_dataset_root(step1)
    n_instances = step1.get("n_instances")
    seed = int(step1.get("seed", cfg.get("seed", 42)))
    size_range = _pick_size_range(step1)

    extra_kwargs: Dict[str, Any] = {}
    params = step1.get("params")
    if isinstance(params, dict):
        extra_kwargs.update(params)
    if step1.get("dataset_dir") is not None:
        extra_kwargs.setdefault("dataset_dir", step1.get("dataset_dir"))
    if step1.get("dataset_path") is not None:
        extra_kwargs.setdefault("dataset_path", step1.get("dataset_path"))

    ignored_keys = {
        "size",
        "output",
        "dataset_root",
        "n_instances",
        "seed",
        "params",
        "size_range",
        "n_nodes",
        "n_items",
        "items_range",
        "n_jobs",
        "n_tasks",
        "n_requests",
        "n_types",
        "n_vertices",
    }
    for key, value in step1.items():
        if key not in ignored_keys:
            extra_kwargs.setdefault(key, value)

    registry_name = REGISTRY_PROBLEM_ALIAS.get(problem_type, problem_type)
    entry = get_problem_entry(registry_name)
    entry.generate(
        size=size,
        out_path=output,
        n_instances=n_instances,
        dataset_root=dataset_root,
        seed=seed,
        size_range=size_range,
        extra_kwargs=extra_kwargs,
    )
    return output
