"""
Problem registry for step1_instance_creation.

Each entry in PROBLEM_REGISTRY describes how to generate instances for a
combinatorial optimisation problem. Call::

    from step1_instance_creation.registry import PROBLEM_REGISTRY, get_problem_entry
    entry = get_problem_entry("MIS")
    entry.generate("S", out_path="out/MIS_S.json", n_instances=50,
                   dataset_root="./step1_instance_creation")
"""

from __future__ import annotations

import json
import os
import random
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SizeConfig:
    """Parameters for one problem size tier (S / M / L)."""
    size_range: Union[int, Tuple[int, int]]   # n_nodes / n_items range
    n_instances: int = 50
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemEntry:
    """
    Registry entry for one combinatorial optimisation problem.

    Callers should use :meth:`generate` rather than ``_make_generator_fn``
    directly.
    """
    name: str
    sizes: Dict[str, SizeConfig]              # "S" | "M" | "L" -> SizeConfig
    _make_generator_fn: Callable              # (dataset_root: str) -> generator

    # ------------------------------------------------------------------
    def generate(
        self,
        size: str,
        out_path: str,
        n_instances: Optional[int] = None,
        dataset_root: str = ".",
        seed: int = 42,
        size_range: Optional[Union[int, Tuple[int, int]]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Generate instances for *size* and save to *out_path*.

        Parameters
        ----------
        size:          ``"S"``, ``"M"``, or ``"L"``.
        out_path:      Destination JSON path (parent dirs created automatically).
        n_instances:   Override the default count stored in :attr:`sizes`.
        dataset_root:  Root directory that contains ``datasets/``.
        seed:          Random seed forwarded to the generator.
        """
        sc = self.sizes[size]
        ni = n_instances if n_instances is not None else sc.n_instances
        size_range_to_use = size_range if size_range is not None else sc.size_range
        merged_kwargs = dict(sc.extra_kwargs)
        if extra_kwargs:
            merged_kwargs.update(extra_kwargs)
        gen = self._make_generator_fn(dataset_root)
        gen(
            size_range=size_range_to_use,
            n_instances=ni,
            out_path=out_path,
            seed=seed,
            **merged_kwargs,
        )


REGISTRY_TO_CONFIG_PROBLEM_ALIAS: Dict[str, str] = {
    "SMTWTP": "SMTWT",
}


def _parse_config_size_range(value: Any) -> Optional[Union[int, Tuple[int, int]]]:
    """Parse a size range from YAML config values like ``5-10`` or ``[5, 10]``."""
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


def _load_yaml_defined_sizes(
    registry_problem_name: str,
    fallback_sizes: Dict[str, "SizeConfig"],
) -> Dict[str, "SizeConfig"]:
    """Load per-scale size configs from ``problems/<problem>/config/*.yaml`` if present."""
    project_root = Path(__file__).resolve().parents[1]
    config_problem_name = REGISTRY_TO_CONFIG_PROBLEM_ALIAS.get(
        registry_problem_name, registry_problem_name
    )
    config_dir = project_root / "problems" / config_problem_name / "config"
    if not config_dir.exists():
        return fallback_sizes

    resolved_sizes: Dict[str, SizeConfig] = {}
    for scale_name, fallback in fallback_sizes.items():
        config_path = config_dir / f"{scale_name}.yaml"
        if not config_path.exists():
            resolved_sizes[scale_name] = fallback
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            resolved_sizes[scale_name] = fallback
            continue

        step1 = cfg.get("step1") or {}
        size_range = None
        for key in (
            "size_range",
            "n_nodes",
            "n_items",
            "items_range",
            "n_jobs",
            "n_tasks",
            "n_requests",
            "n_types",
            "n_vertices",
        ):
            size_range = _parse_config_size_range(step1.get(key))
            if size_range is not None:
                break

        if size_range is None:
            resolved_sizes[scale_name] = fallback
            continue

        n_instances = step1.get("n_instances")
        resolved_sizes[scale_name] = SizeConfig(
            size_range=size_range,
            n_instances=int(n_instances) if n_instances is not None else fallback.n_instances,
            extra_kwargs=dict(fallback.extra_kwargs),
        )

    return resolved_sizes


def _apply_yaml_size_overrides() -> None:
    """Replace hard-coded size tiers with YAML-defined ones when available."""
    for problem_name, entry in PROBLEM_REGISTRY.items():
        entry.sizes = _load_yaml_defined_sizes(problem_name, entry.sizes)


def _resolve_dataset_ref(dataset_root: str, override: Optional[str], default_rel: str) -> str:
    """Resolve dataset path with optional per-config override."""
    ref = override or default_rel
    p = Path(str(ref))
    if p.is_absolute():
        return str(p)
    return str(Path(dataset_root) / p)


def _resolve_configured_dataset_ref(dataset_root: str, kw: Dict[str, Any]) -> Optional[str]:
    """Resolve dataset path only when explicitly configured in YAML/kwargs."""
    ref = kw.get("dataset_dir") or kw.get("dataset_path")
    if not ref:
        return None
    p = Path(str(ref))
    if p.is_absolute():
        return str(p)
    return str(Path(dataset_root) / p)


def _build_instance_generator(ext: Any, mod: Any) -> Any:
    gen_cls = getattr(mod, "InstanceGenerator", None)
    if gen_cls is None:
        from .utils.base import InstanceGenerator as BaseInstanceGenerator
        gen_cls = BaseInstanceGenerator
    return gen_cls(ext)


def _generate_instances_with_flexible_output(gen: Any, size_range: Any, n_instances: int, out_path: str, **kwargs: Any) -> None:
    call_kwargs = {
        "n_nodes": size_range,
        "n_instances": n_instances,
        "out_path": out_path,
        "output_path": out_path,
        **kwargs,
    }
    filtered = _filter_kwargs_for_callable(gen.generate_and_save, call_kwargs)
    gen.generate_and_save(**filtered)


def _filter_kwargs_for_callable(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by callable signature (unless it has **kwargs)."""
    try:
        sig = inspect.signature(fn)
    except Exception:
        return {}

    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return dict(kwargs)

    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _build_extractor_with_fallbacks(
    cls_factory: Callable,
    dataset_roots: list[str],
    ext_kwargs: Dict[str, Any],
):
    last_error: Optional[Exception] = None
    for ds in dataset_roots:
        try:
            return cls_factory(ds, **ext_kwargs)
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Failed to init extractor with dataset '{ds}': {exc}")
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("No dataset candidates provided to extractor builder.")


# ---------------------------------------------------------------------------
# Helper: generic InstanceGenerator wrapper
# ---------------------------------------------------------------------------

def _wrap_instance_generator(extractor_factory: Callable) -> Callable:
    """
    Return a ``_make_generator_fn`` that:
    1. Calls ``extractor_factory(dataset_root)`` to build an extractor.
    2. Wraps it in the matching ``InstanceGenerator``.
    3. Returns a callable matching the ``gen(size_range, n_instances, out_path, seed)`` API.
    """
    def make(dataset_root: str):
        from .utils.base import InstanceGenerator  # noqa – local import

        def _gen(size_range, n_instances, out_path, seed=42, **kw):
            ext = extractor_factory(dataset_root)
            gen = InstanceGenerator(ext)
            gen.generate_and_save(
                n_nodes=size_range,
                n_instances=n_instances,
                out_path=out_path,
                oversample_factor=kw.get("oversample_factor", 1.0),
            )

        return _gen

    return make


# ---------------------------------------------------------------------------
# Per-problem _make_generator_fn factories
# ---------------------------------------------------------------------------

# ── Graph subgraph (RedistrictSet) ──────────────────────────────────────────

def _make_graph_extractor(cls_factory: Callable) -> Callable:
    """Generic factory for SingleGraphSubgraphExtractor problems."""
    def make(dataset_root: str):
        from .utils.base import InstanceGenerator  # noqa

        def _gen(size_range, n_instances, out_path, seed=42, **kw):
            dataset_dir = _resolve_dataset_ref(
                dataset_root,
                kw.get("dataset_dir") or kw.get("dataset_path"),
                "datasets/RedistrictSet",
            )
            dataset_candidates = [dataset_dir]
            for rel in ("datasets/RedistrictSet", "datasets/CitationNetwork", "datasets/StreetNetwork"):
                cand = _resolve_dataset_ref(dataset_root, None, rel)
                if cand not in dataset_candidates:
                    dataset_candidates.append(cand)
            gen_keys = {"oversample_factor", "min_density", "max_density"}
            ext_kwargs = {k: v for k, v in kw.items() if k not in gen_keys and k not in {"dataset_dir", "dataset_path"}}
            gen_kwargs = {k: v for k, v in kw.items() if k in gen_keys}

            ext = _build_extractor_with_fallbacks(cls_factory, dataset_candidates, ext_kwargs)
            gen = InstanceGenerator(ext)
            gen.generate_and_save(
                n_nodes=size_range,
                n_instances=n_instances,
                out_path=out_path,
                **gen_kwargs,
            )

        return _gen

    return make


def _mis_factory(dataset_dir, **kwargs):
    from .problems.mis import MISSubgraphInstanceExtractor
    ctor_kwargs = _filter_kwargs_for_callable(MISSubgraphInstanceExtractor, kwargs)
    return MISSubgraphInstanceExtractor(dataset_dir, **ctor_kwargs)


def _mvc_factory(dataset_dir, **kwargs):
    from .problems.mvc import MVCSubgraphInstanceExtractor
    ctor_kwargs = _filter_kwargs_for_callable(MVCSubgraphInstanceExtractor, kwargs)
    return MVCSubgraphInstanceExtractor(dataset_dir, **ctor_kwargs)


def _mcp_factory(dataset_dir, **kwargs):
    from .problems.mcp import MCPSubgraphInstanceExtractor
    ctor_kwargs = _filter_kwargs_for_callable(MCPSubgraphInstanceExtractor, kwargs)
    return MCPSubgraphInstanceExtractor(dataset_dir, **ctor_kwargs)


def _maxcut_factory(dataset_dir, **kwargs):
    from .problems.maxcut import MaxCutSubgraphInstanceExtractor
    ctor_kwargs = _filter_kwargs_for_callable(MaxCutSubgraphInstanceExtractor, kwargs)
    return MaxCutSubgraphInstanceExtractor(dataset_dir, **ctor_kwargs)


def _gcp_factory(dataset_dir, **kwargs):
    from .problems.gcp import GCPSubgraphInstanceExtractor
    ctor_kwargs = _filter_kwargs_for_callable(GCPSubgraphInstanceExtractor, kwargs)
    return GCPSubgraphInstanceExtractor(dataset_dir, **ctor_kwargs)


def _mds_factory(dataset_dir, **kwargs):
    from .problems.mds import MDSSubgraphInstanceExtractor
    ctor_kwargs = _filter_kwargs_for_callable(MDSSubgraphInstanceExtractor, kwargs)
    return MDSSubgraphInstanceExtractor(dataset_dir, **ctor_kwargs)


# ── Set / cover (ROAD) ──────────────────────────────────────────────────────

def _make_set_extractor(cls_factory: Callable) -> Callable:
    def make(dataset_root: str):
        from .utils.base import InstanceGenerator  # noqa

        def _gen(size_range, n_instances, out_path, seed=42, **kw):
            dataset_dir = _resolve_dataset_ref(
                dataset_root,
                kw.get("dataset_dir") or kw.get("dataset_path"),
                "datasets/ROAD",
            )
            dataset_candidates = [dataset_dir]
            for rel in ("datasets/ROAD", "datasets/RedistrictSet"):
                cand = _resolve_dataset_ref(dataset_root, None, rel)
                if cand not in dataset_candidates:
                    dataset_candidates.append(cand)
            gen_keys = {"oversample_factor", "min_density", "max_density"}
            ext_kwargs = {k: v for k, v in kw.items() if k not in gen_keys and k not in {"dataset_dir", "dataset_path"}}
            gen_kwargs = {k: v for k, v in kw.items() if k in gen_keys}

            ext = _build_extractor_with_fallbacks(cls_factory, dataset_candidates, ext_kwargs)
            gen = InstanceGenerator(ext)
            gen.generate_and_save(
                n_nodes=size_range,
                n_instances=n_instances,
                out_path=out_path,
                **gen_kwargs,
            )

        return _gen

    return make


def _scp_factory(dataset_dir, **kwargs):
    from .problems.scp import GraphToSCPExtractor
    ctor_kwargs = _filter_kwargs_for_callable(GraphToSCPExtractor, kwargs)
    return GraphToSCPExtractor(dataset_dir, **ctor_kwargs)


def _mkc_factory(dataset_dir, **kwargs):
    from .problems.mkc import GraphToMaxCovExtractor
    ctor_kwargs = _filter_kwargs_for_callable(GraphToMaxCovExtractor, kwargs)
    return GraphToMaxCovExtractor(dataset_dir, **ctor_kwargs)


def _sp_factory(dataset_dir, **kwargs):
    from .problems.sp import GraphToSPExtractor
    ctor_kwargs = _filter_kwargs_for_callable(GraphToSPExtractor, kwargs)
    return GraphToSPExtractor(dataset_dir, **ctor_kwargs)


def _hsp_factory(dataset_dir, **kwargs):
    from .problems.hsp import GraphToHSPExtractor
    ctor_kwargs = _filter_kwargs_for_callable(GraphToHSPExtractor, kwargs)
    return GraphToHSPExtractor(dataset_dir, **ctor_kwargs)


# ── Tree (STP dataset) ──────────────────────────────────────────────────────

def _make_tree_gen(cls_factory: Callable, problem_module_attr: str) -> Callable:
    def make(dataset_root: str):
        def _gen(size_range, n_instances, out_path, seed=42, **kw):
            dataset_dir = _resolve_dataset_ref(
                dataset_root,
                kw.get("dataset_dir") or kw.get("dataset_path"),
                "datasets/STP",
            )
            ext = cls_factory(dataset_dir)
            # Each tree problem file has its own InstanceGenerator
            module_path = problem_module_attr.rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_path[0], package="step1_instance_creation")
            InstanceGenerator = getattr(mod, "InstanceGenerator")
            gen = InstanceGenerator(ext)
            gen_keys = {"oversample_factor", "viz_dir", "min_density", "max_density"}
            gen_kwargs = {k: v for k, v in kw.items() if k in gen_keys}
            gen.generate_and_save(
                n_nodes=size_range,
                n_instances=n_instances,
                out_path=out_path,
                **gen_kwargs,
            )

        return _gen

    return make


def _make_stp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.stp import STPSubgraphInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_path") or kw.get("dataset_dir"),
            "datasets/STP",
        )
        ext_kwargs = _filter_kwargs_for_callable(
            STPSubgraphInstanceExtractor,
            {k: v for k, v in kw.items() if k not in {"dataset_path", "dataset_dir"}},
        )
        ext = STPSubgraphInstanceExtractor(dataset_path, **ext_kwargs)
        gen = InstanceGenerator(ext)
        gen.generate_and_save(
            n_nodes=size_range,
            n_instances=n_instances,
            out_path=out_path,
            oversample_factor=kw.get("oversample_factor", 1.0),
            viz_dir=kw.get("viz_dir"),
        )
    return _gen


def _make_kmst(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.kmst import KMSTSubgraphInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_path") or kw.get("dataset_dir"),
            "datasets/STP",
        )
        ext_kwargs = _filter_kwargs_for_callable(
            KMSTSubgraphInstanceExtractor,
            {k: v for k, v in kw.items() if k not in {"dataset_path", "dataset_dir"}},
        )
        ext = KMSTSubgraphInstanceExtractor(dataset_path, **ext_kwargs)
        gen = InstanceGenerator(ext)
        gen.generate_and_save(
            n_nodes=size_range,
            n_instances=n_instances,
            out_path=out_path,
            oversample_factor=kw.get("oversample_factor", 1.0),
            viz_dir=kw.get("viz_dir"),
        )
    return _gen


def _make_sfp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.sfp import STPSubgraphForestInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_path") or kw.get("dataset_dir"),
            "datasets/STP",
        )
        ext_kwargs = _filter_kwargs_for_callable(
            STPSubgraphForestInstanceExtractor,
            {k: v for k, v in kw.items() if k not in {"dataset_path", "dataset_dir"}},
        )
        ext = STPSubgraphForestInstanceExtractor(dataset_path, **ext_kwargs)
        gen = InstanceGenerator(ext)
        gen.generate_and_save(
            n_nodes=size_range,
            n_instances=n_instances,
            out_path=out_path,
            oversample_factor=kw.get("oversample_factor", 1.0),
            viz_dir=kw.get("viz_dir"),
        )
    return _gen


# ── Assignment (external datasets) ─────────────────────────────────────────

def _make_qap(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        import importlib
        mod = importlib.import_module(".problems.qap", package="step1_instance_creation")
        dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
        if dataset_path and Path(dataset_path).exists():
            ExtractorCls = getattr(mod, "QAPInstanceExtractor")
            ext = ExtractorCls(dataset_path)
        else:
            ExtractorCls = getattr(mod, "SyntheticQAPInstanceExtractor")
            ext = ExtractorCls()
        gen = _build_instance_generator(ext, mod)
        _generate_instances_with_flexible_output(
            gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1.0)
        )
    return _gen


def _make_gap(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        import importlib
        mod = importlib.import_module(".problems.gap", package="step1_instance_creation")
        dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
        if dataset_path and Path(dataset_path).exists():
            ExtractorCls = getattr(mod, "GAPInstanceExtractor")
            ext = ExtractorCls(dataset_path)
        else:
            ExtractorCls = getattr(mod, "SyntheticGAPInstanceExtractor")
            ext = ExtractorCls()
        gen = _build_instance_generator(ext, mod)
        _generate_instances_with_flexible_output(
            gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1.0)
        )
    return _gen


def _make_ap3(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        import importlib
        mod = importlib.import_module(".problems.ap3", package="step1_instance_creation")
        dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
        if dataset_path and Path(dataset_path).exists():
            ExtractorCls = getattr(mod, "AP3InstanceExtractor")
            ext = ExtractorCls(dataset_path)
        else:
            ExtractorCls = getattr(mod, "SyntheticAP3InstanceExtractor")
            ext = ExtractorCls()
        gen = _build_instance_generator(ext, mod)
        _generate_instances_with_flexible_output(
            gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1)
        )
    return _gen


# ── Routing ──────────────────────────────────────────────────────────────────

def _make_tsp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.tsp import TSPInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/TSP",
        )
        ext = TSPInstanceExtractor(dataset_path)
        gen = InstanceGenerator(ext)
        gen.generate_and_save(n_nodes=size_range, n_instances=n_instances, out_path=out_path,
                               oversample_factor=kw.get("oversample_factor", 1.0))
    return _gen


def _make_bpp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.bpp import BPPInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/BPP",
        )
        ext = BPPInstanceExtractor(dataset_path)
        gen = InstanceGenerator(ext)
        gen.generate_and_save(n_nodes=size_range, n_instances=n_instances, out_path=out_path,
                               oversample_factor=kw.get("oversample_factor", 1.0))
    return _gen


def _make_pctsp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.pctsp import PCTSPInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/TSP",
        )
        ext = PCTSPInstanceExtractor(dataset_path)
        gen = InstanceGenerator(ext)
        gen.generate_and_save(n_nodes=size_range, n_instances=n_instances, out_path=out_path,
                               oversample_factor=kw.get("oversample_factor", 1.0))
    return _gen


def _make_op(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        import importlib
        mod = importlib.import_module(".problems.op", package="step1_instance_creation")
        dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
        if dataset_path and Path(dataset_path).exists():
            ExtractorCls = getattr(mod, "OPInstanceExtractor")
            ext = ExtractorCls(dataset_path)
        else:
            ExtractorCls = getattr(mod, "SyntheticOPInstanceExtractor")
            ext = ExtractorCls()
        gen = _build_instance_generator(ext, mod)
        _generate_instances_with_flexible_output(
            gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1.0)
        )
    return _gen


def _make_cvrp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.cvrp import CVRPInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/CVRP",
        )
        ext = CVRPInstanceExtractor(str(dataset_path))
        gen = InstanceGenerator(ext)
        gen.generate_and_save(n_nodes=size_range, n_instances=n_instances, out_path=out_path,
                               oversample_factor=kw.get("oversample_factor", 1.0))
    return _gen



def _make_pdp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        import importlib
        mod = importlib.import_module(".problems.pdp", package="step1_instance_creation")
        dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
        if dataset_path and Path(dataset_path).exists():
            ExtractorCls = getattr(mod, "PDPInstanceExtractor")
            ext = ExtractorCls(dataset_path)
        else:
            ExtractorCls = getattr(mod, "SyntheticPDPInstanceExtractor")
            ext = ExtractorCls()
        gen = _build_instance_generator(ext, mod)
        _generate_instances_with_flexible_output(
            gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1.0)
        )
    return _gen


def _make_cmp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        import importlib
        mod = importlib.import_module(".problems.cmp", package="step1_instance_creation")
        dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
        if dataset_path and Path(dataset_path).exists():
            ExtractorCls = getattr(mod, "CMPInstanceExtractor")
            ext = ExtractorCls(dataset_path)
        else:
            ExtractorCls = getattr(mod, "SyntheticCMPInstanceExtractor")
            ext = ExtractorCls()
        gen = _build_instance_generator(ext, mod)
        _generate_instances_with_flexible_output(
            gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1.0)
        )
    return _gen


# ── Location ────────────────────────────────────────────────────────────────

def _make_uflp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.uflp import UFLPInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/UFLP",
        )
        ext = UFLPInstanceExtractor(dataset_path)
        gen = InstanceGenerator(ext)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # use generate_and_save if available
        gen.generate_and_save(n_nodes=size_range, n_instances=n_instances, out_path=out_path,
                               oversample_factor=kw.get("oversample_factor", 1.0))
    return _gen


def _make_location(prob: str) -> Callable:
    """Factory for location problems using dataset extractors when configured."""
    synthetic_cls_names = {
        "UFLP": ("uflp", "SyntheticUFLPInstanceExtractor"),
        "CFLP": ("cflp", "SyntheticCFLPInstanceExtractor"),
        "PMED": ("pmed", "SyntheticPMEDInstanceExtractor"),
        "PCENTER": ("pcenter", "SyntheticPCENTERInstanceExtractor"),
        "MDP": ("mdp", "SyntheticMDPInstanceExtractor"),
    }
    dataset_cls_names = {
        "UFLP": "UFLPInstanceExtractor",
        "CFLP": "CFLPInstanceExtractor",
        "PMED": "PMEDInstanceExtractor",
        "PCENTER": "PCENTERInstanceExtractor",
        "MDP": "MDPInstanceExtractor",
    }

    module_name, cls_name = synthetic_cls_names[prob]
    dataset_cls_name = dataset_cls_names[prob]

    def make(dataset_root: str):
        def _gen(size_range, n_instances, out_path, seed=42, **kw):
            import importlib
            mod = importlib.import_module(
                f".problems.{module_name}", package="step1_instance_creation"
            )
            dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
            if dataset_path and Path(dataset_path).exists():
                ExtractorCls = getattr(mod, dataset_cls_name)
                ext = ExtractorCls(dataset_path)
            else:
                ExtractorCls = getattr(mod, cls_name)
                ext = ExtractorCls()
            gen = _build_instance_generator(ext, mod)
            _generate_instances_with_flexible_output(
                gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1.0)
            )

        return _gen

    return make


# ── Scheduling (Taillard JSP/FSP/OSP) ──────────────────────────────────────

def _make_jsp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.jsp import generate_and_save_jsp
        n_machines = kw.get("n_machines", 3)
        generate_and_save_jsp(
            n_jobs=size_range,
            n_machines=n_machines,
            n_instances=n_instances,
            out_path=out_path,
            seed=seed,
        )
    return _gen


def _make_fsp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.fsp import generate_and_save_fsp
        n_machines = kw.get("n_machines", 3)
        generate_and_save_fsp(
            n_jobs=size_range,
            n_machines=n_machines,
            n_instances=n_instances,
            out_path=out_path,
            seed=seed,
        )
    return _gen


def _make_osp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.osp import generate_and_save_osp
        n_machines = kw.get("n_machines", 3)
        generate_and_save_osp(
            n_jobs=size_range,
            n_machines=n_machines,
            n_instances=n_instances,
            out_path=out_path,
            seed=seed,
        )
    return _gen


def _make_pms(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.pms import generate_save_pms
        n_machines = kw.get("n_machines", None)
        size_category = kw.get("size_category", None)
        dataset_dir = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/PMS",
        )
        generate_save_pms(
            n_instances=n_instances,
            out_path=out_path,
            num_jobs=size_range,
            num_machines=n_machines,
            dataset_dir=dataset_dir,
            seed=seed,
            size_category=size_category,
        )
    return _gen


def _make_rcpsp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.rcpsp import generate_save_rcpsp
        dataset_dir = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/RCPSP",
        )
        generate_save_rcpsp(
            n_instances=n_instances,
            out_path=out_path,
            num_tasks=size_range,
            dataset_dir=dataset_dir,
            seed=seed,
        )
    return _gen


def _make_smtwtp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.smtwtp import generate_save_wt
        generate_save_wt(
            n_jobs=size_range,
            n_instances=n_instances,
            out_path=out_path,
            seed=seed,
        )
    return _gen


def _make_lop(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.lop import LOPInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/LOP",
        )
        ext = LOPInstanceExtractor(dataset_path)
        gen = InstanceGenerator(ext)
        gen.generate_and_save(n_nodes=size_range, n_instances=n_instances, output_path=out_path)
    return _gen


def _make_tsptw(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.tsptw import TSPTWInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/TSP",
        )
        ext = TSPTWInstanceExtractor(dataset_path)
        gen = InstanceGenerator(ext)
        gen.generate_and_save(n_nodes=size_range, n_instances=n_instances, output_path=out_path)
    return _gen


# ── Standalone pure generators ──────────────────────────────────────────────

def _make_kp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.kp import KnapsackGenerator

        rng_seed = seed if seed is not None else 42
        R = int(kw.get("R", 100))
        gen = KnapsackGenerator(R=R, seed=rng_seed)
        alphas = kw.get("alphas")
        alpha = kw.get("alpha")
        if alphas is not None:
            if isinstance(alphas, (list, tuple)):
                alpha_values = [float(a) for a in alphas]
            else:
                alpha_values = [float(alphas)]
        elif alpha is not None:
            alpha_values = [float(alpha)]
        else:
            alpha_values = [0.5]
        time_limit = float(kw.get("time_limit", 60.0))
        threads = int(kw.get("threads", 8))
        gen.generate_and_save(
            n_items=size_range,
            n_instances=n_instances,
            out_path=out_path,
            alpha_values=alpha_values,
            time_limit=time_limit,
            threads=threads,
            seed=rng_seed,
        )

    return _gen


def _make_spp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.spp import SPPInstanceGenerator
        gen = SPPInstanceGenerator(rng_seed=seed)
        gen.generate_and_save(
            items_range=size_range,
            n_instances=n_instances,
            out_path=out_path,
        )
    return _gen


def _make_qspp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.qspp import RostamiQSPPInstanceExtractor, InstanceGenerator, RostamiGenConfig
        cfg = RostamiGenConfig(family="grid2", shape="square", seed=seed)
        ext = RostamiQSPPInstanceExtractor(cfg)
        gen = InstanceGenerator(ext)
        instances = gen.generate_instances(n_nodes=size_range, n_instances=n_instances)
        gen.save_to_json(instances, out_path)
    return _gen


def _make_mlp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        from .problems.mlp import MLPInstanceExtractor, InstanceGenerator
        dataset_path = _resolve_dataset_ref(
            dataset_root,
            kw.get("dataset_dir") or kw.get("dataset_path"),
            "datasets/TSP",
        )
        ext = MLPInstanceExtractor(dataset_path)
        gen = InstanceGenerator(ext)
        gen.generate_and_save(n_nodes=size_range, n_instances=n_instances, output_path=out_path)
    return _gen


def _make_csp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        import importlib
        mod = importlib.import_module(".problems.csp", package="step1_instance_creation")
        dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
        if dataset_path and Path(dataset_path).exists():
            ExtractorCls = getattr(mod, "CSPInstanceExtractor")
            ext = ExtractorCls(dataset_path)
        else:
            ExtractorCls = getattr(mod, "SyntheticCSPInstanceExtractor")
            ext = ExtractorCls()
        gen = _build_instance_generator(ext, mod)
        _generate_instances_with_flexible_output(
            gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1.0)
        )
    return _gen


def _make_2sp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        import importlib
        mod = importlib.import_module(".problems.twosp", package="step1_instance_creation")
        dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
        if dataset_path and Path(dataset_path).exists():
            ExtractorCls = getattr(mod, "TwoSPInstanceExtractor")
            ext = ExtractorCls(dataset_path)
        else:
            ExtractorCls = getattr(mod, "Synthetic2SPInstanceExtractor")
            ext = ExtractorCls()
        gen = _build_instance_generator(ext, mod)
        _generate_instances_with_flexible_output(
            gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1.0)
        )
    return _gen


def _make_qkp(dataset_root: str):
    def _gen(size_range, n_instances, out_path, seed=42, **kw):
        import importlib
        mod = importlib.import_module(".problems.qkp", package="step1_instance_creation")
        dataset_path = _resolve_configured_dataset_ref(dataset_root, kw)
        if dataset_path and Path(dataset_path).exists():
            ExtractorCls = getattr(mod, "QKPInstanceExtractor")
            ext = ExtractorCls(dataset_path)
        else:
            ExtractorCls = getattr(mod, "SyntheticQKPInstanceExtractor")
            ext = ExtractorCls()
        gen = _build_instance_generator(ext, mod)
        _generate_instances_with_flexible_output(
            gen, size_range, n_instances, out_path, oversample_factor=kw.get("oversample_factor", 1.0)
        )
    return _gen


# ---------------------------------------------------------------------------
# PROBLEM_REGISTRY
# ---------------------------------------------------------------------------

PROBLEM_REGISTRY: Dict[str, ProblemEntry] = {

    # ── Graph subgraph ───────────────────────────────────────────────────────
    "MIS": ProblemEntry(
        name="MIS",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_graph_extractor(_mis_factory),
    ),
    "MVC": ProblemEntry(
        name="MVC",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_graph_extractor(_mvc_factory),
    ),
    "MCP": ProblemEntry(
        name="MCP",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_graph_extractor(_mcp_factory),
    ),
    "MAXCUT": ProblemEntry(
        name="MAXCUT",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_graph_extractor(_maxcut_factory),
    ),
    "GCP": ProblemEntry(
        name="GCP",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_graph_extractor(_gcp_factory),
    ),
    "MDS": ProblemEntry(
        name="MDS",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_graph_extractor(_mds_factory),
    ),

    # ── Set / cover ──────────────────────────────────────────────────────────
    "SCP": ProblemEntry(
        name="SCP",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_set_extractor(_scp_factory),
    ),
    "MkC": ProblemEntry(
        name="MkC",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_set_extractor(_mkc_factory),
    ),
    "SP": ProblemEntry(
        name="SP",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_set_extractor(_sp_factory),
    ),
    "HSP": ProblemEntry(
        name="HSP",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_set_extractor(_hsp_factory),
    ),

    # ── Tree ─────────────────────────────────────────────────────────────────
    "STP": ProblemEntry(
        name="STP",
        sizes={
            "S": SizeConfig((10, 15)),
            "M": SizeConfig((16, 20)),
            "L": SizeConfig((21, 30)),
        },
        _make_generator_fn=_make_stp,
    ),
    "KMST": ProblemEntry(
        name="KMST",
        sizes={
            "S": SizeConfig((10, 15)),
            "M": SizeConfig((16, 20)),
            "L": SizeConfig((21, 30)),
        },
        _make_generator_fn=_make_kmst,
    ),
    "SFP": ProblemEntry(
        name="SFP",
        sizes={
            "S": SizeConfig((10, 15)),
            "M": SizeConfig((16, 20)),
            "L": SizeConfig((21, 30)),
        },
        _make_generator_fn=_make_sfp,
    ),

    # ── Assignment ───────────────────────────────────────────────────────────
    "QAP": ProblemEntry(
        name="QAP",
        sizes={
            "S": SizeConfig((3, 5)),
            "M": SizeConfig((6, 8)),
            "L": SizeConfig((9, 12)),
        },
        _make_generator_fn=_make_qap,
    ),
    "GAP": ProblemEntry(
        name="GAP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_gap,
    ),
    "AP3": ProblemEntry(
        name="AP3",
        sizes={
            "S": SizeConfig((3, 5)),
            "M": SizeConfig((6, 8)),
            "L": SizeConfig((9, 12)),
        },
        _make_generator_fn=_make_ap3,
    ),

    # ── Routing ──────────────────────────────────────────────────────────────
    "TSP": ProblemEntry(
        name="TSP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_tsp,
    ),
    "BPP": ProblemEntry(
        name="BPP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_bpp,
    ),
    "PCTSP": ProblemEntry(
        name="PCTSP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_pctsp,
    ),
    "OP": ProblemEntry(
        name="OP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_op,
    ),
    "CVRP": ProblemEntry(
        name="CVRP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_cvrp,
    ),
    "PDP": ProblemEntry(
        name="PDP",
        sizes={
            "S": SizeConfig((3, 5)),
            "M": SizeConfig((6, 8)),
            "L": SizeConfig((9, 14)),
        },
        _make_generator_fn=_make_pdp,
    ),
    "CMP": ProblemEntry(
        name="CMP",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_cmp,
    ),

    # ── Location ─────────────────────────────────────────────────────────────
    "UFLP": ProblemEntry(
        name="UFLP",
        sizes={
            "S": SizeConfig((5, 8)),
            "M": SizeConfig((9, 14)),
            "L": SizeConfig((15, 20)),
        },
        _make_generator_fn=_make_location("UFLP"),
    ),
    "CFLP": ProblemEntry(
        name="CFLP",
        sizes={
            "S": SizeConfig((5, 8)),
            "M": SizeConfig((9, 14)),
            "L": SizeConfig((15, 20)),
        },
        _make_generator_fn=_make_location("CFLP"),
    ),
    "PMED": ProblemEntry(
        name="PMED",
        sizes={
            "S": SizeConfig((5, 8)),
            "M": SizeConfig((9, 14)),
            "L": SizeConfig((15, 20)),
        },
        _make_generator_fn=_make_location("PMED"),
    ),
    "PCENTER": ProblemEntry(
        name="PCENTER",
        sizes={
            "S": SizeConfig((5, 8)),
            "M": SizeConfig((9, 14)),
            "L": SizeConfig((15, 20)),
        },
        _make_generator_fn=_make_location("PCENTER"),
    ),
    "MDP": ProblemEntry(
        name="MDP",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_location("MDP"),
    ),

    # ── Scheduling ───────────────────────────────────────────────────────────
    "JSP": ProblemEntry(
        name="JSP",
        sizes={
            "S": SizeConfig((2, 5), extra_kwargs={"n_machines": (2, 3)}),
            "M": SizeConfig((6, 8), extra_kwargs={"n_machines": (3, 5)}),
            "L": SizeConfig((9, 12), extra_kwargs={"n_machines": (5, 6)}),
        },
        _make_generator_fn=_make_jsp,
    ),
    "FSP": ProblemEntry(
        name="FSP",
        sizes={
            "S": SizeConfig((2, 5), extra_kwargs={"n_machines": (2, 3)}),
            "M": SizeConfig((6, 8), extra_kwargs={"n_machines": (3, 5)}),
            "L": SizeConfig((9, 12), extra_kwargs={"n_machines": (5, 6)}),
        },
        _make_generator_fn=_make_fsp,
    ),
    "OSP": ProblemEntry(
        name="OSP",
        sizes={
            "S": SizeConfig((2, 5), extra_kwargs={"n_machines": (2, 3)}),
            "M": SizeConfig((6, 8), extra_kwargs={"n_machines": (3, 5)}),
            "L": SizeConfig((9, 12), extra_kwargs={"n_machines": (5, 6)}),
        },
        _make_generator_fn=_make_osp,
    ),
    "PMS": ProblemEntry(
        name="PMS",
        sizes={
            "S": SizeConfig((5, 10), extra_kwargs={"size_category": "S", "n_machines": (2, 2)}),
            "M": SizeConfig((11, 15), extra_kwargs={"size_category": "M", "n_machines": (2, 3)}),
            "L": SizeConfig((16, 25), extra_kwargs={"size_category": "L", "n_machines": (2, 5)}),
        },
        _make_generator_fn=_make_pms,
    ),
    "RCPSP": ProblemEntry(
        name="RCPSP",
        sizes={
            "S": SizeConfig((4, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 21)),
        },
        _make_generator_fn=_make_rcpsp,
    ),
    "SMTWTP": ProblemEntry(
        name="SMTWTP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 20)),
            "L": SizeConfig((21, 30)),
        },
        _make_generator_fn=_make_smtwtp,
    ),
    "LOP": ProblemEntry(
        name="LOP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_lop,
    ),
    "TSPTW": ProblemEntry(
        name="TSPTW",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_tsptw,
    ),
    # ── Standalone pure generators ───────────────────────────────────────────
    "KP": ProblemEntry(
        name="KP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 20)),
            "L": SizeConfig((21, 30)),
        },
        _make_generator_fn=_make_kp,
    ),
    "SPP": ProblemEntry(
        name="SPP",
        sizes={
            "S": SizeConfig((8, 12)),
            "M": SizeConfig((13, 18)),
            "L": SizeConfig((19, 25)),
        },
        _make_generator_fn=_make_spp,
    ),
    "QSPP": ProblemEntry(
        name="QSPP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_qspp,
    ),

    # ── New problems (MLP / CSP / 2SP / QKP) ─────────────────────────────────
    "MLP": ProblemEntry(
        name="MLP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_mlp,
    ),
    "CSP": ProblemEntry(
        name="CSP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_csp,
    ),
    "2SP": ProblemEntry(
        name="2SP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_2sp,
    ),
    "QKP": ProblemEntry(
        name="QKP",
        sizes={
            "S": SizeConfig((5, 10)),
            "M": SizeConfig((11, 15)),
            "L": SizeConfig((16, 25)),
        },
        _make_generator_fn=_make_qkp,
    ),
}

_apply_yaml_size_overrides()


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_problem_entry(name: str) -> ProblemEntry:
    """Return the registry entry for *name*, raising KeyError if not found."""
    if name not in PROBLEM_REGISTRY:
        raise KeyError(
            f"Unknown problem {name!r}. Available: {sorted(PROBLEM_REGISTRY.keys())}"
        )
    return PROBLEM_REGISTRY[name]


ALL_PROBLEMS = sorted(PROBLEM_REGISTRY.keys())
