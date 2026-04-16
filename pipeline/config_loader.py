# pipeline/config_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge override into base (override wins).
    Does not mutate input dicts.
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def load_problem_scale_config(
    problem: str,
    scale: str,
    root: str | Path | None = None,
    global_config_path: str | Path | None = None,
) -> dict:
    """
    Load config from:
      problems/<problem>/config/<scale>.yaml

    Optionally merge a global config first:
      cfg = merge(global_cfg, problem_cfg)
    """
    root_path = Path(root) if root is not None else Path.cwd()

    prob_cfg_path = root_path / "problems" / problem / "config" / f"{scale}.yaml"
    if not prob_cfg_path.exists():
        raise FileNotFoundError(prob_cfg_path)

    prob_cfg = load_config(prob_cfg_path)

    if global_config_path:
        global_cfg = load_config(root_path / global_config_path)
        cfg = merge_dicts(prob_cfg, global_cfg)
    else:
        cfg = prob_cfg

    # normalize: ensure problem/scale are present
    cfg.setdefault("problem", problem)
    cfg.setdefault("scale", scale)

    return cfg
