# problems/registry.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from problems._step1_adapter import generate_step1_for_problem
from problems._step2_adapter import generate_step2_for_problem


@dataclass(frozen=True)
class ProblemEntry:
    name: str
    generate: Callable[[dict], str]
    contextualize: Optional[Callable[[dict], str]] = None


def _problems_root() -> Path:
    return Path(__file__).resolve().parent


def list_problem_names() -> list[str]:
    """
    Auto-discover problems as direct subdirectories of `problems/`
    that contain a config directory.
    """
    root = _problems_root()
    names: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if child.name in {"__pycache__"}:
            continue
        if (child / "config").is_dir():
            names.append(child.name)
    return names


def get_problem(name: str) -> ProblemEntry:
    """
    Return the centralized adapters for one problem.
    """
    root = _problems_root()
    pdir = root / name
    if not pdir.is_dir():
        raise KeyError(f"Unknown problem '{name}'. Available: {list_problem_names()}")
    if not (pdir / "config").is_dir():
        raise KeyError(f"Problem '{name}' missing config directory")

    return ProblemEntry(
        name=name,
        generate=lambda cfg, problem=name: generate_step1_for_problem(problem, cfg),
        contextualize=lambda cfg, problem=name: generate_step2_for_problem(problem, cfg),
    )


def get_all_problems() -> Dict[str, ProblemEntry]:
    """
    Eager load all problems (used by run_all).
    """
    return {name: get_problem(name) for name in list_problem_names()}
