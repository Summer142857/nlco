from __future__ import annotations

from typing import Dict

from step2_contextualization.problems._dispatch import run_problem_step2


def generate_step2_for_problem(problem_type: str, cfg: Dict) -> str:
    return run_problem_step2(problem_type, cfg)
