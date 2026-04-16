"""
step1_instance_creation — CO benchmark instance generation package.

Key exports::

    from step1_instance_creation.registry import PROBLEM_REGISTRY, get_problem_entry, ALL_PROBLEMS

CLI::

    python -m step1_instance_creation.generate_cli --list
    python -m step1_instance_creation.generate_cli --problem MIS --sizes S
"""

from .registry import PROBLEM_REGISTRY, get_problem_entry, ALL_PROBLEMS

__all__ = ["PROBLEM_REGISTRY", "get_problem_entry", "ALL_PROBLEMS"]
