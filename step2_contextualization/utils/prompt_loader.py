from __future__ import annotations
import os
from string import Template

class PromptLoader:
    def __init__(self, base_dir: str | None = None):
        here = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = base_dir or os.path.join(here, "prompts")

    def load(self, filename: str) -> str:
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def render(self, filename: str, **vars) -> str:
        """
        Render a prompt file using string.Template with $placeholders.
        Use .substitute(...) to fail fast when a variable is missing.
        NOTE: This does NOT touch '{{INSTANCE_INPUT}}' etc., they remain literal.
        """
        raw = self.load(filename)
        try:
            return Template(raw).substitute(**vars)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(f"Missing variable '${missing}' for prompt '{filename}'")
