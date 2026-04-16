import os
import json
import statistics
from typing import Dict, List

import tiktoken


# -----------------------------
# Configuration
# -----------------------------

ROOT_DIR = "dataset"

# Use the same tokenizer as OpenAI GPT-4 / GPT-4o family
TOKENIZER_NAME = "cl100k_base"

# JSON field name that contains the instruction text
INSTRUCTION_FIELD = "instruction"


# -----------------------------
# Helper Functions
# -----------------------------

def load_json(path: str) -> List[Dict]:
    """Load a JSON file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_tokens(text: str, encoder) -> int:
    """Count tokens in a string using tiktoken."""
    return len(encoder.encode(text))


# -----------------------------
# Main Statistics Logic
# -----------------------------

def compute_instruction_token_stats(root_dir: str):
    """
    For each problem and each size (S/M/L),
    compute min / max / mean token counts of instructions.
    """
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)

    results = {}

    for problem in sorted(os.listdir(root_dir)):
        problem_dir = os.path.join(root_dir, problem)
        if not os.path.isdir(problem_dir):
            continue

        results[problem] = {}

        for size in ["S", "M", "L"]:
            json_name = f"{problem}_{size}_context.json"
            json_path = os.path.join(problem_dir, json_name)

            if not os.path.exists(json_path):
                continue

            data = load_json(json_path)

            token_counts = []
            for item in data:
                if INSTRUCTION_FIELD not in item:
                    continue

                instruction = item[INSTRUCTION_FIELD]
                tokens = count_tokens(instruction, encoder)
                token_counts.append(tokens)

            if not token_counts:
                continue

            results[problem][size] = {
                "count": len(token_counts),
                "min": min(token_counts),
                "max": max(token_counts),
                "mean": round(statistics.mean(token_counts), 2),
            }

    return results


# -----------------------------
# Pretty Print
# -----------------------------

def print_stats(stats: Dict):
    """Print statistics in a readable table-like format."""
    for problem, size_dict in stats.items():
        print(f"\nProblem: {problem}")
        for size, values in size_dict.items():
            print(
                f"  Size {size}: "
                f"count={values['count']}, "
                f"min={values['min']}, "
                f"max={values['max']}, "
                f"mean={values['mean']}"
            )


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    stats = compute_instruction_token_stats(ROOT_DIR)
    print_stats(stats)
