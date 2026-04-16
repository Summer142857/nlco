import json
import random
from pathlib import Path
from typing import Union, Tuple


def parse_range(s: str) -> Union[int, Tuple[int, int]]:
    """Parse '5-10' -> (5,10) or '20' -> 20."""
    if isinstance(s, int):
        return s
    s = str(s).strip()
    if "-" in s:
        a, b = map(int, s.split("-", 1))
        if a < 2:
            a = 2
        return (a, b)
    n = int(s)
    return n


def get_random_dataset_file(dataset_path: str, extensions=None) -> str:
    """Select a random dataset file from path (file or directory)."""
    from pathlib import Path
    dataset_path = Path(dataset_path)
    if dataset_path.is_file():
        return str(dataset_path)
    elif dataset_path.is_dir():
        if extensions is None:
            extensions = ['*.dat', '*.txt', '*.json', '*.tsp', '*.stp']
        all_files = []
        for pattern in extensions:
            all_files.extend(dataset_path.glob(pattern))
        if not all_files:
            # Try all files
            all_files = [f for f in dataset_path.iterdir() if f.is_file()]
        if not all_files:
            raise ValueError(f"No dataset files found in {dataset_path}")
        selected = random.choice(all_files)
        print(f"Selected: {selected.name}")
        return str(selected)
    else:
        raise ValueError(f"Invalid dataset path: {dataset_path}")


def save_to_json(instances: list, out_path) -> None:
    """Save list of instances to JSON file, creating parent dirs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(instances, f, indent=2)
    print(f"[Info] Saved {len(instances)} instances to {out_path}")
