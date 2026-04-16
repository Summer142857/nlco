# ui/runner.py
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator, List, Optional


def run_cmd_stream(cmd: List[str], cwd: Optional[str] = None) -> Iterator[str]:
    """
    Run a command and yield stdout lines in realtime (stderr merged).
    """
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    for line in p.stdout:
        yield line.rstrip("\n")
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
