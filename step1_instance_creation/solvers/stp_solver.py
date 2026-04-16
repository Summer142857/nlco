import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Set


def _write_stp_file(
    path: str,
    num_nodes: int,
    edges: List[Tuple[int, int, float]],
    terminals: Set[int],
    name: str = "stp_subgraph",
) -> None:
    """
    Write a valid STP file (SteinLib format) for SCIP-Jack.
    Node indices MUST start from 1.
    """

    with open(path, "w") as f:
        f.write("33D32945 STP File, STP Format Version 1.0\n")

        f.write("SECTION Comment\n")
        f.write(f"Name \"{name}\"\n")
        f.write("END\n\n")

        f.write("SECTION Graph\n")
        f.write(f"Nodes {num_nodes}\n")
        f.write(f"Edges {len(edges)}\n")
        for u, v, w in edges:
            w_int = int(round(w))
            f.write(f"E {u} {v} {w_int}\n")
        f.write("END\n\n")

        f.write("SECTION Terminals\n")
        f.write(f"Terminals {len(terminals)}\n")
        for t in sorted(terminals):
            f.write(f"T {t}\n")
        f.write("END\n\n")

        f.write("EOF\n")


def _parse_solution_file(sol_path) -> List[Tuple[int, int]]:
    edges = []
    inside = False

    with open(sol_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("SECTION Finalsolution"):
                inside = True
                continue

            if inside:
                if line.startswith("E "):
                    _, u, v = line.split()[:3]
                    edges.append((int(u), int(v)))
                if line.startswith("End"):
                    break

    return edges




def _parse_objective_from_stdout(stdout: str) -> float:
    import re
    for pat in [
        r"Final solution value:\s*([0-9eE\+\-\.]+)",
        r"Primal Bound\s*[:=]\s*([0-9eE\+\-\.]+)",
        r"objective value\s*[:=]\s*([0-9eE\+\-\.]+)",
    ]:
        m = re.search(pat, stdout)
        if m:
            try:
                return float(m.group(1))
            except:
                pass
    return float("nan")


import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Set


def solve_stp_scipjack(
    num_nodes: int,
    edges: List[Tuple[int, int, float]],
    terminals: Set[int],
    time_limit: float = 60.0,
    stp_binary: str = "/home/jing/Downloads/scipoptsuite-6.0.2/scip/applications/STP/bin/stp.linux.x86_64.gnu.opt.spx2",
    workdir: str = "./scipjack_debug",
) -> Tuple[List[Tuple[int, int]], float]:

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    in_path = (workdir / "instance.stp").resolve()
    set_path = (workdir / "write.set").resolve()
    stplog_path = workdir / "instance.stplog"

    _write_stp_file(
        path=str(in_path),
        num_nodes=num_nodes,
        edges=edges,
        terminals=terminals,
        name="stp_subgraph",
    )

    with open(set_path, "w") as f:
        f.write('stp/logfile = "scipjack_debug/instance.stplog"\n')
    cmd = [
        stp_binary,
        "-f", str(in_path),
        "-s", str(set_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=time_limit,
        )
    except subprocess.TimeoutExpired:
        print(f"[Timeout] SCIP-Jack exceeded {time_limit}s.")
        return [], float("nan")


    if proc.returncode != 0:
        print("[Error] SCIP-Jack returned code:", proc.returncode)
        return [], float("nan")


    steiner_edges = _parse_solution_file(stplog_path)
    obj_value = _parse_objective_from_stdout(proc.stdout)

    print("[Debug] Parsed edges:", steiner_edges)
    print("[Debug] Obj:", obj_value)

    return steiner_edges, obj_value
