# ui/app.py
from __future__ import annotations

import difflib
import glob
import html
import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
PROBLEMS_DIR = ROOT / "problems"

# =========================================================
# Step3 keys (edit + validate)
# =========================================================
STEP3_KEYS_PATH = ROOT / "step3_evaluation" / "configs" / "keys.json"

KNOWN_KEY_FIELDS = [
    "openai",
    "anthropic",
    "deepseek",
    "gemini",
    "openrouter",
    "xiaomi",
    "vertex_project_id",
]

PROVIDER_REQUIRED_KEYS = {
    "openai": ["openai"],
    "anthropic": ["anthropic"],
    "deepseek": ["deepseek"],
    "gemini": ["gemini"],
    "openrouter": ["openrouter"],
    # Vertex MaaS usually needs project id (and auth via env/ADC); follow your existing keys.json design:
    "vertex_maas": ["vertex_project_id"],
}


def load_step3_keys() -> Dict[str, str]:
    try:
        if STEP3_KEYS_PATH.exists():
            obj = json.loads(STEP3_KEYS_PATH.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}
    return {}


def save_step3_keys(keys: Dict[str, str]) -> None:
    STEP3_KEYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STEP3_KEYS_PATH.write_text(json.dumps(keys, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def provider_missing_keys(provider: str, keys: Dict[str, str]) -> List[str]:
    need = PROVIDER_REQUIRED_KEYS.get(provider, [])
    missing = []
    for k in need:
        v = str(keys.get(k, "") or "").strip()
        if not v:
            missing.append(k)
    return missing


# =========================================================
# Helpers: filesystem / YAML / CSV
# =========================================================
def list_problems() -> List[str]:
    if not PROBLEMS_DIR.exists():
        return []
    out: List[str] = []
    for p in PROBLEMS_DIR.iterdir():
        if not p.is_dir():
            continue
        cfg_dir = p / "config"
        if not cfg_dir.exists():
            continue
        has_scale = bool(list(cfg_dir.glob("*.yaml")) or list(cfg_dir.glob("*.yml")))
        has_custom = bool(list((cfg_dir / "custom").glob("*.yaml"))) if (cfg_dir / "custom").exists() else False
        if has_scale or has_custom:
            out.append(p.name)
    return sorted(out)


def list_scale_and_custom(problem: str) -> Tuple[List[str], List[str]]:
    cfg_dir = PROBLEMS_DIR / problem / "config"
    if not cfg_dir.exists():
        return [], []

    scale_files = sorted(cfg_dir.glob("*.yaml")) + sorted(cfg_dir.glob("*.yml"))
    scales = [p.stem for p in scale_files]
    order = {"S": 0, "M": 1, "L": 2}
    scales.sort(key=lambda x: order.get(x.upper(), 999))

    custom_dir = cfg_dir / "custom"
    custom_files = sorted(custom_dir.glob("*.yaml")) if custom_dir.exists() else []
    customs = [p.stem for p in custom_files]
    return scales, customs


def cfg_path(problem: str, name: str, kind: str) -> Path:
    cfg_dir = PROBLEMS_DIR / problem / "config"
    if kind == "custom":
        return cfg_dir / "custom" / f"{name}.yaml"

    p1 = cfg_dir / f"{name}.yaml"
    p2 = cfg_dir / f"{name}.yml"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"No scale config: problems/{problem}/config/{name}.yaml")


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def safe_load_yaml(text: str) -> dict:
    obj = yaml.safe_load(text)
    return obj if isinstance(obj, dict) else {}


def validate_yaml(text: str) -> None:
    yaml.safe_load(text)


def safe_read_csv(csv_path: Path, nrows: int = 5000) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, nrows=nrows)
    except Exception:
        return pd.read_csv(csv_path, nrows=nrows, engine="python", encoding="utf-8")


def find_recent_csvs(search_root: Path, limit: int = 200) -> List[Path]:
    if not search_root.exists():
        return []
    csvs = [Path(p) for p in glob.glob(str(search_root / "**/*.csv"), recursive=True)]
    csvs = [p for p in csvs if p.is_file()]
    csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[:limit]


def infer_output_root_dir_from_yaml_text(cfg_text: str) -> Optional[Path]:
    cfg = safe_load_yaml(cfg_text)
    step2 = cfg.get("step2") or {}
    out = step2.get("output_root_dir") or step2.get("output_dir")
    if not out:
        return None
    out_path = Path(str(out))
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    return out_path


def infer_scale_from_yaml_text(cfg_text: str) -> Optional[str]:
    cfg = safe_load_yaml(cfg_text)
    s = cfg.get("scale")
    return str(s) if s else None


def fmt_mtime(p: Path) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
    except Exception:
        return "unknown"


def relpath_or_abs(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def compute_editor_dirty(selected_cfg_path: Optional[Path], editor_text: str) -> bool:
    if not selected_cfg_path or not selected_cfg_path.exists():
        return False
    try:
        disk_text = read_text(selected_cfg_path)
    except Exception:
        return False
    return disk_text.strip() != (editor_text or "").strip()


def compute_run_cfg_path(selected_cfg_path: Optional[Path], editor_text: str, edit_mode: bool) -> Optional[Path]:
    """
    Pure computation (no file writes):
    - if edit_mode and editor != disk => temp path that Run will use
    - else => selected config path
    """
    if not selected_cfg_path:
        return None
    if not edit_mode:
        return selected_cfg_path

    if compute_editor_dirty(selected_cfg_path, editor_text):
        return selected_cfg_path.with_suffix(".tmp.yaml")
    return selected_cfg_path


def cleanup_tmp_yaml_if_exists(base_yaml_path: Path) -> None:
    tmp = base_yaml_path.with_suffix(".tmp.yaml")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass


# =========================================================
# Helpers: live logs (stdout/stderr separated)
# =========================================================
_ERROR_RE = re.compile(r"\b(error|exception|traceback|failed)\b", re.IGNORECASE)
_WARN_RE = re.compile(r"\b(warn|warning)\b", re.IGNORECASE)


def _iter_lines_from_stream(stream, stream_name: str, q: "queue.Queue[Tuple[str, str]]"):
    buf = ""
    for chunk in iter(stream.readline, ""):
        if not chunk:
            break
        chunk = chunk.replace("\r", "\n")
        buf += chunk
        parts = buf.split("\n")
        buf = parts.pop()
        for line in parts:
            q.put((stream_name, line))
    if buf != "":
        q.put((stream_name, buf))


def run_cmd_capture_lines(
    cmd: List[str],
    cwd: Path,
    on_line: Callable[[str, str], None],
    extra_env: Optional[Dict[str, str]] = None,
) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    if len(cmd) >= 1 and cmd[0] in ("python", "python3") and "-u" not in cmd:
        cmd = cmd[:1] + ["-u"] + cmd[1:]

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    t_out = threading.Thread(target=_iter_lines_from_stream, args=(proc.stdout, "stdout", q), daemon=True)
    t_err = threading.Thread(target=_iter_lines_from_stream, args=(proc.stderr, "stderr", q), daemon=True)
    t_out.start()
    t_err.start()

    while True:
        try:
            stream_name, line = q.get(timeout=0.05)
            on_line(stream_name, line)
        except queue.Empty:
            pass

        if proc.poll() is not None:
            while True:
                try:
                    stream_name, line = q.get_nowait()
                    on_line(stream_name, line)
                except queue.Empty:
                    break
            break

    return int(proc.returncode or 0)


def render_log_box(entries: List[object], height_px: int = 320) -> str:
    line_html: List[str] = []
    for e in entries:
        if isinstance(e, dict):
            stream = str(e.get("stream", "stdout"))
            text = str(e.get("text", ""))
        else:
            stream = "stdout"
            text = str(e)

        cls = ["log-line", f"s-{stream}"]
        if _ERROR_RE.search(text):
            cls.append("is-error")
        elif _WARN_RE.search(text):
            cls.append("is-warn")

        esc = html.escape(text)
        line_html.append(f'<div class="{" ".join(cls)}">{esc}</div>')

    lines_joined = "\n".join(line_html)

    return f"""
<div id="logwrap" style="
  height:{int(height_px)}px;
  overflow-y:auto;
  padding:12px 14px;
  background: rgba(245,246,250,0.95);
  border-radius: 12px;
">
  <div class="log-inner">{lines_joined}</div>
</div>

<style>
  .log-inner {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.5;
    color: #111;
  }}
  .log-line {{
    margin: 0 0 3px 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
  .s-stdout {{ color: #111; }}
  .s-stderr {{ color: #8a2d2d; }}
  .is-warn {{
    background: rgba(255, 193, 7, 0.18);
    border-radius: 6px;
    padding: 2px 6px;
  }}
  .is-error {{
    background: rgba(220, 53, 69, 0.16);
    border-radius: 6px;
    padding: 2px 6px;
    font-weight: 600;
  }}
</style>

<script>
  const el = document.getElementById("logwrap");
  if (el) {{
    el.scrollTop = el.scrollHeight;
  }}
</script>
"""


def render_log_into_placeholder(ph: st.delta_generator.DeltaGenerator, entries: List[dict], height_px: int) -> None:
    ph.empty()
    with ph:
        components.html(
            render_log_box(entries, height_px=height_px),
            height=height_px + 40,
            scrolling=False,
        )


def extract_recent_issues(entries: List[dict], limit: int = 30) -> Tuple[List[str], List[str]]:
    errs: List[str] = []
    warns: List[str] = []
    for e in reversed(entries):
        t = str(e.get("text", ""))
        if _ERROR_RE.search(t):
            errs.append(t)
        elif _WARN_RE.search(t):
            warns.append(t)
        if len(errs) >= limit and len(warns) >= limit:
            break
    return list(reversed(errs[:limit])), list(reversed(warns[:limit]))


# =========================================================
# Run stage inference (simple, log-driven)
# =========================================================
STAGE_RULES: List[Tuple[str, List[str]]] = [
    ("Starting", ["running", "start", "begin"]),
    ("Step1: Instance generation", ["step1", "instance generation", "generate instances"]),
    ("Step2: Contextualization", ["step2", "contextual", "contextualization"]),
    ("Writing outputs", ["writing", "saving", "output_root", "output_dir", ".csv", ".json"]),
    ("Done", ["finished", "complete", "rc="]),
]


def infer_stage_from_line(line: str, current: str) -> str:
    s = (line or "").lower()
    for stage, keys in STAGE_RULES:
        for k in keys:
            if k in s:
                # Don't regress from Done
                if current == "Done":
                    return "Done"
                return stage
    return current


def stage_progress(stage: str) -> float:
    order = [
        "Starting",
        "Step1: Instance generation",
        "Step2: Contextualization",
        "Writing outputs",
        "Done",
    ]
    if stage not in order:
        return 0.0
    idx = order.index(stage)
    return min(1.0, max(0.0, idx / (len(order) - 1)))


# =========================================================
# Step3 preset matrix (FIXED model/provider/reasoning)
# =========================================================
EVAL_PRESETS: Dict[str, Dict[str, Any]] = {
    # OpenRouter
    "qwen/qwen3-14b (no reasoning, max_tokens=16384)": {
        "provider": "openrouter",
        "model": "qwen3-14b",
        "reasoning": "disabled",
        "reasoning_effort": None,
        "supports_effort": False,
        "default_max_tokens": 16384,
        "notes": "OpenRouter: add /no_thinking internally for qwen3-14b when reasoning disabled.",
    },
    "qwen/qwen3-14b (reasoning=enabled, effort=high)": {
        "provider": "openrouter",
        "model": "qwen3-14b",
        "reasoning": "enabled",
        "reasoning_effort": "high",
        "supports_effort": True,
        "default_max_tokens": None,
        "notes": "OpenRouter reasoning mode enabled.",
    },
    "qwen/qwq-32b (reasoning=enabled, effort=high)": {
        "provider": "openrouter",
        "model": "qwq-32b",
        "reasoning": "enabled",
        "reasoning_effort": "high",
        "supports_effort": True,
        "default_max_tokens": None,
        "notes": "OpenRouter reasoning mode enabled.",
    },
    "mistralai/ministral-14b-2512 (no reasoning, max_tokens=65536)": {
        "provider": "openrouter",
        "model": "ministral-14b-2512",
        "reasoning": "disabled",
        "reasoning_effort": None,
        "supports_effort": False,
        "default_max_tokens": 65536,
        "notes": "No reasoning.",
    },
    "nvidia/nemotron-3-nano-30b-a3b (reasoning=enabled, effort=high)": {
        "provider": "openrouter",
        "model": "nemotron-3-nano-30b-a3b",
        "reasoning": "enabled",
        "reasoning_effort": "high",
        "supports_effort": True,
        "default_max_tokens": None,
        "notes": "OpenRouter reasoning enabled.",
    },
    "nvidia/nemotron-3-nano-30b-a3b (no reasoning, max_tokens=65536)": {
        "provider": "openrouter",
        "model": "nemotron-3-nano-30b-a3b",
        "reasoning": "disabled",
        "reasoning_effort": None,
        "supports_effort": False,
        "default_max_tokens": 65536,
        "notes": "No reasoning.",
    },
    "xiaomi/mimo-v2-flash (no reasoning, max_tokens=65536)": {
        "provider": "openrouter",
        "model": "mimo-v2-flash",
        "reasoning": "disabled",
        "reasoning_effort": None,
        "supports_effort": False,
        "default_max_tokens": 65536,
        "notes": "OpenRouter route for mimo-v2-flash:free.",
    },
    "x-ai/grok-4.1-fast (reasoning=enabled, effort=high)": {
        "provider": "openrouter",
        "model": "grok-4.1-fast",
        "reasoning": "enabled",
        "reasoning_effort": "high",
        "supports_effort": True,
        "default_max_tokens": None,
        "notes": "OpenRouter reasoning enabled.",
    },
    # Vertex MaaS
    "vertex_maas: meta/llama-4-maverick-17b-128e (no reasoning, max_tokens=8192)": {
        "provider": "vertex_maas",
        "model": "llama-4-maverick-17b-128e-instruct-maas",
        "reasoning": "disabled",
        "reasoning_effort": None,
        "supports_effort": False,
        "default_max_tokens": 8192,
        "notes": "Vertex MaaS OpenAI-compatible endpoint; temperature fixed to 0.",
    },
    "vertex_maas: qwen3-235b-a22b-instruct-2507 (no reasoning, max_tokens=16384)": {
        "provider": "vertex_maas",
        "model": "qwen3-235b-a22b-instruct-2507-maas",
        "reasoning": "disabled",
        "reasoning_effort": None,
        "supports_effort": False,
        "default_max_tokens": 16384,
        "notes": "Vertex MaaS OpenAI-compatible endpoint; temperature fixed to 0.",
    },
    # Anthropic
    "anthropic: claude-sonnet-4-5 (no reasoning, max_tokens=64000)": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "reasoning": "disabled",
        "reasoning_effort": None,
        "supports_effort": False,
        "default_max_tokens": 64000,
        "default_anthropic_budget": None,
        "notes": "Anthropic online evaluation.",
    },
    # OpenAI (Flex)
    "openai: o4-mini (reasoning=enabled, effort=high, flex)": {
        "provider": "openai",
        "model": "o4-mini",
        "reasoning": "enabled",
        "reasoning_effort": "high",
        "supports_effort": True,
        "default_max_tokens": None,
        "notes": "OpenAI uses Responses API (service_tier=flex).",
    },
    "openai: gpt-5.1 (reasoning=enabled, effort=medium, flex)": {
        "provider": "openai",
        "model": "gpt-5.1",
        "reasoning": "enabled",
        "reasoning_effort": "medium",
        "supports_effort": True,
        "default_max_tokens": None,
        "notes": "OpenAI uses Responses API (service_tier=flex).",
    },
    # DeepSeek
    "deepseek: deepseek-reasoner (reasoning=enabled)": {
        "provider": "deepseek",
        "model": "deepseek-reasoner",
        "reasoning": "enabled",
        "reasoning_effort": None,
        "supports_effort": False,
        "default_max_tokens": None,
        "notes": "DeepSeek reasoning enabled.",
    },
    "deepseek: deepseek-chat (no reasoning, max_tokens=8192)": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "reasoning": "disabled",
        "reasoning_effort": None,
        "supports_effort": False,
        "default_max_tokens": 8192,
        "notes": "DeepSeek chat no reasoning.",
    },
    # Gemini
    "gemini: gemini-3-flash-preview (reasoning=enabled, effort=high)": {
        "provider": "gemini",
        "model": "gemini-3-flash-preview",
        "reasoning": "enabled",
        "reasoning_effort": "high",
        "supports_effort": True,
        "default_max_tokens": None,
        "notes": "Gemini 3 thinking enabled.",
    },
}


def _preset_summary(preset: Dict[str, Any]) -> str:
    provider = preset["provider"]
    model = preset["model"]
    reasoning = preset["reasoning"]
    effort = preset.get("reasoning_effort")
    s = f"provider={provider}, model={model}, reasoning={reasoning}"
    if reasoning == "enabled" and effort:
        s += f", effort={effort}"
    if preset.get("default_max_tokens") is not None:
        s += f", max_tokens={preset['default_max_tokens']}"
    return s


def unified_diff_text(a: str, b: str, fromfile: str = "disk", tofile: str = "editor") -> str:
    a_lines = (a or "").splitlines(keepends=True)
    b_lines = (b or "").splitlines(keepends=True)
    diff = difflib.unified_diff(a_lines, b_lines, fromfile=fromfile, tofile=tofile, lineterm="")
    return "\n".join(list(diff))


# =========================================================
# Session init
# =========================================================
def ss_init():
    defaults = {
        "yaml_editor": "",
        "last_selected_key": "",
        "logs": [],
        "log_height": 420,
        "csvs": [],
        "csv_chosen": None,
        "run_status": "",
        "run_stage": "Starting",
        "run_rc": None,
        "run_started_ts": None,
        "run_finished_ts": None,
        "live_logs": True,  # throttle/log refresh mode
        "run_history": [],  # list of dicts for recent runs

        "custom_name": "my_config",
        "edit_mode": True,  # keep for run logic
        "yaml_source": "scale",

        # track what THIS run used (for correct UI display)
        "active_selected_cfg_rel": "",
        "active_selected_cfg_abs": "",
        "active_run_cfg_rel": "",
        "active_run_cfg_abs": "",
        "active_output_root": "",
        "active_csv_count": 0,

        # ---- Step3 eval state ----
        "eval_preset_name": list(EVAL_PRESETS.keys())[0],
        "eval_problem_mode": "current",  # current / all / select
        "eval_selected_problems": [],
        "eval_sizes": ["S", "M", "L"],
        "eval_output_root": str(ROOT / "eval_outputs"),
        "eval_max_concurrency": 50,
        "eval_num_test": None,
        "eval_logs": [],
        "eval_csvs": [],
        "eval_csv_chosen": None,
        "eval_status": "",

        # eval results filter UI
        "eval_filter_problem": "ALL",
        "eval_filter_model": "ALL",

        # CSV preview options
        "csv_preview_cols": [],
        "csv_filter_query": "",
        "csv_max_rows": 2000,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ss_init()

# =========================================================
# UI
# =========================================================
st.set_page_config(page_title="NLCO Pipeline UI", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem;
    }
    h1 {
        margin-top: 0.2rem !important;
        padding-top: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
div.stButton > button { border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("NLCO")
st.markdown(
    """
NLCO targets **end-to-end combinatorial optimization reasoning**: the model receives a **language-language description of a CO problem**
and must output a **structured solution** *without code or external solvers*.
    """
)

problems = list_problems()
if not problems:
    st.error(f"No problems found under: {PROBLEMS_DIR}")
    st.stop()

is_running = st.session_state.run_status == "running"

col_left, col_right = st.columns([0.3, 0.7], gap="large")

# =========================================================
# Left: Parameters
# =========================================================
with col_left:
    st.markdown("### Parameters")
    st.caption("Choose a config and tune YAML fields for NLCO construction.")

    st.markdown(
        """
        <style>
        div[data-baseweb="tab-list"] { gap: 8px; }
        button[data-baseweb="tab"] {
            font-size: 15px;
            padding: 8px 16px;
            height: 40px;
        }
        div[data-baseweb="tab-panel"] {
            max-height: 560px;
            overflow-y: auto;
            padding-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_cfg, tab_edit = st.tabs(["Config", "Edit & Save"])

    # -----------------------------
    # Tab 1: Config
    # -----------------------------
    with tab_cfg:
        with st.expander("Guide", expanded=False):
            st.markdown(
                """
- Select **Problem → Source → YAML** to specify the NLCO benchmark setting.
- (Optional) Edit the configuration in **Edit & Save**.
- Click **🚀 Run** to construct benchmark instances.
- Use **Evaluation** to assess end-to-end solution quality across models.
- Inspect logs and generated CSV files in the workspace.
                """
            )

        # ---- Problem ----
        problem = st.selectbox("Problem", problems, index=0, disabled=is_running)

        scales, customs = list_scale_and_custom(problem)
        if not scales and not customs:
            st.error(f"No YAML files under problems/{problem}/config (or config/custom).")
            st.stop()

        st.caption(f"Configs available — scale: **{len(scales)}**, custom: **{len(customs)}**")

        # ---- Source selection ----
        st.session_state.yaml_source = st.radio(
            "YAML source",
            ["scale", "custom"],
            index=0 if scales else 1,
            horizontal=True,
            disabled=is_running,
        )

        selected_cfg_path: Optional[Path] = None
        selected_kind = st.session_state.yaml_source

        # ---- File selection ----
        if selected_kind == "scale":
            scale = st.selectbox(
                "YAML file",
                scales,
                index=0,
                disabled=is_running,
                help="Preset configs under problems/<problem>/config/",
            )
            selected_cfg_path = cfg_path(problem, scale, "scale")
            selected_name = scale
        else:
            if customs:
                chosen_custom = st.selectbox(
                    "YAML file",
                    customs,
                    index=0,
                    disabled=is_running,
                    help="Custom configs under problems/<problem>/config/custom/",
                )
                selected_cfg_path = cfg_path(problem, chosen_custom, "custom")
                selected_name = chosen_custom
            else:
                st.warning(
                    "No custom YAML found.\n\nCreate one via **Edit & Save → Save as custom**.",
                    icon="⚠️",
                )
                selected_cfg_path = None
                selected_name = ""

        # ---- Load editor when selection changes ----
        selection_key = f"{problem}::{selected_kind}::{selected_name}"
        if selected_cfg_path and selection_key != st.session_state.last_selected_key:
            st.session_state.yaml_editor = read_text(selected_cfg_path)
            st.session_state.last_selected_key = selection_key

        # ---- Selected + Run config info ----
        if selected_cfg_path:
            st.info(f"Selected config: `{relpath_or_abs(selected_cfg_path)}`", icon="🧩")

        # Compute (no side-effects)
        st.session_state.edit_mode = True  # ensure run-safe behavior (no view mode)
        run_cfg_path = compute_run_cfg_path(selected_cfg_path, st.session_state.yaml_editor, st.session_state.edit_mode)
        dirty_now = compute_editor_dirty(selected_cfg_path, st.session_state.yaml_editor)

        if selected_cfg_path and run_cfg_path:
            if dirty_now and run_cfg_path != selected_cfg_path:
                st.warning(
                    f"Run config (unsaved edits): `{relpath_or_abs(run_cfg_path)}`",
                    icon="📝",
                )
            else:
                st.caption("Run config matches disk file (no unsaved edits).")

        # ---- If currently running / finished, show what THIS run actually used ----
        if st.session_state.active_run_cfg_rel:
            st.caption(f"Active run config (last run): `{st.session_state.active_run_cfg_rel}`")

        if is_running:
            st.info("Pipeline is running. Editing and saving are locked.", icon="⏳")

        # ---- Recent runs (session) ----
        if st.session_state.run_history:
            with st.expander("Recent runs", expanded=False):
                for i, r in enumerate(reversed(st.session_state.run_history[-8:]), start=1):
                    when = r.get("when", "")
                    prob = r.get("problem", "")
                    cfg = r.get("run_cfg", "")
                    rc = r.get("rc", "")
                    dur = r.get("duration_s", None)
                    dur_s = f"{dur:.1f}s" if isinstance(dur, (int, float)) else "?"
                    st.markdown(f"- `{when}` | **{prob}** | rc={rc} | {dur_s} | `{cfg}`")

        st.divider()

    # -----------------------------
    # Tab 2: Edit & Save (always editor; no view mode)
    # -----------------------------
    with tab_edit:
        st.session_state.edit_mode = True

        with st.expander("Guide", expanded=False):
            st.markdown(
                """
            YAML controls **benchmark construction**: instance generation and contextualization.

            **Step1 (instance generation)**
            - `step1.dataset_path`: source dataset location (problem-dependent)
            - `step1.n_instances`: how many instances to generate
            - `step1.output`: output JSON path
            - `step1.params`: extra generator parameters (optional)

            **Step2 (contextualization)**
            - `step2.instance_dir`: where Step1 JSON files are
            - `step2.output_root_dir` / `step2.output_dir`: where contextualized JSON/CSV will be saved
            - `step2.k_per_call`: scenarios per LLM call
            - `step2.n_target`: total scenarios to generate
            - `step2.load_cached_contexts`: whether to load pre-generated contexts from cache (e.g., `false` to always regenerate)
            - `step2.contexts_audit_path`: path to cached/audit contexts JSON  

            Tip: edits do **not** need to be saved to run — unsaved edits are executed via a temporary config.
                """
            )

        # ---- Quick edit panel (small, high-value) ----
        with st.expander("Quick edit", expanded=False):
            cfg_obj = safe_load_yaml(st.session_state.yaml_editor or "")
            step1 = cfg_obj.get("step1") or {}
            step2 = cfg_obj.get("step2") or {}

            q1, q2 = st.columns(2, gap="small")
            with q1:
                n_instances = st.number_input(
                    "step1.n_instances",
                    min_value=1,
                    value=int(step1.get("n_instances", 1) or 1),
                    step=1,
                    disabled=is_running,
                )
                step1["n_instances"] = int(n_instances)
            with q2:
                n_target = st.number_input(
                    "step2.n_target",
                    min_value=1,
                    value=int(step2.get("n_target", 1) or 1),
                    step=1,
                    disabled=is_running,
                )
                step2["n_target"] = int(n_target)

            q3, q4 = st.columns(2, gap="small")
            with q3:
                k_per_call = st.number_input(
                    "step2.k_per_call",
                    min_value=1,
                    value=int(step2.get("k_per_call", 1) or 1),
                    step=1,
                    disabled=is_running,
                )
                step2["k_per_call"] = int(k_per_call)
            with q4:
                out_root = st.text_input(
                    "step2.output_root_dir",
                    value=str(step2.get("output_root_dir") or ""),
                    disabled=is_running,
                )
                if out_root.strip():
                    step2["output_root_dir"] = out_root.strip()

            cfg_obj["step1"] = step1
            cfg_obj["step2"] = step2

            apply_quick = st.button("Apply quick edits to YAML text", disabled=is_running, use_container_width=True)
            if apply_quick:
                try:
                    # keep yaml readable
                    st.session_state.yaml_editor = yaml.safe_dump(cfg_obj, sort_keys=False, allow_unicode=True)
                    st.toast("Quick edits applied.", icon="🧩")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to dump YAML: {e}")

        st.session_state.yaml_editor = st.text_area(
            "YAML content",
            value=st.session_state.yaml_editor,
            height=340,
            disabled=is_running,
        )

        st.markdown("#### Save")

        disk_text = read_text(selected_cfg_path) if selected_cfg_path and selected_cfg_path.exists() else ""
        is_dirty = (st.session_state.yaml_editor or "").strip() != (disk_text or "").strip()

        if is_dirty:
            st.warning("Modified (not saved).", icon="📝")
            with st.expander("View diff (disk vs editor)", expanded=False):
                diff_txt = unified_diff_text(disk_text, st.session_state.yaml_editor, fromfile="disk", tofile="editor")
                st.code(diff_txt or "(no diff?)", language="diff")
        else:
            st.caption("No unsaved changes.")

        save_mode = st.radio(
            "Save destination",
            ["Overwrite current file", "Save as custom"],
            index=0,
            horizontal=True,
            disabled=is_running,
        )

        custom_name = None
        if save_mode == "Save as custom":
            st.session_state.custom_name = st.text_input(
                "Custom YAML name",
                value=st.session_state.custom_name,
                help="Saved to problems/<problem>/config/custom/<name>.yaml",
                disabled=is_running,
            )
            custom_name = (st.session_state.custom_name or "my_config").strip() or "my_config"

        c1, c2 = st.columns([1, 1], gap="small")
        with c1:
            btn_save = st.button("💾 Save", use_container_width=True, disabled=is_running)
        with c2:
            btn_discard = st.button("🗑️ Discard changes", use_container_width=True, disabled=is_running)

        if btn_discard:
            if selected_cfg_path and selected_cfg_path.exists():
                st.session_state.yaml_editor = read_text(selected_cfg_path)
                st.toast("Discarded changes.", icon="🗑️")
                st.rerun()
            else:
                st.warning("Nothing to discard.")

        if btn_save:
            try:
                validate_yaml(st.session_state.yaml_editor)
            except Exception as e:
                st.error(f"YAML parse error (not saved):\n\n{e}")
                st.stop()

            if save_mode == "Overwrite current file":
                if not selected_cfg_path:
                    st.error("No YAML selected to overwrite.")
                    st.stop()
                write_text(selected_cfg_path, st.session_state.yaml_editor)
                st.toast(f"Saved: {selected_cfg_path.name}", icon="💾")
                st.rerun()
            else:
                target = PROBLEMS_DIR / problem / "config" / "custom" / f"{custom_name}.yaml"
                write_text(target, st.session_state.yaml_editor)
                st.toast(f"Saved custom: {relpath_or_abs(target)}", icon="🧩")
                st.session_state.yaml_source = "custom"
                st.session_state.last_selected_key = ""
                st.rerun()

        # ---- Delete config option ----
        st.divider()
        st.markdown("#### Delete config")

        if selected_cfg_path and selected_cfg_path.exists():
            is_scale_cfg = (selected_kind == "scale")
            can_delete = True

            if is_scale_cfg:
                st.warning(
                    "You are viewing a **scale (preset)** config. Deleting it may break the repo defaults.",
                    icon="⚠️",
                )
                confirm_scale = st.checkbox(
                    "I understand. Allow deleting scale preset files.",
                    value=False,
                    disabled=is_running,
                )
                can_delete = bool(confirm_scale) and not is_running
            else:
                can_delete = not is_running

            confirm_delete = st.checkbox(
                "Confirm delete (cannot undo)",
                value=False,
                disabled=is_running,
            )
            del_btn = st.button(
                "🗑️ Delete selected config file",
                type="secondary",
                use_container_width=True,
                disabled=(not can_delete) or (not confirm_delete) or is_running,
            )
            if del_btn:
                try:
                    selected_cfg_path.unlink()
                    st.toast(f"Deleted: {relpath_or_abs(selected_cfg_path)}", icon="🗑️")
                    st.session_state.last_selected_key = ""
                    # move selection to scale by default after delete
                    if selected_kind == "custom":
                        st.session_state.yaml_source = "scale"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete file: {e}")
        else:
            st.caption("No selected config file to delete.")


# =========================================================
# Right: Workspace
# =========================================================
with col_right:
    bar = st.columns([0.62, 0.20, 0.18], gap="small")

    with bar[0]:
        st.markdown("### Workspace")
        st.caption(f"Status: `{st.session_state.run_status or 'idle'}`")

    with bar[1]:
        # vertical spacer for alignment
        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

        if st.button("🧹 Clear logs", use_container_width=True, disabled=is_running):
            st.session_state.logs = []
            st.rerun()

    with bar[2]:
        # vertical spacer for alignment
        st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

        run_btn = st.button(
            "🚀 Run",
            type="primary",
            use_container_width=True,
            disabled=is_running,
        )

    st.markdown(
        """
        <style>
        div[data-baseweb="tab-list"] { gap: 8px; }
        button[data-baseweb="tab"] {
            font-size: 15px;
            padding: 8px 16px;
            height: 40px;
        }
        div[data-baseweb="tab-panel"] {
            max-height: 560px;
            overflow-y: auto;
            padding-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_logs, tab_csv, tab_eval = st.tabs(["Logs", "Output CSV Preview", "Evaluation"])

    with tab_logs:
        st.info(
            "Run executes the pipeline and streams logs here. "
            "stderr is colored red; lines containing error/exception/traceback/failed are highlighted.",
            icon="ℹ️",
        )
        log_placeholder = st.empty()
        tail = st.session_state.logs[-1200:]
        render_log_into_placeholder(log_placeholder, tail, height_px=int(st.session_state.log_height))

        # Errors / warnings summary
        errs, warns = extract_recent_issues(st.session_state.logs, limit=25)
        with st.expander(f"Issues summary (errors={len(errs)}, warnings={len(warns)})", expanded=False):
            if errs:
                st.markdown("**Recent errors**")
                st.code("\n".join(errs[-25:]), language="text")
            else:
                st.caption("No errors detected in recent log lines.")
            st.divider()
            if warns:
                st.markdown("**Recent warnings**")
                st.code("\n".join(warns[-25:]), language="text")
            else:
                st.caption("No warnings detected in recent log lines.")

    with tab_csv:
        out_root_hint = infer_output_root_dir_from_yaml_text(st.session_state.yaml_editor) or (ROOT / "step2_contextualization")
        st.info(
            f"CSV are discovered by scanning: `{relpath_or_abs(out_root_hint)}`. "
            "If nothing shows up, check YAML `step2.output_root_dir` / `step2.output_dir`.",
            icon="ℹ️",
        )

        if st.session_state.csvs:
            opts = [str(p.relative_to(ROOT)) if p.is_absolute() else str(p) for p in st.session_state.csvs]
            # (they are absolute paths; normalize to ROOT-relative where possible)
            opts = [relpath_or_abs(Path(o)) for o in st.session_state.csvs]

            default_idx = 0
            if st.session_state.csv_chosen in opts:
                default_idx = opts.index(st.session_state.csv_chosen)

            chosen = st.selectbox("Select CSV", opts, index=default_idx)
            st.session_state.csv_chosen = chosen

            abs_path = ROOT / chosen if not Path(chosen).is_absolute() else Path(chosen)
            st.caption(f"Selected: `{relpath_or_abs(abs_path)}`  |  mtime: `{fmt_mtime(abs_path)}`")

            df = safe_read_csv(abs_path)
            st.dataframe(df, use_container_width=True, height=560)
        else:
            st.warning(
                "No CSV found yet. Run the pipeline, then return here. "
                "If you still see nothing, confirm the scan root in the message above.",
                icon="⚠️",
            )

    with tab_eval:
        with st.expander("Guide", expanded=False):
            st.markdown(
                """
Evaluation runs a Step3 CLI over generated benchmark artifacts.

- Pick a **Model preset** (provider/model/reasoning are fixed by preset).
- Choose **Problems**: current / all / select.
- Choose **Sizes**: S/M/L.
- Confirm `step2 output root` (auto inferred from current YAML).
- Set `output_root` (where evaluation CSVs are written).
- Click **Run Evaluation** and watch eval logs below.

API Keys
- Step3 reads provider keys from `step3_evaluation/configs/keys.json`.
- You can update keys directly in the UI under **API Keys**, or edit the JSON file manually.
                """
            )
        # -----------------------------
        # Step3 API Keys (editable)
        # -----------------------------
        # -----------------------------
        # Step3 API Keys (editable)
        # -----------------------------
        with st.expander("API Keys", expanded=False):
            st.caption(f"Edit: `{relpath_or_abs(STEP3_KEYS_PATH)}`")

            keys_now = load_step3_keys()

            # Only reveal plaintext when user explicitly asks (safer default)
            reveal = st.toggle(
                "Reveal secret keys (unsafe)",
                value=False,
                help="Off by default to avoid leaking keys on screen recordings / screenshots.",
            )


            def _is_set(k: str) -> bool:
                return bool(str(keys_now.get(k, "") or "").strip())


            def _mask(v: str, head: int = 6, tail: int = 4) -> str:
                v = (v or "").strip()
                if not v:
                    return ""
                if len(v) <= head + tail + 3:
                    return v[:2] + "…" + v[-2:]
                return f"{v[:head]}…{v[-tail:]}"


            # status row
            cols = st.columns(3, gap="small")
            with cols[0]:
                st.caption("Status (from keys.json)")
            with cols[1]:
                st.caption("Required keys are checked before **Run Evaluation**.")
            with cols[2]:
                if st.button("↩️ Reload", use_container_width=True):
                    st.rerun()

            st.markdown("#### Current keys")
            for k in KNOWN_KEY_FIELDS:
                if k == "vertex_project_id":
                    shown = str(keys_now.get(k, "") or "").strip() if reveal else _mask(
                        str(keys_now.get(k, "") or "").strip(), 4, 2
                    )
                else:
                    shown = str(keys_now.get(k, "") or "").strip() if reveal else _mask(
                        str(keys_now.get(k, "") or "").strip()
                    )

                if _is_set(k):
                    st.markdown(f"- **{k}**: ✅  `{shown}`")
                else:
                    st.markdown(f"- **{k}**: ⚠️ missing")

            st.divider()

            st.markdown("#### Edit")
            st.caption("Leave blank to keep existing value. Secret inputs are masked.")

            inp_openai = st.text_input(
                "openai",
                value=(str(keys_now.get("openai", "") or "").strip() if reveal else ""),
                type="password",
                help="OpenAI API key",
            )
            inp_anthropic = st.text_input(
                "anthropic",
                value=(str(keys_now.get("anthropic", "") or "").strip() if reveal else ""),
                type="password",
                help="Anthropic API key",
            )
            inp_deepseek = st.text_input(
                "deepseek",
                value=(str(keys_now.get("deepseek", "") or "").strip() if reveal else ""),
                type="password",
                help="DeepSeek API key",
            )
            inp_gemini = st.text_input(
                "gemini",
                value=(str(keys_now.get("gemini", "") or "").strip() if reveal else ""),
                type="password",
                help="Gemini API key",
            )
            inp_openrouter = st.text_input(
                "openrouter",
                value=(str(keys_now.get("openrouter", "") or "").strip() if reveal else ""),
                type="password",
                help="OpenRouter API key",
            )
            inp_xiaomi = st.text_input(
                "xiaomi",
                value=(str(keys_now.get("xiaomi", "") or "").strip() if reveal else ""),
                type="password",
                help="Xiaomi API key",
            )

            inp_vertex_pid = st.text_input(
                "vertex_project_id",
                value=str(keys_now.get("vertex_project_id", "") or "").strip(),
                help="Vertex project id.",
            )

            cA, cB = st.columns([0.25, 0.75], gap="small")
            with cA:
                save_keys_btn = st.button("💾 Save keys", use_container_width=True)
            with cB:
                st.caption("Tip: keep **Reveal** off unless you really need to copy/check the full key.")

            if save_keys_btn:
                updates = {
                    "openai": inp_openai,
                    "anthropic": inp_anthropic,
                    "deepseek": inp_deepseek,
                    "gemini": inp_gemini,
                    "openrouter": inp_openrouter,
                    "xiaomi": inp_xiaomi,
                }

                # If reveal is off, inputs default empty -> means "keep existing"
                for k, v in updates.items():
                    vv = (v or "").strip()
                    if vv:
                        keys_now[k] = vv

                # non-secret
                keys_now["vertex_project_id"] = (inp_vertex_pid or "").strip()

                save_step3_keys(keys_now)
                st.toast("Saved keys.json", icon="🔐")
                st.rerun()

        inferred_step2_root = infer_output_root_dir_from_yaml_text(st.session_state.yaml_editor)
        dataset_root_auto = None
        if inferred_step2_root is not None:
            dataset_root_auto = Path(inferred_step2_root).parent
        else:
            dataset_root_auto = ROOT

        st.session_state["eval_dataset_root_auto"] = str(dataset_root_auto)
        st.caption(f"step2 output root: `{relpath_or_abs(dataset_root_auto)}`")

        preset_names = list(EVAL_PRESETS.keys())
        if st.session_state.eval_preset_name not in EVAL_PRESETS:
            st.session_state.eval_preset_name = preset_names[0]

        st.session_state.eval_preset_name = st.selectbox(
            "Model preset (provider/model/reasoning are fixed)",
            preset_names,
            index=preset_names.index(st.session_state.eval_preset_name),
        )
        preset = EVAL_PRESETS[st.session_state.eval_preset_name]
        st.caption(_preset_summary(preset))
        if preset.get("notes"):
            st.info(preset["notes"], icon="ℹ️")

        # Early key check (immediate UX feedback)
        keys_now = load_step3_keys()
        missing_now = provider_missing_keys(preset["provider"], keys_now)
        if missing_now:
            st.error(
                "Missing API key(s) for this provider: "
                + ", ".join([f"`{m}`" for m in missing_now])
                + "\n\nFix it in **API Keys** above before running evaluation."
            )

        if preset["provider"] == "openai":
            st.warning(
                "OpenAI models here are evaluated via **OpenAI Responses API** with **service_tier=flex** "
                "(configured in `step3_evaluation/utils/llm.py`).",
                icon="⚡",
            )

        st.session_state.eval_problem_mode = st.radio(
            "Problems",
            ["current", "all", "select"],
            index={"current": 0, "all": 1, "select": 2}.get(st.session_state.eval_problem_mode, 0),
            horizontal=True,
        )

        if st.session_state.eval_problem_mode == "current":
            selected_problems = [problem]
            problem_arg = problem
            st.caption(f"Will evaluate: `{problem}`")
        elif st.session_state.eval_problem_mode == "all":
            selected_problems = []
            problem_arg = "all"
            st.caption("Will evaluate: `all`")
        else:
            st.session_state.eval_selected_problems = st.multiselect(
                "Select problems",
                options=problems,
                default=st.session_state.eval_selected_problems or [problem],
            )
            selected_problems = st.session_state.eval_selected_problems or [problem]
            problem_arg = ",".join(selected_problems)
            st.caption(f"Will evaluate: `{problem_arg}`")

        cS, cM, cL = st.columns(3, gap="small")
        s_checked = cS.checkbox("S", value=("S" in st.session_state.eval_sizes))
        m_checked = cM.checkbox("M", value=("M" in st.session_state.eval_sizes))
        l_checked = cL.checkbox("L", value=("L" in st.session_state.eval_sizes))
        sizes = []
        if s_checked:
            sizes.append("S")
        if m_checked:
            sizes.append("M")
        if l_checked:
            sizes.append("L")
        st.session_state.eval_sizes = sizes or ["S"]
        sizes_arg = ",".join(st.session_state.eval_sizes)

        st.session_state.eval_output_root = st.text_input(
            "output_root",
            value=st.session_state.eval_output_root,
            help="Root directory for evaluation CSV outputs",
        )

        c1, c2, c3 = st.columns([0.34, 0.33, 0.33], gap="small")
        with c1:
            st.session_state.eval_max_concurrency = st.number_input(
                "max_concurrency",
                min_value=1,
                value=int(st.session_state.eval_max_concurrency or 50),
                step=1,
            )
        with c2:
            num_test_val = st.text_input(
                "num_test (optional)",
                value="" if st.session_state.eval_num_test is None else str(st.session_state.eval_num_test),
                help="Limit to first N instances for quick testing",
            )
            st.session_state.eval_num_test = int(num_test_val) if num_test_val.strip().isdigit() else None
        with c3:
            if preset.get("default_max_tokens") is not None:
                preset_default = int(preset["default_max_tokens"])
                max_tokens = st.number_input(
                    "max_tokens",
                    min_value=1,
                    value=int(st.session_state.get("eval_max_tokens") or preset_default),
                    step=256,
                    help="This preset supports/needs explicit max_tokens.",
                )
                st.session_state["eval_max_tokens"] = int(max_tokens)
            else:
                st.caption("max_tokens: (not passed for this preset)")
                st.session_state["eval_max_tokens"] = None

        anthropic_budget = preset.get("default_anthropic_budget", None)

        st.divider()

        eval_cols = st.columns([0.25, 0.25, 0.5], gap="small")
        with eval_cols[0]:
            eval_run_btn = st.button("🧪 Run Evaluation", type="primary", use_container_width=True)
        with eval_cols[1]:
            if st.button("🧹 Clear eval logs", use_container_width=True):
                st.session_state.eval_logs = []
                st.rerun()
        with eval_cols[2]:
            st.caption(f"Eval status: `{st.session_state.eval_status or 'idle'}`")

        eval_log_placeholder = st.empty()
        tail_eval = st.session_state.eval_logs[-1200:]
        render_log_into_placeholder(eval_log_placeholder, tail_eval, height_px=420)
        st.divider()

        st.markdown("#### Evaluation Results Preview")
        eval_out_root = Path(str(st.session_state.eval_output_root))
        if not eval_out_root.is_absolute():
            eval_out_root = ROOT / eval_out_root

        refresh = st.button("🔄 Refresh eval outputs", use_container_width=False)
        if refresh or not st.session_state.eval_csvs:
            st.session_state.eval_csvs = find_recent_csvs(eval_out_root)
            if st.session_state.eval_csvs and not st.session_state.eval_csv_chosen:
                st.session_state.eval_csv_chosen = relpath_or_abs(st.session_state.eval_csvs[0])


        def _extract_problem_model(p: Path):
            rel = p
            try:
                rel = p.relative_to(eval_out_root)
            except Exception:
                pass
            parts = list(rel.parts)
            prob = parts[0] if parts else "UNKNOWN"
            return prob


        meta = []
        for p in st.session_state.eval_csvs:
            prob2 = _extract_problem_model(p)
            meta.append((p, prob2))

        prob_opts = ["ALL"] + sorted({m[1] for m in meta if m[1]})

        # --- apply filter ---
        filtered = []
        for p, prob2 in meta:
            if st.session_state.eval_filter_problem != "ALL" and prob2 != st.session_state.eval_filter_problem:
                continue
            filtered.append(p)

        if filtered:
            opts = [relpath_or_abs(p) for p in filtered]

            # keep current selection valid
            if st.session_state.eval_csv_chosen not in opts:
                st.session_state.eval_csv_chosen = opts[0]

            default_idx = 0
            if st.session_state.eval_csv_chosen in opts:
                default_idx = opts.index(st.session_state.eval_csv_chosen)

            c1, c2 = st.columns([0.2, 0.8], gap="small")

            with c1:
                st.session_state.eval_filter_problem = st.selectbox(
                    "Filter: problem",
                    prob_opts,
                    index=prob_opts.index(st.session_state.eval_filter_problem)
                    if st.session_state.eval_filter_problem in prob_opts
                    else 0,
                    key="eval_filter_problem_select",
                )

            with c2:
                chosen = st.selectbox(
                    "Select eval CSV",
                    opts,
                    index=default_idx,
                    key="eval_csv_select",
                )
                st.session_state.eval_csv_chosen = chosen

            abs_path = ROOT / chosen if not Path(chosen).is_absolute() else Path(chosen)
            st.caption(f"Selected: `{relpath_or_abs(abs_path)}`  |  mtime: `{fmt_mtime(abs_path)}`")

            df = safe_read_csv(abs_path)
            st.dataframe(df, use_container_width=True, height=460)
        else:
            st.caption("No eval CSVs match the current filters.")

        if eval_run_btn:
            # --- Key check before running ---
            keys_now = load_step3_keys()
            missing = provider_missing_keys(preset["provider"], keys_now)
            if missing:
                st.error(
                    "Missing API key(s) for this provider: "
                    + ", ".join([f"`{m}`" for m in missing])
                    + f"\n\nEdit `{relpath_or_abs(STEP3_KEYS_PATH)}` in **API Keys (Step3)** above."
                )
                st.stop()

            cmd = [
                "python", "-m", "step3_evaluation.eval_cli",
                "--problem", str(problem_arg),
                "--sizes", str(sizes_arg),
                "--dataset_root", str(st.session_state["eval_dataset_root_auto"]),
                "--output_root", str(st.session_state.eval_output_root),
                "--model", str(preset["model"]),
                "--provider", str(preset["provider"]),
                "--max_concurrency", str(int(st.session_state.eval_max_concurrency)),
                "--reasoning", str(preset["reasoning"]),
            ]

            if preset["reasoning"] == "enabled" and preset.get("supports_effort") and preset.get("reasoning_effort"):
                cmd += ["--reasoning_effort", str(preset["reasoning_effort"])]

            if preset.get("default_max_tokens") is not None and st.session_state.get("eval_max_tokens") is not None:
                cmd += ["--max_tokens", str(int(st.session_state["eval_max_tokens"]))]

            if preset["provider"] == "anthropic" and anthropic_budget is not None:
                cmd += ["--anthropic_budget", str(int(anthropic_budget))]

            if st.session_state.eval_num_test is not None:
                cmd += ["--num_test", str(int(st.session_state.eval_num_test))]

            st.session_state.eval_logs = []
            st.session_state.eval_status = "running"
            render_log_into_placeholder(eval_log_placeholder, [], height_px=420)

            def on_eval_line(stream_name: str, line: str):
                st.session_state.eval_logs.append({"stream": stream_name, "text": line})
                now = time.time()
                last = getattr(st.session_state, "_last_eval_log_render_ts", 0.0)
                if (now - last) >= 0.05:
                    tail_local = st.session_state.eval_logs[-1200:]
                    render_log_into_placeholder(eval_log_placeholder, tail_local, height_px=420)
                    st.session_state._last_eval_log_render_ts = now

            with st.spinner("Running evaluation..."):
                rc = run_cmd_capture_lines(cmd, cwd=ROOT, on_line=on_eval_line)

            tail_eval = st.session_state.eval_logs[-1200:]
            render_log_into_placeholder(eval_log_placeholder, tail_eval, height_px=420)
            st.session_state.eval_status = f"finished (rc={rc})"

            st.session_state.eval_csvs = find_recent_csvs(eval_out_root)
            if st.session_state.eval_csvs:
                st.session_state.eval_csv_chosen = relpath_or_abs(st.session_state.eval_csvs[0])

            st.toast(f"Evaluation finished (rc={rc})", icon="🧪")
            st.rerun()


# =========================================================
# Execute - Step1/2 pipeline run
# =========================================================
if run_btn:
    if selected_kind == "scale":
        if not scales:
            st.error("No scale YAML available.")
            st.stop()
        scale = selected_name
        base_yaml_path = cfg_path(problem, scale, "scale")
    else:
        if not selected_name:
            st.error("No custom YAML selected. Save one first.")
            st.stop()
        base_yaml_path = cfg_path(problem, selected_name, "custom")
        scale = "S"

    # cleanup old temp
    cleanup_tmp_yaml_if_exists(base_yaml_path)

    run_yaml_path = base_yaml_path

    # Decide whether to run temp (and actually WRITE temp file here, not in the display code)
    if st.session_state.edit_mode:
        disk_text = read_text(base_yaml_path) if base_yaml_path.exists() else ""
        if disk_text.strip() != (st.session_state.yaml_editor or "").strip():
            try:
                validate_yaml(st.session_state.yaml_editor)
            except Exception as e:
                st.error(f"YAML parse error (cannot run):\n\n{e}")
                st.stop()
            tmp = base_yaml_path.with_suffix(".tmp.yaml")
            write_text(tmp, st.session_state.yaml_editor)
            run_yaml_path = tmp
            st.toast(f"Running with temp YAML: {tmp.name}", icon="📝")

    inferred_scale = infer_scale_from_yaml_text(st.session_state.yaml_editor)
    if inferred_scale:
        scale = inferred_scale

    # Record what this run uses (for correct UI)
    st.session_state.active_selected_cfg_abs = str(base_yaml_path)
    st.session_state.active_selected_cfg_rel = relpath_or_abs(base_yaml_path)
    st.session_state.active_run_cfg_abs = str(run_yaml_path)
    st.session_state.active_run_cfg_rel = relpath_or_abs(run_yaml_path)

    cmd = ["python", "-m", "pipeline.run", "--problem", problem, "--scale", str(scale)]
    cmd += ["--config", str(run_yaml_path)]

    st.session_state.logs = []
    st.session_state.csvs = []
    st.session_state.csv_chosen = None
    st.session_state.run_status = "running"
    st.session_state.run_stage = "Starting"
    st.session_state.run_rc = None
    st.session_state.run_started_ts = time.time()
    st.session_state.run_finished_ts = None

    render_log_into_placeholder(log_placeholder, [], height_px=int(st.session_state.log_height))

    def on_line(stream_name: str, line: str):
        st.session_state.logs.append({"stream": stream_name, "text": line})
        now = time.time()
        last = getattr(st.session_state, "_last_log_render_ts", 0.0)
        if (now - last) >= 0.05:
            tail_local = st.session_state.logs[-1200:]
            render_log_into_placeholder(log_placeholder, tail_local, height_px=int(st.session_state.log_height))
            st.session_state._last_log_render_ts = now

    with st.spinner("Running pipeline..."):
        rc = run_cmd_capture_lines(cmd, cwd=ROOT, on_line=on_line)

    st.session_state.run_rc = int(rc)
    st.session_state.run_finished_ts = time.time()
    st.session_state.run_stage = "Done"

    tail = st.session_state.logs[-1200:]
    render_log_into_placeholder(log_placeholder, tail, height_px=int(st.session_state.log_height))

    st.session_state.run_status = "finished"

    out_root_path = infer_output_root_dir_from_yaml_text(st.session_state.yaml_editor)
    if out_root_path is None:
        out_root_path = ROOT / "step2_contextualization"

    csvs = find_recent_csvs(out_root_path)
    st.session_state.csvs = csvs
    st.session_state.active_output_root = relpath_or_abs(out_root_path)
    st.session_state.active_csv_count = len(csvs)

    if csvs:
        st.session_state.csv_chosen = relpath_or_abs(csvs[0])

    # add to run history (session)
    started = st.session_state.run_started_ts
    finished = st.session_state.run_finished_ts
    dur = None
    if isinstance(started, (int, float)) and isinstance(finished, (int, float)):
        dur = max(0.0, float(finished) - float(started))
    st.session_state.run_history.append(
        {
            "when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.session_state.run_finished_ts or time.time())),
            "problem": problem,
            "scale": str(scale),
            "selected_cfg": st.session_state.active_selected_cfg_rel,
            "run_cfg": st.session_state.active_run_cfg_rel,
            "rc": int(rc),
            "duration_s": dur,
            "output_root": st.session_state.active_output_root,
            "csv_count": int(st.session_state.active_csv_count or 0),
        }
    )

    st.toast(f"Run finished (rc={rc})", icon="✅")
    st.rerun()
