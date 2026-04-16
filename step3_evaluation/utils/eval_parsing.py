
import json
import re
from typing import Any, Optional, List, Tuple, Dict


def _extract_all_balanced_curly_objects(text: str) -> List[str]:
    """
    Return all balanced {...} substrings in reading order.
    Handles quotes and escapes so braces inside strings don't count.
    """
    objs: List[str] = []
    in_str = False
    esc = False
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        objs.append(text[start : i + 1])
                        start = None

    return objs


def _pick_best_dict(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for d in reversed(candidates):
        if isinstance(d, dict) and "solution" in d:
            return d
    return candidates[-1] if candidates else None



def safe_json_loads(raw: str) -> Any:
    if raw is None:
        raise ValueError("safe_json_loads: raw is None")

    raw_text = raw.strip()
    if not raw_text:
        raise ValueError("safe_json_loads: raw is empty")

    # Build multiple candidate views WITHOUT overriding each other (avoid regression)
    candidates_text: List[str] = []

    # (A) full raw (keeps everything; most robust, but may be long)
    candidates_text.append(raw_text)

    # (B) tail from last '{' (your current strong heuristic)
    last_lbrace = raw_text.rfind("{")
    if last_lbrace != -1:
        candidates_text.append(raw_text[last_lbrace:])

    # (C) last fenced block from full raw (may contain the JSON)
    fenced_blocks = [
        m.group(1).strip()
        for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    ]
    if fenced_blocks:
        candidates_text.append(fenced_blocks[-1].strip())

    def _try_parse_dict_from_text(text: str) -> List[Dict[str, Any]]:
        # light cleanup for common LLM artifacts (same as your original)
        cleaned = text.replace("$", "")
        cleaned = re.sub(r"\\boxed\s*\{", "{", cleaned)  # keep balance

        parsed: List[Dict[str, Any]] = []

        # 1) parse from first '{' to last '}'
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first != -1 and last != -1 and last > first:
            whole = cleaned[first : last + 1].strip()
            for cand in (whole, whole[1:-1].strip() if whole.startswith("{{") and whole.endswith("}}") else None):
                if not cand:
                    continue
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        parsed.append(obj)
                except Exception:
                    pass

        # 2) balanced extraction of all {...}
        blocks = _extract_all_balanced_curly_objects(cleaned)
        for b in blocks:
            for cand in (b, b[1:-1].strip() if b.startswith("{{") and b.endswith("}}") else None):
                if not cand:
                    continue
                try:
                    obj = json.loads(cand)
                    if isinstance(obj, dict):
                        parsed.append(obj)
                except Exception:
                    continue

        return parsed

    # Collect dict candidates from all candidate texts
    all_dicts: List[Dict[str, Any]] = []
    for t in candidates_text:
        all_dicts.extend(_try_parse_dict_from_text(t))

    best = _pick_best_dict(all_dicts)
    if best is not None:
        return best

    # Conservative fallback: boxed list -> {"solution": [...]}
    m = re.search(r"\\boxed\s*\{\s*(\[[0-9,\s]+\])\s*\}", raw_text)
    if m:
        try:
            arr = json.loads(m.group(1))
            if isinstance(arr, list):
                return {"solution": arr}
        except Exception:
            pass

    # Conservative fallback: trailing numeric list -> {"solution": [...]}
    m2 = re.search(r"(\[[0-9,\s]+\])\s*$", raw_text)
    if m2:
        try:
            arr = json.loads(m2.group(1))
            if isinstance(arr, list):
                return {"solution": arr}
        except Exception:
            pass

    raise ValueError(f"safe_json_loads: Could not parse JSON object as dict. raw={raw!r}")


def parse_json_field(field: Any) -> Any:
    """
    Safely parse a JSON-like field from CSV.

    Often it's stored as a string like:
        "[[3, 100], [75, 26], ...]" or "[0, 2, 12, ...]"

    If it's already a list/dict, return as-is.
    If parsing fails, return None.
    """
    if isinstance(field, (list, dict)):
        return field

    if isinstance(field, str):
        text = field.strip()

        # Remove possible markdown fences ```json ... ```
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                # Keep content after the first newline
                text = text.split("\n", 1)[1]

        # Try JSON first
        try:
            return json.loads(text)
        except Exception:
            # Fallback: safe-ish eval (only if data is trusted)
            try:
                return eval(text, {"__builtins__": {}})
            except Exception:
                return None

    return None



def _normalize_label(x: Any) -> Any:
    """
    Normalize a single label:
      - "3" -> 3
      - "  7 " -> 7
      - "A" -> "A"
      - 5 -> 5
    """
    if isinstance(x, str):
        s = x.strip()
        if s.lstrip("-").isdigit():
            return int(s)
        return s
    return x


def _normalize_decision(obj: Any) -> Any:
    """
    Recursively normalize a decision structure that may be:
      - list of labels
      - list of list of labels
      - etc.
    """
    if isinstance(obj, list):
        return [_normalize_decision(v) for v in obj]
    return _normalize_label(obj)

