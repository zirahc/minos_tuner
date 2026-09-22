"""Apply GATK .conf updates without importing minos_subnet."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def update_gatk_conf(path: Path, updates: Dict[str, Any]) -> List[Tuple[str, str, str]] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"ERROR: could not read {path}: {e}", flush=True)
        return None

    pending = dict(updates)
    changed: List[Tuple[str, str, str]] = []
    out_lines: List[str] = []
    for raw in text.splitlines(keepends=True):
        stripped = raw.strip()
        newline = "\n" if raw.endswith("\n") else ""
        body = stripped[:-1] if stripped.endswith("\r") else stripped
        if body and not body.startswith("#") and "=" in body:
            key, _, current = body.partition("=")
            key = key.strip()
            if key in pending:
                new_val = _format_conf_value(pending.pop(key))
                old_val = current.strip()
                indent = raw[: len(raw) - len(raw.lstrip())]
                out_lines.append(f"{indent}{key}={new_val}{newline}")
                if old_val != new_val:
                    changed.append((key, old_val, new_val))
                continue
        out_lines.append(raw)

    if pending:
        if out_lines and not out_lines[-1].endswith("\n"):
            out_lines[-1] += "\n"
        out_lines.append("\n# Added by minos_tuner worker\n")
        for key, value in pending.items():
            new_val = _format_conf_value(value)
            out_lines.append(f"{key}={new_val}\n")
            changed.append((key, "(missing)", new_val))

    try:
        path.write_text("".join(out_lines), encoding="utf-8")
    except OSError as e:
        print(f"ERROR: could not write {path}: {e}", flush=True)
        return None
    return changed


def read_gatk_conf(path: Path) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key:
            params[key] = _parse_conf_value(val)
    return params


def _format_conf_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_conf_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw
