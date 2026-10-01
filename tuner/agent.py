"""Propose the next GATK search box from gatk_config_scores.

Tuner server only. Does not run GATK or edit minos_subnet.

Default is a multi-step agent: diagnose the limiter, review whether the
last category is still moving, choose the next category, tighten bounds
around the best trials. With an API key, a last step may revise that draft.

  python -m tuner.agent
  python -m tuner.agent --no-review
  python -m tuner.agent --heuristic
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

TUNER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TUNER_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(TUNER_ROOT / ".env")
    load_dotenv(TUNER_ROOT / ".env.tuner")
except ImportError:
    pass

from tuner.search_box import (
    MAX_TRIALS,
    enough_trials,
    specs_to_space_json,
    validate_search_box,
    write_search_box,
)
from tuner.spaces import SPACES, agent_reference, category_instruction, space_for
from tuner.steps import _keys_from_experiment, multistep_search_box
from tuner.supabase_scores import fetch_config_scores
from tuner.v2_score import V2_BRIEF

DEFAULT_OUT = TUNER_ROOT / "search_box.json"
HISTORY_LIMIT = 40

SYSTEM_PROMPT = """You are a Minos subnet 107 GATK practice-tuning analyst.

You propose ONE search box for Optuna. You do not invent GATK flags.
You do not tune a live miner. You only choose a category and bounds.

{v2}

Rules:
- Optimize avg_combined_final. The target is 0.9.
- Study the history. Do not repeat a setting already scored there.
- The same experiment with different parameter values is acceptable.
- While the best score is below 0.88, make a large change: several catalog parameters, full coarse ranges, from more than one category if one category is stuck.
- A new experiment name starts with v2_ and lists its keys: v2_{{term}}__{{key}}__{{key}}. term is core, indel, snp, or fp.
- Numeric values use the coarse step in the catalog. Do not ask for adjacent values such as 32, 34, 36 on a 30-100 range.
- Categorical choices must be the full catalog list. Optuna cannot change that list later.
- Keys must already exist in the catalog. Do not invent GATK flags.
- Each parameter note names the v2 metric it moves. The hypothesis must name that metric and the direction.
- n_trials is the product of the coarse grid, not the raw high-low span. Step 10 from 10 to 50 is 5 trials.
- When several parameters are searched together, multiply those grid sizes. 4 PCR models times another parameter's grid. Do not leave that search at 4.
- {max_trials} is the maximum. Use the smaller product when the grid is smaller. Do not set {max_trials} unless the product is at least {max_trials}.
- Failed / missing scores are not a GATK failure; ignore them for ranking.
- Write a short hypothesis that a later review can confirm or reject.

Return ONLY a JSON object with keys:
  search_category, hypothesis, space, constraints, n_trials, optimize
optimize must be "avg_combined_final".
constraints may include min_avg_f1_snp, min_avg_f1_indel, min_avg_combined_final.
""".format(v2=V2_BRIEF.strip(), max_trials=MAX_TRIALS)

REVIEW_PROMPT = """You revise ONE Optuna search box for Minos GATK practice tuning.

{v2}

You are given the scored history, a diagnosis, and a draft box.
The draft is a starting point. Replace it when history shows a larger experiment
will move avg_combined_final toward 0.9.

Study the history before you answer. Each history row's varied map is a setting that was already run. Do not propose that same setting again.
The same experiment with different parameter values is acceptable. Prefer a new combination when the old values did not raise avg_combined_final.
The target is 0.9. While the best score is below 0.88, make a large change: several catalog parameters at once, on their full coarse ranges, from more than one category when one category is stuck. Do not stay on a classic single-category screen, and do not shrink a full coarse range.
You may change search_category, the keys in space, the bounds, n_trials (1..{max_trials}), hypothesis, and constraints.
A new experiment name starts with v2_ and lists its keys: v2_{{term}}__{{key}}__{{key}}. term is core, indel, snp, or fp.
Keys must already exist in the catalog. Use each key's coarse step. Do not invent flags. Do not ask for a 1 or 2 unit change on a wide range.
At 0.88 or above, narrow the range around the best scored values.
n_trials is the product of the coarse grids. Step 10 from 10 to 50 is 5. Use that product when it is below {max_trials}. {max_trials} is only the ceiling.
The hypothesis must name the v2 metric and the direction.
optimize must stay "avg_combined_final".

Return ONLY a JSON object with keys:
  search_category, hypothesis, space, constraints, n_trials, optimize
""".format(v2=V2_BRIEF.strip(), max_trials=MAX_TRIALS)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    rows = fetch_config_scores(
        category=None,
        scored_only=not args.include_failed,
        order="created_at.desc",
    )
    if rows is None:
        return 2
    rows = list(reversed(rows))

    history = _compact_history(rows)
    print(f"   history rows for agent: {len(rows)} (newest {len(history)} kept for the log)", flush=True)

    if args.heuristic:
        try:
            box = validate_search_box(heuristic_search_box(history))
        except ValueError as e:
            print(f"ERROR: heuristic box invalid: {e}", flush=True)
            return 2
        source = "heuristic"
    elif args.once:
        if not _has_llm_key():
            print("ERROR: --once needs ANTHROPIC_API_KEY or OPENAI_API_KEY.", flush=True)
            return 2
        try:
            box = validate_search_box(_llm_search_box(history))
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: LLM search box rejected: {e}", flush=True)
            return 2
        source = "llm"
    else:
        try:
            draft, report = multistep_search_box(rows)
            box = validate_search_box(draft)
        except ValueError as e:
            print(f"ERROR: multi-step box invalid: {e}", flush=True)
            return 2
        _print_report(report)
        source = "steps"
        if _has_llm_key() and not args.no_review:
            try:
                revised = validate_search_box(_llm_review(report, box, history))
            except (ValueError, RuntimeError) as e:
                print(f"   step 5 review rejected ({e}); keeping the draft", flush=True)
            else:
                box = revised
                box["n_trials"] = enough_trials(box["space"])
                source = "steps+llm"
                print(
                    f"   step 5 review: accepted  n_trials={box['n_trials']}",
                    flush=True,
                )
        elif not args.no_review:
            print("   step 5 review: skipped, no API key", flush=True)
        if report.get("mode") == "stop":
            box["run_experiment"] = False

    box["n_trials"] = enough_trials(box["space"])
    box["suggested_by"] = f"agent+{source}"
    out_path = Path(args.out).resolve() if args.out else DEFAULT_OUT
    write_search_box(box, out_path)
    print(f"   wrote {out_path}", flush=True)
    print(json.dumps(box, indent=2, sort_keys=True), flush=True)
    return 0


def heuristic_search_box(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in history if r.get("avg_combined_final") is not None]
    latest = scored[-1] if scored else {}
    f1_snp = _num(latest.get("avg_f1_snp"))
    f1_indel = _num(latest.get("avg_f1_indel"))
    fp = _num(latest.get("avg_fp_per_target"))

    if f1_snp is not None and f1_snp >= 0.98 and fp is not None and fp >= 1.0:
        category = "quality_filters"
        hypothesis = (
            "Target SNP F1 is already high; v2 looks limited by fp_per_target. "
            "Search quality filters and calling confidence."
        )
    elif (
        f1_indel is not None
        and f1_snp is not None
        and f1_indel + 0.02 < f1_snp
    ):
        category = "pcr"
        hypothesis = (
            "INDEL F1 lags SNP F1. Search the PCR indel model before assembly."
        )
    elif not scored:
        category = "quality_filters"
        hypothesis = (
            "No scored history yet. Start with quality filters; they are the usual "
            "lever for extra region calls."
        )
    else:
        category = "quality_filters"
        hypothesis = (
            "Default next box: quality filters, one category, maximize avg_combined_final."
        )

    constraints: Dict[str, float] = {}
    if f1_snp is not None:
        constraints["min_avg_f1_snp"] = round(max(0.0, min(1.0, f1_snp - 0.02)), 4)
    if f1_indel is not None:
        constraints["min_avg_f1_indel"] = round(max(0.0, min(1.0, f1_indel - 0.02)), 4)

    space = specs_to_space_json(space_for(category))
    return {
        "search_category": category,
        "hypothesis": hypothesis,
        "space": space,
        "constraints": constraints,
        "n_trials": enough_trials(space),
        "optimize": "avg_combined_final",
    }


def _compact_history(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in rows[-HISTORY_LIMIT:]:
        compact.append({
            "config_id": row.get("config_id"),
            "experiment": row.get("experiment"),
            "search_category": row.get("search_category"),
            "suggested_by": row.get("suggested_by"),
            "hypothesis": row.get("hypothesis"),
            "status": row.get("status"),
            "n_scored": row.get("n_scored"),
            "avg_combined_final": row.get("avg_combined_final"),
            "avg_core": row.get("avg_core"),
            "avg_germline": row.get("avg_germline"),
            "avg_fp_per_target": row.get("avg_fp_per_target"),
            "avg_f1_snp": row.get("avg_f1_snp"),
            "avg_f1_indel": row.get("avg_f1_indel"),
            "gatk_updates": row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {},
        })
    return compact


def _pass_counts(report: Dict[str, Any]) -> str:
    passes = report.get("full_passes") if isinstance(report.get("full_passes"), dict) else {}
    return " ".join(f"{name}={int(passes.get(name) or 0)}" for name in sorted(passes))


def _experiment_counts(report: Dict[str, Any]) -> str:
    categories = report.get("categories") if isinstance(report.get("categories"), dict) else {}
    return " ".join(
        f"{name}={info.get('n')}"
        for name, info in sorted(categories.items())
        if isinstance(info, dict)
    )


def _print_report(report: Dict[str, Any]) -> None:
    best = report.get("best_avg_combined_final")
    best_txt = f"{best:.4f}" if isinstance(best, float) else "none"
    losses = report.get("losses") if isinstance(report.get("losses"), dict) else {}
    loss_txt = " ".join(
        f"{key}={losses[key]:.4f}" if isinstance(losses.get(key), float) else f"{key}=na"
        for key in ("indel", "snp", "fp")
    )
    print(
        f"   step 1 experiments: {_experiment_counts(report)}\n"
        f"   step 1 diagnose: bottleneck={report.get('bottleneck')} "
        f"best={best_txt} from {report.get('best_category')}  v2 loss {loss_txt}",
        flush=True,
    )
    last = report.get("last_category")
    print(
        f"   step 2 review: last={last} state={report.get('last_state')} "
        f"full-config passes: {_pass_counts(report)}",
        flush=True,
    )
    print(
        f"   step 3 choose: {report.get('choice')} ({report.get('reason')})\n"
        f"   step 3 note: {category_instruction(str(report.get('choice') or ''))}",
        flush=True,
    )
    print(
        f"   step 4 bounds: {report.get('bounds')} n_trials={report.get('n_trials')}",
        flush=True,
    )


def _setting_brief(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scored settings the next experiment must not repeat."""
    brief: List[Dict[str, Any]] = []
    for row in rows[-HISTORY_LIMIT:]:
        name = str(row.get("search_category") or "")
        updates = row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {}
        parsed = _keys_from_experiment(name)
        if parsed:
            varied = {key: updates[key] for key in parsed[1] if key in updates}
        elif name in SPACES:
            varied = {key: updates[key] for key in SPACES[name] if key in updates}
        else:
            varied = {}
        brief.append({
            "search_category": name,
            "experiment": row.get("experiment"),
            "status": row.get("status"),
            "avg_combined_final": row.get("avg_combined_final"),
            "avg_core": row.get("avg_core"),
            "avg_germline": row.get("avg_germline"),
            "avg_fp_per_target": row.get("avg_fp_per_target"),
            "varied": varied,
        })
    return brief


def _llm_review(
    report: Dict[str, Any], draft: Dict[str, Any], history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    user = json.dumps(
        {
            "catalog": agent_reference(),
            "diagnosis": report,
            "history": _setting_brief(history),
            "draft": draft,
            "instruction": (
                "Study the history. Propose a large experiment that does not "
                "repeat a varied setting already scored. The same experiment "
                "with different values is allowed. Return JSON only."
            ),
        },
        default=str,
    )
    return _parse_json_object(_chat(REVIEW_PROMPT, user))


def _llm_search_box(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    return _parse_json_object(_chat(SYSTEM_PROMPT, json.dumps(
        {
            "catalog": agent_reference(),
            "history": history,
            "instruction": (
                "Choose the next search box from the catalog notes and history. "
                "Return JSON only."
            ),
        },
        default=str,
    )))


def _chat(system: str, user: str) -> str:
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if provider == "openai" or (not provider and os.environ.get("OPENAI_API_KEY")):
        return _openai_chat(system, user)
    return _anthropic_chat(system, user)


def _has_llm_key() -> bool:
    return bool(
        (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )


def _anthropic_chat(system: str, user: str) -> str:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    model = (os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-6").strip()
    body = {
        "model": model,
        "max_tokens": 2500,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    raw = _http_json(req)
    blocks = raw.get("content") if isinstance(raw, dict) else None
    if not isinstance(blocks, list):
        raise RuntimeError("unexpected Anthropic response")
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
    return "\n".join(parts)


def _openai_chat(system: str, user: str) -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    model = (os.environ.get("OPENAI_MODEL") or "gpt-4.1").strip()
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        },
    )
    raw = _http_json(req)
    try:
        return raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError("unexpected OpenAI response") from e


def _http_json(req: urllib.request.Request) -> Any:
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            if not (200 <= resp.status < 300):
                raise RuntimeError(f"LLM HTTP {resp.status}")
            return json.loads(payload)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"LLM HTTP {e.code}: {err}") from e
    except (TimeoutError, urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise RuntimeError(f"LLM request failed: {reason}") from e


def _parse_json_object(text: str) -> Dict[str, Any]:
    blob = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, re.DOTALL)
    if fenced:
        blob = fenced.group(1)
    else:
        start = blob.find("{")
        end = blob.rfind("}")
        if start >= 0 and end > start:
            blob = blob[start : end + 1]
    try:
        data = json.loads(blob)
    except ValueError as e:
        raise ValueError("model did not return JSON") from e
    if not isinstance(data, dict):
        raise ValueError("model JSON was not an object")
    return data


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Propose the next GATK Optuna search box from Supabase history.",
    )
    p.add_argument(
        "--heuristic",
        action="store_true",
        help="Use the old single-rule box instead of the multi-step agent.",
    )
    p.add_argument(
        "--no-review",
        action="store_true",
        help="Skip the optional LLM revision of the multi-step draft.",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="One LLM call from raw history. Requires an API key.",
    )
    p.add_argument(
        "--include-failed",
        action="store_true",
        help="Include configs with no scored combined_final in the history payload.",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Write JSON here (default {DEFAULT_OUT})",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
