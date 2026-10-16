"""Upload experiments.json into the Supabase experiments table.

    python upload_experiments.py
    python upload_experiments.py experiments.json

Each entry becomes one row with status pending. Uploading the same name
again keeps that study. A changed space or gatk_config sets the row back to
pending and the fleet queues only combinations that are not already scored.
Run the experiments section of gatk_tuning.sql once before the first upload.
"""
from __future__ import annotations

import json
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuner.experiments import load_experiments
from tuner.jobs import experiment_table
from tuner.supabase_scores import rest_json


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.tuner")


def _existing_row(name: str):
    quoted = urllib.parse.quote(name, safe="")
    data = rest_json(
        "GET",
        experiment_table(),
        query=f"select=*&name=eq.{quoted}&limit=1",
    )
    if data is None:
        return None
    if isinstance(data, list) and data:
        return data[0]
    return False


def _entry_body(entry: dict) -> dict:
    body = {"hypothesis": str(entry.get("hypothesis") or "").strip()}
    if entry.get("search_category"):
        body["search_category"] = str(entry["search_category"])
    if isinstance(entry.get("space"), dict):
        body["space"] = entry["space"]
    if isinstance(entry.get("updates"), dict):
        body["updates"] = entry["updates"]
    if isinstance(entry.get("gatk_config"), dict):
        body["gatk_config"] = entry["gatk_config"]
    return body


def _same_definition(stored: dict, body: dict) -> bool:
    for key in ("hypothesis", "search_category", "space", "updates", "gatk_config"):
        old = stored.get(key)
        new = body.get(key)
        if json.dumps(old, sort_keys=True, default=str) != json.dumps(new, sort_keys=True, default=str):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    _load_env()
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0] if args else ROOT / "experiments.json")
    try:
        load_experiments(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", flush=True)
        return 2
    entries = raw if isinstance(raw, list) else raw.get("experiments", [])
    if not isinstance(entries, list) or not entries:
        print(f"ERROR: {path.name} has no experiments", flush=True)
        return 2
    uploaded = 0
    continued = 0
    for entry in entries:
        if not isinstance(entry, dict):
            print("ERROR: each experiment must be an object", flush=True)
            return 2
        name = str(entry.get("name") or "").strip()
        found = _existing_row(name)
        if found is None:
            return 2
        body = _entry_body(entry)
        if isinstance(found, dict):
            if _same_definition(found, body):
                print(
                    f"   {name} is unchanged  id={found.get('id')}  status={found.get('status')}",
                    flush=True,
                )
                continue
            quoted = urllib.parse.quote(str(found.get("id") or ""), safe="")
            patched = rest_json(
                "PATCH",
                experiment_table(),
                query=f"id=eq.{quoted}",
                body={
                    **body,
                    "status": "pending",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if not isinstance(patched, list) or not patched:
                print(f"ERROR: could not update {name}", flush=True)
                return 2
            print(
                f"   continued {name}  id={found.get('id')}  status=pending  "
                "same study; scored configs are kept",
                flush=True,
            )
            continued += 1
            continue
        inserted = rest_json(
            "POST",
            experiment_table(),
            body={"name": name, "status": "pending", **body},
        )
        if not isinstance(inserted, list) or not inserted:
            print(f"ERROR: could not upload {name}", flush=True)
            return 2
        print(f"   uploaded {name}  id={inserted[0].get('id')}  status=pending", flush=True)
        uploaded += 1
    print(f"   uploaded {uploaded} experiment(s), continued {continued}", flush=True)
    print(
        "   set EXPERIMENT_ID to one id to run only that row. "
        "Leave it empty to run every row that is not done.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
