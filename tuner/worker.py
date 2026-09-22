"""Score pending Optuna jobs on the GATK machine.

Does not modify original minos_subnet modules. Needs MINOS_SUBNET pointing
at that checkout (score_gatk_folders.py + configs/gatk.conf + practice data).

  set MINOS_SUBNET=C:\\Project\\TAO\\minos_subnet
  python -m tuner.worker
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

TUNER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TUNER_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(TUNER_ROOT / ".env")
    load_dotenv(TUNER_ROOT / ".env.tuner")
except ImportError:
    pass

from tuner.gatk_conf import read_gatk_conf, update_gatk_conf
from tuner.jobs import (
    insert_evaluations,
    list_pending,
    load_score_details,
    patch_config,
)


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    subnet = Path(os.environ.get("MINOS_SUBNET") or args.minos_subnet).resolve()
    conf = subnet / "configs" / "gatk.conf"
    scorer = subnet / "score_gatk_folders.py"
    practice = subnet / "datasets" / "practice"
    if not conf.exists() or not scorer.exists():
        print(
            f"ERROR: MINOS_SUBNET={subnet} is missing configs/gatk.conf or "
            "score_gatk_folders.py",
            flush=True,
        )
        return 2

    # Scoring machine .env often lives next to main.py.
    try:
        from dotenv import load_dotenv
        load_dotenv(subnet / ".env")
    except ImportError:
        pass

    poll = int(args.poll)
    print(f"   worker MINOS_SUBNET={subnet}  poll={poll}s", flush=True)
    while True:
        jobs = list_pending(limit=1)
        if not jobs:
            if args.once:
                print("   no pending jobs", flush=True)
                return 0
            time.sleep(max(1, poll))
            continue
        job = jobs[0]
        config_id = str(job.get("id") or "")
        if not config_id:
            print("   skip job with no id", flush=True)
            if args.once:
                return 1
            continue
        rc = _run_job(job, conf=conf, scorer=scorer, practice=practice, subnet=subnet)
        if args.once:
            return rc
        time.sleep(1)


def _run_job(job: dict, *, conf: Path, scorer: Path, practice: Path, subnet: Path) -> int:
    config_id = str(job["id"])
    experiment = job.get("experiment") or config_id
    updates = job.get("gatk_updates") if isinstance(job.get("gatk_updates"), dict) else {}
    print(f"\n   job {config_id}  {experiment}  updates={updates}", flush=True)
    if not patch_config(config_id, {"status": "running"}):
        print("   ERROR: could not mark running", flush=True)
        return 1

    try:
        original = conf.read_text(encoding="utf-8")
    except OSError as e:
        print(f"ERROR: read {conf}: {e}", flush=True)
        patch_config(config_id, {"status": "failed"})
        return 1

    try:
        if updates:
            changed = update_gatk_conf(conf, updates)
            if changed is None:
                patch_config(config_id, {"status": "failed"})
                return 1
            for key, old, new in changed:
                print(f"     {key}: {old} -> {new}", flush=True)
        gatk_params = read_gatk_conf(conf)
        scored = subprocess.run([sys.executable, str(scorer)], cwd=str(subnet))
        scores = load_score_details(practice)
        scored_ok = [s for s in scores if s.get("ok") and s.get("combined_final") is not None]
        status = "scored" if scored_ok else "failed"
        if not insert_evaluations(config_id, scores):
            status = "failed"
        patch_config(config_id, {"status": status, "gatk_config": gatk_params})
        print(
            f"   job {config_id} {status}  folders={len(scores)}  "
            f"scorer_exit={scored.returncode}",
            flush=True,
        )
        return 0 if status == "scored" else 1
    finally:
        try:
            conf.write_text(original, encoding="utf-8")
            print(f"   restored {conf.name} after job", flush=True)
        except OSError as e:
            print(f"ERROR: could not restore {conf}: {e}", flush=True)


def _parse_args(argv: Optional[list]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pending GATK scoring jobs from Supabase.")
    p.add_argument(
        "--minos-subnet",
        default=os.environ.get("MINOS_SUBNET") or r"C:\Project\TAO\minos_subnet",
        help="Path to the minos_subnet checkout (not modified except gatk.conf during a job).",
    )
    p.add_argument("--once", action="store_true", help="Process at most one pending job and exit.")
    p.add_argument("--poll", type=int, default=15, help="Seconds between pending polls.")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
