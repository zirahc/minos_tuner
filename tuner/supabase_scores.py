"""Read gatk_config_scores from Supabase REST. Tuner-side only; no GATK."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_VIEW = "gatk_config_scores"


def supabase_rest() -> Optional[tuple[str, str]]:
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (
        os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    if not url or not key:
        print(
            "ERROR: set SUPABASE_URL and SUPABASE_KEY in .env. "
            "Do not paste those values here.",
            flush=True,
        )
        return None
    if not url.startswith("https://"):
        print("ERROR: SUPABASE_URL must be an https:// project URL.", flush=True)
        return None
    return url, key


def fetch_config_scores(
    category: Optional[str] = None,
    scored_only: bool = True,
    limit: int = 1000,
) -> Optional[List[Dict[str, Any]]]:
    dest = supabase_rest()
    if dest is None:
        return None
    url, key = dest
    view = (os.environ.get("SUPABASE_SCORE_VIEW") or DEFAULT_VIEW).strip()
    params: List[str] = ["select=*", f"limit={int(limit)}", "order=created_at.asc"]
    if category:
        params.append(f"search_category=eq.{urllib.parse.quote(category, safe='')}")
    if scored_only:
        params.append("n_scored=gte.1")
        params.append("avg_combined_final=not.is.null")
    query = "&".join(params)
    endpoint = f"{url}/rest/v1/{urllib.parse.quote(view, safe='')}?{query}"
    req = urllib.request.Request(
        endpoint,
        method="GET",
        headers={
            "Accept": "application/json",
            "apikey": key,
            "Authorization": f"Bearer {key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not (200 <= resp.status < 300):
                print(f"ERROR: HTTP {resp.status} reading {view}", flush=True)
                return None
            data = json.loads(raw) if raw else []
            if not isinstance(data, list):
                print("ERROR: score view did not return a JSON array", flush=True)
                return None
            print(f"   fetched {len(data)} row(s) from {view}", flush=True)
            return data
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500]
        print(f"ERROR: HTTP {e.code} {e.reason}  view={view}", flush=True)
        if err_body:
            print(f"   {err_body}", flush=True)
        if e.code == 404:
            print(
                "   Hint: run gatk_tuning.sql so view gatk_config_scores exists, "
                "and expose it in the Supabase API.",
                flush=True,
            )
        if e.code in (401, 403):
            print(
                "   Hint: use the service-role key, or add a SELECT policy. "
                "Do not paste the key here.",
                flush=True,
            )
        return None
    except urllib.error.URLError as e:
        print(f"ERROR: could not reach Supabase: {e.reason}", flush=True)
        return None
    except ValueError as e:
        print(f"ERROR: invalid JSON from score view: {e}", flush=True)
        return None


def rest_json(
    method: str,
    table: str,
    query: str = "",
    body: Any = None,
    timeout: int = 30,
) -> Optional[Any]:
    dest = supabase_rest()
    if dest is None:
        return None
    url, key = dest
    path = f"{url}/rest/v1/{urllib.parse.quote(table, safe='')}"
    if query:
        path = f"{path}?{query}"
    data = None if body is None else json.dumps(body, default=str).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Prefer": "return=representation",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(path, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not (200 <= resp.status < 300):
                print(f"ERROR: HTTP {resp.status} {method} {table}", flush=True)
                return None
            if not raw:
                return []
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:500]
        print(f"ERROR: HTTP {e.code} {e.reason}  {method} {table}", flush=True)
        if err_body:
            print(f"   {err_body}", flush=True)
        if e.code in (401, 403):
            print(
                "   Hint: use the service-role key, or add a table policy. "
                "Do not paste the key here.",
                flush=True,
            )
        return None
    except urllib.error.URLError as e:
        print(f"ERROR: could not reach Supabase: {e.reason}", flush=True)
        return None
    except ValueError as e:
        print(f"ERROR: invalid JSON from {table}: {e}", flush=True)
        return None
