#!/usr/bin/env python3
"""Download Hugging Face round folders into the practice directory.

Each folder name is a path inside the dataset (default
``Lila-RGB/minos-full-test-vcfs``). Files land under
``$MINOS_SUBNET/datasets/practice/<folder>/``, the same tree
``download_practice_samples.py`` uses.

Pass one folder, or list rounds in ``hf_test_rounds.txt`` (one folder per
line). Chromosomes are read from VCF headers, ``region.txt``, and
``sample.json``. Any missing GRCh38 FASTA and RTG SDF is fetched from the
Minos reference host.

This does not score anything and does not submit to the live miner API.

Examples:
  python download_huggingface.py
  python download_huggingface.py --rounds hf_test_rounds.txt
  python download_huggingface.py 20261010T235400Z
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent


def _load_local_env() -> None:
    """Load tuner .env files. Never print values."""
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
        load_dotenv(REPO_ROOT / ".env.tuner")
        return
    except ImportError:
        pass
    for name in (".env", ".env.tuner"):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env()

DEFAULT_REPO = "Lila-RGB/minos-full-test-vcfs"
DEFAULT_REVISION = "main"
DEFAULT_ROUNDS_FILE = REPO_ROOT / "hf_test_rounds.txt"
DEFAULT_REF_BASE = "https://api.theminos.ai/reference"
HF_USER_AGENT = "minos-tuner/0.1"
# api.theminos.ai/reference rejects the default Python urllib user agent.
REF_USER_AGENT = "minos-installer/0.1 (+https://github.com/minos-protocol/minos_subnet)"
FASTA_EXTS = ("fa", "fa.fai", "dict")
SDF_FILES = (
    "done",
    "mainIndex",
    "nameIndex0",
    "namedata0",
    "namepointer0",
    "progress",
    "seqdata0",
    "seqpointer0",
    "sequenceIndex0",
    "summary.txt",
)
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_CHR_RE = re.compile(r"^chr(?:[1-9]|1[0-9]|2[0-2]|X|Y|M)$")
_CONTIG_RE = re.compile(r"ID=([^,>]+)")
_NEXT_LINK_RE = re.compile(r"<([^>]+)>\s*;\s*rel=\"next\"")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    subnet = _subnet_root()
    if subnet is None:
        return 2
    practice_dir = subnet / "datasets" / "practice"
    reference_dir = subnet / "datasets" / "reference"

    rounds_file = Path(args.rounds) if args.rounds else DEFAULT_ROUNDS_FILE
    if not rounds_file.is_absolute():
        rounds_file = REPO_ROOT / rounds_file
    try:
        if args.folder:
            folders = [_safe_folder(args.folder)]
        else:
            folders = _load_rounds(rounds_file)
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", flush=True)
        return 2
    if not folders:
        print(f"ERROR: no rounds listed in {rounds_file}", flush=True)
        return 2

    repo = (args.repo or os.getenv("HF_DATASET") or DEFAULT_REPO).strip()
    if not _REPO_RE.match(repo):
        print(f"ERROR: dataset repo must look like owner/name, got {repo!r}", flush=True)
        return 2
    revision = (args.revision or DEFAULT_REVISION).strip() or DEFAULT_REVISION
    try:
        repo_kind = _repo_kind(repo)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        print(f"ERROR: {e}", flush=True)
        return 2

    print("=" * 72, flush=True)
    print("  DOWNLOAD HUGGING FACE ROUNDS", flush=True)
    print("=" * 72, flush=True)
    print(f"  Dataset:   {repo} ({repo_kind}) @ {revision}", flush=True)
    if args.folder:
        print(f"  Folder:    {folders[0]}", flush=True)
    else:
        print(f"  Rounds:    {rounds_file}  ({len(folders)})", flush=True)
    print(f"  Practice:  {practice_dir}", flush=True)
    print(f"  Reference: {reference_dir}", flush=True)
    print(flush=True)

    failed: List[str] = []
    chroms: Set[str] = set()
    saved = 0
    for index, folder in enumerate(folders, 1):
        print(f"\n{'#' * 72}", flush=True)
        print(f"  [{index}/{len(folders)}] {folder}", flush=True)
        print(f"{'#' * 72}", flush=True)
        count, folder_chroms, error = _download_folder(
            repo, revision, repo_kind, folder, practice_dir, force=args.force,
        )
        saved += count
        if error:
            print(f"   ERROR: {error}", flush=True)
            failed.append(folder)
            continue
        chroms.update(folder_chroms)
        print(f"   Chromosomes: {', '.join(folder_chroms)}", flush=True)

    if chroms:
        print(flush=True)
        print(f"  Chromosomes: {', '.join(_sort_chroms(chroms))}", flush=True)
        if not _ensure_references(_sort_chroms(chroms), reference_dir):
            return 2
    elif not failed:
        print("ERROR: downloaded rounds but found no chromosome.", flush=True)
        return 2

    print(flush=True)
    print(f"Done. {saved} file(s), {len(folders) - len(failed)} round(s) ok, {len(failed)} failed.", flush=True)
    if failed:
        print("Failed:", flush=True)
        for folder in failed:
            print(f"  - {folder}", flush=True)
        return 1
    return 0


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download Hugging Face round folders into datasets/practice "
            "and fetch any missing chromosome reference."
        ),
    )
    p.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="One dataset folder, e.g. 20261010T235400Z. Omit to read the rounds file.",
    )
    p.add_argument(
        "--rounds",
        default=None,
        help=(
            "Text file of round folders, one per line "
            f"(default when folder is omitted: {DEFAULT_ROUNDS_FILE.name})."
        ),
    )
    p.add_argument(
        "--repo",
        default=None,
        help=f"Hugging Face dataset id (default: HF_DATASET env or {DEFAULT_REPO}).",
    )
    p.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Dataset revision (default: {DEFAULT_REVISION}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when a complete copy is already on disk.",
    )
    return p.parse_args(argv)


def _subnet_root() -> Optional[Path]:
    raw = (os.getenv("MINOS_SUBNET") or "").strip()
    if not raw:
        print(
            "ERROR: set MINOS_SUBNET in .env to the minos_subnet checkout. "
            "Downloads go to that checkout's datasets/ folder.",
            flush=True,
        )
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        print(f"ERROR: MINOS_SUBNET is not a directory: {path}", flush=True)
        return None
    return path


def _load_rounds(path: Path) -> List[str]:
    """One round folder per line. Blank lines and # comments are ignored."""
    if not path.is_file():
        raise ValueError(
            f"rounds file not found: {path}. "
            "Put one Hugging Face folder name per line."
        )
    rounds: List[str] = []
    seen = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        folder = _safe_folder(line)
        if folder not in seen:
            seen.add(folder)
            rounds.append(folder)
    return rounds


def _download_folder(
    repo: str,
    revision: str,
    repo_kind: str,
    folder: str,
    practice_dir: Path,
    force: bool,
) -> tuple:
    """Download one round. Returns (file_count, chromosomes, error)."""
    try:
        files = _list_files(repo, revision, repo_kind, folder)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        return 0, [], f"could not list {folder}: {e}"
    if not files:
        return 0, [], f"no files under {folder!r}"

    print(f"   Files: {len(files)}", flush=True)
    saved = 0
    for i, item in enumerate(files, 1):
        rel = str(item.get("path") or "")
        try:
            dest = _dest_for(practice_dir, rel, folder)
        except ValueError as e:
            return saved, [], str(e)
        expected = _remote_size(item)
        print(f"   [{i}/{len(files)}] {rel}", flush=True)
        if not force and _reuse_ok(dest, expected):
            print(f"      already downloaded — reusing {dest.name}", flush=True)
            saved += 1
            continue
        url = _resolve_url(repo, revision, repo_kind, rel)
        if not _download_file(url, dest, expected_size=expected, headers=_hf_headers()):
            return saved, [], f"failed to download {rel}"
        saved += 1

    chroms = _chroms_in_tree(practice_dir / Path(folder))
    if not chroms:
        return saved, [], (
            f"downloaded {folder} but found no chromosome "
            "(no VCF contig, region.txt, or sample.json)"
        )
    return saved, chroms, None


def _sort_chroms(chroms: Iterable[str]) -> List[str]:
    return sorted(set(chroms), key=_chrom_sort_key)


def _safe_folder(raw: str) -> str:
    folder = (raw or "").strip().replace("\\", "/").strip("/")
    if not folder:
        raise ValueError("folder name is empty")
    parts = folder.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"folder name {raw!r} is not a relative path")
    return folder


def _hf_headers() -> Dict[str, str]:
    headers = {"User-Agent": HF_USER_AGENT}
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _quote_repo(repo: str) -> str:
    return "/".join(urllib.parse.quote(part) for part in repo.split("/"))


def _repo_kind(repo: str) -> str:
    """A Hugging Face id can be a dataset or a model. eliteminer/minos_ch19 is a model."""
    headers = {**_hf_headers(), "Accept": "application/json"}
    quoted = _quote_repo(repo)
    for kind, api in (("dataset", "datasets"), ("model", "models")):
        url = f"https://huggingface.co/api/{api}/{quoted}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if 200 <= resp.status < 300:
                    return kind
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise ValueError(f"could not read {repo}: HTTP {e.code}") from e
    raise ValueError(f"{repo} was not found as a Hugging Face dataset or model")


def _list_files(repo: str, revision: str, repo_kind: str, folder: str) -> List[Dict[str, Any]]:
    api = "datasets" if repo_kind == "dataset" else "models"
    quoted_rev = urllib.parse.quote(revision, safe="")
    quoted_folder = urllib.parse.quote(folder, safe="/")
    page = (
        f"https://huggingface.co/api/{api}/{_quote_repo(repo)}/tree/"
        f"{quoted_rev}/{quoted_folder}?recursive=true"
    )
    files: List[Dict[str, Any]] = []
    seen_pages = 0
    while page:
        seen_pages += 1
        if seen_pages > 50:
            raise ValueError("too many Hugging Face listing pages")
        req = urllib.request.Request(page, headers={**_hf_headers(), "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if not (200 <= resp.status < 300):
                    raise ValueError(f"listing HTTP {resp.status}")
                payload = json.load(resp)
                link = resp.headers.get("Link")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ValueError(f"folder {folder!r} was not found") from e
            raise ValueError(f"listing HTTP {e.code}") from e
        if not isinstance(payload, list):
            raise ValueError("unexpected Hugging Face listing response")
        for item in payload:
            if isinstance(item, dict) and item.get("type") == "file" and item.get("path"):
                files.append(item)
        page = _next_page(link)
    files.sort(key=lambda item: str(item.get("path") or ""))
    return files


def _next_page(link: Optional[str]) -> Optional[str]:
    if not link:
        return None
    match = _NEXT_LINK_RE.search(link)
    return match.group(1) if match else None


def _remote_size(item: Dict[str, Any]) -> int:
    lfs = item.get("lfs")
    if isinstance(lfs, dict) and lfs.get("size"):
        try:
            return int(lfs["size"])
        except (TypeError, ValueError):
            return 0
    try:
        return int(item.get("size") or 0)
    except (TypeError, ValueError):
        return 0


def _resolve_url(repo: str, revision: str, repo_kind: str, path: str) -> str:
    quoted_rev = urllib.parse.quote(revision, safe="")
    quoted_path = urllib.parse.quote(path, safe="/")
    quoted_repo = _quote_repo(repo)
    if repo_kind == "dataset":
        return f"https://huggingface.co/datasets/{quoted_repo}/resolve/{quoted_rev}/{quoted_path}"
    return f"https://huggingface.co/{quoted_repo}/resolve/{quoted_rev}/{quoted_path}"


def _dest_for(practice_dir: Path, repo_path: str, folder: str) -> Path:
    rel = repo_path.replace("\\", "/").lstrip("/")
    if rel != folder and not rel.startswith(folder + "/"):
        raise ValueError(f"refusing path outside {folder}: {repo_path}")
    dest = (practice_dir / Path(*rel.split("/"))).resolve()
    root = practice_dir.resolve()
    if dest != root and root not in dest.parents:
        raise ValueError(f"refusing path outside practice dir: {repo_path}")
    return dest


def _reuse_ok(path: Path, expected_size: int) -> bool:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
    except OSError:
        return False
    if expected_size and path.stat().st_size != expected_size:
        print(f"   {path.name} size does not match — will re-download.", flush=True)
        return False
    return True


def _download_file(
    url: str,
    dest: Path,
    expected_size: int = 0,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 180,
) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers=headers or {"User-Agent": HF_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not (200 <= resp.status < 300):
                return False
            total = 0
            next_report = 32 * 1024 * 1024
            with tmp.open("wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)
                    if total >= next_report:
                        print(f"   {total / (1024 * 1024):.0f} MB", flush=True)
                        next_report += 32 * 1024 * 1024
        if not tmp.is_file() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return False
        if expected_size and tmp.stat().st_size != expected_size:
            print(
                f"   ERROR: {dest.name} is {tmp.stat().st_size} bytes, "
                f"expected {expected_size}",
                flush=True,
            )
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"   ERROR: download failed ({e})", flush=True)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _chroms_in_tree(root: Path) -> List[str]:
    found: Set[str] = set()
    if not root.is_dir():
        return []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        try:
            if name == "region.txt":
                found.update(_chroms_from_region_text(path.read_text(encoding="utf-8")))
            elif name == "sample.json":
                found.update(_chroms_from_sample_json(path))
            elif name.endswith(".vcf.gz") or name.endswith(".vcf"):
                found.update(_chroms_from_vcf(path))
            elif name.endswith(".bam"):
                found.update(_chroms_from_bam_header(path))
        except (OSError, UnicodeError, gzip.BadGzipFile, json.JSONDecodeError):
            continue
    return sorted(found, key=_chrom_sort_key)


def _chrom_sort_key(chrom: str) -> tuple:
    rest = chrom[3:]
    if rest.isdigit():
        return (0, int(rest))
    return (1, rest)


def _accept_chrom(raw: str) -> Optional[str]:
    chrom = _normalize_type(raw)
    if chrom and _CHR_RE.match(chrom):
        return chrom
    return None


def _normalize_type(raw: str) -> str:
    """Accept chr18, CHR18, or 18 → chr18."""
    value = (raw or "").strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered.startswith("chr"):
        rest = lowered[3:]
    else:
        rest = lowered
    if rest in ("x", "y", "m"):
        rest = rest.upper()
    return f"chr{rest}" if rest else ""


def _chroms_from_region_text(text: str) -> Set[str]:
    found: Set[str] = set()
    for line in text.splitlines():
        token = line.strip().split(":", 1)[0]
        chrom = _accept_chrom(token)
        if chrom:
            found.add(chrom)
    return found


def _chroms_from_sample_json(path: Path) -> Set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return set()
    found: Set[str] = set()
    chrom = payload.get("chromosome")
    if isinstance(chrom, str):
        accepted = _accept_chrom(chrom)
        if accepted:
            found.add(accepted)
    region = payload.get("region")
    if isinstance(region, str):
        found.update(_chroms_from_region_text(region))
    return found


def _chroms_from_vcf(path: Path) -> Set[str]:
    found: Set[str] = set()
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("##contig="):
                match = _CONTIG_RE.search(line)
                if match:
                    chrom = _accept_chrom(match.group(1))
                    if chrom:
                        found.add(chrom)
                continue
            if line.startswith("#"):
                continue
            chrom = _accept_chrom(line.split("\t", 1)[0])
            if chrom:
                found.add(chrom)
            break
    return found


def _chroms_from_bam_header(path: Path) -> Set[str]:
    """Read ASCII chromosome names from the start of a BAM header."""
    found: Set[str] = set()
    with path.open("rb") as handle:
        blob = handle.read(2 * 1024 * 1024)
    text = blob.decode("latin-1", errors="ignore")
    for match in re.finditer(r"chr(?:[1-9]|1[0-9]|2[0-2]|X|Y|M)\b", text):
        chrom = _accept_chrom(match.group(0))
        if chrom:
            found.add(chrom)
    return found


def _ensure_references(chroms: Iterable[str], reference_dir: Path) -> bool:
    """Download FASTA + RTG SDF for each chromosome that is not already on disk."""
    chrom_list = [chrom for chrom in chroms if chrom]
    if not chrom_list:
        print("ERROR: no chromosome, so no reference can be fetched.", flush=True)
        return False
    ref_base = (os.getenv("REF_S3_BASE") or DEFAULT_REF_BASE).rstrip("/")
    print(f"  Reference: {reference_dir}", flush=True)
    for chrom in chrom_list:
        if not _ensure_one_reference(chrom, ref_base, reference_dir):
            return False
    return True


def _ensure_one_reference(chrom: str, ref_base: str, reference_dir: Path) -> bool:
    fa_dir = reference_dir / chrom
    fa_path = fa_dir / f"{chrom}.fa"
    sdf_dir = fa_dir / f"{chrom}.sdf"
    fasta_ready = fa_path.is_file() and fa_path.stat().st_size > 0
    sdf_ready = (sdf_dir / "seqdata0").is_file()
    if fasta_ready and sdf_ready and _fasta_sidecars_ready(fa_dir, chrom):
        print(f"  Reference {chrom} already present", flush=True)
        return True

    print(f"  Fetching reference for {chrom}...", flush=True)
    fa_dir.mkdir(parents=True, exist_ok=True)
    for ext in FASTA_EXTS:
        dest = fa_dir / f"{chrom}.{ext}"
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        url = f"{ref_base}/{chrom}/{chrom}.{ext}"
        if not _download_file(url, dest, headers={"User-Agent": REF_USER_AGENT}, timeout=120):
            print(f"   ERROR: could not download {chrom}.{ext}", flush=True)
            return False
    if not sdf_ready:
        sdf_dir.mkdir(parents=True, exist_ok=True)
        for name in SDF_FILES:
            dest = sdf_dir / name
            if dest.is_file() and dest.stat().st_size > 0:
                continue
            url = f"{ref_base}/{chrom}/{chrom}.sdf/{name}"
            if not _download_file(url, dest, headers={"User-Agent": REF_USER_AGENT}, timeout=120):
                print(f"   ERROR: could not download {chrom}.sdf/{name}", flush=True)
                return False
    print(f"  Reference {chrom} ready", flush=True)
    return True


def _fasta_sidecars_ready(fa_dir: Path, chrom: str) -> bool:
    for ext in ("fa.fai", "dict"):
        path = fa_dir / f"{chrom}.{ext}"
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


if __name__ == "__main__":
    sys.exit(main())
