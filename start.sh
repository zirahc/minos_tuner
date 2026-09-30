#!/usr/bin/env bash
# GATK VPS setup. Clone minos_tuner first, then from that folder:
#   bash start.sh 1 18
# First argument is this machine's WORKER_ID. Second is the practice chromosome.
# minos_subnet is cloned next to minos_tuner. No other file is edited.

set -euo pipefail

if [[ $# -ne 2 || -z "${1:-}" || -z "${2:-}" ]]; then
  echo "Usage: bash start.sh WORKER_ID CHR_TYPE"
  echo "Example: bash start.sh 1 18"
  exit 1
fi

WORKER_ID="$1"
CHR_TYPE="$2"
TUNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$TUNER_DIR/.." && pwd)"
SUBNET_DIR="$PARENT_DIR/minos_subnet"

echo "tuner:  $TUNER_DIR"
echo "subnet: $SUBNET_DIR"
echo "worker: $WORKER_ID"
echo "chr:    $CHR_TYPE"

if [[ ! -d "$SUBNET_DIR/.git" ]]; then
  git clone https://github.com/minos-protocol/minos_subnet.git "$SUBNET_DIR"
fi

# install.sh asks two questions. Answer them on a terminal:
# demo miner, then GATK (already the default).
python3 - "$SUBNET_DIR" <<'PY'
import os
import select
import signal
import sys
import time

subnet = sys.argv[1]
master, slave = os.openpty()
pid = os.fork()
if pid == 0:
    os.setsid()
    os.dup2(slave, 0)
    os.dup2(slave, 1)
    os.dup2(slave, 2)
    os.close(master)
    os.close(slave)
    os.chdir(subnet)
    os.environ["TERM"] = "xterm-256color"
    os.execvp("bash", ["bash", "install.sh", "--no-ai-assistant"])

os.close(slave)
buf = b""
role_sent = False
template_sent = False
update_only = False
pending = None
send_at = 0.0
status = None
try:
    while True:
        if pending is not None and time.time() >= send_at:
            os.write(master, pending)
            pending = None
        ready, _, _ = select.select([master], [], [], 0.2)
        if master in ready:
            try:
                chunk = os.read(master, 4096)
            except OSError:
                chunk = b""
            if not chunk:
                break
            os.write(1, chunk)
            buf += chunk
            text = buf.decode("utf-8", "ignore")
            if "Existing installation detected" in text or "Update complete" in text:
                update_only = True
            if "No existing wallets found" in text or (
                "Live miners and validators need a Bittensor hotkey" in text
            ):
                print(
                    "\nERROR: installer selected Miner, not Demo miner.",
                    file=sys.stderr,
                )
                os.kill(pid, signal.SIGTERM)
                sys.exit(1)
            if pending is None and (not role_sent) and ("What are you setting up?" in text):
                # Miner, Validator, Demo miner. Two downs land on Demo miner.
                # A third down wraps back to Miner.
                pending = b"\x1b[B\x1b[B\r"
                send_at = time.time() + 1.0
                role_sent = True
            elif (
                pending is None
                and role_sent
                and (not template_sent)
                and ("Select your variant calling template:" in text)
            ):
                # GATK is the highlighted default. Enter only.
                pending = b"\r"
                send_at = time.time() + 1.0
                template_sent = True
            if len(buf) > 65536:
                buf = buf[-8192:]
        done, waited = os.waitpid(pid, os.WNOHANG)
        if done:
            status = waited
            while True:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                os.write(1, chunk)
            break
finally:
    os.close(master)

if status is None:
    _, status = os.waitpid(pid, 0)

if not update_only and (not role_sent or not template_sent):
    print(
        "ERROR: install.sh did not ask for demo miner and GATK.",
        file=sys.stderr,
    )
    sys.exit(1)
if os.WIFEXITED(status):
    code = os.WEXITSTATUS(status)
else:
    code = 1
sys.exit(code)
PY

cp "$TUNER_DIR/score_gatk_folders.py" "$SUBNET_DIR/score_gatk_folders.py"

if [[ ! -f "$TUNER_DIR/.env" ]]; then
  cp "$TUNER_DIR/.env.example" "$TUNER_DIR/.env"
fi

python3 - "$TUNER_DIR/.env" "$WORKER_ID" "$SUBNET_DIR" <<'PY'
import sys
from pathlib import Path

path, worker, subnet = sys.argv[1:]
file = Path(path)
lines = file.read_text(encoding="utf-8").splitlines()

def set_key(lines, key, value):
    out = []
    found = False
    prefix = key + "="
    commented = "# " + prefix
    commented_tight = "#" + prefix
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix) or stripped.startswith(commented) or stripped.startswith(commented_tight):
            if not found:
                out.append(f"{key}={value}")
                found = True
            continue
        out.append(line)
    if not found:
        out.append(f"{key}={value}")
    return out

lines = set_key(lines, "MINOS_SUBNET", subnet)
lines = set_key(lines, "WORKER_ID", worker)
file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"set WORKER_ID={worker}")
print(f"set MINOS_SUBNET={subnet}")
PY

if [[ ! -f "$SUBNET_DIR/.venv/bin/activate" ]]; then
  echo "ERROR: $SUBNET_DIR/.venv is missing. install.sh did not finish."
  exit 1
fi

# shellcheck disable=SC1091
source "$SUBNET_DIR/.venv/bin/activate"
cd "$TUNER_DIR"
pip install -r requirements.txt
python download_practice_samples.py --type "$CHR_TYPE"
exec python main.py
