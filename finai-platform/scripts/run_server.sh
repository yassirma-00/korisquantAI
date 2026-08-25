#!/usr/bin/env bash
# Start the KorisQuant AI backend (serves the API and the bundled dashboard).
#
# Usage:  ./scripts/run_server.sh {start|stop|restart|logs}
# Tip: if you get "Permission denied" (the execute bit is not always preserved
#      by git/zip/docker COPY), either run `chmod +x scripts/run_server.sh`
#      once, or invoke it directly as `bash scripts/run_server.sh start`.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SELF")/.." && pwd)"
cd "$ROOT/backend"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG="${LOG:-/tmp/korisquant_server.log}"

export PYTHONPATH="$ROOT/backend"

# Which interpreter to run under.
#
# Prefer the project's own virtualenv when one exists, even if it was never
# activated. Hard-coding `python3` meant a fully installed .venv sitting right
# there was ignored, and the server died on ModuleNotFoundError while the user
# could see the packages were installed. An explicitly activated environment
# still wins, so `source other/bin/activate` is not overridden.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/venv/bin/python" ]; then
  PYTHON="$ROOT/venv/bin/python"
else
  PYTHON="python3"
fi

# Fail with the one instruction that fixes it, rather than a stack trace.
if ! "$PYTHON" -c "import fastapi" 2>/dev/null; then
  # A .venv that exists but records a different path is a *moved* project, not
  # a missing environment. Telling that user to "create a virtualenv" sends
  # them to rebuild something they already have.
  stale=""
  if [ -f "$ROOT/.venv/bin/activate" ]; then
    recorded="$(sed -n 's/^[[:space:]]*export VIRTUAL_ENV=//p' "$ROOT/.venv/bin/activate" \
                | head -1 | tr -d '"'"'"'')"
    [ -n "$recorded" ] && [ "$recorded" != "$ROOT/.venv" ] && stale="yes"
  fi

  if [ -n "$stale" ]; then
    cat >&2 <<EOF
This project's virtualenv still points at its old location:

  recorded: $recorded
  actual:   $ROOT/.venv

That happens when the folder is moved or renamed. Nothing is lost — the
packages are still there, only the paths are stale. Repair it in place:

  bash scripts/fix_venv.sh
EOF
  else
    cat >&2 <<EOF
Dependencies are missing for: $PYTHON

Install them into a virtualenv (Kali and Debian refuse system-wide pip
installs — that is the "externally-managed-environment" error):

  cd "$ROOT"
  python3 -m venv .venv
  source .venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt

A .venv in the project root is picked up automatically, activated or not.
EOF
  fi
  exit 1
fi

# self-heal the execute bit so `./scripts/run_server.sh` keeps working after a
# checkout or archive extraction that dropped file modes.
chmod +x "$SELF" 2>/dev/null || true

case "${1:-start}" in
  start)
    if curl -s -m 2 -o /dev/null "http://$HOST:$PORT/health" 2>/dev/null; then
      echo "Already running on http://$HOST:$PORT"
      exit 0
    fi
    setsid "$PYTHON" -m uvicorn app.main:app --host "$HOST" --port "$PORT" \
      >"$LOG" 2>&1 < /dev/null &
    echo $! > /tmp/korisquant_server.pid
    for _ in $(seq 1 40); do
      sleep 1
      if curl -s -m 2 -o /dev/null "http://$HOST:$PORT/health" 2>/dev/null; then
        echo "KorisQuant AI running -> http://$HOST:$PORT  (docs: /docs, log: $LOG)"
        exit 0
      fi
    done
    echo "Server failed to start; last log lines:" >&2
    tail -20 "$LOG" >&2
    exit 1
    ;;
  stop)
    if [ -f /tmp/korisquant_server.pid ]; then
      kill "$(cat /tmp/korisquant_server.pid)" 2>/dev/null || true
      rm -f /tmp/korisquant_server.pid
      echo "Stopped."
    else
      echo "No PID file; nothing to stop."
    fi
    ;;
  restart)
    bash "$SELF" stop || true
    sleep 2
    bash "$SELF" start
    ;;
  logs)
    tail -f "$LOG"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|logs}"
    exit 1
    ;;
esac
