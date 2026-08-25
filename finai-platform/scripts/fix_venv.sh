#!/usr/bin/env bash
# Repair a virtualenv that stopped working after the project folder was moved
# or renamed.
#
# Why this is needed
# ------------------
# `python -m venv` writes its own absolute path into `.venv/bin/activate` and
# into the shebang of every console script it installs. Move or rename any
# parent directory and those paths point at a folder that no longer exists.
# The failure is quiet and misleading: `activate` still sets the prompt to
# `(.venv)`, but `pip` resolves to the system one, and Kali/Debian then refuse
# it with "error: externally-managed-environment" (PEP 668).
#
# The environment is NOT destroyed by a move. Every installed package is still
# on disk and importable — only the recorded paths are stale. So this repairs
# in place instead of deleting, which avoids re-downloading ~2 GB of wheels.
#
#   1. `python3 -m venv .venv` over the existing directory rewrites activate
#      and the interpreter symlinks, leaving site-packages untouched.
#   2. That step does NOT regenerate console scripts such as bin/pip, whose
#      shebang was written by ensurepip at creation time. Reinstalling pip with
#      --force-reinstall makes it rewrite its own launcher.
#
# Usage:  bash scripts/fix_venv.sh
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF/.." && pwd)"
VENV="${VENV_DIR:-$ROOT/.venv}"

echo "Project root : $ROOT"
echo "Virtualenv   : $VENV"

if [ ! -d "$VENV" ]; then
  echo
  echo "No virtualenv found. Creating one:"
  python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  echo
  echo "Created. Now install the dependencies (CPU-only torch first, or pip"
  echo "pulls ~2 GB of CUDA you do not need):"
  echo
  echo "  source .venv/bin/activate"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
  echo "  pip install -r requirements.txt"
  exit 0
fi

# What the environment currently believes about itself.
recorded="$(sed -n 's/^[[:space:]]*export VIRTUAL_ENV=//p' "$VENV/bin/activate" \
            | head -1 | tr -d '"'"'"'')"
echo "Recorded path: ${recorded:-<none>}"

if [ "$recorded" = "$VENV" ] && "$VENV/bin/pip" --version >/dev/null 2>&1; then
  echo
  echo "Nothing to repair — this virtualenv already points at itself."
  echo "If pip still fails, check that you activated THIS one:  which pip"
  exit 0
fi

echo
echo "Stale paths detected. Repairing in place (packages are kept)…"

# Step 1: rewrite activate and the interpreter symlinks.
python3 -m venv "$VENV"

# Step 2: make pip rewrite its own launcher shebang. `venv --upgrade` and
# `ensurepip --upgrade` both leave an existing, current pip alone, so neither
# fixes the stale shebang on its own.
"$VENV/bin/python" -m pip install --quiet --force-reinstall --no-cache-dir pip

echo
echo "Verifying…"
"$VENV/bin/pip" --version
if "$VENV/bin/python" -c "import fastapi" 2>/dev/null; then
  echo "Dependencies are present. Start the server with:"
  echo "  bash scripts/run_server.sh start"
else
  echo
  echo "The virtualenv works again, but the project dependencies are missing."
  echo "Install them (CPU-only torch FIRST to avoid ~2 GB of CUDA):"
  echo
  echo "  source .venv/bin/activate"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
  echo "  pip install -r requirements.txt"
fi
