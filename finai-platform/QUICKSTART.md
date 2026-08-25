# Quickstart

Three ways to run KorisQuant AI. Pick one. Every command below was executed and verified.

---

## Option A — local Python (fastest, recommended)

All commands run from the **project root** — the folder containing `README.md`:

```bash
cd /path/to/the/project        # the folder containing README.md

# 1. Create a virtualenv. Debian, Ubuntu and Kali refuse system-wide pip
#    installs (PEP 668): without this, pip stops with
#    "error: externally-managed-environment".
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (~30s). The CPU-only torch wheel keeps this small.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Start the server
bash scripts/run_server.sh start
```

> `requirements.txt` exists at both the root and in `backend/` (the root one
> just forwards to it), so either directory works.

> **Moved or renamed the project folder? Run `bash scripts/fix_venv.sh`.**
>
> `venv` writes its own absolute path into `.venv/bin/activate` and into the
> shebang of every console script. Move any parent directory and those paths
> point at a folder that no longer exists. The failure is quiet: the prompt
> still shows `(.venv)` while `pip` is silently the system one — and Kali then
> refuses it with `externally-managed-environment`. The give-away is
> `which pip` printing `/usr/bin/pip` or `/usr/local/bin/pip`.
>
> **Nothing is lost.** The packages are still on disk and importable; only the
> recorded paths are stale, so the repair is in place and takes seconds rather
> than re-downloading gigabytes:
>
> ```bash
> bash scripts/fix_venv.sh
> ```
>
> Doing it by hand is two commands — recreating over the existing directory
> rewrites `activate`, then reinstalling pip rewrites its own launcher, which
> the first step leaves alone:
>
> ```bash
> python3 -m venv .venv
> ./.venv/bin/python -m pip install --force-reinstall pip
> ```
>
> `scripts/run_server.sh` uses `.venv/` in the project root automatically, so
> forgetting to activate it is not fatal.

Expected output:

```
KorisQuant AI running -> http://127.0.0.1:8000  (docs: /docs, log: /tmp/korisquant_server.log)
```

Then open **<http://localhost:8000>**.

### Optional: enable the AI assistant

The chat panel on every page needs Ollama. Two ways:

**Cloud** — get a key at <https://ollama.com/settings/keys>:

```bash
cat >> .env <<'EOF'
OLLAMA_API_KEY=your_key_here
OLLAMA_BASE_URL=https://ollama.com/v1
OLLAMA_MODEL=gpt-oss:20b
EOF
bash scripts/run_server.sh restart
```

**Local** — no key, no cost, nothing leaves your machine:

```bash
ollama serve &            # install from https://ollama.com/download
ollama pull llama3.1      # any model with tool-calling support

cat >> .env <<'EOF'
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1
EOF
bash scripts/run_server.sh restart
```

Verify with `python3 scripts/check_install.py`, or open the panel — the header
shows the active model and tool count when it works, and names the exact fix when
it does not (`ollama serve`, `ollama pull <model>`, a bad key, or a model that
needs a paid plan).

Skip this and everything else still runs; the panel simply reports itself as
unavailable. The key is read server-side only and never sent to the browser.

### Managing the server

```bash
bash scripts/run_server.sh start      # start (idempotent)
bash scripts/run_server.sh stop       # stop
bash scripts/run_server.sh restart    # restart after code changes
bash scripts/run_server.sh logs       # tail the log
```

> The script is committed with the execute bit set, so `./scripts/run_server.sh start`
> also works. Git/zip/Docker don't always preserve that bit — if you see
> `Permission denied`, either prefix with `bash` (as above) or run
> `chmod +x scripts/run_server.sh` once.

### Prefer plain uvicorn?

```bash
cd backend
uvicorn app.main:app --reload --port 8000     # --reload for development
```

---

## Option B — Docker

```bash
cd /path/to/the/project        # the folder containing README.md
docker compose up --build            # API + dashboard on :8000
docker compose --profile full up     # + PostgreSQL + Redis
docker compose down                  # stop
```

---

## Option C — no server (library / notebook use)

Every service works standalone — useful for research or a report:

```python
import sys; sys.path.insert(0, "backend")

from app.services.data.market_data import market_data_service
from app.services.indicators.technical import compute_indicators, signal_summary

series = market_data_service.get_history("AAPL", period="2y")
print(series.source, len(series.df), "bars")

enriched = compute_indicators(series.df, ["rsi", "macd", "bbands"])
print(signal_summary(enriched)["consensus"])
```

---

## Where to go in the UI

| URL | Page |
|---|---|
| `/` | Market overview — indices, watchlist, heatmap, AI screener |
| `/analysis.html` | Candlesticks, 17 indicators, statistics, news, correlations |
| `/forecast.html` | Train LSTM/GRU/TCN/Transformer, forecast with confidence bands |
| `/rl.html` | Train RL agents, equity curve vs Buy & Hold, live action |
| `/signals.html` | Fused recommendation with SHAP + rationale |
| `/xai.html` | SHAP / LIME / counterfactuals / sentiment explorer |
| `/portfolio.html` | Paper trading, optimiser, efficient frontier |
| `/risk.html` | Anomalies, crash risk, VaR, alerts |
_API reference (Swagger) is hidden from the site. For local development run
`EXPOSE_API_DOCS=true bash scripts/run_server.sh restart`, then open `/docs`._

---

## First 5 minutes

The platform works immediately — **no API keys needed** (Yahoo Finance is free).
Deep-learning and RL signals only appear once you train them:

1. Open **<http://localhost:8000>** — live prices load right away.
2. Go to **AI Forecasting** → click **Train Model** (~5–15s for LSTM on AAPL).
3. Go to **RL Agent** → click **Train Agent** (~20–40s for 15 episodes).
4. Go to **Recommendations** → click **Generate**. All four signals now contribute.

Prefer to pre-train from the CLI?

```bash
python scripts/seed_demo.py             # 1 forecaster + 1 agent for AAPL
python scripts/seed_demo.py --full      # all 5 architectures + 3 agents
python scripts/seed_demo.py --symbol MSFT
```

Until models are trained, the recommendation engine reports
`missing_signals: [forecast, rl]` and renormalises weights across the signals it
does have — it never invents a score.

---

## Verify your install

```bash
python scripts/check_install.py     # confirms this copy has the latest fixes
```

Run this first whenever a symptom persists after an update — it tells you
whether the code that is *running* is the code that was *fixed*, and prints the
exact command to repair anything that is out of date.

```bash
cd backend && pytest tests/ -q      # 182 tests, ~2min, fully offline
ruff check app tests                # lint
curl localhost:8000/health          # should report "healthy"
```

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `Permission denied` on the script | Execute bit lost in transit → `bash scripts/run_server.sh start` |
| `No module named uvicorn` | Dependencies not installed → rerun the `pip install` step in Option A |
| Port 8000 busy | `PORT=8080 bash scripts/run_server.sh start` |
| Badges read `SIMULATED` | No network reached Yahoo. The synthetic engine took over so nothing breaks — data is fake but plausible. Set `DATA_MODE=live` to fail loudly instead. |
| `409 model_not_trained` | Expected before training. Train on the Forecast/RL page or run `seed_demo.py`. |
| `Could not open requirements file` | You're in the wrong directory. `cd` to the project root (where `README.md` is), or use `pip install -r backend/requirements.txt`. |
| Old dates / inflated metrics on the RL page | Agents trained before the train/test split fix keep their contaminated metadata. Run `python scripts/purge_leaky_agents.py --delete`, then retrain. Affected agents are also flagged **stale** in the Trained Agents table. |
| `api.X is not a function` | A stale cached script. Assets are now content-hashed (`api.js?v=b6f79b`), so a normal reload fixes it automatically. If you are running an **older copy of the code**, pull the latest and restart: `bash scripts/run_server.sh restart`. |
| Server won't start | `tail -30 /tmp/korisquant_server.log` |
| Slow first request | Cold cache + live provider fetch; subsequent calls are cached for 5 min. |
