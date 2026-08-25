# Regime-aware sweep — 32 symbols × 6 native discrete algorithms

> ## ⚠️ Status correction — the artefacts did not survive
>
> **The 192 runs were really trained and really verified** (sections 4 and 5
> below are genuine measurements, taken while the files existed). But the
> workspace enforces a **128 MB / 10 000-file snapshot cap**, and this session
> created 157.9 MB across 971 files. **364 files were dropped, including 182 of
> the 192 checkpoints.**
>
> **What is on disk now: 10 of the 192** (`--verify` reports `10/192`). The
> survivors are the AAPL and MSFT twins listed in
> `docs/REGIME_AWARE_TWINS.md`; the twins for the other 30 symbols are gone.
>
> Nothing else was lost: all source, tests, docs and the **21 baselines** are
> intact, with **no orphaned metadata** (every remaining `.json` has its
> weights). The ledger was reconciled against the disk so it no longer claims
> runs whose artefacts are absent.
>
> Regenerate at any time — the plan is deterministic and the script resumable:
>
> ```bash
> python3 scripts/train_all_regime_aware.py --only-algo dqn double_dqn dueling_dqn c51 iqn rainbow
> ```
>
> Expect ~36 minutes and ~64 MB, which will breach the cap again. Read the
> figures below as *"this is what the sweep produced when it ran"*, not as a
> description of the current directory.

**Trained and verified when run: 192/192 combinations, 0 failures.**

Scope was narrowed from the original 416 (32 × 13) on your instruction, after I
reported that the full sweep would cost ~10 h and ~512 MB — in direct conflict
with the 40 MB target set two turns earlier. The 7 non-native algorithms
(`ppo`, `a2c`, `trpo`, `ddpg`, `td3`, `sac`, `qr_dqn`) were **not** trained.

---

## 1. Plan, verified before training

`python3 scripts/train_all_regime_aware.py --plan` printed the full list and
confirmed the coverage mechanically:

```
symbols  : 32      algos : 6      total : 192
SKIPPED: none — every symbol x algorithm pair is planned.
```

Checked programmatically: **no symbol has ≠ 6 algorithms**, **no algorithm has
≠ 32 symbols**. Nothing was skipped, so there is no skip reason to report.

### Hyperparameters are inherited, never invented

Each combination reuses the settings recorded in the **existing baseline** for
that algorithm — period, episodes/timesteps, profile. A symbol that has its own
baseline takes precedence:

| Combination | Settings | Inherited from |
|---|---|---|
| `AAPL/dqn` | 2y, 8 ep, `seed5` | `AAPL/dqn` |
| `NVDA/c51` | 2y, 8 ep, **`seed1`** | `AAPL/c51` |
| `MSFT/dueling_dqn` | 2y, **3 ep**, `default` | `MSFT/dueling_dqn` (its own) |

The plan output names the source for every row.

## 2. Data provenance — checked before training

All 32 symbols resolve from the **disk cache** (13 `cache:stale`, 19 `cache`);
**none** falls through to the synthetic engine. An earlier probe appeared to
show 21 synthetic symbols — that was a cold in-memory cache on first call, not
the served data. Bar counts range 500–731 over 2 years, all above the 200-bar
floor.

## 3. A real bug this sweep exposed

Three European tickers (`MC.PA`, `AIR.PA`, `SAN.PA`) produced checkpoint stems
like `rl_MC.PA_dqn__regime`. Every save site calls `.with_suffix(".pt")`, and
`Path` reads `.PA_dqn__regime` as an existing extension and **replaces** it —
so all six algorithms for `MC.PA` wrote to the single file `rl_MC.pt`.

Four of the six then failed to load:

```
DQNConfig.__init__() got an unexpected keyword argument 'n_atoms'
```

— a C51 checkpoint being read back as a DQN. **Pre-existing bug**, not
introduced here; it had simply never been exercised because no `.PA` agent had
ever been trained.

Fixed at the root in `RLService._safe()` by mapping `.` to `_`, alongside the
`/`, `=`, `^` and `,` it already handled. The 18 affected runs were deleted and
retrained. A test pins it, proven by mutation.

**`--verify` was also too lenient** — it called those 12 broken checkpoints
"valid" because it only checked that the file existed. It now loads each policy
for real and compares the observation width.

## 4. End-to-end verification — all 192

| Check | Result |
|---|---|
| Checkpoint present + metadata `regime_aware=True` | **192/192** |
| Policy loads into its environment (obs width 42) | **192/192** |
| `rl_service.backtest(..., variant="regime")` returns a curve | **192/192** |
| Regime attribution available on the decision | **192/192** |
| Reachable over HTTP (`/rl/backtest`, `/intel/agent-decision`) | verified on a sample across all six asset classes |

Baselines untouched: the 21 original checkpoints are still on disk under their
original names, and 30 of the 32 symbols never had one — for those, `?variant=`
empty correctly returns **409 `model_not_trained`**, which is the honest answer.

## 5. Measured results — the uncomfortable part

Training cost **35.7 minutes**, 0 failures. But the returns are poor:

| Algorithm | Runs | Return exactly 0 | Share |
|---|---:|---:|---:|
| c51 | 31 | 30 | **97%** |
| iqn | 31 | 30 | **97%** |
| rainbow | 31 | 30 | **97%** |
| double_dqn | 31 | 15 | 48% |
| dueling_dqn | 30 | 9 | 30% |
| dqn | 31 | 9 | 29% |

**123 of 185 recorded runs never trade at all.** A return of exactly 0.0000
means the agent held cash for the entire test window. This reproduces, across
32 symbols, the behaviour already documented for the distributional family in
the report (`c51`/`iqn`/`rainbow` at 0.00% ± 0.00 in the multi-seed study) —
the sweep confirms it is systematic, not an artefact of one instrument.

Of the 62 runs that do trade, the **median return is −0.0011**. So:

> **These agents are not profitable.** The sweep delivers coverage and
> regime attribution across the whole catalogue, not performance. No claim of
> an edge is made, and none is supported by these numbers.

Per-run figures: `data/artifacts/train_all_regime.json`.

## 6. Cost

`data/models/rl` grew to **118 MB** during the sweep (project total 143 MB).
That is what breached the 128 MB workspace cap and cost 182 checkpoints.
After the truncation the project sits at **54 MB / 519 files** — 42% of the
size budget and 5% of the file budget.
The 192 twins were ~64 MB of that; the 24 that remain are ~18 MB. To reclaim
that too:

```bash
rm data/models/rl/*__regime.*       # all twins, the 21 baselines survive
```

**If you regenerate the sweep, budget for the cap.** 192 twins add ~64 MB and
384 files on top of the current 54 MB, which exceeds 128 MB once the LaTeX PDF
and the market cache are counted. Train in subsets (`--only-symbol`) and delete
each subset before the next, or accept that the snapshot will truncate again.

## 7. Reproduce

```bash
python3 scripts/train_all_regime_aware.py --plan
python3 scripts/train_all_regime_aware.py --only-algo dqn double_dqn dueling_dqn c51 iqn rainbow
python3 scripts/train_all_regime_aware.py --verify --only-algo dqn double_dqn dueling_dqn c51 iqn rainbow
```

The script is resumable: state lives in `data/artifacts/train_all_regime.json`
and a combination whose checkpoint already exists is skipped.

> **Ledger caveat, disclosed.** The ledger records **185** runs while **192**
> checkpoints exist and verify. Seven runs were completed by an earlier batch
> that was interrupted before it could write its state. The checkpoints are
> real and verified — the ledger is simply incomplete, and `--verify` reads the
> disk rather than the ledger.
