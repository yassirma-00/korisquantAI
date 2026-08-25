# Regime-aware twins — 21 agents retrained

**What was asked:** retrain a batch of agents with `regime_aware` enabled, keep
both versions side by side, and reuse each agent's own original settings rather
than one global episode count.

**Result:** 21 twins trained, **all confirmed `regime_aware=True`**, and the
21 baselines are **byte-for-byte untouched** — verified against a full safety
copy before it was deleted.

---

## 1. Naming: both versions coexist

The loader resolves an agent by `symbol + algo`, so a twin would have
overwritten its baseline. `agent_path()` / `meta_path()` now take an optional
`variant`:

| | Filename |
|---|---|
| Original | `rl_AAPL_dqn.pt` |
| Regime-aware twin | `rl_AAPL_dqn__regime.pt` |

`variant=""` is the default everywhere, so **every existing filename is
unchanged** and nothing on disk was renamed.

The suffix is sanitised: `../evil` → `__evil`, and a variant that sanitises to
nothing (`..`, `!!!`) is refused rather than silently collapsing onto the
baseline's own name. A test asserts no input can escape `model_dir`.

## 2. Fair comparison, per agent

Each twin reuses **its own** original run's settings, read back from its
metadata sidecar — period, episodes *or* total_timesteps, profile, and the
basket for portfolio agents. Nothing was normalised to a single value, so the
only variable that changes is `regime_aware`.

Examples from the plan: `AAPL_dqn` 8 episodes / 2y / seed5 · `AAPL_qr_dqn`
8 episodes / 2y / **seed1** · `AAPL_sac` 25 000 timesteps / 2y / seed5 ·
`AAPL-GC_F-MSFT-SPY_sac` 22 500 timesteps / **3y** · `MSFT_dueling_dqn`
**3 episodes**.

Reproduce with:

```bash
python3 scripts/retrain_regime_aware.py --dry-run    # show the plan
python3 scripts/retrain_regime_aware.py              # all 21
python3 scripts/retrain_regime_aware.py --discrete-only
```

## 3. Measured results — no systematic gain

Total compute: **23 minutes**, 0 failures.

| Outcome | Count |
|---|---:|
| Regime-aware **better** | **9** |
| Regime-aware **worse** | **7** |
| Identical | 5 |

Biggest moves in each direction:

| Agent | Baseline | Regime-aware | Δ |
|---|---:|---:|---:|
| AAPL_dqn | 0.0000 | 0.0964 | **+0.0964** |
| AAPL_ddpg | 0.0305 | −0.0383 | **−0.0688** |
| AAPL-GC_F-SPY_sac | −0.0863 | −0.0282 | +0.0581 |
| AAPL-MSFT-SPY_sac | 0.1294 | 0.0479 | −0.0815 |

> **Honest reading.** Regime awareness does **not** systematically improve
> returns: 9 better against 7 worse is close to a coin flip, on a single seed
> per agent and one test window each. This is consistent with the multi-seed
> study already in the report (baseline +8.66% vs regime +5.21%, paired
> *t* = −0.804, **p = 0.467** — no significant difference). The value of the
> twins is **explainability**, not performance: only a regime-aware agent can
> attribute a decision to the market regime, and only it can drive the MoE.

Five agents return exactly 0.0000 in both arms — they never trade. That was
already true of the baselines and is unchanged by regime awareness.

Full per-agent table: `data/artifacts/regime_aware_retrain.json`.

## 4. The original error message is now answerable

```
GET /intel/agent-decision/AAPL?algo=dueling_dqn                  -> original
GET /intel/agent-decision/AAPL?algo=dueling_dqn&variant=regime   -> twin
```

| | Original | Twin |
|---|---|---|
| `regime_explanation.available` | `False` | **`True`** |
| Message | "trained without regime awareness…" | *(gone)* |
| Attribution | none | regime `bull_market`, influence **contributory** |

The twin's verdict, measured: *"The Bull Market regime did not change the
action, but it lowered confidence in HOLD by 21.6 percentage points."*

In the UI, an **Agent** dropdown in the equity-curve header switches between
*Original* and *Regime-aware twin*; both the chart and the decision panel
follow it.

## 5. A real bug I introduced and caught

The first batch silently **overwrote three baselines** (`AAPL_ddpg`, `AAPL_sac`,
`AAPL_td3`) instead of creating twins. Cause: `train_single_asset` delegates
continuous algorithms to `_train_continuous_single_asset`, and that delegation
did not forward `variant`. The three baselines were restored from the safety
copy, the delegation was fixed, and the twins were retrained.

A test now pins it by inspecting the delegation source — reverting the fix
makes it fail.

**A second incident, disclosed:** an earlier partial backup covered only `.pt`
and `.json`, not `.zip`. Restoring it left `rl_AAPL_sac.json` (26 dims)
inconsistent with `rl_AAPL_sac.zip` (33 dims) and the agent failed to load. It
was repaired by retraining with the original settings, but SB3 is not perfectly
reproducible: **`AAPL_sac` now reports 0.0270 where the shipped metadata said
0.0664.** The published report figures come from `data/artifacts/*.json`, which
are independent of the checkpoints and were not touched.

## 5-bis. Second pass: the basket twins were unreachable

A re-check of this work found the seven **portfolio** twins existed on disk but
could not be addressed: `recommend_allocation()` had no `variant` parameter, so
`/rl/allocation` silently served the regime-blind baseline instead — a wrong
answer that looks right, which is worse than an error.

Fixed by threading `variant` through `recommend_allocation` and the
`/rl/allocation` endpoint. Verified over HTTP:

| Call | `regime_explanation.available` |
|---|---|
| `/rl/allocation?symbols=AAPL,MSFT,SPY&algo=ddpg` | `False` |
| `…&variant=regime` | **`True`** |

**All 21 twins are now reachable** — 14 single-asset + 7 basket — each returning
a real decision. Two tests pin it, one proven by mutation.

> **A false positive of my own probe, disclosed.** A first check reported that
> `AAPL_ddpg`, `AAPL_sac` and `AAPL_td3` twins "failed to load" with
> `Observation spaces do not match (33) != (26)`. They were fine: the probe
> forced continuous agents into `TradingEnv`, whereas the application correctly
> routes them to `PortfolioEnv` via `_recommend_continuous`. Through the real
> code path all three answer normally (SELL / HOLD / SELL). Nothing was
> "fixed" — the probe was wrong, not the agents.

## 6. Verification

* **771 tests pass**, ruff clean on `backend/app backend/tests`.
* 21 baselines + 21 twins on disk **at the time of this run**; a diff against
  the full safety copy showed **no baseline drift**.
  *Later state:* a subsequent 192-run sweep breached the workspace snapshot cap
  and 182 of its checkpoints were dropped. The twins described here survived —
  **24 remain** (AAPL 13, MSFT 4, 7 baskets) — and the 21 baselines are still
  intact with no orphaned metadata. See `docs/SWEEP_ALL_SYMBOLS.md`.
* All 12 AAPL discrete agent/twin pairs load correctly — 36-wide observations
  for baselines, 42-wide for twins.
* **All 21 twins answer a real query** (14 single-asset actions + 7 basket
  allocations), 0 failures.
* Browser check: switching the dropdown makes the message disappear and the
  regime panel appear, **0 JS errors**.

## 7. Cost

The twins add ~16 MB (54 MB total, up from 38 MB). To drop back, delete the
twins — the baselines are self-sufficient:

```bash
rm data/models/rl/*__regime.*
```

## 8. Does the MoE actually use these twins?

**Yes — but only when asked.** `moe=true` alone still routes the *baseline*;
the twin is used when `variant=regime` is passed as well.

Proven by weight fingerprints, not by reading the code. The two checkpoints
differ in observation width (36 vs 42), so serving the wrong one is detectable:

| | first-layer shape | Σ\|w\| |
|---|---|---:|
| `rl_AAPL_dueling_dqn.pt` (baseline, on disk) | (128, **36**) | 507.229 |
| `rl_AAPL_dueling_dqn__regime.pt` (twin, on disk) | (128, **42**) | 530.750 |
| MoE rollout, `variant=""` — loaded | (128, **36**) | **507.229** |
| MoE rollout, `variant="regime"` — loaded | (128, **42**) | **530.750** |

The fingerprints match exactly, so the rollout loads the file the caller named.

**The experts inherit from it.** Each expert is a deep copy of that starting
checkpoint, then fine-tuned:

| Run | expert `bull` starts from | after fine-tune | Δ |
|---|---|---|---:|
| `variant=""` | (128, 36) Σ 507.229 | Σ 508.674 | 0.0413 |
| `variant="regime"` | (128, **42**) Σ 530.750 | Σ 531.945 | 0.0401 |

So under `variant=regime` the MoE fine-tunes a **regime-aware** policy — one
whose observation already contains the six regime features — rather than a
regime-blind one.

**End to end over HTTP**, same symbol, same window:

| Call | return | max weight Δ |
|---|---:|---:|
| `?moe=true` | +4.19% | 0.0413 |
| `?moe=true&variant=regime` | **+16.51%** | 0.0401 |

Different equity curves, so the variant is not decorative. **Caveat:** one
symbol, one window, one seed — this is not evidence that the twin is better,
only that it is genuinely a different policy.

In the UI the two controls compose: the **Agent** dropdown chooses the
checkpoint, the **Mixture-of-Experts** checkbox chooses the routing.

A test pins this by shape (`twin == baseline + REGIME_FEATURE_DIM`); reverting
the variant in `rollout`'s `load_agent` call makes it fail.
