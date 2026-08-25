# MoE Integration Report

**Question asked:** is the Regime-Aware Mixture-of-Experts now used by the real
application, with the minimum possible change and without disturbing anything
that already worked?

**Answer:** yes. Below is what was changed, how it is activated, why the
baseline is provably intact, and what the mechanism actually achieves — including
where it does not help.

---

## 1. The problem found on inspection

`backend/app/services/rl/moe.py` was **dead code**. An AST-level check of every
import in the repository found no importer outside the module's own tests:

```
grep -rn "services.rl.moe" --include=*.py .   ->  only backend/tests/test_api.py
```

The mechanism was correct — real DQN experts, causal regime detection, routing,
genuine fine-tuning, K-5 — but **not one line of it ever executed** in the
running platform. `expert_factory` was an extension point whose only callers
were test lambdas.

## 2. Integration point chosen, and why

The MoE only means something where a policy acts **sequentially, bar after
bar**, on real data. In this codebase there is exactly one such place:

| Candidate | Verdict |
|---|---|
| `RLService.recommend_action()` | Single point-in-time decision — no sequence, so no regime *switch* to react to. |
| `RLService.train_single_asset()` | Would alter how agents are produced, changing the baseline itself. Out of scope. |
| **`RLService.backtest()` → `GET /rl/backtest/{symbol}`** | **Chosen.** The only sequential replay over real bars, and already the endpoint the UI calls for the equity curve. |

A fact that made this clean, verified rather than assumed:

```
len(df) == len(env.raw) and df.index.equals(env.raw.index)   # True on 1y, 2y, 5y
env.t enumerates contiguous positions of env.raw             # verified
```

`env.t` indexes exactly the same positions as the frame the regime provider
classifies, so routing and stepping cannot disagree about which bar is bar *t* —
no re-indexing layer was needed.

### Why a loop rather than `agent.evaluate(env)`

`evaluate()` holds **one fixed policy for the whole episode by construction**.
The entire point of a MoE is that the acting policy changes *mid-episode*.
Expressing that through `evaluate()` would have required modifying `evaluate()`
— putting MoE code on the **baseline's execution path**, which is precisely what
this task forbade. The new loop copies `evaluate()`'s decision rule verbatim
(`argmax` over `q_values`), which is why the no-expert case reproduces it
exactly.

## 3. Files modified — the complete list

| File | Change | Evidence |
|---|---|---|
| `backend/app/services/rl/moe.py` | **+265 lines appended.** The original 500 lines are preserved **verbatim as a prefix**. | `after.startswith(before) == True` |
| `backend/app/api/v1/endpoints/rl.py` | 2 optional query params + one lazily-imported branch, on the `backtest` endpoint only. | 23-line diff, shown below |
| `backend/tests/test_api.py` | +8 integration tests (752 → 760). | — |
| `README.md`, `docs/RAPPORT_COMPLET.md` | Documentation + corrected stale figures. | §5 below |
| `frontend/auth.html`, `frontend/landing.html` | Advertised test count 752 → 760. | guarded by `test_the_advertised_test_count_is_true` |

**Files deliberately NOT touched — verified byte-identical (md5):**

`services/rl/service.py` · `services/rl/environment.py` · `services/rl/agents/dqn.py`
· `services/rl/regime_features.py` · `services/risk/regime.py`

No change to the RL architecture, environment, reward, XAI, risk metrics, or any
frontend logic. The RL page was loaded in a real browser after the change:
**0 JS errors**, rendering identical.

The whole API-layer diff:

```python
+    moe: bool = Query(False, description=(...)),
+    moe_adapt: bool = Query(True, description=(...)),
 ):
+    env_overrides = {"initial_balance": initial_balance,
+                     "transaction_cost": transaction_cost}
+    if moe:
+        from app.services.rl.moe import rollout          # lazy: off-path
+        return rollout(symbol, algo=algo, period=period,
+                       env_overrides=env_overrides, adapt=moe_adapt)
     return rl_service.backtest(symbol, algo=algo, period=period,
-                               env_overrides={...})
+                               env_overrides=env_overrides)
```

No new endpoint was added: the API still exposes **127 paths / 135 operations**,
exactly as before.

## 4-bis. In the user interface

The MoE is operable from the RL page, not only from a hand-typed URL:

* **Checkbox "Mixture-of-Experts"** in the equity-curve card header. Off by
  default. Ticking it re-runs the backtest through the routed experts and opens
  a panel below the chart with the routing trace.
* The toggle **disables itself** for SB3 algorithms (PPO/A2C/SAC/TD3/TRPO) and
  its tooltip states why, instead of letting the user trigger the 422.
* **Checkbox "Regime-aware"** in the training row — see §9.

The panel deliberately shows the failures next to the successes: how many
switches produced no reaction, both K-5 readings, and — when a mean rests on a
single observation — the words *"single event, not an average"*. Measured on
AAPL/1y it reports 8 switches, **1** real fine-tune, K-5 60 bars over 1 of 8.

## 4. How MoE is activated

```bash
# Baseline — unchanged, and the default
GET /rl/backtest/AAPL?algo=dueling_dqn&period=2y

# MoE: route + fine-tune + measure K-5
GET /rl/backtest/AAPL?algo=dueling_dqn&period=2y&moe=true

# Control: route the experts, apply no gradient
GET /rl/backtest/AAPL?algo=dueling_dqn&period=2y&moe=true&moe_adapt=false
```

`moe=true` returns the same keys as the baseline (`performance`, `baselines`,
`equity_curve`, `trades`) **plus** a `moe` block containing the routing trace,
the adaptation records, and K-5. A caller that ignores the extra block receives
a payload it already knows how to render.

## 5. How the baseline is proven unchanged

Three independent guarantees, each pinned by a test:

1. **Identical response.** `moe=false` and the parameterless call return
   byte-equal JSON, and the baseline response gains **neither** a `moe` key
   **nor** a `mode` key. Existing consumers see no contract change.
2. **Bit-for-bit numerical parity.** With the expert threshold made unreachable
   (`min_expert_bars=10**9`), the MoE rollout reproduces the baseline exactly:

   | | `total_return` | `n_trades` | equity curve |
   |---|---:|---:|---|
   | `rl_service.backtest` | 0.0419 | 37 | — |
   | `moe.rollout` (no expert eligible) | 0.0419 | 37 | **identical** |

3. **Not even imported.** The MoE import sits *inside* the `if moe:` branch, so
   the default path cannot be affected by anything in the module.

## 6. Does the real application now use the MoE? — measured

Yes, over live HTTP against the running server (AAPL, `dueling_dqn`):

| Window | Baseline | MoE | Buy & Hold | Bars driven by an expert | Real fine-tunes (max weight Δ) |
|---|---:|---:|---:|---|---:|
| 2y | +23.84% | **+30.22%** | +38.65% | bull 241 / 480 | 1 (0.047) |
| 5y | +1.19% | **+6.64%** | +111.76% | bull 895 · bear 85 · stress 10 / 1234 | 13 (0.329) |

The weight deltas are measured on the network parameters themselves, before vs
after — not inferred from the fact that `train()` was called.

**KPI K-5, published two ways so neither can be quoted alone:**

| Window | Global mean | Strict mean (real expert changes only) | Max | Unadapted |
|---|---:|---:|---:|---:|
| 2y | 0.00 bars | 0.00 bars | 0 | 10 / 19 |
| 5y | 0.19 bars | **0.77 bars** | 10 | **26 / 79** |

The global reading counts a switch that needed no new expert as a zero-bar
reaction — true, but flattering. The strict reading is the honest one.

### What this does not show

* **The MoE is still crushed by Buy & Hold** (+6.64% vs +111.76% on 5y). It
  beats the single policy on the two windows tried, but that is **one
  instrument, one window, no multi-seed repetition** — not evidence of an edge,
  and not claimed as one. The demonstrated gain is adaptation *latency*.
* **26 of 79 switches on 5y went unadapted**: the target expert had fewer than
  90 bars of its own regime. Below that floor the replay buffer never reaches
  `min_buffer = 1000`, `learn_step` never runs, and the "fine-tuned" network
  returns bit-identical. These are counted as **failures to react**, not dropped.
* **SB3 policies are refused, not faked.** PPO/A2C/SAC/TD3/TRPO expose
  `predict`, not `q_values`, and have no fine-tune path here, so `moe=true`
  returns **422** with the reason. The six native discrete agents (`dqn`,
  `double_dqn`, `dueling_dqn`, `c51`, `iqn`, `rainbow`) were each verified
  working end-to-end.
* **MAML is not implemented** — as instructed.
* No walk-forward and no multi-seed study for the MoE path.

## 7. Bugs found in my own integration and fixed

* **Self-contradictory audit trace.** `from_expert` was taken from the previous
  regime's *owner* while `expert_changed` compared against the expert *actually
  acting*, producing records reading `bull -> bull, expert_changed=True`. Fixed
  to report the acting policy; pinned by a consistency test.
* **A counter that could lie.** `bars_acted_by` reported the label, so a
  mutation that advanced the label without swapping the policy still claimed an
  expert drove 241 bars while the base agent decided every one. It now counts
  the object actually queried (`acting is not base_agent`).
* **"Superseded" hid the real cause.** Unadapted switches said only
  "superseded"; they now state how many bars of its regime the expert had and
  how many it needed.

## 8. Test evidence

**760 tests pass** (752 before, +8), **ruff clean** on `backend/app backend/tests`.

Six mutations were injected to prove the new tests actually catch regressions:

| # | Mutation | Result |
|---|---|---|
| 1 | Baseline path leaks the MoE block | **caught** (2 tests) |
| 2 | Fine-tune reads bars up to and including the switch (leakage) | **caught** |
| 3 | `adapt=False` silently fine-tunes anyway | **caught** |
| 4 | Experts never actually take control | **survived → test strengthened → caught** |
| 5 | Revert the `from_expert` bug | **caught** |
| 6 | Report a failed fine-tune as a success | **survived → branch unreachable → forced with a stub → caught** |

Mutation 6 is worth noting: it survived not because the test was weak but
because that branch is **never reached on the offline fixtures** — measured,
zero refused fine-tunes across 1y, 2y and 5y. Rather than leave it unverified,
a test now forces the failure with `monkeypatch` and asserts the switch is
recorded as unadapted with no expert taking control.

---

## 9. "This agent was trained without regime awareness" — cause and fix

**The message was not a bug.** `regime_explain.py:79` emits it when the loaded
agent's environment has `regime_aware=False`, which is the honest answer: the
regime genuinely did not enter that agent's decision, and fabricating an
attribution would have been worse.

**The real defect was that the advice could not be followed.** The message told
the user to "retrain with regime_aware enabled", but:

* `RL_REGIME_AWARE` defaults to `False` (`config.py:164`), and
* the RL training form had **no control** for it — verified: `grep regime_aware
  frontend/` returned only two read-only badges on the Training page.

So every agent trained from the UI came out regime-blind, and the panel could
only ever show that message. Measured across the 21 shipped checkpoints:
**17 have `regime_aware=False`**, 3 unset, 1 true.

**Fix.** A "Regime-aware" checkbox in the training row, checked by default, whose
value is forwarded in the `trainRL` / `trainPortfolioRL` payload.

**Verified end to end**, not assumed:

| | Before | After |
|---|---|---|
| Agent | AAPL `dueling_dqn` (regime-blind) | MSFT `dueling_dqn` trained with the box ticked |
| `regime_explanation.available` | `False` | **`True`** |
| Message | "trained without regime awareness…" | *(gone)* |
| Attribution | none | regime **high_volatility**, influence *negligible*, **6 feature contributions** |

The new agent's verdict is *"the High Volatility regime left this decision
unchanged"* — a real measured result, including when the answer is "no effect".

**Note on existing agents.** The 17 regime-blind checkpoints already on disk are
untouched and will keep showing the message, correctly. Only agents trained
after this change carry the regime block. Nothing was retrained automatically:
silently replacing a user's saved agents would discard measured results.

## 10. Chart regression fixed along the way

The legend and Plotly's range buttons were both at `y: 1.12`, both anchored
left, so they were drawn on top of each other. Measured in Chromium: the legend
entry "MoE agent" spanned x **383–435** while "6M"/"All" sat at **386–434**.
Anchoring the buttons right (`x: 1, xanchor: 'right'`) moved them to
**933–1015**. Pre-existing bug, visible on every chart using the shared time
range — not introduced by the MoE work.
