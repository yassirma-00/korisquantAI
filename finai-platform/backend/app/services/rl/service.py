"""High-level Reinforcement-Learning service.

Responsibilities
----------------
* build train/test environments with a strict chronological split
* train any supported algorithm (DQN family natively, PPO/A2C/SAC/TD3 via SB3)
* persist / reload agents
* produce an out-of-sample backtest and a live action recommendation
* compare an agent against Buy&Hold and other baselines
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config import settings
from app.core.exceptions import InvalidRequestError, ModelNotTrainedError
from app.core.logging import get_logger
from app.services.data.market_data import market_data_service
from app.services.rl.agents.distributional import (
    DISTRIBUTIONAL_AGENTS,
    DistributionalConfig,
)
from app.services.rl.agents.dqn import DQNAgent, DQNConfig
from app.services.rl.agents.policy_gradient import (
    SB3_AVAILABLE,
    NativePPOAgent,
    PGConfig,
    SB3Agent,
)
from app.services.rl.catalogue import BY_KEY, get_algorithm
from app.services.rl.environment import ACTION_NAMES, EnvConfig, PortfolioEnv, TradingEnv

logger = get_logger(__name__)

# Discrete agents implemented natively in this repo (no SB3 dependency)
NATIVE_DQN = {"dqn", "double_dqn", "dueling_dqn"}
NATIVE_DISTRIBUTIONAL = set(DISTRIBUTIONAL_AGENTS)          # c51, iqn, rainbow
NATIVE_DISCRETE = NATIVE_DQN | NATIVE_DISTRIBUTIONAL

SUPPORTED_ALGOS = [a.key for a in BY_KEY.values()]
DISCRETE_ONLY = {k for k, a in BY_KEY.items() if a.action_space == "discrete"}
CONTINUOUS_ONLY = {k for k, a in BY_KEY.items() if a.action_space == "continuous"}


class RLService:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = Path(model_dir or settings.MODEL_DIR) / "rl"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- paths
    def _safe(self, name: str) -> str:
        """Filesystem-safe stem for a symbol or basket key.

        The dot matters as much as the slash. European tickers carry one
        (`MC.PA`, `AIR.PA`, `SAN.PA`) and every save site calls
        `.with_suffix(".pt")` on the result — which treats `.PA_dqn__regime`
        as an existing extension and *replaces* it, so `rl_MC.PA_dqn__regime`
        collapsed to `rl_MC.pt`. Measured: the six algorithms trained for
        `MC.PA` all wrote to that one file, and four of them then failed to
        load with `DQNConfig.__init__() got an unexpected keyword argument
        'n_atoms'` — a C51 checkpoint being read back as a DQN.
        """
        return (name.upper().replace("/", "_").replace("=", "_")
                .replace("^", "idx_").replace(",", "-").replace(".", "_"))

    def agent_path(self, symbol: str, algo: str, variant: str = "") -> Path:
        return self.model_dir / f"rl_{self._safe(symbol)}_{algo}{self._suffix(variant)}"

    def meta_path(self, symbol: str, algo: str, variant: str = "") -> Path:
        return self.model_dir / f"rl_{self._safe(symbol)}_{algo}{self._suffix(variant)}.json"

    @staticmethod
    def _suffix(variant: str) -> str:
        """Filename tail for a named variant of the same symbol+algo pair.

        Empty by default, so every existing checkpoint keeps the exact name it
        already has on disk — renaming them would orphan 21 trained agents and
        silently change what the dashboard lists.

        The only current use is `regime` : a regime-aware twin trained with the
        same period, episodes and profile as its baseline, kept side by side so
        the two can be compared instead of one replacing the other.
        """
        v = (variant or "").strip().lower()
        if not v:
            return ""
        safe = "".join(c for c in v if c.isalnum() or c == "_")
        if not safe:
            raise InvalidRequestError(f"invalid agent variant: {variant!r}")
        return f"__{safe}"

    @staticmethod
    def _has_leaky_split(meta: dict) -> bool:
        """True when stored metadata came from the old overlapping split.

        Agents trained before the split fix recorded a test window that begins
        before training ended, so their metrics are inflated. Flag them instead
        of replaying those numbers as if they were valid.
        """
        train_window = meta.get("train_window")
        test_window = meta.get("test_window")
        if not train_window or not test_window:
            return False
        return str(test_window[0]) <= str(train_window[1])

    def list_agents(self) -> list[dict]:
        out = []
        for meta in sorted(self.model_dir.glob("*.json")):
            try:
                payload = json.loads(meta.read_text())
            except Exception:
                continue
            if self._has_leaky_split(payload):
                payload["stale"] = True
                payload["stale_reason"] = (
                    "Trained with an overlapping train/test split; its reported "
                    "performance is inflated. Retrain before relying on it.")
            out.append(payload)
        return out

    # ------------------------------------------------------------ env build
    def _split(self, df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Strictly chronological, non-overlapping train/test split.

        An earlier version handed the test set 60 bars of *training* data
        (``df.iloc[split - 60:]``) to warm up indicators. That leaked: those
        bars were both fitted on and scored, inflating out-of-sample results by
        making ~28% of the test window already-seen data.

        The warm-up is unnecessary anyway — ``TradingEnv`` starts trading at
        ``t = lookback``, so the first bars of the test set are consumed as
        context and never scored. The split is now clean.
        """
        if len(df) < 200:
            raise InvalidRequestError(f"Need at least 200 bars for RL, got {len(df)}")
        split = int(len(df) * (1 - test_fraction))
        train_df, test_df = df.iloc[:split], df.iloc[split:]
        if not train_df.index.intersection(test_df.index).empty:  # pragma: no cover
            raise RuntimeError("train/test overlap detected - this must never happen")
        return train_df, test_df

    def _env_config(self, overrides: dict | None = None) -> EnvConfig:
        cfg = EnvConfig(
            initial_balance=settings.RL_INITIAL_BALANCE,
            transaction_cost=settings.RL_TRANSACTION_COST,
            regime_aware=settings.RL_REGIME_AWARE,
        )
        for key, value in (overrides or {}).items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)
        return cfg

    # ------------------------------------------------- hyperparameter wiring
    @staticmethod
    def _resolve_hyperparams(algo: str, profile: str, hyperparams: dict | None,
                             env_overrides: dict | None,
                             episodes: int | None = None,
                             total_timesteps: int | None = None,
                             test_fraction: float | None = None):
        """Merge configs/ with anything the caller passed explicitly.

        Explicit arguments outrank the YAML so every existing call site keeps
        its exact behaviour — this migration must not silently retune agents
        that someone is already training.
        """
        from app.services.rl.hyperparams import hyperparameters

        overrides: dict = dict(hyperparams or {})
        if episodes is not None:
            overrides.setdefault("training", {})
            overrides["training"] = {**overrides.get("training", {}),
                                     "episodes": episodes}
        if total_timesteps is not None:
            overrides["training"] = {**overrides.get("training", {}),
                                     "total_timesteps": total_timesteps}
        if test_fraction is not None:
            overrides["training"] = {**overrides.get("training", {}),
                                     "test_fraction": test_fraction}

        # `env_overrides` is the legacy channel and still the one the API uses.
        # Map its keys onto the config sections that now own them, so a request
        # that sets transaction_cost keeps working unchanged.
        env_map = {
            "initial_balance": "environment", "transaction_cost": "environment",
            "slippage": "environment", "lookback": "environment",
            "trade_fraction": "environment", "reward_scaling": "environment",
            "allow_short": "environment",
            "risk_penalty": "risk", "drawdown_penalty": "risk",
            "turnover_penalty": "risk", "cvar_penalty": "risk",
            "cvar_alpha": "risk", "regime_aware": "risk",
            "regime_step": "risk", "regime_window": "risk",
            "regime_reward_weight": "risk",
        }
        for key, value in (env_overrides or {}).items():
            section = env_map.get(key)
            if section and value is not None:
                overrides[section] = {**overrides.get(section, {}), key: value}

        return hyperparameters.resolve(algo, profile, overrides)

    @staticmethod
    def _env_config_from(cfg) -> EnvConfig:
        """Build the environment from the resolved config."""
        env = cfg.section("environment")
        risk = cfg.section("risk")
        return EnvConfig(
            initial_balance=float(env.get("initial_balance", settings.RL_INITIAL_BALANCE)),
            transaction_cost=float(env.get("transaction_cost", settings.RL_TRANSACTION_COST)),
            slippage=float(env.get("slippage", 0.0005)),
            lookback=int(env.get("lookback", 20)),
            trade_fraction=float(env.get("trade_fraction", 0.25)),
            reward_scaling=float(env.get("reward_scaling", 100.0)),
            allow_short=bool(env.get("allow_short", False)),
            risk_penalty=float(risk.get("risk_penalty", 0.15)),
            drawdown_penalty=float(risk.get("drawdown_penalty", 0.35)),
            turnover_penalty=float(risk.get("turnover_penalty", 0.02)),
            cvar_penalty=float(risk.get("cvar_penalty", 0.10)),
            cvar_alpha=float(risk.get("cvar_alpha", 0.05)),
            regime_aware=bool(risk.get("regime_aware", settings.RL_REGIME_AWARE)),
            regime_step=int(risk.get("regime_step", 5)),
            regime_window=int(risk.get("regime_window", 252)),
            regime_reward_weight=float(risk.get("regime_reward_weight", 1.0)),
        )

    @staticmethod
    def _dqn_config_from(cfg, decay_steps: int) -> DQNConfig:
        opt, net, rep, exp = (cfg.section("optimizer"), cfg.section("network"),
                              cfg.section("replay"), cfg.section("exploration"))
        return DQNConfig(
            hidden=tuple(net.get("hidden", (128, 128))),
            gamma=float(opt.get("gamma", 0.99)),
            lr=float(opt.get("learning_rate", 5e-4)),
            batch_size=int(opt.get("batch_size", 64)),
            grad_clip=float(opt.get("grad_clip", 10.0)),
            buffer_size=int(rep.get("buffer_size", 50_000)),
            min_buffer=int(rep.get("min_buffer", 1_000)),
            target_update=int(rep.get("target_update", 250)),
            train_freq=int(rep.get("train_freq", 1)),
            epsilon_start=float(exp.get("epsilon_start", 1.0)),
            epsilon_end=float(exp.get("epsilon_end", 0.05)),
            epsilon_decay_steps=decay_steps,
            double=bool(net.get("double", True)),
            dueling=bool(net.get("dueling", True)),
            seed=int(cfg.get("training.seed", 42)),
            device=str(cfg.get("training.device", "cpu")),
        )

    @staticmethod
    def _distributional_config_from(cfg, decay_steps: int) -> DistributionalConfig:
        opt, net, rep, exp = (cfg.section("optimizer"), cfg.section("network"),
                              cfg.section("replay"), cfg.section("exploration"))
        dist = cfg.section("distributional")
        return DistributionalConfig(
            hidden=tuple(net.get("hidden", (128, 128))),
            gamma=float(opt.get("gamma", 0.99)),
            lr=float(opt.get("learning_rate", 5e-5)),
            batch_size=int(opt.get("batch_size", 64)),
            grad_clip=float(opt.get("grad_clip", 10.0)),
            buffer_size=int(rep.get("buffer_size", 50_000)),
            min_buffer=int(rep.get("min_buffer", 1_000)),
            target_update=int(rep.get("target_update", 250)),
            train_freq=int(rep.get("train_freq", 1)),
            epsilon_start=float(exp.get("epsilon_start", 1.0)),
            epsilon_end=float(exp.get("epsilon_end", 0.05)),
            epsilon_decay_steps=decay_steps,
            n_atoms=int(dist.get("n_atoms", 51)),
            v_min=float(dist.get("v_min", -10.0)),
            v_max=float(dist.get("v_max", 10.0)),
            n_quantiles=int(dist.get("n_quantiles", 32)),
            n_quantile_targets=int(dist.get("n_quantile_targets", 32)),
            embedding_dim=int(dist.get("embedding_dim", 64)),
            risk_distortion=str(dist.get("risk_distortion", "neutral")),
            cvar_alpha=float(dist.get("cvar_alpha", 0.25)),
            n_step=int(dist.get("n_step", 3)),
            per_alpha=float(dist.get("per_alpha", 0.5)),
            per_beta=float(dist.get("per_beta", 0.4)),
            noisy=bool(dist.get("noisy", True)),
            seed=int(cfg.get("training.seed", 42)),
            device=str(cfg.get("training.device", "cpu")),
        )

    @staticmethod
    def _pg_config_from(cfg, algo: str, total_timesteps: int) -> PGConfig:
        opt, net = cfg.section("optimizer"), cfg.section("network")
        pg, rep = cfg.section("policy_gradient"), cfg.section("replay")
        off = cfg.section("off_policy")
        return PGConfig(
            algo=algo,
            total_timesteps=int(total_timesteps),
            learning_rate=float(opt.get("learning_rate", 3e-4)),
            gamma=float(opt.get("gamma", 0.99)),
            batch_size=int(opt.get("batch_size", 64)),
            n_steps=int(pg.get("n_steps") or 512),
            n_epochs=int(pg.get("n_epochs") or 10),
            gae_lambda=float(pg.get("gae_lambda") or 0.95),
            clip_range=float(pg.get("clip_range") or 0.2),
            ent_coef=float(pg.get("ent_coef") or 0.0),
            vf_coef=float(pg.get("vf_coef") or 0.5),
            buffer_size=int(rep.get("buffer_size", 50_000)),
            learning_starts=int(rep.get("learning_starts", 500)),
            target_update=int(rep.get("target_update", 250)),
            exploration_fraction=float(cfg.get("exploration.exploration_fraction", 0.3)),
            exploration_final_eps=float(cfg.get("exploration.epsilon_end", 0.05)),
            tau=float(off.get("tau", 0.005)),
            seed=int(cfg.get("training.seed", 42)),
            device=str(cfg.get("training.device", "cpu")),
            policy_kwargs={"net_arch": list(net.get("hidden", [128, 128]))},
        )

    @staticmethod
    def _ohlcv_for(symbols: list[str], period: str) -> dict:
        """True OHLCV per symbol, for the regime detector.

        Fails soft per symbol: one unreachable instrument must not abort a
        whole training run, and `PortfolioRegimeProvider` synthesises a frame
        from close for anything missing — recording that it did so.
        """
        frames: dict = {}
        for sym in symbols:
            try:
                frames[sym.upper()] = market_data_service.get_history(
                    sym, period=period).df
            except Exception as exc:      # pragma: no cover - provider dependent
                logger.info("OHLCV unavailable for %s, will synthesise: %s", sym, exc)
        return frames

    def _env_config_for_agent(self, symbol: str, algo: str,
                              overrides: dict | None = None,
                              variant: str = "") -> EnvConfig:
        """Rebuild the environment a saved agent was actually trained in.

        Inference has to reproduce the *training* observation width, not
        today's default. A regime-aware agent handed a 36-wide vector — or a
        legacy agent handed a 42-wide one — fails with

            mat1 and mat2 shapes cannot be multiplied (1x42 and 36x128)

        so the flag is read back from the checkpoint's metadata. Agents saved
        before this feature existed have no `regime_aware` key and correctly
        default to False.
        """
        cfg = self._env_config(overrides)
        meta_path = self.meta_path(symbol, algo, variant)
        if meta_path.exists():
            try:
                saved = json.loads(meta_path.read_text()).get("env_config") or {}
            except Exception:      # pragma: no cover - unreadable metadata
                saved = {}
            for key in ("regime_aware", "regime_step", "regime_window",
                        "lookback", "feature_columns"):
                if key in saved and saved[key] is not None:
                    setattr(cfg, key, saved[key])
        return cfg

    # ----------------------------------------------------------- training
    def train_single_asset(
        self,
        symbol: str,
        period: str = "3y",
        algo: str = "dueling_dqn",
        episodes: int | None = None,
        total_timesteps: int | None = None,
        env_overrides: dict | None = None,
        test_fraction: float | None = None,
        profile: str = "default",
        hyperparams: dict | None = None,
        variant: str = "",
    ) -> dict:
        algo = algo.lower().strip()
        # Every training parameter now comes from configs/, not from literals
        # here. Explicit arguments still win so existing callers are unaffected.
        cfg = self._resolve_hyperparams(algo, profile, hyperparams, env_overrides,
                                        episodes, total_timesteps, test_fraction)
        episodes = int(cfg.get("training.episodes"))
        test_fraction = float(cfg.get("training.test_fraction"))
        from app.services.rl.hyperparams import hyperparameters
        experiment_id = hyperparameters.experiment_id()
        spec = get_algorithm(algo)
        if spec is None:
            raise InvalidRequestError(f"Unsupported algo '{algo}'. Supported: {sorted(SUPPORTED_ALGOS)}")
        if algo in CONTINUOUS_ONLY:
            # SAC/TD3/DDPG emit a weight vector, not BUY/HOLD/SELL. Rather than
            # refuse them outright, run them on a single-asset PortfolioEnv:
            # the agent then chooses a target exposure in [0, 1] for this one
            # instrument, which is a perfectly meaningful decision and is later
            # mapped onto a discrete signal for the recommendation engine.
            steps = cfg.get("training.total_timesteps") or max(
                episodes * int(cfg.get("training.continuous_timesteps_per_episode")),
                int(cfg.get("training.continuous_timesteps_floor")))
            return self._train_continuous_single_asset(
                symbol, period, algo, int(steps), env_overrides, test_fraction,
                cfg=cfg, variant=variant)
        if not spec.available:
            raise InvalidRequestError(
                f"'{spec.name}' requires the '{spec.backend}' backend, which is not installed.",
                details={"algorithm": algo, "install": {
                    "sb3": "pip install stable-baselines3",
                    "sb3_contrib": "pip install sb3-contrib",
                    "rllib": "pip install 'ray[rllib]'",
                }.get(spec.backend)},
            )

        series = market_data_service.get_history(symbol, period=period)
        train_df, test_df = self._split(series.df, test_fraction)
        env_cfg = self._env_config_from(cfg)
        train_env = TradingEnv(train_df, env_cfg)
        test_env = TradingEnv(test_df, env_cfg)
        obs_dim = train_env.observation_space.shape[0]
        n_actions = train_env.action_space.n

        # Exploration decay is scaled to the data, floored by the config: a
        # fixed step count would anneal far too fast on a long series and never
        # finish annealing on a short one.
        decay_steps = max(len(train_df) * max(episodes // 3, 1),
                          int(cfg.get("exploration.epsilon_decay_floor", 2000)))

        # eval_freq / checkpoint_interval come from configs/. They evaluate on
        # `test_env` — the held-out window — so the curve shows generalisation
        # rather than how well the agent memorised its training slice.
        from app.services.rl.monitor import make_monitor

        monitor = make_monitor(
            cfg, eval_env=test_env,
            checkpoint_dir=self.model_dir / "checkpoints",
            run_id=f"{self._safe(symbol)}_{algo}_{experiment_id}")

        if algo in NATIVE_DQN:
            agent = DQNAgent(obs_dim, n_actions,
                             self._dqn_config_from(cfg, decay_steps))
            history = agent.train(train_env, episodes=episodes, monitor=monitor)
            agent.save(self.agent_path(symbol, algo, variant).with_suffix(".pt"))
        elif algo in NATIVE_DISTRIBUTIONAL:
            agent = DISTRIBUTIONAL_AGENTS[algo](
                obs_dim, n_actions,
                self._distributional_config_from(cfg, decay_steps))
            history = agent.train(train_env, episodes=episodes, monitor=monitor)
            agent.save(self.agent_path(symbol, algo, variant).with_suffix(".pt"))
        else:
            steps = cfg.get("training.total_timesteps") or max(
                len(train_df) * episodes,
                int(cfg.get("training.timesteps_per_episode_floor", 10_000)))
            pg_cfg = self._pg_config_from(cfg, algo, int(steps))
            agent = SB3Agent(train_env, pg_cfg) if SB3_AVAILABLE else NativePPOAgent(train_env, pg_cfg)
            history = agent.train(episodes=episodes,
                                  total_timesteps=pg_cfg.total_timesteps,
                                  monitor=monitor)
            agent.save(self.agent_path(symbol, algo, variant))

        evaluation = agent.evaluate(test_env, deterministic=True)
        baselines = self._baselines(test_df, env_cfg)

        meta = {
            "symbol": symbol.upper(),
            "algo": algo,
            "period": period,
            "data_source": series.source,
            "episodes": episodes,
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "train_window": [str(train_df.index[0].date()), str(train_df.index[-1].date())],
            "test_window": [str(test_df.index[0].date()), str(test_df.index[-1].date())],
            "env_config": asdict(env_cfg),
            "training_history": history,
            "test_performance": evaluation["performance"],
            "baselines": baselines,
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "agent_path": str(self.agent_path(symbol, algo, variant)),
            "sb3": SB3_AVAILABLE and algo not in DISCRETE_ONLY,
            "variant": variant or None,
            "regime_aware": bool(env_cfg.regime_aware),
            # ---- reproducibility record -------------------------------
            # The *fully materialised* parameter set, not a diff against the
            # YAML: the files can change afterwards and this run must still be
            # replayable from its own metadata. The fingerprint identifies the
            # configuration, the experiment id identifies this particular run.
            "experiment_id": experiment_id,
            "profile": cfg.profile,
            "hyperparameters": cfg.params,
            "hyperparameter_fingerprint": cfg.fingerprint,
            "config_sources": cfg.sources,
            "seed": int(cfg.get("training.seed", 42)),
            # What the periodic evaluation / checkpointing actually did. An
            # empty record when both are 0 is the honest report that they were
            # switched off, not that they ran and found nothing.
            "monitoring": monitor.summary(),
            # Which regimes the agent actually met while training. A policy
            # that never saw a crash cannot be trusted to handle one, and this
            # is what lets the UI say so instead of implying broad competence.
            "regime_exposure": (train_env.regime.summary()
                                if getattr(train_env, "regime", None) else None),
        }
        self.meta_path(symbol, algo, variant).write_text(json.dumps(meta, indent=2, default=str))
        logger.info("RL %s/%s trained | test return %.2f%% vs B&H %.2f%%", symbol, algo,
                    evaluation["performance"]["total_return"] * 100,
                    evaluation["performance"]["buy_and_hold_return"] * 100)
        return meta

    def train_portfolio(
        self,
        symbols: list[str],
        period: str = "3y",
        algo: str = "ppo",
        total_timesteps: int | None = None,
        env_overrides: dict | None = None,
        test_fraction: float | None = None,
        profile: str = "default",
        hyperparams: dict | None = None,
        variant: str = "",
    ) -> dict:
        algo = algo.lower().strip()
        cfg = self._resolve_hyperparams(algo, profile, hyperparams, env_overrides,
                                        None, total_timesteps, test_fraction)
        total_timesteps = int(cfg.get("training.total_timesteps")
                              or cfg.get("training.timesteps_per_episode_floor", 25_000))
        test_fraction = float(cfg.get("training.test_fraction"))
        from app.services.rl.hyperparams import hyperparameters
        experiment_id = hyperparameters.experiment_id()
        if algo in DISCRETE_ONLY:
            raise InvalidRequestError(f"'{algo}' is discrete-only -> use /rl/train for a single asset")
        if not SB3_AVAILABLE and algo != "ppo":
            raise InvalidRequestError(f"'{algo}' requires stable-baselines3; only native PPO is available")

        matrix = market_data_service.get_price_matrix(symbols, period=period)
        split = int(len(matrix) * (1 - test_fraction))
        # Non-overlapping, for the same reason as _split(): PortfolioEnv also
        # begins at t = lookback, so it warms up on its own test data.
        train_m, test_m = matrix.iloc[:split], matrix.iloc[split:]
        env_cfg = self._env_config_from(cfg)
        # The price matrix is close-only, but the regime detector reads OHLCV
        # (ADX needs high/low). Measured, close-only disagreed with the true
        # verdict once in eight symbols and reported far lower confidence, so
        # the real frames are passed down rather than synthesised.
        ohlcv = self._ohlcv_for(symbols, period) if env_cfg.regime_aware else None
        train_env = PortfolioEnv(train_m, env_cfg, ohlcv=ohlcv)
        test_env = PortfolioEnv(test_m, env_cfg, ohlcv=ohlcv)

        from app.services.rl.monitor import make_monitor

        key = ",".join(sorted(s.upper() for s in symbols))
        monitor = make_monitor(
            cfg, eval_env=test_env,
            checkpoint_dir=self.model_dir / "checkpoints",
            run_id=f"{self._safe(key)}_{algo}_{experiment_id}")

        pg_cfg = self._pg_config_from(cfg, algo, total_timesteps)
        agent = SB3Agent(train_env, pg_cfg) if SB3_AVAILABLE else NativePPOAgent(train_env, pg_cfg)
        # Episode count is unknown up front on a timestep-driven run; the
        # callback counts them from `dones` and only uses this for the
        # is-last-episode check, which a 0 correctly never satisfies.
        history = agent.train(episodes=0, total_timesteps=total_timesteps,
                              monitor=monitor)
        agent.save(self.agent_path(key, algo, variant))
        evaluation = agent.evaluate(test_env, deterministic=True)

        meta = {
            "symbols": list(matrix.columns),
            "portfolio_key": key,
            "algo": algo,
            "period": period,
            "total_timesteps": total_timesteps,
            "train_bars": len(train_m),
            "test_bars": len(test_m),
            "env_config": asdict(env_cfg),
            "training_history": history,
            "test_performance": evaluation["performance"],
            "weight_path": evaluation["actions"][-5:],
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "variant": variant or None,
            "regime_aware": bool(env_cfg.regime_aware),
            "experiment_id": experiment_id,
            "profile": cfg.profile,
            "hyperparameters": cfg.params,
            "hyperparameter_fingerprint": cfg.fingerprint,
            "config_sources": cfg.sources,
            "seed": int(cfg.get("training.seed", 42)),
            "monitoring": monitor.summary(),
            # Per-asset regime exposure during training, plus whether any
            # asset's OHLC had to be synthesised from close.
            "regime_exposure": (train_env.regime.summary()
                                if getattr(train_env, "regime", None) else None),
            "agent_path": str(self.agent_path(key, algo, variant)),
        }
        self.meta_path(key, algo, variant).write_text(json.dumps(meta, indent=2, default=str))
        return meta

    # ------------------------------------------------------------ baselines
    def _baselines(self, df: pd.DataFrame, env_cfg: EnvConfig) -> dict:
        """Buy&Hold, always-cash and a simple SMA-crossover reference strategy."""
        prices = df["close"].values
        start = env_cfg.lookback
        buy_hold = float(prices[-1] / prices[start] - 1)

        from app.services.indicators.technical import sma
        fast, slow = sma(df["close"], 20), sma(df["close"], 50)
        env = TradingEnv(df, env_cfg)
        obs, _ = env.reset()
        done = False
        while not done:
            i = env.t
            f, s = fast.iloc[i] if i < len(fast) else np.nan, slow.iloc[i] if i < len(slow) else np.nan
            action = 1
            if np.isfinite(f) and np.isfinite(s):
                action = 2 if f > s else 0
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        sma_perf = env.performance()

        return {
            "buy_and_hold": {"total_return": round(buy_hold, 4)},
            "sma_crossover": {"total_return": sma_perf["total_return"],
                              "sharpe_ratio": sma_perf["sharpe_ratio"],
                              "max_drawdown": sma_perf["max_drawdown"],
                              "n_trades": sma_perf["n_trades"]},
            "cash": {"total_return": 0.0},
        }

    # ------------------------------------- continuous agents, single asset
    def _train_continuous_single_asset(
        self, symbol: str, period: str, algo: str, total_timesteps: int,
        env_overrides: dict | None, test_fraction: float, cfg=None,
        variant: str = "",
    ) -> dict:
        """Train SAC/TD3/DDPG on one instrument as a 1-asset allocation problem.

        The action is a target weight in [0, 1] for the instrument (the rest is
        cash). That is a genuine decision - "hold 70% exposure" - and it lets the
        whole continuous family appear alongside the discrete ones instead of
        being restricted to multi-asset baskets.
        """
        series = market_data_service.get_history(symbol, period=period)
        matrix = pd.DataFrame({symbol.upper(): series.df["close"]}).dropna()
        split = int(len(matrix) * (1 - test_fraction))
        train_m, test_m = matrix.iloc[:split], matrix.iloc[split:]
        if train_m.index.intersection(test_m.index).size:  # pragma: no cover
            raise RuntimeError("train/test overlap detected")

        from app.services.rl.hyperparams import hyperparameters
        if cfg is None:
            cfg = self._resolve_hyperparams(algo, "default", None, env_overrides)
        env_cfg = self._env_config_from(cfg)
        ohlcv = {symbol.upper(): series.df} if env_cfg.regime_aware else None
        train_env = PortfolioEnv(train_m, env_cfg, ohlcv=ohlcv)
        test_env = PortfolioEnv(test_m, env_cfg, ohlcv=ohlcv)

        from app.services.rl.monitor import make_monitor

        monitor = make_monitor(
            cfg, eval_env=test_env,
            checkpoint_dir=self.model_dir / "checkpoints",
            run_id=f"{self._safe(symbol)}_{algo}_cont")

        pg_cfg = self._pg_config_from(cfg, algo, total_timesteps)
        agent = SB3Agent(train_env, pg_cfg) if SB3_AVAILABLE else NativePPOAgent(train_env, pg_cfg)
        history = agent.train(episodes=0, total_timesteps=total_timesteps,
                              monitor=monitor)
        agent.save(self.agent_path(symbol, algo, variant))
        evaluation = agent.evaluate(test_env, deterministic=True)

        perf = evaluation["performance"]
        # Report against the same baselines the discrete agents use so the two
        # families stay directly comparable.
        #
        # The start bar matters. `PortfolioEnv` cannot act until it has a full
        # observation window, so the agent's first possible trade is at
        # `env_cfg.lookback`. Measuring Buy & Hold from row 0 credited the
        # benchmark with `lookback` bars the agent was never allowed to trade:
        # on AAPL/2y that read +22.72% against the discrete path's +16.45% for
        # the same window, a 6.3-point gap flowing straight into
        # `alpha_vs_buy_hold` and making discrete agents look better than
        # continuous ones on identical data.
        bh_start = min(env_cfg.lookback, len(test_m) - 1)
        buy_hold = float(test_m.iloc[-1, 0] / test_m.iloc[bh_start, 0] - 1)
        perf["buy_and_hold_return"] = round(buy_hold, 4)
        perf["alpha_vs_buy_hold"] = round(perf["total_return"] - buy_hold, 4)

        spec = get_algorithm(algo)
        meta = {
            "symbol": symbol.upper(), "algo": algo,
            "algorithm_name": spec.name if spec else algo.upper(),
            "period": period, "data_source": series.source,
            "action_space": "continuous", "mode": "single_asset_allocation",
            "total_timesteps": total_timesteps,
            "train_bars": len(train_m), "test_bars": len(test_m),
            "train_window": [str(train_m.index[0].date()), str(train_m.index[-1].date())],
            "test_window": [str(test_m.index[0].date()), str(test_m.index[-1].date())],
            "env_config": asdict(env_cfg),
            "training_history": history,
            "test_performance": perf,
            "baselines": {"buy_and_hold": {"total_return": round(buy_hold, 4)},
                          "cash": {"total_return": 0.0}},
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "agent_path": str(self.agent_path(symbol, algo, variant)),
            "regime_aware": bool(env_cfg.regime_aware),
            # Same reproducibility record as the discrete path. Omitting it
            # here would make SAC/TD3/DDPG runs unreplayable purely because
            # they take a different branch.
            "experiment_id": hyperparameters.experiment_id(),
            "profile": cfg.profile,
            "hyperparameters": cfg.params,
            "hyperparameter_fingerprint": cfg.fingerprint,
            "config_sources": cfg.sources,
            "seed": int(cfg.get("training.seed", 42)),
            "monitoring": monitor.summary(),
        }
        self.meta_path(symbol, algo, variant).write_text(json.dumps(meta, indent=2, default=str))
        logger.info("RL %s/%s (continuous, 1 asset) | test return %.2f%%",
                    symbol, algo, perf["total_return"] * 100)
        return meta

    # ------------------------------------------------------------- loading
    def load_agent(self, symbol: str, algo: str, env, variant: str = ""):
        algo = algo.lower()
        if algo in NATIVE_DQN:
            path = self.agent_path(symbol, algo, variant).with_suffix(".pt")
            if not path.exists():
                raise ModelNotTrainedError(f"No trained {algo} agent for {symbol}")
            return DQNAgent.load(path)
        if algo in NATIVE_DISTRIBUTIONAL:
            path = self.agent_path(symbol, algo, variant).with_suffix(".pt")
            if not path.exists():
                raise ModelNotTrainedError(f"No trained {algo} agent for {symbol}")
            return DISTRIBUTIONAL_AGENTS[algo].load(path)
        path = self.agent_path(symbol, algo, variant).with_suffix(".zip")
        if not path.exists():
            raise ModelNotTrainedError(f"No trained {algo} agent for {symbol}")
        if not SB3_AVAILABLE:
            raise ModelNotTrainedError("stable-baselines3 required to reload this agent")
        return SB3Agent.load(self.agent_path(symbol, algo, variant), env, algo=algo)

    # --------------------------------------------------------- inference
    def recommend_action(self, symbol: str, algo: str = "dueling_dqn", period: str = "1y",
                         variant: str = "") -> dict:
        algo = algo.lower().strip()
        if algo in CONTINUOUS_ONLY:
            return self._recommend_continuous(symbol, algo, period)

        series = market_data_service.get_history(symbol, period=period)
        env_cfg = self._env_config_for_agent(symbol, algo, None, variant)
        env = TradingEnv(series.df, env_cfg)
        agent = self.load_agent(symbol, algo, env, variant)

        obs, _ = env.reset()
        env.t = len(env.prices) - 1        # jump to the most recent bar
        obs = env._observation()

        distribution = None
        if hasattr(agent, "action_distribution"):
            # Distributional agents expose per-action risk, not just a mean
            q = agent.q_values(obs)
            action = int(np.argmax(q))
            exp = np.exp(q - q.max())
            probs = exp / exp.sum()
            confidence = float(probs[action])
            q_map = {ACTION_NAMES[i]: round(float(v), 4) for i, v in enumerate(q)}
            raw_dist = agent.action_distribution(obs)
            distribution = {ACTION_NAMES[i]: v for i, v in raw_dist.items() if i in ACTION_NAMES}
        elif isinstance(agent, DQNAgent):
            q = agent.q_values(obs)
            action = int(np.argmax(q))
            exp = np.exp(q - q.max())
            probs = exp / exp.sum()
            confidence = float(probs[action])
            q_map = {ACTION_NAMES[i]: round(float(v), 4) for i, v in enumerate(q)}
        else:
            action = int(agent.act(obs, deterministic=True))
            confidence, q_map = 0.65, {}

        meta_file = self.meta_path(symbol, algo, variant)
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        spec = get_algorithm(algo)
        action_name = ACTION_NAMES.get(action, str(action))
        last_price = float(series.df["close"].iloc[-1])

        trade_plan = self._trade_plan(series.df, action_name, confidence,
                                      meta.get("test_performance", {}), distribution)

        # How the detected regime bore on this specific decision, measured by
        # counterfactual rather than asserted by template.
        from app.services.rl.regime_explain import explain_regime_influence
        regime_explanation = explain_regime_influence(
            agent, env, obs, action, ACTION_NAMES)

        from app.services.rl.audit import model_version
        version = model_version(meta.get("agent_path"), meta.get("trained_at"))

        return {
            "symbol": symbol.upper(),
            "algo": algo,
            "algorithm_name": spec.name if spec else algo.upper(),
            "algorithm_family": spec.family if spec else None,
            "action": action_name,
            "action_id": action,
            "confidence": round(confidence, 4),
            "regime_explanation": regime_explanation,
            "regime_aware": bool(env_cfg.regime_aware),
            "model_version": version,
            "regime_exposure": meta.get("regime_exposure"),
            "q_values": q_map,
            "return_distribution": distribution,
            "as_of": str(series.df.index[-1].date()),
            "last_price": round(last_price, 4),
            "agent_test_performance": meta.get("test_performance", {}),
            "baselines": meta.get("baselines", {}),
            "trained_at": meta.get("trained_at"),
            **trade_plan,
        }

    # ------------------------------------------------------- decision detail
    def _trade_plan(self, df, action: str, confidence: float,
                    perf: dict, distribution: dict | None) -> dict:
        """Turn a bare BUY/HOLD/SELL into an actionable, explained trade plan.

        Stop-loss and take-profit are derived from ATR rather than fixed
        percentages: a 2% stop is noise on a crypto pair and a catastrophe on a
        currency pair. ATR adapts the levels to each instrument's own volatility.
        """
        from app.services.indicators.technical import atr as atr_fn

        close = df["close"]
        last_price = float(close.iloc[-1])
        atr_value = float(atr_fn(df, 14).iloc[-1])
        atr_pct = atr_value / last_price * 100 if last_price else 2.0
        returns = close.pct_change().dropna()
        ann_vol = float(returns.tail(63).std() * np.sqrt(252)) if len(returns) > 63 else 0.25

        # Risk assessment blends realised volatility with the agent's own
        # out-of-sample drawdown - a strategy can be risky even on a calm asset.
        agent_dd = abs(perf.get("max_drawdown", 0.0) or 0.0)
        risk_score = float(np.clip(0.55 * min(ann_vol / 0.6, 1.0) + 0.45 * min(agent_dd / 0.35, 1.0), 0, 1))
        risk_level = ("low" if risk_score < 0.3 else "moderate" if risk_score < 0.55
                      else "high" if risk_score < 0.78 else "critical")

        # Volatility-targeted sizing, cut by conviction and risk
        target_vol = 0.15
        base_weight = min(target_vol / max(ann_vol, 0.05), 1.0)
        haircut = {"low": 1.0, "moderate": 0.75, "high": 0.45, "critical": 0.2}[risk_level]
        weight = float(np.clip(base_weight * confidence * haircut, 0.0, 0.35))

        if action == "BUY":
            stop_pct = min(atr_pct * 2.0, 15.0)
            take_pct = min(atr_pct * 3.5, 30.0)
            plan = {
                "position_size_pct": round(weight * 100, 2),
                "stop_loss_price": round(last_price * (1 - stop_pct / 100), 4),
                "stop_loss_pct": round(-stop_pct, 2),
                "take_profit_price": round(last_price * (1 + take_pct / 100), 4),
                "take_profit_pct": round(take_pct, 2),
                "risk_reward_ratio": round(take_pct / stop_pct, 2) if stop_pct else None,
            }
        elif action == "SELL":
            plan = {
                "position_size_pct": 0.0,
                "reduce_existing_pct": round(min(confidence * 100, 100), 1),
                "stop_loss_price": None, "stop_loss_pct": None,
                "take_profit_price": None, "take_profit_pct": None,
                "risk_reward_ratio": None,
            }
        else:  # HOLD
            plan = {
                "position_size_pct": 0.0,
                "stop_loss_price": round(last_price * (1 - min(atr_pct * 2.5, 18.0) / 100), 4),
                "stop_loss_pct": round(-min(atr_pct * 2.5, 18.0), 2),
                "take_profit_price": None, "take_profit_pct": None,
                "risk_reward_ratio": None,
            }

        # Horizon scales with volatility: fast-moving assets invalidate a thesis sooner
        horizon_days = int(np.clip(round(21 * (0.25 / max(ann_vol, 0.05))), 3, 90))

        return {
            "risk": {
                "score": round(risk_score, 3), "level": risk_level,
                "annualised_volatility": round(ann_vol, 4),
                "atr_pct": round(atr_pct, 2),
                "agent_max_drawdown": round(-agent_dd, 4),
            },
            "trade_plan": plan,
            "investment_horizon": {
                "days": horizon_days,
                "label": ("short-term (days)" if horizon_days <= 10 else
                          "medium-term (weeks)" if horizon_days <= 45 else "long-term (months)"),
                "rationale": (f"Scaled to {ann_vol:.0%} annualised volatility: higher volatility "
                              "invalidates a thesis faster, so the horizon shortens."),
            },
            "explanation": self._explain_decision(action, confidence, risk_level, atr_pct,
                                                  ann_vol, perf, distribution),
        }

    @staticmethod
    def _explain_decision(action: str, confidence: float, risk_level: str, atr_pct: float,
                          ann_vol: float, perf: dict, distribution: dict | None) -> dict:
        drivers: list[str] = []
        if distribution and action in distribution:
            d = distribution[action]
            drivers.append(
                f"The distributional critic puts the expected risk-adjusted value of {action} at "
                f"{d['mean']:+.3f} with a 5% worst-case (CVaR) of {d['cvar_5pct']:+.3f}.")
        drivers.append(
            f"Realised volatility is {ann_vol:.1%} annualised (ATR {atr_pct:.2f}% of price), "
            f"placing this trade in the {risk_level} risk band.")
        if perf:
            drivers.append(
                f"On unseen data the agent returned {perf.get('total_return', 0):.2%} versus "
                f"{perf.get('buy_and_hold_return', 0):.2%} for buy-and-hold "
                f"(Sharpe {perf.get('sharpe_ratio', 0):.2f}, max drawdown "
                f"{perf.get('max_drawdown', 0):.2%}).")
        verdict = {
            "BUY": "The agent sees positive expected risk-adjusted reward from opening or adding to a position.",
            "SELL": "The agent expects negative risk-adjusted reward; this is an instruction to reduce, not to short.",
            "HOLD": "No action has a clear edge; the agent prefers to wait rather than pay transaction costs.",
        }[action]
        caveat = ("Confidence is low - the agent is close to indifferent between actions."
                  if confidence < 0.45 else
                  "Confidence is moderate." if confidence < 0.65 else
                  "The agent strongly prefers this action over the alternatives.")
        return {
            "summary": f"{verdict} {caveat}",
            "drivers": drivers,
            "disclaimer": ("A reinforcement-learning policy optimises a historical reward signal. "
                           "It cannot anticipate regime changes it has never seen."),
        }

    def _recommend_continuous(self, symbol: str, algo: str, period: str) -> dict:
        """Turn a continuous agent's target weight into a BUY/HOLD/SELL signal.

        The agent outputs a desired exposure w in [0, 1]. Mapping it onto a
        discrete action needs a reference point: neutral is taken as 50%, i.e.
        half invested. Well above that is an accumulate signal, well below it is
        a reduce signal, and the band in between is HOLD so small oscillations
        in the weight do not churn the recommendation.
        """
        series = market_data_service.get_history(symbol, period=period)
        matrix = pd.DataFrame({symbol.upper(): series.df["close"]}).dropna()
        env_cfg = self._env_config_for_agent(symbol, algo)
        ohlcv = {symbol.upper(): series.df} if env_cfg.regime_aware else None
        env = PortfolioEnv(matrix, env_cfg, ohlcv=ohlcv)
        agent = self.load_agent(symbol, algo, env)

        obs, _ = env.reset()
        env.t = len(env.prices) - 1
        obs = env._observation()
        raw = agent.act(obs, deterministic=True)
        exp = np.exp(np.asarray(raw, dtype=float) - np.max(raw))
        weights = exp / exp.sum()
        target = float(weights[0])                 # exposure to the instrument

        NEUTRAL, BAND = 0.5, 0.15
        if target >= NEUTRAL + BAND:
            action = "BUY"
        elif target <= NEUTRAL - BAND:
            action = "SELL"
        else:
            action = "HOLD"
        # Confidence grows with distance from neutral, saturating at the edges
        confidence = float(np.clip(abs(target - NEUTRAL) / NEUTRAL, 0.0, 1.0))

        meta_file = self.meta_path(symbol, algo)
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        spec = get_algorithm(algo)
        last_price = float(series.df["close"].iloc[-1])
        perf = meta.get("test_performance", {})
        trade_plan = self._trade_plan(series.df, action, confidence, perf, None)

        return {
            "symbol": symbol.upper(), "algo": algo,
            "algorithm_name": spec.name if spec else algo.upper(),
            "algorithm_family": spec.family if spec else None,
            "action": action,
            "action_id": {"SELL": 0, "HOLD": 1, "BUY": 2}[action],
            "confidence": round(confidence, 4),
            "q_values": {},
            "target_exposure": round(target, 4),
            "cash_weight": round(float(weights[-1]), 4),
            "action_space": "continuous",
            "mapping_note": (
                f"The agent targets {target:.0%} exposure. Above "
                f"{NEUTRAL + BAND:.0%} reads as BUY, below {NEUTRAL - BAND:.0%} as SELL; "
                "in between the position is left unchanged."),
            "return_distribution": None,
            "as_of": str(series.df.index[-1].date()),
            "last_price": round(last_price, 4),
            "agent_test_performance": perf,
            "baselines": meta.get("baselines", {}),
            "trained_at": meta.get("trained_at"),
            **trade_plan,
        }

    def recommend_allocation(self, symbols: list[str], algo: str = "sac",
                             period: str = "1y", variant: str = "") -> dict:
        """Current target weights from a trained multi-asset agent.

        `/rl/portfolio/train` could train a basket agent but nothing could ever
        ask it what to hold — five trained baskets sat on disk with no way to
        query them. This closes that loop and attributes the regime's influence
        on the weights per asset.
        """
        key = ",".join(sorted(s.upper().strip() for s in symbols))
        ordered = key.split(",")

        matrix = market_data_service.get_price_matrix(ordered, period=period)
        if matrix.empty or len(matrix.columns) < 2:
            raise InvalidRequestError(
                "A multi-asset allocation needs at least two instruments with "
                "overlapping history.")

        env_cfg = self._env_config_for_agent(key, algo, None, variant)
        ohlcv = self._ohlcv_for(list(matrix.columns), period) if env_cfg.regime_aware else None
        env = PortfolioEnv(matrix, env_cfg, ohlcv=ohlcv)
        agent = self.load_agent(key, algo, env, variant)

        env.reset()
        env.t = len(env.prices) - 1
        obs = env._observation()
        raw = np.asarray(agent.act(obs, deterministic=True), dtype=float).ravel()
        exp = np.exp(raw - raw.max())
        weights = exp / exp.sum()

        from app.services.rl.allocation_explain import explain_allocation_influence
        regime_explanation = explain_allocation_influence(agent, env, obs, weights)

        meta_file = self.meta_path(key, algo, variant)
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        spec = get_algorithm(algo)

        from app.services.rl.audit import model_version
        version = model_version(meta.get("agent_path"), meta.get("trained_at"))

        assets = list(matrix.columns)
        allocation = [
            {"symbol": sym,
             "weight": round(float(w), 4),
             "last_price": round(float(matrix[sym].iloc[-1]), 4)}
            for sym, w in zip(assets, weights[:len(assets)], strict=False)
        ]
        allocation.sort(key=lambda a: a["weight"], reverse=True)

        return {
            "symbols": assets,
            "portfolio_key": key,
            "algo": algo,
            "algorithm_name": spec.name if spec else algo.upper(),
            "algorithm_family": spec.family if spec else None,
            "action_space": "continuous",
            "allocation": allocation,
            "cash_weight": round(float(weights[-1]), 4),
            # Concentration is the number a risk function asks for first.
            "largest_position": allocation[0] if allocation else None,
            "regime_aware": bool(env_cfg.regime_aware),
            "regime_explanation": regime_explanation,
            "model_version": version,
            "regime_exposure": meta.get("regime_exposure"),
            "as_of": str(matrix.index[-1].date()),
            "agent_test_performance": meta.get("test_performance", {}),
            "trained_at": meta.get("trained_at"),
        }

    def backtest(self, symbol: str, algo: str = "dueling_dqn", period: str = "1y",
                 env_overrides: dict | None = None, variant: str = "") -> dict:
        series = market_data_service.get_history(symbol, period=period)
        env_cfg = self._env_config_for_agent(symbol, algo, env_overrides, variant)
        env = TradingEnv(series.df, env_cfg)
        agent = self.load_agent(symbol, algo, env, variant)
        result = agent.evaluate(env, deterministic=True)
        dates = [str(d.date()) for d in series.df.index[env_cfg.lookback: env_cfg.lookback + len(result["equity_curve"])]]
        return {
            "symbol": symbol.upper(), "algo": algo, "period": period,
            "performance": result["performance"],
            "baselines": self._baselines(series.df, env_cfg),
            "equity_curve": [{"date": d, "value": round(float(v), 2)}
                             for d, v in zip(dates, result["equity_curve"], strict=False)],
            "trades": result["trades"][:200],
            "n_actions": len(result["actions"]),
        }


rl_service = RLService()
