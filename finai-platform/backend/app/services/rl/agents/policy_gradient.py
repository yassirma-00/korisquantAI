"""Policy-gradient / actor-critic agents (PPO, A2C, SAC, TD3).

Stable-Baselines3 is used when installed. A compact native PPO implementation
is provided as a fallback so the platform keeps working in a slim deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from app.core.logging import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
    SB3_MODELS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3,
                  "ddpg": DDPG, "sb3_dqn": DQN}
except Exception:  # pragma: no cover
    SB3_AVAILABLE = False
    SB3_MODELS = {}

try:  # pragma: no cover - QR-DQN and TRPO live in sb3-contrib
    from sb3_contrib import QRDQN, TRPO

    SB3_CONTRIB_AVAILABLE = True
    SB3_MODELS.update({"qr_dqn": QRDQN, "trpo": TRPO})
except Exception:  # pragma: no cover
    SB3_CONTRIB_AVAILABLE = False

DISCRETE_ALGOS = {"ppo", "a2c", "dqn"}
CONTINUOUS_ALGOS = {"ppo", "a2c", "sac", "td3"}


@dataclass
class PGConfig:
    algo: str = "ppo"
    total_timesteps: int = 30_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    n_steps: int = 512          # PPO/A2C rollout length
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    seed: int = 42
    device: str = "cpu"
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [128, 128]})
    # Off-policy settings (SAC / TD3 / DDPG / QR-DQN). These were literals
    # inside the SB3 kwargs block, so no profile could reach them.
    buffer_size: int = 50_000
    learning_starts: int = 500
    tau: float = 0.005
    target_update: int = 250            # QR-DQN target sync interval
    exploration_fraction: float = 0.3   # QR-DQN epsilon schedule
    exploration_final_eps: float = 0.05


# ---------------------------------------------------------------- SB3 path
class SB3Agent:
    """Thin wrapper giving SB3 models the same interface as ``DQNAgent``."""

    def __init__(self, env, config: PGConfig | None = None) -> None:
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 is not installed")
        self.cfg = config or PGConfig()
        algo = self.cfg.algo.lower()
        if algo not in SB3_MODELS:
            raise ValueError(f"Unsupported SB3 algorithm '{algo}'. Available: {sorted(SB3_MODELS)}")

        self.env = DummyVecEnv([lambda: Monitor(env)])
        cls = SB3_MODELS[algo]
        kwargs: dict = {
            "policy": "MlpPolicy", "env": self.env, "learning_rate": self.cfg.learning_rate,
            "gamma": self.cfg.gamma, "seed": self.cfg.seed, "device": self.cfg.device,
            "verbose": 0, "policy_kwargs": self.cfg.policy_kwargs,
        }
        if algo in ("ppo", "a2c"):
            kwargs["n_steps"] = self.cfg.n_steps
            if algo == "ppo":
                kwargs.update({"batch_size": self.cfg.batch_size, "n_epochs": self.cfg.n_epochs,
                               "gae_lambda": self.cfg.gae_lambda, "clip_range": self.cfg.clip_range,
                               "ent_coef": self.cfg.ent_coef, "vf_coef": self.cfg.vf_coef})
            else:
                kwargs.update({"gae_lambda": self.cfg.gae_lambda, "ent_coef": self.cfg.ent_coef,
                               "vf_coef": self.cfg.vf_coef})
        elif algo == "trpo":
            kwargs["n_steps"] = self.cfg.n_steps
            kwargs["gae_lambda"] = self.cfg.gae_lambda
            kwargs.pop("ent_coef", None)
        else:  # SAC / TD3 / DDPG / QR-DQN - off-policy, replay-based
            kwargs.update({"batch_size": self.cfg.batch_size,
                           "buffer_size": self.cfg.buffer_size,
                           "learning_starts": self.cfg.learning_starts})
            if algo == "qr_dqn":
                # discrete off-policy: exploration is epsilon-greedy, not action noise
                kwargs.update({"exploration_fraction": self.cfg.exploration_fraction,
                               "exploration_final_eps": self.cfg.exploration_final_eps,
                               "target_update_interval": self.cfg.target_update})
            else:
                # Polyak coefficient for the target networks. QR-DQN uses a hard
                # sync instead, and SB3 rejects `tau` for it.
                kwargs["tau"] = self.cfg.tau
        self.model = cls(**kwargs)
        self.algo = algo

    def train(self, episodes: int | None = None, total_timesteps: int | None = None,
              monitor=None) -> dict:
        steps = total_timesteps or self.cfg.total_timesteps
        # SB3 owns its loop, so the monitor is inverted into a callback. It
        # counts completed episodes from `dones`, not timesteps, so eval_freq
        # keeps meaning "every N episodes" here as it does natively.
        callback = None
        if monitor is not None and monitor.active:
            from app.services.rl.monitor import make_sb3_callback
            callback = make_sb3_callback(monitor, self, episodes or 0)
        self.model.learn(total_timesteps=steps, progress_bar=False, callback=callback)
        rewards = list(self.env.envs[0].get_episode_rewards())
        lengths = list(self.env.envs[0].get_episode_lengths())
        return {"episode_rewards": [round(float(r), 3) for r in rewards],
                "episode_lengths": lengths, "timesteps": steps, "algo": self.algo}

    def act(self, obs: np.ndarray, deterministic: bool = True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def evaluate(self, env, deterministic: bool = True) -> dict:
        obs, _ = env.reset()
        done, actions, equity = False, [], []
        discrete = hasattr(env.action_space, "n")
        while not done:
            action = self.act(obs, deterministic=deterministic)
            step_action = int(action) if discrete else np.asarray(action).ravel()
            obs, _, terminated, truncated, info = env.step(step_action)
            done = terminated or truncated
            equity.append(info["portfolio_value"])
            entry = {"portfolio_value": round(info["portfolio_value"], 2)}
            if discrete:
                entry.update({"date": str(env.raw.index[min(env.t, len(env.raw) - 1)].date()),
                              "action": info["action"], "price": round(info["price"], 4)})
            else:
                entry.update({"date": str(env.prices_df.index[min(env.t, len(env.prices_df) - 1)].date()),
                              "weights": info["weights"]})
            actions.append(entry)
        return {"performance": env.performance(), "equity_curve": equity,
                "actions": actions, "trades": getattr(env, "trades", [])}

    def save(self, path) -> None:
        self.model.save(str(Path(path).with_suffix("")))

    @classmethod
    def load(cls, path, env, algo: str = "ppo", device: str = "cpu") -> SB3Agent:
        if not SB3_AVAILABLE:
            raise RuntimeError("stable-baselines3 is not installed")
        obj = cls.__new__(cls)
        obj.cfg = PGConfig(algo=algo, device=device)
        obj.env = DummyVecEnv([lambda: Monitor(env)])
        obj.model = SB3_MODELS[algo.lower()].load(str(Path(path).with_suffix("")), env=obj.env, device=device)
        obj.algo = algo.lower()
        return obj


# ------------------------------------------------------- native PPO fallback
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, discrete: bool, hidden: int = 128) -> None:
        super().__init__()
        self.discrete = discrete
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
        if not discrete:
            self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def distribution(self, obs: torch.Tensor):
        h = self.body(obs)
        logits = self.actor(h)
        if self.discrete:
            return Categorical(logits=logits), self.critic(h).squeeze(-1)
        return Normal(torch.tanh(logits), self.log_std.exp()), self.critic(h).squeeze(-1)


class NativePPOAgent:
    """Minimal PPO used when Stable-Baselines3 is unavailable."""

    def __init__(self, env, config: PGConfig | None = None) -> None:
        self.cfg = config or PGConfig()
        self.env = env
        self.discrete = hasattr(env.action_space, "n")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.device = torch.device(self.cfg.device)
        torch.manual_seed(self.cfg.seed)
        self.net = ActorCritic(obs_dim, action_dim, self.discrete).to(self.device)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)

    def act(self, obs: np.ndarray, deterministic: bool = True):
        with torch.no_grad():
            dist, _ = self.net.distribution(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            action = dist.mode if deterministic and hasattr(dist, "mode") else dist.sample()
        action = action.cpu().numpy().ravel()
        return int(action[0]) if self.discrete else action

    def train(self, episodes: int | None = None, total_timesteps: int | None = None,
              monitor=None) -> dict:
        steps_target = total_timesteps or self.cfg.total_timesteps
        obs, _ = self.env.reset()
        rewards_hist, collected = [], 0
        ep_reward = 0.0
        # Episodes, not timesteps: `eval_freq` must mean the same thing here as
        # in the native loops, or "every 5" would fire hundreds of times per
        # episode on this path.
        episode_count = 0

        while collected < steps_target:
            buf_obs, buf_act, buf_logp, buf_rew, buf_val, buf_done = [], [], [], [], [], []
            for _ in range(self.cfg.n_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    dist, value = self.net.distribution(obs_t)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                    if not self.discrete:
                        logp = logp.sum(-1)
                act_np = action.cpu().numpy().ravel()
                step_action = int(act_np[0]) if self.discrete else act_np
                next_obs, reward, terminated, truncated, _ = self.env.step(step_action)

                buf_obs.append(obs)
                buf_act.append(act_np)
                buf_logp.append(float(logp))
                buf_rew.append(reward)
                buf_val.append(float(value))
                buf_done.append(terminated or truncated)
                ep_reward += reward
                obs = next_obs
                collected += 1
                if terminated or truncated:
                    rewards_hist.append(round(ep_reward, 3))
                    ep_reward = 0.0
                    obs, _ = self.env.reset()
                    episode_count += 1
                    if monitor is not None:
                        monitor.on_episode_end(episode_count, episodes or 0, self)

            with torch.no_grad():
                _, last_value = self.net.distribution(
                    torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
            advantages, gae = np.zeros(len(buf_rew), dtype=np.float32), 0.0
            values = buf_val + [float(last_value)]
            for t in reversed(range(len(buf_rew))):
                mask = 0.0 if buf_done[t] else 1.0
                delta = buf_rew[t] + self.cfg.gamma * values[t + 1] * mask - values[t]
                gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
                advantages[t] = gae
            returns = advantages + np.array(buf_val, dtype=np.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_t = torch.as_tensor(np.array(buf_obs), dtype=torch.float32, device=self.device)
            act_t = torch.as_tensor(np.array(buf_act), device=self.device)
            if self.discrete:
                act_t = act_t.squeeze(-1).long()
            old_logp = torch.as_tensor(buf_logp, dtype=torch.float32, device=self.device)
            adv_t = torch.as_tensor(advantages, device=self.device)
            ret_t = torch.as_tensor(returns, device=self.device)

            for _ in range(self.cfg.n_epochs):
                idx = torch.randperm(len(obs_t), device=self.device)
                for start in range(0, len(idx), self.cfg.batch_size):
                    b = idx[start: start + self.cfg.batch_size]
                    dist, value = self.net.distribution(obs_t[b])
                    logp = dist.log_prob(act_t[b])
                    if not self.discrete:
                        logp = logp.sum(-1)
                    ratio = (logp - old_logp[b]).exp()
                    surr1 = ratio * adv_t[b]
                    surr2 = torch.clamp(ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range) * adv_t[b]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = ((value - ret_t[b]) ** 2).mean()
                    entropy = dist.entropy().mean()
                    loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy
                    self.optimiser.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.optimiser.step()

        return {"episode_rewards": rewards_hist, "timesteps": collected, "algo": "ppo_native"}

    def evaluate(self, env, deterministic: bool = True) -> dict:
        obs, _ = env.reset()
        done, actions, equity = False, [], []
        while not done:
            action = self.act(obs, deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            equity.append(info["portfolio_value"])
            actions.append({"portfolio_value": round(info["portfolio_value"], 2),
                            **({"action": info["action"]} if "action" in info else {"weights": info.get("weights")})})
        return {"performance": env.performance(), "equity_curve": equity,
                "actions": actions, "trades": getattr(env, "trades", [])}

    def save(self, path) -> None:
        torch.save({"state_dict": self.net.state_dict(), "config": self.cfg.__dict__}, path)
