"""Native DQN family implementation (PyTorch).

Supports the three variants required by the specification through flags:

* ``DQN``          -- vanilla deep Q-network with target network + replay
* ``Double DQN``   -- decouples action selection from evaluation (``double=True``)
* ``Dueling DQN``  -- separate value/advantage streams (``dueling=True``)

Kept dependency-free (no Stable-Baselines3) so the discrete-action agent always
works, even in a minimal deployment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DQNConfig:
    hidden: tuple[int, ...] = (128, 128)
    gamma: float = 0.99
    lr: float = 5e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer: int = 1_000
    target_update: int = 250          # gradient steps between hard target syncs
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8_000
    double: bool = True
    dueling: bool = True
    grad_clip: float = 10.0
    seed: int = 42
    device: str = "cpu"
    train_freq: int = 1
    history: dict = field(default_factory=dict)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...], dueling: bool) -> None:
        super().__init__()
        self.dueling = dueling
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.body = nn.Sequential(*layers)
        if dueling:
            self.value = nn.Sequential(nn.Linear(last, last // 2), nn.ReLU(), nn.Linear(last // 2, 1))
            self.advantage = nn.Sequential(nn.Linear(last, last // 2), nn.ReLU(), nn.Linear(last // 2, n_actions))
        else:
            self.q = nn.Linear(last, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        if not self.dueling:
            return self.q(h)
        v, a = self.value(h), self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.idx
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch_size: int):
        n = len(self)
        idx = np.random.randint(0, n, size=min(batch_size, n))
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx])


class DQNAgent:
    """Deep Q-Network agent with Double and Dueling extensions."""

    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig | None = None) -> None:
        self.cfg = config or DQNConfig()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(self.cfg.device)

        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        self.online = QNetwork(obs_dim, n_actions, self.cfg.hidden, self.cfg.dueling).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, self.cfg.hidden, self.cfg.dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim)
        self.steps = 0
        self.grad_steps = 0

    # ------------------------------------------------------------- policy
    def epsilon(self) -> float:
        frac = min(self.steps / max(self.cfg.epsilon_decay_steps, 1), 1.0)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.online(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            q = self.online(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
        return q.cpu().numpy().ravel()

    # ------------------------------------------------------------ learning
    def learn_step(self) -> float | None:
        if len(self.buffer) < max(self.cfg.min_buffer, self.cfg.batch_size):
            return None
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.cfg.batch_size)
        obs_t = torch.as_tensor(obs, device=self.device)
        next_t = torch.as_tensor(next_obs, device=self.device)
        act_t = torch.as_tensor(actions, device=self.device)
        rew_t = torch.as_tensor(rewards, device=self.device)
        done_t = torch.as_tensor(dones, device=self.device)

        q_sa = self.online(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.cfg.double:
                next_actions = self.online(next_t).argmax(dim=1, keepdim=True)
                next_q = self.target(next_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_t).max(dim=1).values
            target = rew_t + self.cfg.gamma * next_q * (1.0 - done_t)

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimiser.step()

        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def train(self, env, episodes: int = 30, max_steps: int | None = None,
              log_every: int = 5, progress_cb=None, monitor=None) -> dict:
        rewards_hist, value_hist, loss_hist, sharpe_hist = [], [], [], []
        for ep in range(episodes):
            obs, _ = env.reset()
            done, total_reward, ep_losses, step = False, 0.0, [], 0
            while not done:
                action = self.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.buffer.add(obs, action, reward, next_obs, terminated)
                obs = next_obs
                total_reward += reward
                self.steps += 1
                step += 1
                if self.steps % self.cfg.train_freq == 0:
                    loss = self.learn_step()
                    if loss is not None:
                        ep_losses.append(loss)
                if max_steps and step >= max_steps:
                    break

            perf = env.performance()
            rewards_hist.append(round(total_reward, 3))
            value_hist.append(perf["final_value"])
            sharpe_hist.append(perf["sharpe_ratio"])
            loss_hist.append(round(float(np.mean(ep_losses)), 5) if ep_losses else 0.0)

            if progress_cb:
                progress_cb(ep + 1, episodes, perf)
            # Periodic evaluation / checkpointing, driven by eval_freq and
            # checkpoint_interval from configs/. No monitor -> unchanged loop.
            if monitor is not None:
                monitor.on_episode_end(ep + 1, episodes, self)
            if (ep + 1) % log_every == 0 or ep == episodes - 1:
                logger.info("DQN ep %d/%d | reward=%.2f value=%.0f sharpe=%.2f eps=%.3f",
                            ep + 1, episodes, total_reward, perf["final_value"],
                            perf["sharpe_ratio"], self.epsilon())

        self.cfg.history = {"episode_rewards": rewards_hist, "final_values": value_hist,
                            "losses": loss_hist, "sharpe": sharpe_hist}
        return self.cfg.history

    # ---------------------------------------------------------- evaluation
    def evaluate(self, env, deterministic: bool = True) -> dict:
        obs, _ = env.reset()
        done = False
        actions: list[dict] = []
        equity: list[float] = []
        while not done:
            q = self.q_values(obs)
            action = int(np.argmax(q)) if deterministic else self.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            equity.append(info["portfolio_value"])
            actions.append({
                "date": str(env.raw.index[min(env.t, len(env.raw) - 1)].date()),
                "action": info["action"], "price": round(info["price"], 4),
                "portfolio_value": round(info["portfolio_value"], 2),
                "q_values": [round(float(v), 4) for v in q],
            })
        return {"performance": env.performance(), "equity_curve": equity,
                "actions": actions, "trades": env.trades}

    # -------------------------------------------------------- persistence
    def save(self, path) -> None:
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "config": self.cfg.__dict__,
            "obs_dim": self.obs_dim, "n_actions": self.n_actions,
            "steps": self.steps,
        }, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> DQNAgent:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = dict(ckpt["config"])
        cfg_dict.pop("history", None)
        cfg = DQNConfig(**cfg_dict)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], cfg)
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        agent.steps = ckpt.get("steps", 0)
        return agent
