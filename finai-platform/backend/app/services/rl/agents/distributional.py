"""Distributional RL agents: C51, IQN and Rainbow.

Why distributional methods matter in finance
--------------------------------------------
A standard DQN learns E[return]. Two actions with the same expected value can
have wildly different downside: one steady, one with a fat left tail. An
expectation-based agent cannot tell them apart. Distributional agents learn the
whole return distribution, so the platform can show *risk* per action, not just
a scalar score — and can act on quantiles (CVaR) rather than the mean.

Implemented natively because Stable-Baselines3 ships none of these
(sb3-contrib provides only QR-DQN).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.core.logging import get_logger
from app.services.rl.agents.dqn import ReplayBuffer

logger = get_logger(__name__)


# ============================================================ configuration
@dataclass
class DistributionalConfig:
    hidden: tuple[int, ...] = (128, 128)
    gamma: float = 0.99
    lr: float = 5e-5
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer: int = 1_000
    target_update: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8_000
    grad_clip: float = 10.0
    seed: int = 42
    device: str = "cpu"
    train_freq: int = 1
    # C51 / Rainbow
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    # IQN
    n_quantiles: int = 32
    n_quantile_targets: int = 32
    embedding_dim: int = 64
    risk_distortion: str = "neutral"     # neutral | cvar | wang
    cvar_alpha: float = 0.25
    # Rainbow extras
    n_step: int = 3
    per_alpha: float = 0.5
    per_beta: float = 0.4
    noisy: bool = True
    history: dict = field(default_factory=dict)


# ================================================================ NoisyNet
class NoisyLinear(nn.Module):
    """Factorised Gaussian noise layer (Fortunato et al. 2018).

    Replaces ε-greedy with *learned* exploration: the network decides how much
    noise each weight needs, so exploration anneals itself.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)


def _linear(in_f: int, out_f: int, noisy: bool) -> nn.Module:
    return NoisyLinear(in_f, out_f) if noisy else nn.Linear(in_f, out_f)


# ==================================================================== C51
class CategoricalNetwork(nn.Module):
    """Outputs a categorical distribution over `n_atoms` fixed return values."""

    def __init__(self, obs_dim: int, n_actions: int, n_atoms: int,
                 hidden: tuple[int, ...], dueling: bool = True, noisy: bool = False) -> None:
        super().__init__()
        self.n_actions, self.n_atoms, self.dueling = n_actions, n_atoms, dueling
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.body = nn.Sequential(*layers)
        if dueling:
            self.value = nn.Sequential(_linear(last, last // 2, noisy), nn.ReLU(),
                                       _linear(last // 2, n_atoms, noisy))
            self.advantage = nn.Sequential(_linear(last, last // 2, noisy), nn.ReLU(),
                                           _linear(last // 2, n_actions * n_atoms, noisy))
        else:
            self.head = _linear(last, n_actions * n_atoms, noisy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities, shape (batch, n_actions, n_atoms)."""
        h = self.body(x)
        if self.dueling:
            v = self.value(h).view(-1, 1, self.n_atoms)
            a = self.advantage(h).view(-1, self.n_actions, self.n_atoms)
            logits = v + a - a.mean(dim=1, keepdim=True)
        else:
            logits = self.head(h).view(-1, self.n_actions, self.n_atoms)
        return F.log_softmax(logits, dim=2)

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class C51Agent:
    """Categorical DQN. Learns P(return) on a fixed support."""

    algo_name = "c51"

    def __init__(self, obs_dim: int, n_actions: int,
                 config: DistributionalConfig | None = None) -> None:
        self.cfg = config or DistributionalConfig()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(self.cfg.device)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        self.support = torch.linspace(self.cfg.v_min, self.cfg.v_max,
                                      self.cfg.n_atoms, device=self.device)
        self.delta_z = (self.cfg.v_max - self.cfg.v_min) / (self.cfg.n_atoms - 1)

        self.online = CategoricalNetwork(obs_dim, n_actions, self.cfg.n_atoms,
                                         self.cfg.hidden, noisy=self.cfg.noisy).to(self.device)
        self.target = CategoricalNetwork(obs_dim, n_actions, self.cfg.n_atoms,
                                         self.cfg.hidden, noisy=self.cfg.noisy).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=self.cfg.lr, eps=1.5e-4)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim)
        self.steps = 0
        self.grad_steps = 0

    # -------------------------------------------------------------- policy
    def epsilon(self) -> float:
        if self.cfg.noisy:
            return 0.0                      # NoisyNets handle exploration
        frac = min(self.steps / max(self.cfg.epsilon_decay_steps, 1), 1.0)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            probs = self.online(
                torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)).exp()
            q = (probs * self.support).sum(dim=2)
        return q.cpu().numpy().ravel()

    def action_distribution(self, obs: np.ndarray) -> dict:
        """Full return distribution per action - the point of using C51."""
        with torch.no_grad():
            probs = self.online(
                torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            ).exp().squeeze(0).cpu().numpy()
        support = self.support.cpu().numpy()
        out = {}
        for a in range(self.n_actions):
            p = probs[a]
            mean = float((p * support).sum())
            var = float((p * (support - mean) ** 2).sum())
            cdf = np.cumsum(p)
            idx5 = int(np.searchsorted(cdf, 0.05))
            var5 = float(support[min(idx5, len(support) - 1)])
            tail = support[: idx5 + 1]
            tail_p = p[: idx5 + 1]
            cvar5 = float((tail * tail_p).sum() / max(tail_p.sum(), 1e-9)) if tail_p.sum() > 0 else var5
            out[a] = {
                "mean": round(mean, 4), "std": round(float(np.sqrt(max(var, 0))), 4),
                "var_5pct": round(var5, 4), "cvar_5pct": round(cvar5, 4),
                "support": [round(float(s), 3) for s in support[::5]],
                "probabilities": [round(float(x), 5) for x in p[::5]],
            }
        return out

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and not self.cfg.noisy and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_values(obs)))

    # ------------------------------------------------------------ learning
    def _project_distribution(self, next_obs: torch.Tensor, rewards: torch.Tensor,
                              dones: torch.Tensor) -> torch.Tensor:
        """Project the Bellman-updated distribution back onto the fixed support."""
        batch = len(rewards)
        with torch.no_grad():
            next_probs_online = self.online(next_obs).exp()
            next_q = (next_probs_online * self.support).sum(dim=2)
            next_actions = next_q.argmax(dim=1)                       # Double DQN selection
            next_probs = self.target(next_obs).exp()
            next_dist = next_probs[range(batch), next_actions]        # (batch, n_atoms)

            tz = rewards.unsqueeze(1) + self.cfg.gamma * (1 - dones).unsqueeze(1) * self.support
            tz = tz.clamp(self.cfg.v_min, self.cfg.v_max)
            b = (tz - self.cfg.v_min) / self.delta_z
            lower, upper = b.floor().long(), b.ceil().long()
            # Guard the degenerate case where b lands exactly on an integer
            lower[(upper > 0) & (lower == upper)] -= 1
            upper[(lower < self.cfg.n_atoms - 1) & (lower == upper)] += 1

            projected = torch.zeros_like(next_dist)
            offset = (torch.arange(batch, device=self.device).unsqueeze(1)
                      * self.cfg.n_atoms).expand(batch, self.cfg.n_atoms)
            projected.view(-1).index_add_(
                0, (lower + offset).view(-1), (next_dist * (upper.float() - b)).view(-1))
            projected.view(-1).index_add_(
                0, (upper + offset).view(-1), (next_dist * (b - lower.float())).view(-1))
        return projected

    def learn_step(self) -> float | None:
        if len(self.buffer) < max(self.cfg.min_buffer, self.cfg.batch_size):
            return None
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.cfg.batch_size)
        obs_t = torch.as_tensor(obs, device=self.device)
        next_t = torch.as_tensor(next_obs, device=self.device)
        act_t = torch.as_tensor(actions, device=self.device)
        rew_t = torch.as_tensor(rewards, device=self.device)
        done_t = torch.as_tensor(dones, device=self.device)

        target_dist = self._project_distribution(next_t, rew_t, done_t)
        log_probs = self.online(obs_t)[range(len(act_t)), act_t]
        loss = -(target_dist * log_probs).sum(dim=1).mean()

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimiser.step()
        if self.cfg.noisy:
            self.online.reset_noise()
            self.target.reset_noise()

        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def train(self, env, episodes: int = 20, log_every: int = 5, progress_cb=None,
              monitor=None) -> dict:
        rewards_hist, value_hist, loss_hist, sharpe_hist = [], [], [], []
        for ep in range(episodes):
            obs, _ = env.reset()
            done, total, losses = False, 0.0, []
            while not done:
                action = self.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.buffer.add(obs, action, reward, next_obs, terminated)
                obs = next_obs
                total += reward
                self.steps += 1
                if self.steps % self.cfg.train_freq == 0:
                    loss = self.learn_step()
                    if loss is not None:
                        losses.append(loss)
            perf = env.performance()
            rewards_hist.append(round(total, 3))
            value_hist.append(perf["final_value"])
            sharpe_hist.append(perf["sharpe_ratio"])
            loss_hist.append(round(float(np.mean(losses)), 5) if losses else 0.0)
            if progress_cb:
                progress_cb(ep + 1, episodes, perf)
            # Periodic evaluation / checkpointing from configs/.
            if monitor is not None:
                monitor.on_episode_end(ep + 1, episodes, self)
            if (ep + 1) % log_every == 0 or ep == episodes - 1:
                logger.info("%s ep %d/%d | reward=%.2f value=%.0f sharpe=%.2f",
                            self.algo_name.upper(), ep + 1, episodes, total,
                            perf["final_value"], perf["sharpe_ratio"])
        self.cfg.history = {"episode_rewards": rewards_hist, "final_values": value_hist,
                            "losses": loss_hist, "sharpe": sharpe_hist}
        return self.cfg.history

    def evaluate(self, env, deterministic: bool = True) -> dict:
        from app.services.rl.environment import ACTION_NAMES

        self.online.eval()
        obs, _ = env.reset()
        done, actions, equity = False, [], []
        while not done:
            q = self.q_values(obs)
            action = int(np.argmax(q))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            equity.append(info["portfolio_value"])
            actions.append({
                "date": str(env.raw.index[min(env.t, len(env.raw) - 1)].date()),
                "action": info["action"], "price": round(info["price"], 4),
                "portfolio_value": round(info["portfolio_value"], 2),
                "q_values": [round(float(v), 4) for v in q],
            })
        self.online.train()
        return {"performance": env.performance(), "equity_curve": equity,
                "actions": actions, "trades": env.trades,
                "action_names": ACTION_NAMES}

    def save(self, path) -> None:
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "config": self.cfg.__dict__, "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions, "steps": self.steps,
                    "algo": self.algo_name}, path)

    @classmethod
    def load(cls, path, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = dict(ckpt["config"])
        cfg_dict.pop("history", None)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], DistributionalConfig(**cfg_dict))
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        agent.steps = ckpt.get("steps", 0)
        return agent


# ==================================================================== IQN
class ImplicitQuantileNetwork(nn.Module):
    """Maps (state, quantile level τ) -> value. Continuous in τ."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...],
                 embedding_dim: int = 64) -> None:
        super().__init__()
        self.n_actions, self.embedding_dim = n_actions, embedding_dim
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.body = nn.Sequential(*layers)
        self.feature_dim = last
        self.tau_embed = nn.Sequential(nn.Linear(embedding_dim, last), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(last, last // 2), nn.ReLU(),
                                  nn.Linear(last // 2, n_actions))

    def forward(self, x: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """(batch, obs) x (batch, n_tau) -> (batch, n_tau, n_actions)."""
        batch, n_tau = taus.shape
        features = self.body(x)                                    # (batch, feature)
        i = torch.arange(1, self.embedding_dim + 1, device=x.device, dtype=torch.float32)
        cos = torch.cos(taus.unsqueeze(-1) * i * np.pi)            # (batch, n_tau, embed)
        tau_features = self.tau_embed(cos)                         # (batch, n_tau, feature)
        merged = features.unsqueeze(1) * tau_features              # multiplicative interaction
        return self.head(merged)


class IQNAgent(C51Agent):
    """Implicit Quantile Network with optional risk-averse (CVaR) policy."""

    algo_name = "iqn"

    def __init__(self, obs_dim: int, n_actions: int,
                 config: DistributionalConfig | None = None) -> None:
        cfg = config or DistributionalConfig()
        # Deliberately skip C51Agent.__init__ (different network + no fixed support)
        self.cfg = cfg
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(cfg.device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.online = ImplicitQuantileNetwork(obs_dim, n_actions, cfg.hidden,
                                              cfg.embedding_dim).to(self.device)
        self.target = ImplicitQuantileNetwork(obs_dim, n_actions, cfg.hidden,
                                              cfg.embedding_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimiser = torch.optim.Adam(self.online.parameters(), lr=cfg.lr, eps=1e-4)
        self.buffer = ReplayBuffer(cfg.buffer_size, obs_dim)
        self.steps = 0
        self.grad_steps = 0

    def _sample_taus(self, batch: int, n: int) -> torch.Tensor:
        taus = torch.rand(batch, n, device=self.device)
        # Risk distortion: bias sampling toward the lower tail to make the
        # policy explicitly risk-averse (Dabney et al. 2018, section 5).
        if self.cfg.risk_distortion == "cvar":
            taus = taus * self.cfg.cvar_alpha
        elif self.cfg.risk_distortion == "wang":
            normal = torch.distributions.Normal(0.0, 1.0)
            taus = normal.cdf(normal.icdf(taus.clamp(1e-4, 1 - 1e-4)) - 0.75)
        return taus

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            taus = self._sample_taus(1, self.cfg.n_quantiles)
            q = self.online(x, taus).mean(dim=1)
        return q.cpu().numpy().ravel()

    def action_distribution(self, obs: np.ndarray) -> dict:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            taus = torch.linspace(0.01, 0.99, 99, device=self.device).unsqueeze(0)
            quantiles = self.online(x, taus).squeeze(0).cpu().numpy()   # (99, n_actions)
        out = {}
        for a in range(self.n_actions):
            q = np.sort(quantiles[:, a])
            out[a] = {
                "mean": round(float(q.mean()), 4),
                "std": round(float(q.std()), 4),
                "var_5pct": round(float(np.percentile(q, 5)), 4),
                "cvar_5pct": round(float(q[q <= np.percentile(q, 5)].mean()), 4),
                "median": round(float(np.median(q)), 4),
                "quantiles": [round(float(v), 4) for v in q[::10]],
            }
        return out

    def learn_step(self) -> float | None:
        if len(self.buffer) < max(self.cfg.min_buffer, self.cfg.batch_size):
            return None
        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.cfg.batch_size)
        batch = len(actions)
        obs_t = torch.as_tensor(obs, device=self.device)
        next_t = torch.as_tensor(next_obs, device=self.device)
        act_t = torch.as_tensor(actions, device=self.device)
        rew_t = torch.as_tensor(rewards, device=self.device)
        done_t = torch.as_tensor(dones, device=self.device)

        taus = self._sample_taus(batch, self.cfg.n_quantiles)
        current = self.online(obs_t, taus)                                  # (b, nq, a)
        current = current.gather(2, act_t.view(-1, 1, 1).expand(batch, self.cfg.n_quantiles, 1)).squeeze(2)

        with torch.no_grad():
            next_taus = self._sample_taus(batch, self.cfg.n_quantile_targets)
            next_q_online = self.online(next_t, next_taus).mean(dim=1)
            next_actions = next_q_online.argmax(dim=1)                      # Double selection
            next_quant = self.target(next_t, next_taus)
            next_quant = next_quant.gather(
                2, next_actions.view(-1, 1, 1).expand(batch, self.cfg.n_quantile_targets, 1)).squeeze(2)
            target = rew_t.unsqueeze(1) + self.cfg.gamma * (1 - done_t).unsqueeze(1) * next_quant

        # Quantile Huber loss
        td = target.unsqueeze(1) - current.unsqueeze(2)                     # (b, nq, nqt)
        huber = torch.where(td.abs() <= 1.0, 0.5 * td.pow(2), td.abs() - 0.5)
        weight = (taus.unsqueeze(2) - (td.detach() < 0).float()).abs()
        loss = (weight * huber).sum(dim=1).mean()

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimiser.step()
        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def epsilon(self) -> float:
        frac = min(self.steps / max(self.cfg.epsilon_decay_steps, 1), 1.0)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_values(obs)))


# ================================================================ Rainbow
class PrioritisedReplayBuffer(ReplayBuffer):
    """Proportional prioritised replay (Schaul et al. 2016).

    Samples surprising transitions more often, then corrects the resulting bias
    with importance-sampling weights.
    """

    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.5) -> None:
        super().__init__(capacity, obs_dim)
        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, obs, action, reward, next_obs, done) -> None:
        self.priorities[self.idx] = self.max_priority
        super().add(obs, action, reward, next_obs, done)

    def sample_prioritised(self, batch_size: int, beta: float = 0.4):
        n = len(self)
        if n == 0:
            raise ValueError("empty buffer")
        prios = self.priorities[:n] ** self.alpha
        probs = prios / prios.sum()
        idx = np.random.choice(n, size=min(batch_size, n), p=probs)
        weights = (n * probs[idx]) ** (-beta)
        weights = weights / weights.max()
        return (self.obs[idx], self.actions[idx], self.rewards[idx],
                self.next_obs[idx], self.dones[idx], idx, weights.astype(np.float32))

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        prios = np.abs(td_errors) + 1e-5
        self.priorities[idx] = prios
        self.max_priority = max(self.max_priority, float(prios.max()))


class RainbowAgent(C51Agent):
    """C51 + Double + Dueling + Prioritised Replay + n-step + NoisyNets."""

    algo_name = "rainbow"

    def __init__(self, obs_dim: int, n_actions: int,
                 config: DistributionalConfig | None = None) -> None:
        cfg = config or DistributionalConfig(noisy=True, n_step=3)
        cfg.noisy = True
        super().__init__(obs_dim, n_actions, cfg)
        self.buffer = PrioritisedReplayBuffer(cfg.buffer_size, obs_dim, cfg.per_alpha)
        self.n_step_queue: list[tuple] = []

    def _push_n_step(self, obs, action, reward, next_obs, done) -> None:
        """Accumulate an n-step return before writing to the replay buffer."""
        self.n_step_queue.append((obs, action, reward, next_obs, done))
        if len(self.n_step_queue) < self.cfg.n_step and not done:
            return
        r, nxt, d = 0.0, next_obs, done
        for i, (_, _, ri, ni, di) in enumerate(self.n_step_queue):
            r += (self.cfg.gamma ** i) * ri
            nxt, d = ni, di
            if di:
                break
        obs0, act0 = self.n_step_queue[0][0], self.n_step_queue[0][1]
        self.buffer.add(obs0, act0, r, nxt, d)
        self.n_step_queue.pop(0)
        if done:
            self.n_step_queue.clear()

    def learn_step(self) -> float | None:
        if len(self.buffer) < max(self.cfg.min_buffer, self.cfg.batch_size):
            return None
        obs, actions, rewards, next_obs, dones, idx, weights = \
            self.buffer.sample_prioritised(self.cfg.batch_size, self.cfg.per_beta)
        obs_t = torch.as_tensor(obs, device=self.device)
        next_t = torch.as_tensor(next_obs, device=self.device)
        act_t = torch.as_tensor(actions, device=self.device)
        rew_t = torch.as_tensor(rewards, device=self.device)
        done_t = torch.as_tensor(dones, device=self.device)
        w_t = torch.as_tensor(weights, device=self.device)

        # n-step discounting inside the projection
        gamma_n = self.cfg.gamma ** self.cfg.n_step
        with torch.no_grad():
            batch = len(rew_t)
            next_probs_online = self.online(next_t).exp()
            next_actions = (next_probs_online * self.support).sum(dim=2).argmax(dim=1)
            next_dist = self.target(next_t).exp()[range(batch), next_actions]
            tz = (rew_t.unsqueeze(1) + gamma_n * (1 - done_t).unsqueeze(1) * self.support
                  ).clamp(self.cfg.v_min, self.cfg.v_max)
            b = (tz - self.cfg.v_min) / self.delta_z
            lower, upper = b.floor().long(), b.ceil().long()
            lower[(upper > 0) & (lower == upper)] -= 1
            upper[(lower < self.cfg.n_atoms - 1) & (lower == upper)] += 1
            projected = torch.zeros_like(next_dist)
            offset = (torch.arange(batch, device=self.device).unsqueeze(1)
                      * self.cfg.n_atoms).expand(batch, self.cfg.n_atoms)
            projected.view(-1).index_add_(
                0, (lower + offset).view(-1), (next_dist * (upper.float() - b)).view(-1))
            projected.view(-1).index_add_(
                0, (upper + offset).view(-1), (next_dist * (b - lower.float())).view(-1))

        log_probs = self.online(obs_t)[range(len(act_t)), act_t]
        per_sample = -(projected * log_probs).sum(dim=1)
        loss = (w_t * per_sample).mean()

        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.cfg.grad_clip)
        self.optimiser.step()
        self.buffer.update_priorities(idx, per_sample.detach().cpu().numpy())
        self.online.reset_noise()
        self.target.reset_noise()

        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach())

    def train(self, env, episodes: int = 20, log_every: int = 5, progress_cb=None,
              monitor=None) -> dict:
        rewards_hist, value_hist, loss_hist, sharpe_hist = [], [], [], []
        for ep in range(episodes):
            obs, _ = env.reset()
            self.n_step_queue.clear()
            done, total, losses = False, 0.0, []
            while not done:
                action = self.act(obs)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self._push_n_step(obs, action, reward, next_obs, terminated)
                obs = next_obs
                total += reward
                self.steps += 1
                if self.steps % self.cfg.train_freq == 0:
                    loss = self.learn_step()
                    if loss is not None:
                        losses.append(loss)
            perf = env.performance()
            rewards_hist.append(round(total, 3))
            value_hist.append(perf["final_value"])
            sharpe_hist.append(perf["sharpe_ratio"])
            loss_hist.append(round(float(np.mean(losses)), 5) if losses else 0.0)
            if progress_cb:
                progress_cb(ep + 1, episodes, perf)
            # Periodic evaluation / checkpointing from configs/.
            if monitor is not None:
                monitor.on_episode_end(ep + 1, episodes, self)
            if (ep + 1) % log_every == 0 or ep == episodes - 1:
                logger.info("RAINBOW ep %d/%d | reward=%.2f value=%.0f sharpe=%.2f",
                            ep + 1, episodes, total, perf["final_value"], perf["sharpe_ratio"])
        self.cfg.history = {"episode_rewards": rewards_hist, "final_values": value_hist,
                            "losses": loss_hist, "sharpe": sharpe_hist}
        return self.cfg.history


DISTRIBUTIONAL_AGENTS = {"c51": C51Agent, "iqn": IQNAgent, "rainbow": RainbowAgent}
