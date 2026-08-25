"""Centralised hyperparameter management for the RL module.

Why this exists
---------------
Every training parameter used to be a literal somewhere in the source: the
learning rate in ``DQNConfig``, the replay buffer size in an SB3 kwargs dict,
the episode count in a function signature, ``v_min=-10.0`` passed inline at the
call site. Changing any of them meant editing Python, and a completed run
recorded only *some* of them — so a result could not be reproduced from its own
metadata.

This module makes the configuration data rather than code:

* ``configs/defaults.yaml`` — shared baseline.
* ``configs/algorithms/<algo>.yaml`` — only what that algorithm does differently.
* ``configs/profiles/<name>.yaml`` — user-editable presets.

Resolution is a deep merge, lowest priority first::

    defaults -> algorithm -> profile -> explicit overrides

**Deep**, not shallow: a profile that sets ``risk.risk_penalty`` must not wipe
out the rest of the ``risk`` block. A shallow ``dict.update`` would do exactly
that, silently reverting `cvar_penalty` and `regime_aware` to nothing.

Two properties matter for reproducibility:

* ``resolve()`` returns the **fully materialised** parameter set, not a diff.
  A run records what it actually used, so it can be replayed even if the YAML
  files change afterwards.
* ``fingerprint()`` hashes that set. Two runs sharing a fingerprint used
  identical hyperparameters — which a version string someone has to remember to
  bump cannot promise.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from app.core.config import PROJECT_ROOT
from app.core.exceptions import InvalidRequestError
from app.core.logging import get_logger

logger = get_logger(__name__)

CONFIG_DIR = PROJECT_ROOT / "configs"
ALGO_DIR = CONFIG_DIR / "algorithms"
PROFILE_DIR = CONFIG_DIR / "profiles"

# Sections a resolved config always carries, so a caller can index them without
# defensive `.get(..., {})` at every use site.
SECTIONS = (
    "meta", "training", "optimizer", "network", "replay", "exploration",
    "environment", "risk", "evaluation", "policy_gradient", "off_policy",
    "distributional",
)

# Profile names are used as filenames. Anything outside this alphabet could
# escape the profiles directory (`../../etc/passwd`) or collide on a
# case-insensitive filesystem.
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,48}$")

# Bounds enforced on user-supplied values. A learning rate of 50 does not fail
# loudly — it silently produces a useless agent, hours later.
BOUNDS: dict[str, tuple[float, float]] = {
    "optimizer.learning_rate": (1e-7, 1.0),
    "optimizer.gamma": (0.0, 1.0),
    "optimizer.batch_size": (1, 4096),
    "optimizer.grad_clip": (0.0, 1e4),
    "training.episodes": (1, 1000),
    "training.total_timesteps": (100, 10_000_000),
    "training.test_fraction": (0.05, 0.9),
    "training.seed": (0, 2**31 - 1),
    "replay.buffer_size": (100, 5_000_000),
    "replay.min_buffer": (1, 1_000_000),
    "replay.train_freq": (1, 1000),
    "replay.target_update": (1, 100_000),
    "exploration.epsilon_start": (0.0, 1.0),
    "exploration.epsilon_end": (0.0, 1.0),
    "exploration.epsilon_decay_steps": (1, 10_000_000),
    "environment.initial_balance": (100.0, 1e12),
    "environment.transaction_cost": (0.0, 0.05),
    "environment.slippage": (0.0, 0.05),
    "environment.lookback": (2, 512),
    "environment.trade_fraction": (0.01, 1.0),
    "environment.reward_scaling": (0.01, 1e6),
    "risk.risk_penalty": (0.0, 5.0),
    "risk.drawdown_penalty": (0.0, 5.0),
    "risk.turnover_penalty": (0.0, 5.0),
    "risk.cvar_penalty": (0.0, 5.0),
    "risk.cvar_alpha": (0.001, 0.5),
    "risk.regime_reward_weight": (0.0, 5.0),
    "risk.regime_step": (1, 63),
    "risk.regime_window": (60, 2520),
    "policy_gradient.n_steps": (8, 100_000),
    "policy_gradient.n_epochs": (1, 100),
    "policy_gradient.gae_lambda": (0.0, 1.0),
    "policy_gradient.clip_range": (0.0, 1.0),
    "policy_gradient.ent_coef": (0.0, 1.0),
    "policy_gradient.vf_coef": (0.0, 10.0),
    "distributional.n_atoms": (2, 501),
    "distributional.n_quantiles": (2, 512),
    "distributional.cvar_alpha": (0.01, 1.0),
}


@dataclass(frozen=True)
class ResolvedConfig:
    """A fully materialised hyperparameter set plus its provenance."""

    algo: str
    profile: str
    params: dict
    sources: list[str]
    fingerprint: str

    def section(self, name: str) -> dict:
        return dict(self.params.get(name) or {})

    def get(self, path: str, default: Any = None) -> Any:
        """Read a dotted path, e.g. ``optimizer.learning_rate``."""
        node: Any = self.params
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def to_dict(self) -> dict:
        return {
            "algo": self.algo,
            "profile": self.profile,
            "params": copy.deepcopy(self.params),
            "sources": list(self.sources),
            "fingerprint": self.fingerprint,
        }


# ------------------------------------------------------------------ helpers
def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` onto ``base`` without mutating either.

    A shallow update would let a profile that sets one risk coefficient delete
    every other key in that section.
    """
    out = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise InvalidRequestError(
            f"{path.name} is not valid YAML: {str(exc)[:200]}") from exc
    if not isinstance(loaded, dict):
        raise InvalidRequestError(f"{path.name} must contain a YAML mapping.")
    return loaded


def safe_profile_name(name: str) -> str:
    """Validate a profile name before it becomes a filename."""
    slug = str(name or "").strip().lower().replace(" ", "_")
    if not _SAFE_NAME.match(slug):
        raise InvalidRequestError(
            "A profile name must be 1-49 characters of lowercase letters, "
            "digits, '_' or '-', starting with a letter or digit.",
            details={"received": name})
    return slug


def fingerprint(params: dict) -> str:
    """Stable hash of a parameter set.

    ``meta`` is excluded: a description edit must not read as a different
    experiment. Sorted keys make the hash independent of dict ordering.
    """
    material = {k: v for k, v in params.items() if k != "meta"}
    blob = json.dumps(material, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _validate(params: dict) -> list[str]:
    """Bounds-check the resolved set. Returns human-readable problems."""
    problems: list[str] = []
    for path, (low, high) in BOUNDS.items():
        section, _, key = path.partition(".")
        value = (params.get(section) or {}).get(key)
        if value is None or isinstance(value, bool):
            continue          # null = "not applicable to this algorithm"
        if not isinstance(value, int | float):
            problems.append(f"{path} must be a number, got {type(value).__name__}")
            continue
        if not (low <= value <= high):
            problems.append(f"{path} must be between {low} and {high}, got {value}")

    # Cross-field checks a per-field bound cannot express.
    eps_start = (params.get("exploration") or {}).get("epsilon_start")
    eps_end = (params.get("exploration") or {}).get("epsilon_end")
    if (isinstance(eps_start, int | float) and isinstance(eps_end, int | float)
            and eps_end > eps_start):
        problems.append(
            f"exploration.epsilon_end ({eps_end}) exceeds epsilon_start "
            f"({eps_start}): exploration would increase over training")

    replay = params.get("replay") or {}
    if (isinstance(replay.get("min_buffer"), int)
            and isinstance(replay.get("buffer_size"), int)
            and replay["min_buffer"] > replay["buffer_size"]):
        problems.append(
            f"replay.min_buffer ({replay['min_buffer']}) exceeds buffer_size "
            f"({replay['buffer_size']}): training would never start")

    hidden = (params.get("network") or {}).get("hidden")
    if hidden is not None and (
            not isinstance(hidden, list) or not hidden
            or not all(isinstance(h, int) and 1 <= h <= 8192 for h in hidden)):
        problems.append(
            "network.hidden must be a non-empty list of layer widths (1-8192)")
    return problems


# ------------------------------------------------------------------ manager
class HyperparameterManager:
    """Loads, resolves and persists RL hyperparameter configurations."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = Path(config_dir or CONFIG_DIR)
        self.algo_dir = self.config_dir / "algorithms"
        self.profile_dir = self.config_dir / "profiles"

    # ------------------------------------------------------- provisioning
    def ensure_configs(self) -> dict:
        """Create any missing configuration file from its built-in template.

        Called at startup so a user never has to place a YAML file by hand.
        Three properties matter:

        * **Additive.** A file that exists is left untouched, so an edit made
          through the dashboard survives every restart. Re-seeding on boot
          would silently revert the user's own tuning.
        * **Self-healing.** A deleted or never-created file comes back, rather
          than failing every training request with "No configuration file for
          algorithm 'ppo'" and no route to recovery from the UI.
        * **Never fatal.** A read-only or full filesystem must not stop the
          application from booting; the failure is logged and reported.
        """
        from app.services.rl import config_templates as templates

        created: list[str] = []
        failed: list[str] = []

        def _seed(path: Path, payload: dict) -> None:
            if path.exists():
                return
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(payload, sort_keys=False,
                                               default_flow_style=False))
                created.append(str(path.relative_to(self.config_dir)))
            except OSError as exc:      # pragma: no cover - filesystem dependent
                logger.warning("could not create %s: %s", path, exc)
                failed.append(path.name)

        _seed(self.config_dir / "defaults.yaml", templates.DEFAULTS)
        for algo, payload in templates.ALGORITHMS.items():
            _seed(self.algo_dir / f"{algo}.yaml", payload)
        for profile, payload in templates.PROFILES.items():
            _seed(self.profile_dir / f"{profile}.yaml", payload)

        if created:
            logger.info("hyperparameter configs provisioned: %s", ", ".join(created))
        return {"created": created, "failed": failed,
                "profiles": [p["key"] for p in self.profiles()]}

    # -------------------------------------------------------------- reading
    def defaults(self) -> dict:
        return _read_yaml(self.config_dir / "defaults.yaml")

    def algorithms(self) -> list[str]:
        return sorted(p.stem for p in self.algo_dir.glob("*.yaml"))

    def algorithm_config(self, algo: str) -> dict:
        path = self.algo_dir / f"{algo.lower().strip()}.yaml"
        if not path.exists():
            raise InvalidRequestError(
                f"No configuration file for algorithm '{algo}'.",
                details={"available": self.algorithms()})
        return _read_yaml(path)

    def profiles(self) -> list[dict]:
        out = []
        for path in sorted(self.profile_dir.glob("*.yaml")):
            data = _read_yaml(path)
            meta = data.get("meta") or {}
            out.append({
                "key": path.stem,
                "name": meta.get("name", path.stem.replace("_", " ").title()),
                "description": meta.get("description", ""),
                "builtin": bool(meta.get("builtin", False)),
                "sections": sorted(k for k in data if k != "meta"),
                "modified": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=UTC).isoformat(),
            })
        return out

    def profile_config(self, profile: str) -> dict:
        slug = safe_profile_name(profile)
        path = self.profile_dir / f"{slug}.yaml"
        if not path.exists():
            raise InvalidRequestError(
                f"No profile named '{profile}'.",
                details={"available": [p["key"] for p in self.profiles()]})
        return _read_yaml(path)

    # ------------------------------------------------------------ resolving
    def resolve(self, algo: str, profile: str = "default",
                overrides: dict | None = None) -> ResolvedConfig:
        """Materialise the full parameter set for one training run."""
        algo = algo.lower().strip()
        slug = safe_profile_name(profile)

        merged = self.defaults()
        sources = ["defaults.yaml"]

        algo_cfg = self.algorithm_config(algo)
        merged = deep_merge(merged, algo_cfg)
        sources.append(f"algorithms/{algo}.yaml")

        profile_cfg = self.profile_config(slug)
        # A profile describes a policy stance, not an algorithm identity. Its
        # meta block must not overwrite the algorithm's own meta, or every
        # resolved config would claim to be the profile.
        profile_cfg = {k: v for k, v in profile_cfg.items() if k != "meta"}
        merged = deep_merge(merged, profile_cfg)
        sources.append(f"profiles/{slug}.yaml")

        if overrides:
            merged = deep_merge(merged, self._expand(overrides))
            sources.append("request overrides")

        for section in SECTIONS:
            merged.setdefault(section, {})

        problems = _validate(merged)
        if problems:
            raise InvalidRequestError(
                "The resolved hyperparameters are out of range.",
                details={"problems": problems, "algo": algo, "profile": slug})

        return ResolvedConfig(algo=algo, profile=slug, params=merged,
                              sources=sources, fingerprint=fingerprint(merged))

    @staticmethod
    def _expand(overrides: dict) -> dict:
        """Accept both nested dicts and dotted paths.

        The API takes ``{"optimizer": {"learning_rate": 3e-4}}``; a caller
        patching one value finds ``{"optimizer.learning_rate": 3e-4}`` far more
        convenient. Supporting both avoids a second endpoint.
        """
        nested: dict = {}
        for key, value in overrides.items():
            if "." in key:
                section, _, leaf = key.partition(".")
                nested.setdefault(section, {})[leaf] = value
            else:
                nested[key] = value
        return nested

    # -------------------------------------------------------------- writing
    def save_profile(self, name: str, config: dict,
                     description: str = "", merge: bool = True) -> dict:
        """Create or replace a user profile.

        Built-in profiles are protected: overwriting `default.yaml` would leave
        no baseline to compare against and no way back without a reinstall.
        """
        slug = safe_profile_name(name)
        existing_path = self.profile_dir / f"{slug}.yaml"
        if existing_path.exists():
            current_meta = (_read_yaml(existing_path).get("meta") or {})
            if current_meta.get("builtin"):
                raise InvalidRequestError(
                    f"'{slug}' is a built-in profile and cannot be overwritten. "
                    f"Duplicate it under a new name instead.")

        incoming = {k: v for k, v in (config or {}).items() if k != "meta"}
        # Merge onto what the profile already holds instead of replacing it.
        #
        # The dashboard sends only the fields the user actually touched, so a
        # wholesale replace silently deleted everything else: duplicating
        # Conservative and then editing one learning rate left a profile
        # containing *only* `optimizer`, with its risk penalties, trade
        # fraction and regime settings gone — and training then quietly ran on
        # the defaults those values were meant to override.
        #
        # `merge=False` restores the replacing behaviour for callers that
        # genuinely send a whole document, which is what import does.
        existing = {}
        if merge and existing_path.exists():
            existing = {k: v for k, v in _read_yaml(existing_path).items()
                        if k != "meta"}
        payload = deep_merge(existing, incoming)

        # Validate against a real resolution rather than in isolation: a value
        # is only meaningful once merged onto the defaults it modifies.
        probe = deep_merge(self.defaults(), payload)
        for section in SECTIONS:
            probe.setdefault(section, {})
        problems = _validate(probe)
        if problems:
            raise InvalidRequestError(
                "This profile contains out-of-range values.",
                details={"problems": problems})

        document = {
            "meta": {
                "name": str(config.get("meta", {}).get("name") or
                            name.replace("_", " ").title()),
                "description": description or
                               str(config.get("meta", {}).get("description") or ""),
                "builtin": False,
                "updated": datetime.now(UTC).isoformat(),
            },
            **payload,
        }
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        existing_path.write_text(yaml.safe_dump(document, sort_keys=False,
                                           default_flow_style=False))
        logger.info("hyperparameter profile saved: %s", slug)
        return {"key": slug, **document["meta"]}

    def duplicate_profile(self, source: str, new_name: str) -> dict:
        """Copy a profile under a new name — the supported way to edit a built-in."""
        config = self.profile_config(source)
        config.pop("meta", None)
        return self.save_profile(
            new_name, config,
            description=f"Copied from '{safe_profile_name(source)}'.")

    def delete_profile(self, name: str) -> dict:
        slug = safe_profile_name(name)
        path = self.profile_dir / f"{slug}.yaml"
        if not path.exists():
            raise InvalidRequestError(f"No profile named '{name}'.")
        if (_read_yaml(path).get("meta") or {}).get("builtin"):
            raise InvalidRequestError(
                f"'{slug}' is a built-in profile and cannot be deleted.")
        path.unlink()
        logger.info("hyperparameter profile deleted: %s", slug)
        return {"deleted": slug}

    def export_profile(self, name: str) -> str:
        """YAML text, for download."""
        slug = safe_profile_name(name)
        return yaml.safe_dump(self.profile_config(slug), sort_keys=False,
                              default_flow_style=False)

    def import_profile(self, name: str, yaml_text: str) -> dict:
        """Load a profile from uploaded YAML.

        ``safe_load`` is not optional here: ``yaml.load`` on user input can
        construct arbitrary Python objects.
        """
        try:
            data = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError as exc:
            raise InvalidRequestError(
                f"Not valid YAML: {str(exc)[:200]}") from exc
        if not isinstance(data, dict):
            raise InvalidRequestError("A profile must be a YAML mapping.")
        unknown = [k for k in data if k not in SECTIONS]
        if unknown:
            raise InvalidRequestError(
                f"Unknown configuration section(s): {', '.join(sorted(unknown))}",
                details={"allowed": list(SECTIONS)})
        # Import replaces rather than merges: an imported file is a complete
        # document, and merging it onto whatever happened to be there would
        # produce a profile matching neither the file nor the previous state —
        # which defeats the reproducibility this feature exists for.
        return self.save_profile(name, data, merge=False)

    # ---------------------------------------------------------- experiments
    @staticmethod
    def experiment_id() -> str:
        """A short, sortable identifier for one training run."""
        return f"exp_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


hyperparameters = HyperparameterManager()
