"""
utils.py

Shared utilities used by every training script (DQN, REINFORCE, A2C, PPO):
    - `make_env`: constructs a fresh WasteSegregationEnv instance.
    - `EpisodeCSVLogger`: a Stable-Baselines3 callback that appends one row
      per completed episode (timestep, episode reward, episode length) to a
      CSV file, giving a uniform log format across all four algorithms
      regardless of which one natively supports TensorBoard.
    - `ensure_dir`: small filesystem helper.
"""

from __future__ import annotations

import csv
import os
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

from environment.custom_env import WasteSegregationEnv


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_env(seed: Optional[int] = None, **env_kwargs) -> WasteSegregationEnv:
    """Construct a WasteSegregationEnv with sensible training defaults.

    Args:
        seed: RNG seed for reproducibility.
        **env_kwargs: forwarded to WasteSegregationEnv's constructor,
            allowing callers to override defaults (e.g. max_steps) per run.

    Returns:
        A new WasteSegregationEnv instance.
    """
    env = WasteSegregationEnv(seed=seed, **env_kwargs)
    return env


def load_agent(algo: str, model_path: str, obs_dim: int = 21, n_actions: int = 7):
    """Load a trained agent for inference, regardless of which algorithm
    trained it.

    Every returned object exposes `.predict(obs, deterministic=True) ->
    (action, state)`, matching the Stable-Baselines3 inference API, so
    calling code (play.py, main.py) does not need to branch on algorithm.

    Args:
        algo: one of "dqn", "reinforce", "a2c", "ppo".
        model_path: path to the saved model file (.zip for SB3 algorithms,
            .pt for REINFORCE).
        obs_dim: observation dimensionality (only used to construct the
            REINFORCE network before loading its weights).
        n_actions: action count (only used for REINFORCE).

    Returns:
        A loaded model/agent object with a `.predict()` method.
    """
    algo = algo.lower()
    if algo == "dqn":
        from stable_baselines3 import DQN
        return DQN.load(model_path)
    if algo == "a2c":
        from stable_baselines3 import A2C
        return A2C.load(model_path)
    if algo == "ppo":
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    if algo == "reinforce":
        from training.reinforce_training import ReinforceAgent
        agent = ReinforceAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(model_path)
        return agent
    raise ValueError(f"Unknown algorithm '{algo}'.")


# Uniform per-episode log schema shared by all four algorithms. Columns that
# a given algorithm does not produce are written as empty cells, so
# results/analysis.py can read every run's CSV with one code path.
LOG_COLUMNS = [
    "timestep", "episode", "episode_reward", "episode_length",
    "loss", "entropy", "value_loss",
]


class EpisodeCSVLogger(BaseCallback):
    """Stable-Baselines3 callback that logs one row per completed episode.

    Writes to `<log_dir>/<run_id>.csv` with columns (see LOG_COLUMNS):
        timestep, episode, episode_reward, episode_length,
        loss, entropy, value_loss

    Beyond reward/length, it snapshots the algorithm's most recent training
    diagnostics from `model.logger.name_to_value` at each episode boundary:

        - DQN:      `loss`  <- train/loss  (the TD objective curve)
        - A2C/PPO:  `entropy` <- -train/entropy_loss  (policy entropy, the
                    exploration signal), `value_loss` <- train/value_loss,
                    `loss` <- train/policy_gradient_loss or train/policy_loss

    This gives DQN/A2C/PPO the exact same per-episode log format as the custom
    REINFORCE trainer (which fills the same three extra columns directly), so
    results/analysis.py can plot DQN objective curves and PG entropy curves
    for every algorithm uniformly -- and, critically, all of it lives in the
    CSV that gets zipped back from Colab.
    """

    def __init__(self, log_dir: str, run_id: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_dir = log_dir
        self.run_id = run_id
        self.episode_count = 0
        self._csv_path = os.path.join(log_dir, f"{run_id}.csv")

    def _on_training_start(self) -> None:
        ensure_dir(self.log_dir)
        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)

    def _current_diagnostics(self) -> tuple:
        """Pull (loss, entropy, value_loss) from SB3's latest logged values.

        Returns empty strings for metrics the current algorithm does not log,
        so the CSV column set stays uniform across all algorithms.
        """
        vals = getattr(self.model.logger, "name_to_value", {}) or {}

        # DQN logs train/loss; on/off-policy actor-critics log policy losses.
        loss = vals.get("train/loss")
        if loss is None:
            loss = vals.get("train/policy_gradient_loss")
        if loss is None:
            loss = vals.get("train/policy_loss")

        # SB3 logs entropy_loss = -mean(entropy); recover mean entropy itself.
        entropy_loss = vals.get("train/entropy_loss")
        entropy = -entropy_loss if entropy_loss is not None else None

        value_loss = vals.get("train/value_loss")

        def fmt(x):
            return "" if x is None else float(x)

        return fmt(loss), fmt(entropy), fmt(value_loss)

    def _on_step(self) -> bool:
        # SB3's Monitor wrapper populates `infos[i]["episode"]` exactly on
        # the timestep an episode ends.
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_count += 1
                loss, entropy, value_loss = self._current_diagnostics()
                with open(self._csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        self.num_timesteps,
                        self.episode_count,
                        ep_info["r"],
                        ep_info["l"],
                        loss, entropy, value_loss,
                    ])
        return True
