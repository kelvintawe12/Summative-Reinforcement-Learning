"""
hyperparameters.py

Canonical, versioned hyperparameter sweep definitions for all four algorithms
(DQN, REINFORCE, A2C, PPO). Each algorithm gets exactly ten configurations,
matching the assignment's required hyperparameter-table deliverable.

These are defined once here (not typed ad hoc into notebooks) so that:
    - the Colab notebooks, local CLI training scripts, and the results
      tables all read from a single source of truth;
    - every run is fully reproducible from the repository alone.

Each entry is a plain dict of keyword arguments accepted by the
corresponding Stable-Baselines3 algorithm constructor (or, for REINFORCE,
by the custom `ReinforceAgent` in training/reinforce_training.py), plus a
"run_id" and "total_timesteps" field used by the training scripts.
"""

from __future__ import annotations

DQN_PRESETS = [
    {"run_id": "dqn_01", "total_timesteps": 60_000, "learning_rate": 1e-3, "buffer_size": 20_000, "batch_size": 32, "gamma": 0.99, "exploration_fraction": 0.30, "exploration_final_eps": 0.05, "target_update_interval": 500, "train_freq": 4, "learning_starts": 1_000},
    {"run_id": "dqn_02", "total_timesteps": 60_000, "learning_rate": 5e-4, "buffer_size": 20_000, "batch_size": 32, "gamma": 0.99, "exploration_fraction": 0.30, "exploration_final_eps": 0.05, "target_update_interval": 500, "train_freq": 4, "learning_starts": 1_000},
    {"run_id": "dqn_03", "total_timesteps": 60_000, "learning_rate": 2.5e-4, "buffer_size": 20_000, "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.30, "exploration_final_eps": 0.05, "target_update_interval": 500, "train_freq": 4, "learning_starts": 1_000},
    {"run_id": "dqn_04", "total_timesteps": 60_000, "learning_rate": 1e-3, "buffer_size": 50_000, "batch_size": 64, "gamma": 0.99, "exploration_fraction": 0.30, "exploration_final_eps": 0.05, "target_update_interval": 500, "train_freq": 4, "learning_starts": 1_000},
    {"run_id": "dqn_05", "total_timesteps": 60_000, "learning_rate": 1e-3, "buffer_size": 20_000, "batch_size": 32, "gamma": 0.95, "exploration_fraction": 0.30, "exploration_final_eps": 0.05, "target_update_interval": 500, "train_freq": 4, "learning_starts": 1_000},
    {"run_id": "dqn_06", "total_timesteps": 60_000, "learning_rate": 1e-3, "buffer_size": 20_000, "batch_size": 32, "gamma": 0.999, "exploration_fraction": 0.30, "exploration_final_eps": 0.05, "target_update_interval": 500, "train_freq": 4, "learning_starts": 1_000},
    {"run_id": "dqn_07", "total_timesteps": 60_000, "learning_rate": 1e-3, "buffer_size": 20_000, "batch_size": 32, "gamma": 0.99, "exploration_fraction": 0.10, "exploration_final_eps": 0.02, "target_update_interval": 500, "train_freq": 4, "learning_starts": 1_000},
    {"run_id": "dqn_08", "total_timesteps": 60_000, "learning_rate": 1e-3, "buffer_size": 20_000, "batch_size": 32, "gamma": 0.99, "exploration_fraction": 0.50, "exploration_final_eps": 0.10, "target_update_interval": 500, "train_freq": 4, "learning_starts": 1_000},
    {"run_id": "dqn_09", "total_timesteps": 60_000, "learning_rate": 1e-3, "buffer_size": 20_000, "batch_size": 32, "gamma": 0.99, "exploration_fraction": 0.30, "exploration_final_eps": 0.05, "target_update_interval": 100, "train_freq": 4, "learning_starts": 1_000},
    {"run_id": "dqn_10", "total_timesteps": 60_000, "learning_rate": 1e-3, "buffer_size": 20_000, "batch_size": 32, "gamma": 0.99, "exploration_fraction": 0.30, "exploration_final_eps": 0.05, "target_update_interval": 2_000, "train_freq": 4, "learning_starts": 1_000},
]

# REINFORCE is implemented from scratch (SB3 has no REINFORCE), so its
# preset keys map to constructor arguments of ReinforceAgent.
REINFORCE_PRESETS = [
    {"run_id": "reinforce_01", "total_timesteps": 60_000, "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64, "baseline": True, "entropy_coef": 0.0},
    {"run_id": "reinforce_02", "total_timesteps": 60_000, "learning_rate": 5e-4, "gamma": 0.99, "hidden_size": 64, "baseline": True, "entropy_coef": 0.0},
    {"run_id": "reinforce_03", "total_timesteps": 60_000, "learning_rate": 1e-4, "gamma": 0.99, "hidden_size": 64, "baseline": True, "entropy_coef": 0.0},
    {"run_id": "reinforce_04", "total_timesteps": 60_000, "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 128, "baseline": True, "entropy_coef": 0.0},
    {"run_id": "reinforce_05", "total_timesteps": 60_000, "learning_rate": 1e-3, "gamma": 0.95, "hidden_size": 64, "baseline": True, "entropy_coef": 0.0},
    {"run_id": "reinforce_06", "total_timesteps": 60_000, "learning_rate": 1e-3, "gamma": 0.999, "hidden_size": 64, "baseline": True, "entropy_coef": 0.0},
    {"run_id": "reinforce_07", "total_timesteps": 60_000, "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64, "baseline": False, "entropy_coef": 0.0},
    {"run_id": "reinforce_08", "total_timesteps": 60_000, "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64, "baseline": True, "entropy_coef": 0.01},
    {"run_id": "reinforce_09", "total_timesteps": 60_000, "learning_rate": 1e-3, "gamma": 0.99, "hidden_size": 64, "baseline": True, "entropy_coef": 0.05},
    {"run_id": "reinforce_10", "total_timesteps": 60_000, "learning_rate": 2e-3, "gamma": 0.99, "hidden_size": 64, "baseline": True, "entropy_coef": 0.0},
]

A2C_PRESETS = [
    {"run_id": "a2c_01", "total_timesteps": 60_000, "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "ent_coef": 0.0, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run_id": "a2c_02", "total_timesteps": 60_000, "learning_rate": 3e-4, "n_steps": 5, "gamma": 0.99, "ent_coef": 0.0, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run_id": "a2c_03", "total_timesteps": 60_000, "learning_rate": 1e-3, "n_steps": 5, "gamma": 0.99, "ent_coef": 0.0, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run_id": "a2c_04", "total_timesteps": 60_000, "learning_rate": 7e-4, "n_steps": 16, "gamma": 0.99, "ent_coef": 0.0, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run_id": "a2c_05", "total_timesteps": 60_000, "learning_rate": 7e-4, "n_steps": 32, "gamma": 0.99, "ent_coef": 0.0, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run_id": "a2c_06", "total_timesteps": 60_000, "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.95, "ent_coef": 0.0, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run_id": "a2c_07", "total_timesteps": 60_000, "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.999, "ent_coef": 0.0, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run_id": "a2c_08", "total_timesteps": 60_000, "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "ent_coef": 0.01, "vf_coef": 0.5, "gae_lambda": 1.0},
    {"run_id": "a2c_09", "total_timesteps": 60_000, "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "ent_coef": 0.0, "vf_coef": 0.25, "gae_lambda": 1.0},
    {"run_id": "a2c_10", "total_timesteps": 60_000, "learning_rate": 7e-4, "n_steps": 5, "gamma": 0.99, "ent_coef": 0.0, "vf_coef": 0.5, "gae_lambda": 0.90},
]

PPO_PRESETS = [
    {"run_id": "ppo_01", "total_timesteps": 60_000, "learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    {"run_id": "ppo_02", "total_timesteps": 60_000, "learning_rate": 1e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    {"run_id": "ppo_03", "total_timesteps": 60_000, "learning_rate": 1e-3, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    {"run_id": "ppo_04", "total_timesteps": 60_000, "learning_rate": 3e-4, "n_steps": 512, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    {"run_id": "ppo_05", "total_timesteps": 60_000, "learning_rate": 3e-4, "n_steps": 256, "batch_size": 128, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    {"run_id": "ppo_06", "total_timesteps": 60_000, "learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 4, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    {"run_id": "ppo_07", "total_timesteps": 60_000, "learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 20, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    {"run_id": "ppo_08", "total_timesteps": 60_000, "learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "gamma": 0.95, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0},
    {"run_id": "ppo_09", "total_timesteps": 60_000, "learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.1, "ent_coef": 0.0},
    {"run_id": "ppo_10", "total_timesteps": 60_000, "learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01},
]


def get_presets(algo: str) -> list[dict]:
    """Return the ten hyperparameter presets for the given algorithm name."""
    algo = algo.lower()
    mapping = {
        "dqn": DQN_PRESETS,
        "reinforce": REINFORCE_PRESETS,
        "a2c": A2C_PRESETS,
        "ppo": PPO_PRESETS,
    }
    if algo not in mapping:
        raise ValueError(f"Unknown algorithm '{algo}'. Choose from {list(mapping)}.")
    return mapping[algo]
