"""
dqn_training.py

Trains DQN (Stable-Baselines3) on WasteSegregationEnv across the ten
hyperparameter presets in training/hyperparameters.py::DQN_PRESETS.

Usage:
    python -m training.dqn_training --run all
    python -m training.dqn_training --run dqn_03
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from training.hyperparameters import DQN_PRESETS
from training.utils import EpisodeCSVLogger, ensure_dir, make_env


def train_single_run(preset: dict, log_dir: str, model_dir: str, seed: int = 0) -> str:
    ensure_dir(log_dir)
    ensure_dir(model_dir)

    env = make_env(seed=seed)
    env = Monitor(env)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=preset["learning_rate"],
        buffer_size=preset["buffer_size"],
        batch_size=preset["batch_size"],
        gamma=preset["gamma"],
        exploration_fraction=preset["exploration_fraction"],
        exploration_final_eps=preset["exploration_final_eps"],
        target_update_interval=preset["target_update_interval"],
        train_freq=preset["train_freq"],
        learning_starts=preset["learning_starts"],
        seed=seed,
        verbose=0,
        tensorboard_log=log_dir,
    )

    callback = EpisodeCSVLogger(log_dir=log_dir, run_id=preset["run_id"])
    model.learn(total_timesteps=preset["total_timesteps"], callback=callback, tb_log_name=preset["run_id"])

    model_path = os.path.join(model_dir, f"{preset['run_id']}.zip")
    model.save(model_path)
    env.close()
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN on WasteSegregationEnv.")
    parser.add_argument("--run", default="all", help="'all' or a specific run_id (e.g. dqn_03).")
    parser.add_argument("--log-dir", default="logs/dqn")
    parser.add_argument("--model-dir", default="models/dqn")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    presets = DQN_PRESETS if args.run == "all" else [p for p in DQN_PRESETS if p["run_id"] == args.run]
    if not presets:
        raise ValueError(f"No preset found with run_id '{args.run}'")

    for preset in presets:
        print(f"Training {preset['run_id']} ...")
        path = train_single_run(preset, args.log_dir, args.model_dir, seed=args.seed)
        print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
