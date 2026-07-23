"""
ppo_training.py

Trains PPO (Stable-Baselines3) on WasteSegregationEnv across the ten
hyperparameter presets in training/hyperparameters.py::PPO_PRESETS.

Usage:
    python -m training.ppo_training --run all
    python -m training.ppo_training --run ppo_03
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from training.hyperparameters import PPO_PRESETS
from training.utils import EpisodeCSVLogger, ensure_dir, make_env


def train_single_run(preset: dict, log_dir: str, model_dir: str, seed: int = 0) -> str:
    ensure_dir(log_dir)
    ensure_dir(model_dir)

    env = make_env(seed=seed)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=preset["learning_rate"],
        n_steps=preset["n_steps"],
        batch_size=preset["batch_size"],
        n_epochs=preset["n_epochs"],
        gamma=preset["gamma"],
        gae_lambda=preset["gae_lambda"],
        clip_range=preset["clip_range"],
        ent_coef=preset["ent_coef"],
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
    parser = argparse.ArgumentParser(description="Train PPO on WasteSegregationEnv.")
    parser.add_argument("--run", default="all", help="'all' or a specific run_id (e.g. ppo_03).")
    parser.add_argument("--log-dir", default="logs/ppo")
    parser.add_argument("--model-dir", default="models/ppo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    presets = PPO_PRESETS if args.run == "all" else [p for p in PPO_PRESETS if p["run_id"] == args.run]
    if not presets:
        raise ValueError(f"No preset found with run_id '{args.run}'")

    for preset in presets:
        print(f"Training {preset['run_id']} ...")
        path = train_single_run(preset, args.log_dir, args.model_dir, seed=args.seed)
        print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
