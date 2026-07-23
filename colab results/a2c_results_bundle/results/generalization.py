"""
generalization.py

Generalization test: the models are trained on the environment's *default*
demand profile (the mix of dominant material categories arriving on the
conveyor). This module evaluates each algorithm's best trained model under
several *shifted* demand profiles it never saw during training, to measure how
well the learned policy transfers rather than overfits to one item
distribution.

Profiles evaluated (all are [organic, plastic, metal, glass, contaminant]
probabilities that a given item's dominant material is that category):

    - "train"          : the exact profile used for training (baseline).
    - "high_contam"    : far more non-recyclable contaminant items.
    - "plastic_surge"  : a plastics-heavy waste stream.
    - "uniform"        : every material equally likely (maximally mixed).

For each (algorithm, profile) pair we roll out several fixed-seed evaluation
episodes and record mean reward. The gap between "train" and the shifted
profiles is the generalization signal.

Usage:
    python -m results.generalization
    python -m results.generalization --episodes 20
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment.custom_env import WasteSegregationEnv
from training.utils import load_agent

ALGORITHMS = ["dqn", "reinforce", "a2c", "ppo"]
MODEL_EXTENSION = {"dqn": "zip", "a2c": "zip", "ppo": "zip", "reinforce": "pt"}

# Shifted demand profiles the agent never trained on.
PROFILES: dict[str, np.ndarray] = {
    "train": np.array([0.42, 0.20, 0.10, 0.10, 0.18]),
    "high_contam": np.array([0.25, 0.15, 0.08, 0.07, 0.45]),
    "plastic_surge": np.array([0.20, 0.50, 0.10, 0.10, 0.10]),
    "uniform": np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
}


def _best_run_id(algo: str, logs_root: str) -> str | None:
    """Highest mean-reward-over-last-10% run for this algorithm, or None."""
    log_dir = os.path.join(logs_root, algo)
    best_id, best_mean = None, -float("inf")
    for path in glob.glob(os.path.join(log_dir, f"{algo}_*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        tail = df.tail(max(1, len(df) // 10))
        m = tail["episode_reward"].mean()
        if m > best_mean:
            best_mean, best_id = m, os.path.basename(path)[:-4]
    return best_id


def evaluate_profile(agent, category_probs: np.ndarray, episodes: int, seed0: int) -> float:
    """Mean total reward over `episodes` fixed-seed episodes under one profile."""
    totals = []
    for ep in range(episodes):
        env = WasteSegregationEnv(seed=seed0 + ep, category_probs=category_probs)
        obs, _ = env.reset(seed=seed0 + ep)
        done, total = False, 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
        env.close()
        totals.append(total)
    return float(np.mean(totals))


def run_generalization(
    logs_root: str = "logs",
    models_root: str = "models",
    out_root: str = "results",
    episodes: int = 10,
    seed0: int = 4000,
) -> pd.DataFrame | None:
    rows = []
    for algo in ALGORITHMS:
        best_id = _best_run_id(algo, logs_root)
        if best_id is None:
            continue
        model_path = os.path.join(models_root, algo, f"{best_id}.{MODEL_EXTENSION[algo]}")
        if not os.path.exists(model_path):
            continue
        agent = load_agent(algo, model_path)
        row = {"algorithm": algo.upper(), "run_id": best_id}
        for name, probs in PROFILES.items():
            row[name] = round(evaluate_profile(agent, probs, episodes, seed0), 2)
        rows.append(row)

    if not rows:
        print("No trained models found; run training first.")
        return None

    df = pd.DataFrame(rows)
    out_dir = os.path.join(out_root, "hyperparameter_tables")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "generalization_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[all] generalization table -> {csv_path}")

    _plot(df, out_root)
    return df


def _plot(df: pd.DataFrame, out_root: str) -> str:
    profiles = list(PROFILES.keys())
    x = np.arange(len(profiles))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.bar(x + i * width, [row[p] for p in profiles], width, label=row["algorithm"])

    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(profiles)
    ax.set_xlabel("Demand profile (train = seen during training; others are shifted)")
    ax.set_ylabel(f"Mean episode reward")
    ax.set_title("Generalization: best model per algorithm under shifted demand profiles")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()

    out_dir = os.path.join(out_root, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generalization.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[all] generalization plot -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalization test across shifted demand profiles.")
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--out-root", default="results")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()
    run_generalization(args.logs_root, args.models_root, args.out_root, args.episodes)


if __name__ == "__main__":
    main()
