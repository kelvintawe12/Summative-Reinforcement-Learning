"""
analysis.py

Post-training analysis utilities:
    - Loads per-run episode-reward CSV logs produced by every training
      script (logs/<algo>/<run_id>.csv).
    - Builds the four required hyperparameter comparison tables (one per
      algorithm), joining each run's hyperparameters with its final
      training performance, saved to results/hyperparameter_tables/.
    - Plots reward-over-episodes learning curves, per algorithm and a
      combined comparison across all four, saved to results/plots/.

Usage:
    python -m results.analysis --logs-root logs --out-root results
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from training.hyperparameters import get_presets

ALGORITHMS = ["dqn", "reinforce", "a2c", "ppo"]


def load_run_csv(log_dir: str, run_id: str) -> pd.DataFrame | None:
    path = os.path.join(log_dir, f"{run_id}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def smoothed(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def build_hyperparameter_table(algo: str, logs_root: str, out_root: str) -> pd.DataFrame:
    presets = get_presets(algo)
    log_dir = os.path.join(logs_root, algo)
    rows = []
    for preset in presets:
        df = load_run_csv(log_dir, preset["run_id"])
        row = dict(preset)
        if df is not None:
            tail = df.tail(max(1, len(df) // 10))  # last ~10% of episodes
            row["episodes_completed"] = int(df["episode"].iloc[-1])
            row["mean_reward_last_10pct"] = round(float(tail["episode_reward"].mean()), 2)
            row["best_episode_reward"] = round(float(df["episode_reward"].max()), 2)
            row["final_episode_reward"] = round(float(df["episode_reward"].iloc[-1]), 2)
        else:
            row["episodes_completed"] = None
            row["mean_reward_last_10pct"] = None
            row["best_episode_reward"] = None
            row["final_episode_reward"] = None
        rows.append(row)

    table = pd.DataFrame(rows)
    out_dir = os.path.join(out_root, "hyperparameter_tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{algo}_hyperparameters.csv")
    table.to_csv(out_path, index=False)
    return table


def plot_algorithm_curves(algo: str, logs_root: str, out_root: str) -> str | None:
    presets = get_presets(algo)
    log_dir = os.path.join(logs_root, algo)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False
    for preset in presets:
        df = load_run_csv(log_dir, preset["run_id"])
        if df is None:
            continue
        any_data = True
        ax.plot(df["episode"], smoothed(df["episode_reward"]), label=preset["run_id"], linewidth=1.4)

    if not any_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward (10-episode rolling mean)")
    ax.set_title(f"{algo.upper()} - reward across hyperparameter sweep")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_dir = os.path.join(out_root, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{algo}_reward_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _best_run_id(algo: str, logs_root: str) -> str | None:
    """Return the run_id with the highest mean reward over its last 10% of
    episodes, or None if no logs exist for this algorithm."""
    best_id, best_mean = None, -float("inf")
    log_dir = os.path.join(logs_root, algo)
    for preset in get_presets(algo):
        df = load_run_csv(log_dir, preset["run_id"])
        if df is None:
            continue
        tail = df.tail(max(1, len(df) // 10))
        m = tail["episode_reward"].mean()
        if m > best_mean:
            best_mean, best_id = m, preset["run_id"]
    return best_id


def plot_metric_curves(
    algo: str, metric: str, ylabel: str, title: str, filename: str,
    logs_root: str, out_root: str,
) -> str | None:
    """Plot one logged diagnostic column (e.g. 'loss' or 'entropy') across all
    ten runs of an algorithm. Runs that never logged the metric are skipped.

    Used for the DQN objective (loss) curve and the policy-gradient entropy
    curves the rubric explicitly asks for.
    """
    presets = get_presets(algo)
    log_dir = os.path.join(logs_root, algo)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False
    for preset in presets:
        df = load_run_csv(log_dir, preset["run_id"])
        if df is None or metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        if series.notna().sum() == 0:
            continue
        any_data = True
        ax.plot(df["episode"], smoothed(series), label=preset["run_id"], linewidth=1.4)

    if not any_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_dir = os.path.join(out_root, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_convergence_subplots(logs_root: str, out_root: str) -> str | None:
    """One figure, four subplots (DQN / REINFORCE / A2C / PPO), each showing
    the cumulative-reward learning curve for that algorithm's best run.

    This is the rubric's "cumulative reward curves (all methods in subplots)"
    and doubles as the convergence panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    any_data = False

    for ax, algo in zip(axes.ravel(), ALGORITHMS):
        best_id = _best_run_id(algo, logs_root)
        ax.set_title(f"{algo.upper()}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode reward")
        ax.grid(alpha=0.25)
        if best_id is None:
            ax.text(0.5, 0.5, "no logs yet", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue
        df = load_run_csv(os.path.join(logs_root, algo), best_id)
        any_data = True
        ax.plot(df["episode"], df["episode_reward"], color="0.75",
                linewidth=0.8, label="raw")
        ax.plot(df["episode"], smoothed(df["episode_reward"], window=20),
                linewidth=1.8, label=f"{best_id} (smoothed)")
        ax.legend(fontsize=8, loc="lower right")

    if not any_data:
        plt.close(fig)
        return None

    fig.suptitle("Convergence: best-run reward curve per algorithm", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_dir = os.path.join(out_root, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "convergence_subplots.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_best_run_comparison(logs_root: str, out_root: str) -> str | None:
    """Overlay the single best run from each algorithm on one figure."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_data = False

    for algo in ALGORITHMS:
        presets = get_presets(algo)
        log_dir = os.path.join(logs_root, algo)
        best_df, best_mean = None, -float("inf")
        for preset in presets:
            df = load_run_csv(log_dir, preset["run_id"])
            if df is None:
                continue
            tail = df.tail(max(1, len(df) // 10))
            m = tail["episode_reward"].mean()
            if m > best_mean:
                best_mean = m
                best_df = df
        if best_df is not None:
            any_data = True
            ax.plot(
                best_df["episode"], smoothed(best_df["episode_reward"]),
                label=f"{algo.upper()} (best run)", linewidth=1.8,
            )

    if not any_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward (10-episode rolling mean)")
    ax.set_title("Best run per algorithm - comparison")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_dir = os.path.join(out_root, "plots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "algorithm_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_full_analysis(logs_root: str = "logs", out_root: str = "results") -> None:
    for algo in ALGORITHMS:
        table = build_hyperparameter_table(algo, logs_root, out_root)
        table_path = os.path.join(out_root, "hyperparameter_tables", f"{algo}_hyperparameters.csv")
        print(f"[{algo}] hyperparameter table -> {table_path} "
              f"({table['mean_reward_last_10pct'].notna().sum()}/{len(table)} runs with logs)")
        curve_path = plot_algorithm_curves(algo, logs_root, out_root)
        print(f"[{algo}] reward curves -> {curve_path}")

    # DQN objective (TD-loss) curves.
    dqn_loss = plot_metric_curves(
        "dqn", "loss", "DQN training loss (10-ep rolling mean)",
        "DQN - objective (TD loss) across hyperparameter sweep",
        "dqn_objective_curves.png", logs_root, out_root,
    )
    print(f"[dqn] objective curves -> {dqn_loss}")

    # Policy-gradient entropy curves (exploration signal) for the three PG methods.
    for algo in ("reinforce", "a2c", "ppo"):
        ent = plot_metric_curves(
            algo, "entropy", "Policy entropy (10-ep rolling mean)",
            f"{algo.upper()} - policy entropy across hyperparameter sweep",
            f"{algo}_entropy_curves.png", logs_root, out_root,
        )
        print(f"[{algo}] entropy curves -> {ent}")

    convergence_path = plot_convergence_subplots(logs_root, out_root)
    print(f"[all] convergence subplots -> {convergence_path}")

    comparison_path = plot_best_run_comparison(logs_root, out_root)
    print(f"[all] best-run comparison -> {comparison_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hyperparameter tables and reward plots.")
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--out-root", default="results")
    args = parser.parse_args()
    run_full_analysis(args.logs_root, args.out_root)


if __name__ == "__main__":
    main()
