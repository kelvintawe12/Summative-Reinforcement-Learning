"""
notebook_viz.py

Presentation helpers for the Colab training notebooks. Kept in the package
(not inlined in notebooks) so the plotting/bundling logic is single-source,
importable, and unit-testable, while the notebook cells stay short and
readable.

Provides:
    - training_dashboard(algo): a 4-panel figure (reward, smoothed reward,
      the algorithm's objective/entropy diagnostic, and episode length) built
      from the per-run CSV logs -- rendered inline in the notebook.
    - show_hyperparameter_table(algo): the 10-row hyperparameter table joined
      with each run's final performance, as a styled pandas DataFrame.
    - bundle_results(...): zips logs/, models/, and results/ (tables + plots)
      into a single downloadable archive, and (in Colab) triggers a browser
      download.

All functions degrade gracefully when logs are missing, so a notebook cell
never hard-crashes mid-presentation.
"""

from __future__ import annotations

import os
import zipfile
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The extra diagnostic each algorithm family exposes in its per-episode CSV.
_DIAGNOSTIC = {
    "dqn": ("loss", "TD loss (objective)"),
    "reinforce": ("entropy", "Policy entropy"),
    "a2c": ("entropy", "Policy entropy"),
    "ppo": ("entropy", "Policy entropy"),
}

# Consistent per-run colour cycling.
_CMAP = plt.get_cmap("viridis")


def _smoothed(series: pd.Series, window: int = 10) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(window=window, min_periods=1).mean()


def _load_runs(algo: str, logs_root: str = "logs") -> list[tuple[str, pd.DataFrame]]:
    """Load every present run CSV for an algorithm, sorted by run_id."""
    from training.hyperparameters import get_presets

    log_dir = os.path.join(logs_root, algo)
    runs = []
    for preset in get_presets(algo):
        path = os.path.join(log_dir, f"{preset['run_id']}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                runs.append((preset["run_id"], df))
    return runs


def training_dashboard(algo: str, logs_root: str = "logs", save_to: Optional[str] = None):
    """Render a 4-panel training dashboard for one algorithm, inline.

    Panels: (1) raw episode reward, (2) smoothed reward, (3) the algorithm's
    diagnostic (DQN loss / PG entropy), (4) episode length. One coloured line
    per hyperparameter run.
    """
    runs = _load_runs(algo, logs_root)
    if not runs:
        print(f"[{algo}] no logs found yet -- train first, then re-run this cell.")
        return None

    diag_col, diag_label = _DIAGNOSTIC.get(algo, ("loss", "Loss"))
    colors = [_CMAP(i / max(len(runs) - 1, 1)) for i in range(len(runs))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{algo.upper()} training dashboard  ({len(runs)} runs)", fontsize=16, weight="bold")

    for (run_id, df), c in zip(runs, colors):
        ep = df["episode"]
        axes[0, 0].plot(ep, df["episode_reward"], color=c, alpha=0.45, linewidth=0.8)
        axes[0, 1].plot(ep, _smoothed(df["episode_reward"], 20), color=c, linewidth=1.6, label=run_id)
        if diag_col in df.columns:
            series = pd.to_numeric(df[diag_col], errors="coerce")
            if series.notna().sum() > 0:
                axes[1, 0].plot(ep, _smoothed(series), color=c, linewidth=1.4)
        axes[1, 1].plot(ep, _smoothed(df["episode_length"]), color=c, linewidth=1.2)

    axes[0, 0].set(title="Episode reward (raw)", xlabel="Episode", ylabel="Reward")
    axes[0, 1].set(title="Episode reward (smoothed)", xlabel="Episode", ylabel="Reward")
    axes[1, 0].set(title=diag_label + " (smoothed)", xlabel="Episode", ylabel=diag_label)
    axes[1, 1].set(title="Episode length", xlabel="Episode", ylabel="Steps")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    axes[0, 1].legend(fontsize=7, ncol=2, loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.savefig(save_to, dpi=150)
    plt.show()
    return fig


def show_hyperparameter_table(algo: str, logs_root: str = "logs") -> pd.DataFrame:
    """Return the 10-row hyperparameter + performance table for a notebook to
    display. In a notebook the returned DataFrame renders as a styled table."""
    from results.analysis import build_hyperparameter_table

    table = build_hyperparameter_table(algo, logs_root, out_root="results")
    perf_cols = ["mean_reward_last_10pct", "best_episode_reward", "final_episode_reward"]
    if all(col in table.columns for col in perf_cols) and table["mean_reward_last_10pct"].notna().any():
        best_idx = table["mean_reward_last_10pct"].astype(float).idxmax()
        print(f"[{algo}] best run: {table.loc[best_idx, 'run_id']} "
              f"(mean reward last 10% = {table.loc[best_idx, 'mean_reward_last_10pct']})")
    return table


def bundle_results(
    out_path: str = "rl_results_bundle.zip",
    include=("logs", "models", "results"),
    logs_root: str = ".",
    download: bool = True,
) -> str:
    """Zip logs/, models/, and results/ (tables + plots) into one archive for
    download at the end of a notebook run.

    Before zipping it (re)builds the full analysis so the archive always
    contains fresh tables and plots. In Colab, triggers a browser download.

    Returns the path to the created zip.
    """
    # Regenerate tables + plots so the bundle is always complete and current.
    try:
        from results.analysis import run_full_analysis
        run_full_analysis(logs_root="logs", out_root="results")
    except Exception as exc:  # keep the bundle step robust in a live demo
        print(f"(analysis step skipped: {exc})")

    n_files = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for top in include:
            root = os.path.join(logs_root, top)
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn == ".gitkeep":
                        continue
                    full = os.path.join(dirpath, fn)
                    arcname = os.path.relpath(full, logs_root)
                    zf.write(full, arcname)
                    n_files += 1

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Bundled {n_files} files into {out_path} ({size_mb:.1f} MB)")

    if download:
        try:
            from google.colab import files  # type: ignore
            files.download(out_path)
        except Exception:
            print("(not in Colab -- find the zip in the working directory)")
    return out_path
