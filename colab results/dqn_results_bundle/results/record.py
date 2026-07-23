"""
record.py

Render one episode of an agent acting in WasteSegregationEnv to an animated
GIF (and optionally MP4), by capturing the Pygame dashboard as `rgb_array`
frames. This produces the "agent acting accordingly" visualization for the
report and README without needing a live display, so it also works headless
(Colab, CI, SSH).

Two agent sources are supported:
    - a trained model:      --algo ppo --model models/ppo/ppo_07.zip
    - the capacity-aware heuristic (default, no model needed): --heuristic

The heuristic path lets you see a *correct* agent immediately -- routing each
item to its best-matching, non-full, non-jammed bin, rejecting contaminants,
holding when it should -- before any real training is done on Colab. After
training, re-run with --algo/--model to record the actual learned policy; the
output format is identical.

Usage:
    uv run python -m results.record --heuristic --out assets/agent_heuristic.gif
    uv run python -m results.record --algo ppo --model models/ppo/ppo_07.zip \
        --out assets/agent_ppo.gif
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from environment.custom_env import WasteSegregationEnv
from results.diagnostics import capacity_aware_policy


def _write_gif(frames: list[np.ndarray], out_path: str, fps: int) -> None:
    """Write RGB frames to an animated GIF via matplotlib (Pillow backend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    h, w, _ = frames[0].shape
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(i):
        im.set_array(frames[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=True
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        try:
            anim.save(out_path, writer="ffmpeg", fps=fps)
        except Exception as exc:
            raise SystemExit(f"MP4 export needs ffmpeg installed: {exc}")
    else:
        anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


def record_episode(
    agent_fn,
    out_path: str = "assets/agent_episode.gif",
    seed: int = 123,
    fps: int = 8,
    max_frames: int = 400,
) -> tuple[str, float, dict]:
    """Roll out one episode, capturing a frame per step, and save a GIF/MP4.

    `agent_fn(env, obs) -> int` returns the action for the current state.
    Returns (out_path, total_reward, episode_stats).
    """
    env = WasteSegregationEnv(render_mode="rgb_array", seed=seed)
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    done, total, info = False, 0.0, {}

    while not done and len(frames) < max_frames:
        action = int(agent_fn(env, obs))
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        frames.append(env.render())
        done = terminated or truncated

    env.close()
    _write_gif(frames, out_path, fps)
    return out_path, total, info.get("episode_stats", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Record an agent acting in WasteSegregationEnv to a GIF/MP4.")
    parser.add_argument("--algo", choices=["dqn", "reinforce", "a2c", "ppo"], default=None)
    parser.add_argument("--model", default=None, help="Path to a trained model file.")
    parser.add_argument("--heuristic", action="store_true",
                        help="Use the capacity-aware heuristic instead of a trained model.")
    parser.add_argument("--out", default="assets/agent_episode.gif")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    if args.algo and args.model:
        from training.utils import load_agent
        agent = load_agent(args.algo, args.model)

        def agent_fn(env, obs):
            action, _ = agent.predict(obs, deterministic=True)
            return int(action)

        label = f"trained {args.algo}"
    else:
        if not args.heuristic:
            print("No --algo/--model given; defaulting to the capacity-aware heuristic. "
                  "(Pass --heuristic to silence this, or --algo/--model after training.)")
        agent_fn = capacity_aware_policy
        label = "capacity-aware heuristic"

    out, total, stats = record_episode(agent_fn, out_path=args.out, seed=args.seed, fps=args.fps)
    print(f"Recorded {label} -> {out}")
    print(f"  episode reward: {total:.2f}")
    print(f"  episode stats : {stats}")


if __name__ == "__main__":
    main()
