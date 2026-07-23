"""
play.py

Renders a trained agent acting in WasteSegregationEnv, either live in a
Pygame window (render_mode="human") or as a step-by-step verbose transcript
in the terminal (always printed, regardless of render mode) -- useful both
for the required demonstration video and for headless debugging.

Usage:
    python play.py --algo ppo --model models/ppo/ppo_07.zip
    python play.py --algo reinforce --model models/reinforce/reinforce_04.pt --episodes 3
    python play.py --algo dqn --model models/dqn/dqn_01.zip --no-render
"""

from __future__ import annotations

import argparse

from environment.custom_env import ACTION_NAMES, WasteSegregationEnv
from training.utils import load_agent


def play_episode(env: WasteSegregationEnv, agent, verbose: bool = True) -> float:
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if verbose:
            print(
                f"step {step:4d} | action={ACTION_NAMES[action]:14s} | "
                f"reward={reward:+7.2f} | total={total_reward:+9.2f} | "
                f"bins fill={[round(f, 1) for f in info['bin_fill']]}"
            )

        if env.render_mode is not None:
            env.render()

        done = terminated or truncated

    if verbose:
        print(f"\nEpisode finished in {step} steps. Total reward: {total_reward:.2f}")
        print(f"Episode stats: {info['episode_stats']}")

    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained agent play WasteSegregationEnv.")
    parser.add_argument("--algo", required=True, choices=["dqn", "reinforce", "a2c", "ppo"])
    parser.add_argument("--model", required=True, help="Path to the saved model file.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--no-render", action="store_true", help="Disable the Pygame window.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    render_mode = None if args.no_render else "human"
    env = WasteSegregationEnv(render_mode=render_mode, seed=args.seed)
    agent = load_agent(
        args.algo, args.model,
        obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n,
    )

    rewards = []
    for ep in range(args.episodes):
        print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
        rewards.append(play_episode(env, agent, verbose=True))

    env.close()

    if len(rewards) > 1:
        print(f"\nMean reward over {len(rewards)} episodes: {sum(rewards) / len(rewards):.2f}")


if __name__ == "__main__":
    main()
