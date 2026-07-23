"""
diagnostics.py

Behavioral diagnostics for WasteSegregationEnv policies. The question this
answers: does a policy *collapse into a rut* (spam one or two actions
regardless of the incoming item), or does it use the full action set in
response to what it sees?

For each policy we roll out several episodes and report:
    - action histogram (fraction of steps spending each of the 7 actions),
    - action entropy in bits, normalized to [0, 1] (1 = uniform over 7
      actions, 0 = always the same action),
    - the longest run of an identical action, and the fraction of steps that
      simply repeat the previous action,
    - a crude "input-responsiveness" check: of the steps where the agent
      sorted, how often it sorted into the bin matching the item's true
      dominant material.

This is the evidence used to decide whether an exploration mechanism is even
needed: a healthy policy should show moderate-to-high entropy and high
input-responsiveness. A collapsed policy shows near-zero entropy and low
responsiveness -- and the fix for that is the exploration hyperparameters
(entropy_coef / exploration_fraction), not a history-based reward hack.

Usage:
    python -m results.diagnostics                       # random + heuristic baselines
    python -m results.diagnostics --algo ppo --model models/ppo/ppo_07.zip
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from environment.custom_env import (
    ACTION_HOLD,
    ACTION_NAMES,
    ACTION_REJECT,
    N_ACTIONS,
    WasteSegregationEnv,
)


def _action_entropy_bits(counts: np.ndarray) -> tuple[float, float]:
    """Return (entropy_bits, normalized_entropy in [0,1])."""
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0
    p = counts / total
    nz = p[p > 0]
    h = float(-(nz * np.log2(nz)).sum())
    return h, h / math.log2(N_ACTIONS)


def rollout_stats(policy_fn, n_episodes: int = 10, seed0: int = 7000) -> dict:
    counts = np.zeros(N_ACTIONS, dtype=np.int64)
    total_steps = 0
    repeats = 0
    longest_run = 0
    correct_sorts = 0
    total_sorts = 0
    episode_rewards = []

    for ep in range(n_episodes):
        env = WasteSegregationEnv(seed=seed0 + ep)
        obs, _ = env.reset(seed=seed0 + ep)
        done, total = False, 0.0
        prev_action = None
        cur_run = 0

        while not done:
            action = int(policy_fn(env, obs))
            counts[action] += 1
            total_steps += 1

            if prev_action is not None and action == prev_action:
                repeats += 1
                cur_run += 1
            else:
                cur_run = 1
            longest_run = max(longest_run, cur_run)

            # Input-responsiveness: did a sort go to the item's true best bin?
            if action < 4:
                total_sorts += 1
                true_best = int(np.argmax(env.current_item_true[:4]))
                if action == true_best:
                    correct_sorts += 1

            prev_action = action
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated

        env.close()
        episode_rewards.append(total)

    h_bits, h_norm = _action_entropy_bits(counts)
    return {
        "episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "action_fractions": {ACTION_NAMES[i]: round(counts[i] / max(total_steps, 1), 3)
                             for i in range(N_ACTIONS)},
        "action_entropy_bits": round(h_bits, 3),
        "action_entropy_normalized": round(h_norm, 3),
        "repeat_prev_action_fraction": round(repeats / max(total_steps - 1, 1), 3),
        "longest_identical_run": longest_run,
        "sort_accuracy": round(correct_sorts / total_sorts, 3) if total_sorts else None,
    }


def capacity_aware_policy(env, obs) -> int:
    """The same heuristic used in tests: route to the best-matching, non-full,
    non-jammed bin; reject clear contaminants; otherwise hold."""
    true = env.current_item_true
    order = np.argsort(-true[:4])
    for idx in order:
        if true[idx] < 0.15:
            break
        fill_frac = env.bin_fill[idx] / env.bin_capacity
        if env.bin_jam_cooldown[idx] == 0 and fill_frac < 0.80:
            return int(idx)
    if true[4] >= 0.4:
        return ACTION_REJECT
    if env.consecutive_holds < env.cfg.max_consecutive_holds:
        return ACTION_HOLD
    return ACTION_REJECT


def _print_report(name: str, stats: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"  mean reward                : {stats['mean_reward']:.2f} "
          f"over {stats['episodes']} episodes")
    print(f"  action entropy (norm 0-1)  : {stats['action_entropy_normalized']} "
          f"({stats['action_entropy_bits']} bits)")
    print(f"  repeats prev action        : {stats['repeat_prev_action_fraction']}")
    print(f"  longest identical run      : {stats['longest_identical_run']}")
    print(f"  sort accuracy (true bin)   : {stats['sort_accuracy']}")
    print(f"  action distribution:")
    for a, f in stats["action_fractions"].items():
        bar = "#" * int(f * 40)
        print(f"    {a:14s} {f:5.3f} {bar}")
    # Interpretation hint.
    hn = stats["action_entropy_normalized"]
    if hn < 0.15:
        print("  >> COLLAPSED: near-single-action policy (a rut).")
    elif hn < 0.4:
        print("  >> LOW diversity: check exploration hyperparameters.")
    else:
        print("  >> Healthy action diversity.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavioral rut diagnostics for a policy.")
    parser.add_argument("--algo", choices=["dqn", "reinforce", "a2c", "ppo"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    if args.algo and args.model:
        from training.utils import load_agent
        agent = load_agent(args.algo, args.model)

        def trained_policy(env, obs):
            action, _ = agent.predict(obs, deterministic=True)
            return int(action)

        _print_report(f"trained {args.algo} ({args.model})",
                      rollout_stats(trained_policy, args.episodes))
    else:
        print("No --algo/--model given; reporting baselines (random + heuristic).")
        _print_report("random policy",
                      rollout_stats(lambda env, obs: env.action_space.sample(), args.episodes))
        _print_report("capacity-aware heuristic",
                      rollout_stats(capacity_aware_policy, args.episodes))


if __name__ == "__main__":
    main()
