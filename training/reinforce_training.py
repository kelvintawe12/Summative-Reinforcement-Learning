"""
reinforce_training.py

A from-scratch implementation of REINFORCE (Monte Carlo policy gradient),
since Stable-Baselines3 does not provide one. Supports an optional learned
state-value baseline (variance reduction) and an optional entropy bonus
(exploration regularization), both of which are swept in
`training/hyperparameters.py::REINFORCE_PRESETS`.

Algorithm (per episode):
    1. Roll out one full episode under the current stochastic policy.
    2. Compute discounted returns G_t = sum_{k>=t} gamma^(k-t) * r_k.
    3. If a baseline is used, compute advantages A_t = G_t - V(s_t) and fit
       V towards G_t via MSE; otherwise A_t = G_t (normalized).
    4. Policy loss = -mean( log pi(a_t | s_t) * A_t ) - entropy_coef * H(pi).
    5. Backpropagate and step the optimizer(s).

This file can be run directly for local/CLI training:
    python -m training.reinforce_training --run all
    python -m training.reinforce_training --run reinforce_03
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment.custom_env import WasteSegregationEnv
from training.hyperparameters import REINFORCE_PRESETS
from training.utils import LOG_COLUMNS, ensure_dir, make_env

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    """Two-hidden-layer MLP mapping observations to an action distribution."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # logits


class ValueNetwork(nn.Module):
    """Two-hidden-layer MLP baseline estimating state value V(s)."""

    def __init__(self, obs_dim: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ReinforceAgent:
    """Monte Carlo policy-gradient (REINFORCE) agent with optional baseline.

    Exposes a `.predict(obs)` method with the same call signature as a
    Stable-Baselines3 model's `.predict()`, so play.py and main.py can treat
    all four trained agents uniformly.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_size: int = 64,
        baseline: bool = True,
        entropy_coef: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.gamma = gamma
        self.use_baseline = baseline
        self.entropy_coef = entropy_coef

        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_size).to(DEVICE)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        if self.use_baseline:
            self.value_fn = ValueNetwork(obs_dim, hidden_size).to(DEVICE)
            self.value_optim = torch.optim.Adam(self.value_fn.parameters(), lr=learning_rate)
        else:
            self.value_fn = None
            self.value_optim = None

    def select_action(self, obs: np.ndarray, greedy: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return int(action.item()), dist.log_prob(action).squeeze(0), dist.entropy().squeeze(0)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """SB3-compatible inference API: returns (action, state=None)."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits = self.policy(obs_t)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())
        return action, None

    def update(self, log_probs, entropies, rewards, observations):
        returns = self._discounted_returns(rewards)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        if self.use_baseline:
            obs_t = torch.as_tensor(np.array(observations), dtype=torch.float32, device=DEVICE)
            values = self.value_fn(obs_t)
            advantages = returns_t - values.detach()
            value_loss = F.mse_loss(values, returns_t)
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()
        else:
            advantages = returns_t

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mean_entropy = entropies_t.mean()
        policy_loss = -(log_probs_t * advantages).mean() - self.entropy_coef * mean_entropy
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        value_loss_val = float(value_loss.item()) if self.use_baseline else ""
        return float(policy_loss.item()), float(mean_entropy.item()), value_loss_val

    def _discounted_returns(self, rewards: list[float]) -> list[float]:
        returns = [0.0] * len(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            returns[t] = running
        return returns

    def save(self, path: str) -> None:
        payload = {"policy_state_dict": self.policy.state_dict()}
        if self.use_baseline:
            payload["value_state_dict"] = self.value_fn.state_dict()
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=DEVICE)
        self.policy.load_state_dict(payload["policy_state_dict"])
        if self.use_baseline and "value_state_dict" in payload:
            self.value_fn.load_state_dict(payload["value_state_dict"])


def train_single_run(preset: dict, log_dir: str, model_dir: str, seed: int = 0) -> str:
    """Train one REINFORCE configuration end to end and persist model + logs.

    Returns the path to the saved model checkpoint.
    """
    ensure_dir(log_dir)
    ensure_dir(model_dir)

    env = make_env(seed=seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ReinforceAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        learning_rate=preset["learning_rate"],
        gamma=preset["gamma"],
        hidden_size=preset["hidden_size"],
        baseline=preset["baseline"],
        entropy_coef=preset["entropy_coef"],
        seed=seed,
    )

    csv_path = os.path.join(log_dir, f"{preset['run_id']}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(LOG_COLUMNS)

    total_timesteps = preset["total_timesteps"]
    timestep = 0
    episode = 0

    while timestep < total_timesteps:
        obs, _ = env.reset(seed=seed + episode)
        done = False
        log_probs, entropies, rewards, observations = [], [], [], []

        while not done:
            action, log_prob, entropy = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            observations.append(obs)
            obs = next_obs
            done = terminated or truncated
            timestep += 1

        policy_loss, mean_entropy, value_loss = agent.update(
            log_probs, entropies, rewards, observations
        )
        episode += 1

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                timestep, episode, sum(rewards), len(rewards),
                policy_loss, mean_entropy, value_loss,
            ])

        if timestep >= total_timesteps:
            break

    model_path = os.path.join(model_dir, f"{preset['run_id']}.pt")
    agent.save(model_path)
    env.close()
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train REINFORCE on WasteSegregationEnv.")
    parser.add_argument(
        "--run", default="all",
        help="'all' to run every preset, or a specific run_id (e.g. reinforce_03).",
    )
    parser.add_argument("--log-dir", default="logs/reinforce")
    parser.add_argument("--model-dir", default="models/reinforce")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    presets = REINFORCE_PRESETS if args.run == "all" else [
        p for p in REINFORCE_PRESETS if p["run_id"] == args.run
    ]
    if not presets:
        raise ValueError(f"No preset found with run_id '{args.run}'")

    for preset in presets:
        print(f"Training {preset['run_id']} ...")
        path = train_single_run(preset, args.log_dir, args.model_dir, seed=args.seed)
        print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
