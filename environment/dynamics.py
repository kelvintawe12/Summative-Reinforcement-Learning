"""
dynamics.py

Implements the five core mathematical dynamics that give the WasteSegregationEnv
its non-trivial, sequential-decision character:

    1. Bin value decay      - value_multiplier()
    2. Sensor noise         - sensor_noise_sigma()
    3. Contamination        - contamination_penalty_factor()
    4. Conveyor stalling    - stall_penalty()
    5. Equipment jamming    - jam_probability()

Every function is pure (no hidden state) so it can be unit tested in isolation
and reused identically by the environment and by the test suite. All default
constants are grounded in the ranges discussed in the project design notes:

    - Organic fraction of municipal solid waste in African cities: 30-61%.
    - Optical/NIR sorting sensors on well-separated single-material items
      achieve >95% classification accuracy; accuracy degrades toward noisy,
      unreliable readings as items become more mixed/contaminated.
    - Real conveyor sorting lines run at roughly 60-100 picks/minute, which
      is the basis for the per-item decision-window budget.

All constants are exposed as function parameters (not hardcoded) so they can
be swept as part of hyperparameter/ablation experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsConfig:
    """Tunable constants for all five dynamics models.

    Grouped in a single dataclass so the environment, training scripts, and
    tests all share one canonical source of truth and so sweeps can vary a
    whole configuration at once.
    """

    # --- 1. Bin value decay ---
    decay_start_fraction: float = 0.70   # theta: fill level where decay begins
    decay_strength: float = 0.80         # k: how far value falls by fill=1.0
    min_value_multiplier: float = 0.20   # floor so value never hits zero

    # --- 2. Sensor noise ---
    sigma_min: float = 0.05              # noise for a clean, unambiguous item
    sigma_max: float = 0.18              # noise for a maximally ambiguous item
    scan_noise_reduction: float = 0.45   # multiplicative reduction from scan_closely

    # --- 3. Contamination penalty ---
    contamination_exponent: float = 2.0  # p: convexity of the penalty curve

    # --- 4. Conveyor stalling ---
    stall_beta: float = 0.18             # beta: quadratic stalling penalty coefficient
    max_consecutive_holds: int = 4       # forced action beyond this limit

    # --- 5. Equipment jamming ---
    # Jam risk is driven by *excess* contamination above an expected baseline
    # (real single-stream recyclables are rarely 100% pure), not by raw
    # contamination itself -- otherwise routinely-expected impurity would
    # jam bins even under a correct, well-run policy.
    jam_baseline_contamination: float = 0.38  # expected contamination under correct routing
    jam_kappa: float = 1.4               # kappa: sensitivity of jam probability to excess contamination
    jam_ewma_alpha: float = 0.30         # smoothing factor for recent-contamination EWMA
    jam_cooldown_steps: int = 6          # steps a jammed bin stays offline


def value_multiplier(
    fill_fraction: float,
    theta: float = DynamicsConfig.decay_start_fraction,
    k: float = DynamicsConfig.decay_strength,
    floor: float = DynamicsConfig.min_value_multiplier,
) -> float:
    """Dynamic value decay for a bin as it approaches capacity.

    Value stays at 1.0 until the bin is `theta` full, then decays linearly,
    bottoming out at `floor` rather than at zero. This avoids a hard cliff at
    100% capacity and instead creates a soft, anticipatable pressure: a
    rational policy starts diverting material to other bins (or holding)
    before a bin is actually full.

    Args:
        fill_fraction: current fill level of the bin, in [0, 1].
        theta: fill fraction at which decay begins.
        k: total fractional drop in value by fill_fraction == 1.0.
        floor: minimum multiplier (never exactly zero).

    Returns:
        A value multiplier in [floor, 1.0].
    """
    f = min(max(fill_fraction, 0.0), 1.0)
    if f < theta:
        return 1.0
    span = max(1.0 - theta, 1e-8)
    raw = 1.0 - k * (f - theta) / span
    return max(raw, floor)


def sensor_noise_sigma(
    ambiguity: float,
    sigma_min: float = DynamicsConfig.sigma_min,
    sigma_max: float = DynamicsConfig.sigma_max,
) -> float:
    """Observation noise standard deviation as a function of item ambiguity.

    Ambiguity is defined as `1 - max(true_composition)`: an item that is
    almost entirely one material has low ambiguity (easy to classify); an
    item with no dominant material has high ambiguity (hard to classify).
    Noise scales linearly between sigma_min and sigma_max as ambiguity rises,
    matching the real-world pattern where optical/NIR sorters are reliable on
    clean single-material items and degrade on mixed/contaminated ones.

    Args:
        ambiguity: 1 - max(composition vector), in [0, 1].
        sigma_min: noise floor for a fully unambiguous item.
        sigma_max: noise ceiling for a maximally ambiguous item.

    Returns:
        Standard deviation to use when perturbing the composition observation.
    """
    a = min(max(ambiguity, 0.0), 1.0)
    return sigma_min + (sigma_max - sigma_min) * a


def contamination_penalty_factor(
    contamination_fraction: float,
    p: float = DynamicsConfig.contamination_exponent,
) -> float:
    """Convex realized-value multiplier for a bin's contamination level.

    Returns (1 - c)^p. Because p > 1, small contamination costs little but
    the penalty accelerates as contamination climbs, so there is no safe
    "just under a threshold" exploit -- there is no threshold, only a
    steepening slope.

    Args:
        contamination_fraction: contaminant mass / total mass in the bin, [0, 1].
        p: convexity exponent, p > 1.

    Returns:
        A multiplier in [0, 1] applied to the bin's base realized value.
    """
    c = min(max(contamination_fraction, 0.0), 1.0)
    return (1.0 - c) ** p


def stall_penalty(
    consecutive_holds: int,
    beta: float = DynamicsConfig.stall_beta,
) -> float:
    """Superlinear penalty for consecutive hold/scan (non-committal) actions.

    Penalty grows with the square of consecutive stalling actions, so an
    occasional single hold (legitimately waiting for a bin to free up) stays
    cheap, while indefinite stalling is always strictly worse than committing
    to an imperfect but fast decision.

    Args:
        consecutive_holds: number of consecutive non-committal actions taken.
        beta: quadratic penalty coefficient.

    Returns:
        A non-negative penalty magnitude (subtract this from reward).
    """
    n = max(consecutive_holds, 0)
    return beta * (n ** 2)


def jam_probability(
    recent_contamination_fraction_ewma: float,
    baseline: float = DynamicsConfig.jam_baseline_contamination,
    kappa: float = DynamicsConfig.jam_kappa,
) -> float:
    """Probability that a bin's sorting mechanism jams on this step.

    Modeled as 1 - exp(-kappa * excess), where `excess` is how far an
    exponentially-weighted moving average of per-item contamination
    *fraction* routed into the bin sits above `baseline` -- the level of
    contamination expected even under correct, well-run sorting (real
    single-stream recyclables are rarely 100% pure). Only *excess*
    contamination, caused by genuinely poor routing decisions, drives jam
    risk; routine, expected impurity does not.

    This makes jamming a consequence of the agent's own recent decisions (a
    causal, history-dependent hazard) rather than an exogenous random event,
    which is what gives the environment genuine temporal credit assignment:
    mistakes several steps ago can take a bin offline now.

    Args:
        recent_contamination_fraction_ewma: smoothed recent per-item
            contamination fraction (contaminant mass / item mass) routed
            into the bin, in [0, 1].
        baseline: contamination fraction expected under correct routing.
        kappa: sensitivity constant; higher kappa means bins jam more readily
            once contamination exceeds baseline.

    Returns:
        Probability in [0, 1) that the bin jams this step.
    """
    excess = max(recent_contamination_fraction_ewma - baseline, 0.0)
    return 1.0 - math.exp(-kappa * excess)


def update_ewma(previous: float, new_value: float, alpha: float = DynamicsConfig.jam_ewma_alpha) -> float:
    """Exponentially-weighted moving average update.

    Used to track each bin's recent contamination history for the jamming
    model: recent decisions matter more than old ones, and old contamination
    "decays out" over time, acting as a proxy for periodic maintenance.

    Args:
        previous: previous EWMA value.
        new_value: newest observation to fold in.
        alpha: smoothing factor in (0, 1]; higher alpha weights recent data more.

    Returns:
        Updated EWMA value.
    """
    return alpha * new_value + (1.0 - alpha) * previous
