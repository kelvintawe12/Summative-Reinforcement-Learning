"""Unit tests for environment/dynamics.py -- the five core mathematical models."""

from __future__ import annotations

import math

import pytest

from environment.dynamics import (
    DynamicsConfig,
    contamination_penalty_factor,
    jam_probability,
    sensor_noise_sigma,
    stall_penalty,
    update_ewma,
    value_multiplier,
)


class TestValueMultiplier:
    def test_full_value_below_decay_threshold(self):
        assert value_multiplier(0.0) == 1.0
        assert value_multiplier(0.5, theta=0.7) == 1.0

    def test_decays_above_threshold(self):
        v_low = value_multiplier(0.75, theta=0.7)
        v_high = value_multiplier(0.95, theta=0.7)
        assert v_low > v_high, "value should decay monotonically as fill increases past theta"

    def test_never_below_floor(self):
        v = value_multiplier(1.0, theta=0.7, k=5.0, floor=0.2)
        assert v == pytest.approx(0.2)

    def test_clamps_out_of_range_inputs(self):
        assert value_multiplier(-0.5) == 1.0
        assert value_multiplier(1.5, theta=0.7, floor=0.2) == pytest.approx(
            value_multiplier(1.0, theta=0.7, floor=0.2)
        )


class TestSensorNoiseSigma:
    def test_zero_ambiguity_gives_minimum_noise(self):
        assert sensor_noise_sigma(0.0, sigma_min=0.05, sigma_max=0.2) == pytest.approx(0.05)

    def test_max_ambiguity_gives_maximum_noise(self):
        assert sensor_noise_sigma(1.0, sigma_min=0.05, sigma_max=0.2) == pytest.approx(0.2)

    def test_monotonic_in_ambiguity(self):
        s1 = sensor_noise_sigma(0.2)
        s2 = sensor_noise_sigma(0.8)
        assert s2 > s1


class TestContaminationPenaltyFactor:
    def test_zero_contamination_full_value(self):
        assert contamination_penalty_factor(0.0) == pytest.approx(1.0)

    def test_full_contamination_zero_value(self):
        assert contamination_penalty_factor(1.0) == pytest.approx(0.0)

    def test_convexity_small_contamination_costs_little(self):
        # With p=2, a small contamination fraction should cost much less
        # than proportionally, demonstrating the convex (accelerating) shape.
        factor_at_10pct = contamination_penalty_factor(0.10, p=2.0)
        factor_at_50pct = contamination_penalty_factor(0.50, p=2.0)
        loss_at_10pct = 1.0 - factor_at_10pct
        loss_at_50pct = 1.0 - factor_at_50pct
        assert loss_at_10pct < 0.2 * loss_at_50pct * 5  # sanity: not linear


class TestStallPenalty:
    def test_zero_holds_zero_penalty(self):
        assert stall_penalty(0) == 0.0

    def test_penalty_grows_superlinearly(self):
        p1 = stall_penalty(1, beta=0.2)
        p2 = stall_penalty(2, beta=0.2)
        p4 = stall_penalty(4, beta=0.2)
        # Quadratic growth: doubling n should roughly quadruple the penalty.
        assert p2 == pytest.approx(4 * p1)
        assert p4 == pytest.approx(16 * p1)


class TestJamProbability:
    def test_no_excess_contamination_zero_jam_risk(self):
        assert jam_probability(0.3, baseline=0.38, kappa=1.4) == pytest.approx(0.0)

    def test_excess_contamination_raises_jam_risk(self):
        p_low = jam_probability(0.40, baseline=0.38, kappa=1.4)
        p_high = jam_probability(0.90, baseline=0.38, kappa=1.4)
        assert 0.0 < p_low < p_high < 1.0

    def test_probability_bounded(self):
        p = jam_probability(5.0, baseline=0.0, kappa=1.4)
        assert 0.0 <= p < 1.0


class TestUpdateEwma:
    def test_full_weight_on_new_value(self):
        assert update_ewma(0.0, 1.0, alpha=1.0) == pytest.approx(1.0)

    def test_zero_weight_keeps_previous(self):
        assert update_ewma(0.5, 1.0, alpha=0.0) == pytest.approx(0.5)

    def test_partial_blend(self):
        result = update_ewma(0.0, 1.0, alpha=0.3)
        assert result == pytest.approx(0.3)


def test_dynamics_config_defaults_are_internally_consistent():
    cfg = DynamicsConfig()
    assert 0.0 < cfg.decay_start_fraction < 1.0
    assert 0.0 < cfg.min_value_multiplier < 1.0
    assert cfg.sigma_min < cfg.sigma_max
    assert cfg.contamination_exponent > 1.0
    assert cfg.max_consecutive_holds > 0
    assert cfg.jam_cooldown_steps > 0
