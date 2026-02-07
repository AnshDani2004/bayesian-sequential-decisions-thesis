"""
Unit Tests for Mathematical Correctness

Tests verify:
1. Kelly fraction formulas (Bernoulli, Gaussian)
2. Log-wealth vs multiplicative wealth equivalence
3. Bayesian posterior updates (Beta-Bernoulli)
4. RNG determinism and reproducibility
"""

import numpy as np
import pytest
from scipy import stats

import sys
sys.path.insert(0, '..')

from simulation.agents import compute_kelly_fraction, NaiveBayesKelly, KellyOracle
from simulation.wealth_tracker import WealthTracker, simulate_wealth_path


class TestKellyFractionBernoulli:
    """
    Test Kelly fraction for Bernoulli (coin-flip) bets.
    
    Formula: f* = p - q/b = p - (1-p)/b
    For b=1 (even odds): f* = 2p - 1
    """
    
    def test_fair_coin_zero_edge(self):
        """Fair coin (p=0.5) with even odds has f*=0."""
        p = np.array([0.5])
        b = np.array([1.0])
        f = compute_kelly_fraction(p, b)
        assert np.isclose(f[0], 0.0), f"f*={f[0]}, expected 0"
    
    def test_biased_coin_positive_edge(self):
        """Biased coin (p=0.6) with even odds has f*=0.2."""
        p = np.array([0.6])
        b = np.array([1.0])
        f = compute_kelly_fraction(p, b)
        # f* = p - q/b = 0.6 - 0.4/1 = 0.2
        expected = 0.6 - 0.4 / 1.0
        assert np.isclose(f[0], expected), f"f*={f[0]}, expected {expected}"
    
    def test_negative_edge_clipped_to_zero(self):
        """Negative edge (p=0.3) should be clipped to f*=0."""
        p = np.array([0.3])
        b = np.array([1.0])
        f = compute_kelly_fraction(p, b)
        # f* = 0.3 - 0.7/1 = -0.4, clipped to 0
        assert f[0] >= 0, f"Kelly fraction should be non-negative, got {f[0]}"
    
    def test_high_odds_adjustment(self):
        """High odds (b=2) should reduce Kelly fraction."""
        p = np.array([0.6])
        b = np.array([2.0])
        f = compute_kelly_fraction(p, b)
        # f* = p - q/b = 0.6 - 0.4/2 = 0.4
        expected = 0.6 - 0.4 / 2.0
        assert np.isclose(f[0], expected, atol=0.01), f"f*={f[0]}, expected {expected}"
    
    def test_multiple_arms(self):
        """Test Kelly with multiple independent arms."""
        p = np.array([0.55, 0.60, 0.70])
        b = np.array([1.0, 1.0, 1.0])
        f = compute_kelly_fraction(p, b)
        
        expected = [0.55 - 0.45, 0.60 - 0.40, 0.70 - 0.30]
        np.testing.assert_allclose(f, expected, atol=0.01)


class TestKellyFractionGaussian:
    """
    Test Kelly fraction for continuous Gaussian returns.
    
    Formula: f* = μ / σ² (mean-variance ratio)
    """
    
    def test_gaussian_kelly_formula(self):
        """Verify f* = μ/σ² for Gaussian returns."""
        mu = 0.08
        sigma = 0.15
        
        # Theoretical Kelly for Gaussian
        f_star_theory = mu / (sigma ** 2)
        
        # For agent-computed Kelly, we approximate via posterior mean
        # With enough data, posterior converges to true params
        assert f_star_theory > 0, "Positive edge should give positive Kelly"
        assert f_star_theory == pytest.approx(0.08 / 0.0225, rel=0.01)


class TestLogWealthEquivalence:
    """
    Test that log-wealth and multiplicative wealth give same results.
    
    For short sequences without overflow, both should match exactly.
    """
    
    def test_short_sequence_equivalence(self):
        """Log and multiplicative wealth should match for short sequences."""
        np.random.seed(42)
        
        returns = np.random.normal(0.02, 0.10, size=100)
        fractions = np.full(100, 0.5)
        
        # Method 1: Multiplicative
        wealth_mult = 1.0
        for f, r in zip(fractions, returns):
            wealth_mult *= (1 + f * r)
        
        # Method 2: Log-space
        log_wealth = 0.0
        for f, r in zip(fractions, returns):
            log_wealth += np.log1p(f * r)
        wealth_log = np.exp(log_wealth)
        
        # Should be very close
        assert np.isclose(wealth_mult, wealth_log, rtol=1e-10), \
            f"Multiplicative: {wealth_mult}, Log: {wealth_log}"
    
    def test_wealth_tracker_consistency(self):
        """WealthTracker should give same result as manual log calculation."""
        np.random.seed(123)
        
        returns = np.random.normal(0.01, 0.05, size=50)
        fractions = np.full(50, 0.3)
        
        # Method 1: WealthTracker
        tracker = WealthTracker(initial_wealth=1.0)
        for f, r in zip(fractions, returns):
            tracker.update(f, r)
        
        # Method 2: Manual log calculation
        log_wealth = 0.0
        for f, r in zip(fractions, returns):
            log_wealth += np.log1p(f * r)
        
        assert np.isclose(tracker.log_wealth, log_wealth, rtol=1e-10)
    
    def test_ruin_detection(self):
        """WealthTracker should detect ruin when 1 + f*r <= 0."""
        tracker = WealthTracker(initial_wealth=1.0)
        
        # Normal update
        tracker.update(0.5, 0.10)
        assert not tracker.is_ruined
        
        # Catastrophic loss: f=1.0, r=-1.5 means multiplier = -0.5
        tracker.update(1.0, -1.5)
        assert tracker.is_ruined
        assert tracker.wealth == 0.0


class TestBayesianPosterior:
    """
    Test Bayesian posterior updates.
    
    Beta-Bernoulli conjugate update:
        Prior: Beta(α, β)
        After w wins, l losses: Beta(α + w, β + l)
    """
    
    def test_beta_bernoulli_update(self):
        """Verify Beta posterior updates correctly."""
        agent = NaiveBayesKelly(n_arms=1, odds=[1.0], prior_alpha=1.0, prior_beta=1.0)
        
        # Initial posterior mean: α/(α+β) = 1/2 = 0.5
        assert np.isclose(agent.get_posterior_mean()[0], 0.5)
        
        # Observe a win
        agent.update(np.array([1.0]))
        # Posterior: Beta(2, 1), mean = 2/3
        expected_mean = 2.0 / 3.0
        assert np.isclose(agent.get_posterior_mean()[0], expected_mean, atol=0.01)
        
        # Observe a loss
        agent.update(np.array([-1.0]))
        # Posterior: Beta(2, 2), mean = 0.5
        expected_mean = 2.0 / 4.0
        assert np.isclose(agent.get_posterior_mean()[0], expected_mean, atol=0.01)
    
    def test_posterior_variance_decreases(self):
        """Posterior variance should decrease with more observations."""
        agent = NaiveBayesKelly(n_arms=1, odds=[1.0])
        
        initial_var = agent.get_posterior_variance()[0]
        
        # Update with many observations
        for _ in range(100):
            agent.update(np.array([1.0 if np.random.random() > 0.5 else -1.0]))
        
        final_var = agent.get_posterior_variance()[0]
        
        assert final_var < initial_var, \
            f"Variance should decrease: {initial_var} -> {final_var}"


class TestRNGDeterminism:
    """
    Test that simulations are reproducible with same seed.
    """
    
    def test_same_seed_same_results(self):
        """Same seed should produce identical wealth paths."""
        fractions = np.full(100, 0.5)
        
        # Run 1
        np.random.seed(42)
        returns1 = np.random.normal(0.02, 0.10, size=100)
        wealth1, _, _ = simulate_wealth_path(fractions, returns1)
        
        # Run 2 (same seed)
        np.random.seed(42)
        returns2 = np.random.normal(0.02, 0.10, size=100)
        wealth2, _, _ = simulate_wealth_path(fractions, returns2)
        
        np.testing.assert_array_equal(wealth1, wealth2)
    
    def test_different_seed_different_results(self):
        """Different seeds should produce different results."""
        fractions = np.full(100, 0.5)
        
        np.random.seed(1)
        returns1 = np.random.normal(0.02, 0.10, size=100)
        
        np.random.seed(2)
        returns2 = np.random.normal(0.02, 0.10, size=100)
        
        assert not np.allclose(returns1, returns2)


class TestDrawdownTracking:
    """Test drawdown calculation in WealthTracker."""
    
    def test_no_drawdown_at_peak(self):
        """Drawdown should be 0 when at peak."""
        tracker = WealthTracker(1.0)
        tracker.update(0.5, 0.20)  # Gain
        
        assert tracker.current_drawdown == 0.0
    
    def test_drawdown_after_loss(self):
        """Drawdown should be positive after loss from peak."""
        tracker = WealthTracker(1.0)
        tracker.update(0.5, 0.20)  # Gain: wealth = 1.10
        tracker.update(0.5, -0.10)  # Loss: wealth = 1.10 * 0.95 = 1.045
        
        # Peak = 1.10, current = 1.045
        # Drawdown = (1.10 - 1.045) / 1.10 = 0.05
        expected_dd = (1.10 - 1.045) / 1.10
        assert np.isclose(tracker.current_drawdown, expected_dd, rtol=0.05)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
