"""
Unit Tests for MarketEnvironment.

Tests verify the "physics" of the simulation before running agents:
1. Fat Tails: Student-t (ν=3) produces excess kurtosis > 3
2. Regime Switching: Sticky regimes produce expected run lengths
3. Dimensions: Output shapes match specifications

Reference: theory/model_definition.tex (Section 3: The Environment)
"""

import numpy as np
import pytest
from scipy import stats

import sys
sys.path.insert(0, '..')

from simulation.environment import (
    MarketEnvironment,
    create_bull_bear_environment,
    create_single_regime_environment,
    SimulationResult,
)
from simulation.distributions import (
    sample_markov_chain,
    sample_student_t,
    sample_gaussian,
    generate_random_samples,
)


class TestFatTails:
    """
    Test 1: Fat Tails Verification
    
    Generate 10^6 samples from Student-t (ν=3) and verify that the
    empirical kurtosis is significantly higher than the Normal (kurtosis=3).
    
    For Student-t with ν=3, the theoretical excess kurtosis is undefined
    (infinite 4th moment), but empirical samples show very high kurtosis.
    """
    
    def test_student_t_excess_kurtosis(self):
        """Student-t (ν=3) should have empirical kurtosis >> 3 (Normal)."""
        # Configuration: Single regime with Student-t, ν=3
        config = {
            'crisis': {'mu': 0.0, 'sigma': 0.20, 'df': 3}
        }
        trans_matrix = np.array([[1.0]])  # Single regime, always stay
        
        env = MarketEnvironment(config, trans_matrix, seed=42)
        
        # Generate 10^6 samples (1000 sims × 1000 steps)
        n_sims = 1000
        t_steps = 1000
        result = env.simulate_batch(n_sims=n_sims, t_steps=t_steps)
        
        # Flatten to get all samples
        all_returns = result.returns.flatten()
        
        # Compute empirical kurtosis (Fisher's definition, excess kurtosis)
        empirical_kurtosis = stats.kurtosis(all_returns, fisher=True)
        
        # Normal distribution has excess kurtosis = 0 (kurtosis = 3)
        # Student-t (ν=3) should have much higher excess kurtosis
        print(f"Empirical excess kurtosis: {empirical_kurtosis:.2f}")
        print(f"Expected for Normal: 0.0")
        
        # Assert excess kurtosis is significantly greater than 0 (i.e., kurtosis > 3)
        # For ν=3 Student-t, we expect very high values (typically 10-50+)
        assert empirical_kurtosis > 3.0, (
            f"Student-t (ν=3) should have excess kurtosis >> 0, "
            f"got {empirical_kurtosis:.2f}"
        )
    
    def test_gaussian_kurtosis_near_normal(self):
        """Gaussian regime should have kurtosis ≈ 3 (excess kurtosis ≈ 0)."""
        config = {
            'normal': {'mu': 0.0, 'sigma': 0.20}  # No df = Gaussian
        }
        trans_matrix = np.array([[1.0]])
        
        env = MarketEnvironment(config, trans_matrix, seed=42)
        
        result = env.simulate_batch(n_sims=1000, t_steps=1000)
        all_returns = result.returns.flatten()
        
        empirical_kurtosis = stats.kurtosis(all_returns, fisher=True)
        
        print(f"Gaussian empirical excess kurtosis: {empirical_kurtosis:.3f}")
        
        # For Gaussian, excess kurtosis should be close to 0
        assert abs(empirical_kurtosis) < 0.2, (
            f"Gaussian should have excess kurtosis ≈ 0, got {empirical_kurtosis:.3f}"
        )


class TestRegimeSwitching:
    """
    Test 2: Regime Switching Verification
    
    Initialize an HMM with a "sticky" regime (P_11 = 0.99) and verify that
    the generated regime path stays in state 1 for long durations.
    
    Expected average run length = 1 / (1 - P_11) = 1 / 0.01 = 100 steps
    """
    
    def test_sticky_regime_run_length(self):
        """Sticky regime (P_11 = 0.99) should have average run length ≈ 100."""
        p_stay = 0.99
        
        config = {
            'regime_0': {'mu': 0.05, 'sigma': 0.10},
            'regime_1': {'mu': -0.05, 'sigma': 0.20}
        }
        
        # Sticky transition matrix: mostly stay in current state
        trans_matrix = np.array([
            [p_stay, 1 - p_stay],  # State 0: 99% stay, 1% switch
            [1 - p_stay, p_stay]   # State 1: 1% switch, 99% stay
        ])
        
        env = MarketEnvironment(config, trans_matrix, initial_regime=0, seed=123)
        
        # Generate many paths to estimate run length
        n_sims = 100
        t_steps = 10000
        result = env.simulate_batch(n_sims=n_sims, t_steps=t_steps)
        
        # Calculate average run length (consecutive steps in same state)
        run_lengths = []
        for i in range(n_sims):
            path = result.regimes[i, :]
            current_run = 1
            for t in range(1, t_steps):
                if path[t] == path[t-1]:
                    current_run += 1
                else:
                    run_lengths.append(current_run)
                    current_run = 1
            run_lengths.append(current_run)  # Don't forget last run
        
        avg_run_length = np.mean(run_lengths)
        expected_run_length = 1 / (1 - p_stay)  # = 100
        
        print(f"Average run length: {avg_run_length:.1f}")
        print(f"Expected (theoretical): {expected_run_length:.1f}")
        
        # Allow 20% tolerance due to random sampling
        assert abs(avg_run_length - expected_run_length) / expected_run_length < 0.20, (
            f"Average run length {avg_run_length:.1f} should be close to "
            f"expected {expected_run_length:.1f}"
        )
    
    def test_transition_probabilities_empirical(self):
        """Verify empirical transition probabilities match the matrix."""
        p_bull_to_bear = 0.10  # 10% chance to switch
        
        config = {
            'bull': {'mu': 0.05, 'sigma': 0.10},
            'bear': {'mu': -0.05, 'sigma': 0.20}
        }
        
        trans_matrix = np.array([
            [1 - p_bull_to_bear, p_bull_to_bear],
            [p_bull_to_bear, 1 - p_bull_to_bear]
        ])
        
        env = MarketEnvironment(config, trans_matrix, initial_regime=0, seed=456)
        
        result = env.simulate_batch(n_sims=100, t_steps=10000)
        
        # Count transitions from state 0 to state 1
        transitions_0_to_1 = 0
        total_from_0 = 0
        
        for i in range(result.regimes.shape[0]):
            for t in range(result.regimes.shape[1] - 1):
                if result.regimes[i, t] == 0:
                    total_from_0 += 1
                    if result.regimes[i, t + 1] == 1:
                        transitions_0_to_1 += 1
        
        empirical_p_0_to_1 = transitions_0_to_1 / total_from_0 if total_from_0 > 0 else 0
        
        print(f"Empirical P(0→1): {empirical_p_0_to_1:.4f}")
        print(f"Expected P(0→1): {p_bull_to_bear:.4f}")
        
        # Allow 10% relative tolerance
        assert abs(empirical_p_0_to_1 - p_bull_to_bear) / p_bull_to_bear < 0.10


class TestDimensions:
    """
    Test 3: Output Dimension Verification
    
    Ensure simulate_batch(n_sims, t_steps) returns matrices of exactly
    the specified shape (n_sims, t_steps).
    """
    
    def test_output_dimensions_exact(self):
        """simulate_batch(100, 50) should return exactly (100, 50) matrices."""
        env = create_bull_bear_environment(seed=789)
        
        n_sims = 100
        t_steps = 50
        
        result = env.simulate_batch(n_sims=n_sims, t_steps=t_steps)
        
        assert result.returns.shape == (n_sims, t_steps), (
            f"Returns shape {result.returns.shape} != expected ({n_sims}, {t_steps})"
        )
        
        assert result.regimes.shape == (n_sims, t_steps), (
            f"Regimes shape {result.regimes.shape} != expected ({n_sims}, {t_steps})"
        )
    
    def test_various_dimensions(self):
        """Test multiple dimension combinations."""
        env = create_single_regime_environment(seed=101)
        
        test_cases = [
            (1, 1),
            (10, 100),
            (1000, 10),
            (500, 500),
        ]
        
        for n_sims, t_steps in test_cases:
            result = env.simulate_batch(n_sims=n_sims, t_steps=t_steps)
            
            assert result.returns.shape == (n_sims, t_steps), (
                f"Shape mismatch for ({n_sims}, {t_steps})"
            )
    
    def test_output_types(self):
        """Verify output types are correct."""
        env = create_bull_bear_environment(seed=202)
        result = env.simulate_batch(n_sims=10, t_steps=20)
        
        assert isinstance(result, SimulationResult)
        assert isinstance(result.returns, np.ndarray)
        assert isinstance(result.regimes, np.ndarray)
        assert result.returns.dtype == np.float64
        assert result.regimes.dtype == np.int32


class TestReproducibility:
    """
    Test reproducibility via seeding.
    
    The same seed should produce identical results.
    """
    
    def test_same_seed_same_results(self):
        """Identical seeds should produce identical returns."""
        config = {
            'bull': {'mu': 0.05, 'sigma': 0.10},
            'bear': {'mu': -0.05, 'sigma': 0.20, 'df': 3}
        }
        trans_matrix = np.array([[0.95, 0.05], [0.10, 0.90]])
        
        env1 = MarketEnvironment(config, trans_matrix, seed=42)
        env2 = MarketEnvironment(config, trans_matrix, seed=42)
        
        result1 = env1.simulate_batch(n_sims=50, t_steps=100)
        result2 = env2.simulate_batch(n_sims=50, t_steps=100)
        
        np.testing.assert_array_equal(result1.returns, result2.returns)
        np.testing.assert_array_equal(result1.regimes, result2.regimes)
    
    def test_different_seeds_different_results(self):
        """Different seeds should produce different returns."""
        env1 = create_bull_bear_environment(seed=1)
        env2 = create_bull_bear_environment(seed=2)
        
        result1 = env1.simulate_batch(n_sims=10, t_steps=100)
        result2 = env2.simulate_batch(n_sims=10, t_steps=100)
        
        # Results should be different
        assert not np.allclose(result1.returns, result2.returns)


class TestValidation:
    """
    Test input validation and error handling.
    """
    
    def test_invalid_transition_matrix_rows(self):
        """Transition matrix rows must sum to 1."""
        config = {'regime': {'mu': 0.0, 'sigma': 0.1}}
        
        # Shape (1, 1) but sums to 0.9, not 1
        bad_trans_matrix = np.array([[0.9]])
        
        with pytest.raises(ValueError, match="rows must sum to 1"):
            MarketEnvironment(config, bad_trans_matrix)
    
    def test_invalid_transition_matrix_shape(self):
        """Transition matrix dimensions must match number of regimes."""
        config = {
            'regime_0': {'mu': 0.0, 'sigma': 0.1},
            'regime_1': {'mu': 0.0, 'sigma': 0.1}
        }
        
        # 3x3 matrix for 2 regimes - mismatch!
        bad_trans_matrix = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8]
        ])
        
        with pytest.raises(ValueError, match="does not match"):
            MarketEnvironment(config, bad_trans_matrix)
    
    def test_negative_transition_probabilities(self):
        """Transition probabilities cannot be negative."""
        config = {'regime': {'mu': 0.0, 'sigma': 0.1}}
        
        # Shape (1, 1), sums to 1, but has negative entry
        # This is impossible for a single regime, so use 2 regimes
        config2 = {
            'regime_0': {'mu': 0.0, 'sigma': 0.1},
            'regime_1': {'mu': 0.0, 'sigma': 0.1}
        }
        bad_trans_matrix = np.array([[-0.1, 1.1], [0.5, 0.5]])
        
        with pytest.raises(ValueError, match="negative"):
            MarketEnvironment(config2, bad_trans_matrix)


class TestMeanVariance:
    """
    Statistical verification that generated returns match specified parameters.
    """
    
    def test_mean_returns_match_mu(self):
        """Generated returns should have mean close to μ."""
        mu = 0.08
        sigma = 0.15
        
        env = create_single_regime_environment(mu=mu, sigma=sigma, seed=999)
        result = env.simulate_batch(n_sims=1000, t_steps=1000)
        
        empirical_mean = result.returns.mean()
        
        print(f"Empirical mean: {empirical_mean:.4f}")
        print(f"Expected mean (μ): {mu:.4f}")
        
        # Allow 1% absolute tolerance for mean
        assert abs(empirical_mean - mu) < 0.01, (
            f"Mean {empirical_mean:.4f} should be close to μ={mu:.4f}"
        )
    
    def test_std_returns_match_sigma(self):
        """Generated returns should have std close to σ."""
        mu = 0.0
        sigma = 0.20
        
        env = create_single_regime_environment(mu=mu, sigma=sigma, seed=888)
        result = env.simulate_batch(n_sims=1000, t_steps=1000)
        
        empirical_std = result.returns.std()
        
        print(f"Empirical std: {empirical_std:.4f}")
        print(f"Expected std (σ): {sigma:.4f}")
        
        # Allow 5% relative tolerance for std
        assert abs(empirical_std - sigma) / sigma < 0.05, (
            f"Std {empirical_std:.4f} should be close to σ={sigma:.4f}"
        )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
