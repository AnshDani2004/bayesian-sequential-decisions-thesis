"""
Numba-Optimized Distribution Samplers for Market Simulation.

This module implements the core random number generation functions optimized
with Numba JIT compilation for high-performance simulation of 10^5+ paths.

Reference: theory/model_definition.tex (Section 3: The Environment)

Mathematical Notation:
    - S_t: Hidden regime state at time t
    - r_t: Return at time t
    - P: Transition matrix where P[i,j] = P(S_{t+1} = j | S_t = i)
    - θ_s = (μ_s, σ_s, ν_s): Parameters for regime s
"""

import numpy as np
import numba
from numba import njit, prange
from typing import Tuple


# =============================================================================
# Task 1: Core Generators (Numba-optimized)
# =============================================================================

@njit(cache=True)
def sample_markov_chain(
    n_steps: int,
    trans_matrix: np.ndarray,
    initial_state: int,
    random_state: np.ndarray
) -> np.ndarray:
    """
    Generate a sequence of hidden states S_1, ..., S_T from a Markov chain.
    
    This implements the hidden state process from model_definition.tex:
        P_{ij} = P(S_{t+1} = j | S_t = i)
    
    Args:
        n_steps: Number of time steps T
        trans_matrix: Transition matrix P of shape (M, M) where M is # regimes
        initial_state: Starting state S_0 (0-indexed)
        random_state: Pre-generated uniform random numbers of shape (n_steps,)
    
    Returns:
        states: Array of hidden states [S_1, S_2, ..., S_T] of shape (n_steps,)
                Each S_t ∈ {0, 1, ..., M-1}
    
    Note:
        We use pre-generated random numbers for reproducibility and to enable
        Common Random Numbers (CRN) methodology.
    """
    n_regimes = trans_matrix.shape[0]
    states = np.empty(n_steps, dtype=np.int32)
    
    current_state = initial_state
    
    for t in range(n_steps):
        # Get transition probabilities from current state
        probs = trans_matrix[current_state, :]
        
        # Cumulative sum for sampling
        cumsum = np.zeros(n_regimes + 1)
        for j in range(n_regimes):
            cumsum[j + 1] = cumsum[j] + probs[j]
        
        # Sample next state using pre-generated uniform random number
        u = random_state[t]
        next_state = 0
        for j in range(n_regimes):
            if u < cumsum[j + 1]:
                next_state = j
                break
        
        states[t] = next_state
        current_state = next_state
    
    return states


@njit(cache=True)
def sample_student_t(
    n_steps: int,
    df: float,
    mu: float,
    sigma: float,
    normal_samples: np.ndarray,
    chi2_samples: np.ndarray
) -> np.ndarray:
    """
    Generate heavy-tailed returns from a Student-t distribution.
    
    This implements the heavy-tailed regime from model_definition.tex:
        r_t ~ t_ν(μ, σ²)
    
    Scaling formula: r = μ + σ * T_ν
    where T_ν is a standard Student-t random variable.
    
    Implementation uses the representation:
        T_ν = Z / sqrt(V/ν)
    where Z ~ N(0,1) and V ~ χ²(ν).
    
    Args:
        n_steps: Number of samples
        df: Degrees of freedom ν (typically 3 for heavy tails)
        mu: Location parameter μ
        sigma: Scale parameter σ
        normal_samples: Pre-generated N(0,1) samples of shape (n_steps,)
        chi2_samples: Pre-generated χ²(ν) samples of shape (n_steps,)
    
    Returns:
        returns: Array of Student-t samples [r_1, ..., r_T] of shape (n_steps,)
    
    Note:
        For ν = 3, the distribution has infinite 4th moment (excess kurtosis),
        making variance estimation unreliable - a key thesis finding.
    """
    returns = np.empty(n_steps, dtype=np.float64)
    
    for t in range(n_steps):
        # Standard Student-t: T_ν = Z / sqrt(V/ν)
        z = normal_samples[t]
        v = chi2_samples[t]
        
        # Avoid division by zero
        if v <= 0:
            v = 1e-10
        
        t_standard = z / np.sqrt(v / df)
        
        # Scale and shift: r = μ + σ * T_ν
        returns[t] = mu + sigma * t_standard
    
    return returns


@njit(cache=True)
def sample_gaussian(
    n_steps: int,
    mu: float,
    sigma: float,
    normal_samples: np.ndarray
) -> np.ndarray:
    """
    Generate Gaussian returns.
    
    This implements the Gaussian regime from model_definition.tex:
        r_t ~ N(μ, σ²)
    
    Args:
        n_steps: Number of samples
        mu: Mean return μ
        sigma: Standard deviation σ
        normal_samples: Pre-generated N(0,1) samples of shape (n_steps,)
    
    Returns:
        returns: Array of Gaussian samples [r_1, ..., r_T] of shape (n_steps,)
    """
    returns = np.empty(n_steps, dtype=np.float64)
    
    for t in range(n_steps):
        returns[t] = mu + sigma * normal_samples[t]
    
    return returns


@njit(parallel=True, cache=True)
def sample_markov_chain_batch(
    n_sims: int,
    n_steps: int,
    trans_matrix: np.ndarray,
    initial_state: int,
    random_states: np.ndarray
) -> np.ndarray:
    """
    Generate multiple Markov chain paths in parallel.
    
    Args:
        n_sims: Number of simulation paths N_{sims}
        n_steps: Number of time steps T
        trans_matrix: Transition matrix P of shape (M, M)
        initial_state: Starting state S_0
        random_states: Pre-generated uniforms of shape (n_sims, n_steps)
    
    Returns:
        states: Matrix of hidden states of shape (n_sims, n_steps)
    """
    states = np.empty((n_sims, n_steps), dtype=np.int32)
    
    for i in prange(n_sims):
        states[i, :] = sample_markov_chain(
            n_steps, trans_matrix, initial_state, random_states[i, :]
        )
    
    return states


@njit(parallel=True, cache=True)
def sample_regime_returns_batch(
    n_sims: int,
    n_steps: int,
    states: np.ndarray,
    regime_params: np.ndarray,
    normal_samples: np.ndarray,
    chi2_samples: np.ndarray
) -> np.ndarray:
    """
    Generate returns based on regime-dependent distributions.
    
    Implements: r_t | S_t = s ~ p(r | θ_s)
    
    Args:
        n_sims: Number of simulation paths
        n_steps: Number of time steps
        states: Matrix of hidden states (n_sims, n_steps)
        regime_params: Array of shape (M, 4) where each row is
                      [mu, sigma, df, is_student_t]
                      is_student_t: 1.0 if Student-t, 0.0 if Gaussian
        normal_samples: Pre-generated N(0,1) of shape (n_sims, n_steps)
        chi2_samples: Pre-generated χ²(df) of shape (n_sims, n_steps)
    
    Returns:
        returns: Matrix of returns (n_sims, n_steps)
    """
    returns = np.empty((n_sims, n_steps), dtype=np.float64)
    
    for i in prange(n_sims):
        for t in range(n_steps):
            state = states[i, t]
            mu = regime_params[state, 0]
            sigma = regime_params[state, 1]
            df = regime_params[state, 2]
            is_student_t = regime_params[state, 3]
            
            z = normal_samples[i, t]
            
            if is_student_t > 0.5:
                # Student-t distribution
                v = chi2_samples[i, t]
                if v <= 0:
                    v = 1e-10
                t_standard = z / np.sqrt(v / df)
                returns[i, t] = mu + sigma * t_standard
            else:
                # Gaussian distribution
                returns[i, t] = mu + sigma * z
    
    return returns


# =============================================================================
# Utility Functions (for pre-generating random samples)
# =============================================================================

def generate_random_samples(
    n_sims: int,
    n_steps: int,
    df: float,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre-generate all random samples needed for simulation.
    
    This enables Common Random Numbers (CRN) methodology by using
    the same random draws across different agent strategies.
    
    Args:
        n_sims: Number of simulation paths
        n_steps: Number of time steps
        df: Degrees of freedom for χ² samples (used for Student-t)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of:
            - uniform_samples: For Markov chain (n_sims, n_steps)
            - normal_samples: For returns (n_sims, n_steps)
            - chi2_samples: For Student-t (n_sims, n_steps)
    """
    rng = np.random.default_rng(seed)
    
    uniform_samples = rng.uniform(0, 1, size=(n_sims, n_steps))
    normal_samples = rng.standard_normal(size=(n_sims, n_steps))
    chi2_samples = rng.chisquare(df=df, size=(n_sims, n_steps))
    
    return uniform_samples, normal_samples, chi2_samples
