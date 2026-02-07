"""
Simulation package for Bayesian Sequential Decision-Making thesis.

This package contains the core simulation infrastructure:
- distributions: Numba-optimized random number generators
- environment: MarketEnvironment class (HMM regime switching)
- market_env: MarketEnv class (multi-armed bandit)
- agents: Agent hierarchy (Kelly, Thompson, HMM, etc.)
"""

from .distributions import (
    sample_markov_chain,
    sample_student_t,
    sample_gaussian,
    sample_markov_chain_batch,
    sample_regime_returns_batch,
    generate_random_samples,
)
from .environment import MarketEnvironment
from .market_env import MarketEnv, generate_pregenerated_outcomes
from .agents import (
    BaseAgent,
    KellyOracle,
    NaiveBayesKelly,
    ThompsonKellyAgent,
    FixedFraction,
    ConvexRiskAgent,
    HMMKellyAgent,
    FractionalKelly,
    compute_kelly_fraction,
    create_half_kelly,
    create_double_kelly,
    create_fractional_kelly,
)

__all__ = [
    # Distributions
    'sample_markov_chain',
    'sample_student_t',
    'sample_gaussian',
    'sample_markov_chain_batch',
    'sample_regime_returns_batch',
    'generate_random_samples',
    # Environments
    'MarketEnvironment',
    'MarketEnv',
    'generate_pregenerated_outcomes',
    # Agents
    'BaseAgent',
    'KellyOracle',
    'NaiveBayesKelly',
    'ThompsonKellyAgent',
    'FixedFraction',
    'ConvexRiskAgent',
    'HMMKellyAgent',
    'FractionalKelly',
    'compute_kelly_fraction',
    'create_half_kelly',
    'create_double_kelly',
    'create_fractional_kelly',
]

