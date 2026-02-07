"""
Simulation package for Bayesian Sequential Decision-Making thesis.

This package contains the core simulation infrastructure:
- distributions: Numba-optimized random number generators
- environment: MarketEnvironment class (the "physics engine")
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

__all__ = [
    'sample_markov_chain',
    'sample_student_t',
    'sample_gaussian',
    'sample_markov_chain_batch',
    'sample_regime_returns_batch',
    'generate_random_samples',
    'MarketEnvironment',
]
