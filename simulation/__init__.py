"""
Simulation package for Bayesian Sequential Decision-Making thesis.

This package contains the core simulation infrastructure:
- distributions: Numba-optimized random number generators
- environment: MarketEnvironment class (HMM regime switching)
- market_env: MarketEnv class (multi-armed bandit)
- agents: Agent hierarchy (Kelly, Thompson, HMM, etc.)
- hmm_refined: Vol-Augmented HMM for regime detection (Phase 5)
- risk_constrained: CPPI-like drawdown-constrained Kelly (Phase 5)
- student_t_env: Heavy-tailed environments (Phase 5)
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

# Phase 5 modules
from .hmm_refined import (
    VolAugmentedHMM,
    VolAugmentedHMMKelly,
    create_vol_augmented_hmm_kelly,
)
from .risk_constrained import (
    RiskConstrainedKelly,
    create_risk_constrained_kelly,
)
from .student_t_env import (
    StudentTEnvironment,
    RegimeSwitchingStudentT,
    create_student_t_env,
    create_regime_switching_student_t,
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
    # Phase 5 - Refined HMM
    'VolAugmentedHMM',
    'VolAugmentedHMMKelly',
    'create_vol_augmented_hmm_kelly',
    # Phase 5 - Risk Constrained
    'RiskConstrainedKelly',
    'create_risk_constrained_kelly',
    # Phase 5 - Student-t Environments
    'StudentTEnvironment',
    'RegimeSwitchingStudentT',
    'create_student_t_env',
    'create_regime_switching_student_t',
]
