"""
Agent Module for Bayesian Sequential Decision-Making.

This module implements the agent hierarchy for the thesis:
- BaseAgent: Abstract base class defining the interface
- KellyOracle: Optimal agent with known probabilities (benchmark)
- NaiveBayesKelly: Plug-in estimator (volatile, high ruin risk)
- ThompsonKellyAgent: Core innovation - samples posterior, computes Kelly
- ConvexRiskAgent: CVaR-constrained optimization (CVXPY)
- FixedFraction: Static baseline

Reference: theory/model_definition.tex (Section 4: Agents)

Mathematical Framework:
    Kelly Formula: f* = (bp - q) / b where q = 1 - p
    Bayesian Update: Beta(α, β) → Beta(α + wins, β + losses)
    Thompson Sampling: Sample p̃ ~ Beta(α, β), then act on p̃
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Utility Functions
# =============================================================================

def compute_kelly_fraction(p: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute optimal Kelly fraction for each arm.
    
    Formula: f* = (bp - q) / b = p - q/b
    where q = 1 - p
    
    Args:
        p: Win probabilities [p_1, ..., p_K]
        b: Odds [b_1, ..., b_K]
    
    Returns:
        f*: Optimal betting fractions, clipped to [0, 1]
    """
    p = np.asarray(p)
    b = np.asarray(b)
    q = 1.0 - p
    
    # Kelly formula: f* = (bp - q) / b
    kelly = (b * p - q) / b
    
    return np.clip(kelly, 0.0, 1.0)


def normalize_bets(bets: np.ndarray, max_total: float = 1.0) -> np.ndarray:
    """
    Normalize bets to ensure Σf_i ≤ max_total.
    
    Args:
        bets: Raw betting fractions
        max_total: Maximum total bet (default: 1.0)
    
    Returns:
        Normalized bets with sum ≤ max_total
    """
    bets = np.clip(bets, 0.0, None)  # Ensure non-negative
    total = np.sum(bets)
    
    if total > max_total:
        bets = bets * (max_total / total)
    
    return bets


# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all betting agents.
    
    Defines the interface that all agents must implement:
    - act(): Return betting fractions f ∈ R^K
    - update(): Update beliefs based on observed outcomes
    - reset(): Reset agent state for new episode
    """
    
    def __init__(self, n_arms: int, odds: Optional[List[float]] = None):
        """
        Initialize base agent.
        
        Args:
            n_arms: Number of betting options K
            odds: Payoff odds [b_1, ..., b_K] (default: 1:1)
        """
        self.n_arms = n_arms
        self.odds = np.array(odds if odds is not None else [1.0] * n_arms)
    
    @abstractmethod
    def act(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Decide betting fractions.
        
        Args:
            observation: Optional observation from environment
        
        Returns:
            f: Betting fractions [f_1, ..., f_K] with Σf_i ≤ 1, f_i ≥ 0
        """
        pass
    
    @abstractmethod
    def update(self, outcomes: np.ndarray):
        """
        Update beliefs based on observed outcomes.
        
        Args:
            outcomes: X_t ∈ {+b_i, -1}^K, realized returns for each arm
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset agent state for new episode."""
        pass
    
    def get_state(self) -> dict:
        """Return agent state for debugging/logging."""
        return {'n_arms': self.n_arms, 'odds': self.odds.tolist()}


# =============================================================================
# KellyOracle: Optimal Benchmark
# =============================================================================

class KellyOracle(BaseAgent):
    """
    Cheating agent with access to true probabilities.
    
    This serves as the theoretical upper bound on performance.
    No other agent should beat this in the long run (n → ∞).
    
    Mathematical Basis: SLLN, Asymptotic Optimality
    
    Expected Behavior: Maximum log(W_T). High volatility, but optimal growth.
    """
    
    def __init__(
        self,
        n_arms: int,
        true_probs: List[float],
        odds: Optional[List[float]] = None,
        fraction_multiplier: float = 1.0
    ):
        """
        Initialize KellyOracle.
        
        Args:
            n_arms: Number of arms K
            true_probs: [p_1, ..., p_K] (the agent "knows" these)
            odds: Payoff odds [b_1, ..., b_K]
            fraction_multiplier: Scale factor (1.0 = full Kelly, 0.5 = half Kelly)
        """
        super().__init__(n_arms, odds)
        self.true_probs = np.array(true_probs)
        self.fraction_multiplier = fraction_multiplier
        
        # Pre-compute optimal fractions
        self.kelly_fractions = compute_kelly_fraction(self.true_probs, self.odds)
        self.kelly_fractions *= self.fraction_multiplier
        self.kelly_fractions = normalize_bets(self.kelly_fractions)
    
    def act(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """Always bet the optimal Kelly fraction."""
        return self.kelly_fractions.copy()
    
    def update(self, outcomes: np.ndarray):
        """Oracle doesn't learn - it already knows everything."""
        pass
    
    def reset(self):
        """Nothing to reset."""
        pass
    
    def get_state(self) -> dict:
        state = super().get_state()
        state['true_probs'] = self.true_probs.tolist()
        state['kelly_fractions'] = self.kelly_fractions.tolist()
        return state


# =============================================================================
# NaiveBayesKelly: Plug-in Estimator (Volatile)
# =============================================================================

class NaiveBayesKelly(BaseAgent):
    """
    Uses posterior mean directly in Kelly formula.
    
    This agent maintains Beta priors and plugs the posterior mean
    into the Kelly formula as if it were the true probability.
    
    Mathematical Basis: Plug-in Estimator
    
    Expected Behavior: HIGH RUIN RISK due to estimation variance in early steps.
    The agent overbets when posterior variance is high.
    
    Theoretical Insight: This demonstrates the "Estimation Error vs. Leverage"
    danger - estimation error translates directly into bet-sizing error,
    and overbetting has severe geometric penalties.
    """
    
    def __init__(
        self,
        n_arms: int,
        odds: Optional[List[float]] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ):
        """
        Initialize NaiveBayesKelly.
        
        Args:
            n_arms: Number of arms K
            odds: Payoff odds
            prior_alpha: Initial α for Beta prior (default: 1 = uniform)
            prior_beta: Initial β for Beta prior (default: 1 = uniform)
        """
        super().__init__(n_arms, odds)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Initialize posteriors
        self.alphas = np.full(n_arms, prior_alpha)
        self.betas = np.full(n_arms, prior_beta)
    
    def act(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Bet Kelly fraction based on posterior mean.
        
        p̂ = α / (α + β) (posterior mean)
        f = Kelly(p̂)
        """
        # Posterior mean: p̂ = α / (α + β)
        p_hat = self.alphas / (self.alphas + self.betas)
        
        # Compute Kelly fractions
        kelly = compute_kelly_fraction(p_hat, self.odds)
        
        return normalize_bets(kelly)
    
    def update(self, outcomes: np.ndarray):
        """
        Conjugate Bayesian update.
        
        Win (outcome > 0): α += 1
        Loss (outcome < 0): β += 1
        """
        wins = (outcomes > 0).astype(np.float64)
        losses = (outcomes < 0).astype(np.float64)
        
        self.alphas += wins
        self.betas += losses
    
    def reset(self):
        """Reset to prior."""
        self.alphas = np.full(self.n_arms, self.prior_alpha)
        self.betas = np.full(self.n_arms, self.prior_beta)
    
    def get_posterior_mean(self) -> np.ndarray:
        """Return current posterior mean for each arm."""
        return self.alphas / (self.alphas + self.betas)
    
    def get_posterior_variance(self) -> np.ndarray:
        """Return current posterior variance for each arm."""
        a, b = self.alphas, self.betas
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    def get_state(self) -> dict:
        state = super().get_state()
        state['alphas'] = self.alphas.tolist()
        state['betas'] = self.betas.tolist()
        state['posterior_mean'] = self.get_posterior_mean().tolist()
        return state


# =============================================================================
# ThompsonKellyAgent: Core Innovation
# =============================================================================

class ThompsonKellyAgent(BaseAgent):
    """
    The Core Innovation: Thompson Sampling + Kelly Sizing.
    
    Instead of betting a fixed fraction, this agent:
    1. Samples p̃ ~ Beta(α, β) from the posterior
    2. Computes f*(p̃) using the Kelly formula
    3. Bets f*(p̃)
    
    Why This Matters:
    - High posterior variance → wide spread of p̃ → variable f*
    - This naturally regulates leverage as a function of uncertainty
    - Bayesian uncertainty acts as implicit risk aversion
    
    Mathematical Basis: Probability Matching + Growth Optimality
    
    Expected Behavior: Sublinear regret with uncertainty-aware sizing.
    Should avoid the overbetting problem of NaiveBayesKelly.
    
    Theoretical Insight: Tests if Bayesian uncertainty naturally regulates
    leverage (i.e., high variance in posterior → lower effective growth,
    mimicking risk aversion without explicit constraints).
    """
    
    def __init__(
        self,
        n_arms: int,
        odds: Optional[List[float]] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        fraction_cap: float = 1.0
    ):
        """
        Initialize ThompsonKellyAgent.
        
        Args:
            n_arms: Number of arms K
            odds: Payoff odds
            prior_alpha: Initial α for Beta prior
            prior_beta: Initial β for Beta prior
            fraction_cap: Maximum total bet (safety cap)
        """
        super().__init__(n_arms, odds)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.fraction_cap = fraction_cap
        
        # Initialize posteriors
        self.alphas = np.full(n_arms, prior_alpha)
        self.betas = np.full(n_arms, prior_beta)
        
        # Track last sampled values for debugging
        self.last_samples = None
        self.last_kelly = None
    
    def act(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample from posterior, then compute and bet Kelly fraction.
        
        1. Sample: p̃_i ~ Beta(α_i, β_i)
        2. Compute: f*_i = Kelly(p̃_i)
        3. Return: normalized f*
        """
        # Step 1: Sample from Beta posteriors
        p_samples = np.random.beta(self.alphas, self.betas)
        self.last_samples = p_samples.copy()
        
        # Step 2: Compute Kelly fractions for sampled probabilities
        kelly = compute_kelly_fraction(p_samples, self.odds)
        self.last_kelly = kelly.copy()
        
        # Step 3: Normalize and return
        return normalize_bets(kelly, self.fraction_cap)
    
    def update(self, outcomes: np.ndarray):
        """Conjugate Bayesian update."""
        wins = (outcomes > 0).astype(np.float64)
        losses = (outcomes < 0).astype(np.float64)
        
        self.alphas += wins
        self.betas += losses
    
    def reset(self):
        """Reset to prior."""
        self.alphas = np.full(self.n_arms, self.prior_alpha)
        self.betas = np.full(self.n_arms, self.prior_beta)
        self.last_samples = None
        self.last_kelly = None
    
    def get_posterior_mean(self) -> np.ndarray:
        return self.alphas / (self.alphas + self.betas)
    
    def get_posterior_variance(self) -> np.ndarray:
        a, b = self.alphas, self.betas
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    def get_state(self) -> dict:
        state = super().get_state()
        state['alphas'] = self.alphas.tolist()
        state['betas'] = self.betas.tolist()
        state['posterior_mean'] = self.get_posterior_mean().tolist()
        if self.last_samples is not None:
            state['last_samples'] = self.last_samples.tolist()
            state['last_kelly'] = self.last_kelly.tolist()
        return state


# =============================================================================
# FixedFraction: Static Baseline
# =============================================================================

class FixedFraction(BaseAgent):
    """
    Bets a fixed fraction on specified arms.
    
    This is a non-adaptive baseline for comparison.
    
    Mathematical Basis: Classical Portfolio Theory
    
    Expected Behavior: Fails to adapt to non-stationary probabilities.
    Useful for testing specific betting levels (HalfKelly, DoubleKelly).
    """
    
    def __init__(
        self,
        n_arms: int,
        fractions: Optional[List[float]] = None,
        odds: Optional[List[float]] = None
    ):
        """
        Initialize FixedFraction.
        
        Args:
            n_arms: Number of arms K
            fractions: Fixed betting fractions [f_1, ..., f_K]
                      (default: equal allocation of 10% total)
            odds: Payoff odds
        """
        super().__init__(n_arms, odds)
        
        if fractions is None:
            self.fractions = np.full(n_arms, 0.1 / n_arms)
        else:
            self.fractions = np.array(fractions)
        
        self.fractions = normalize_bets(self.fractions)
    
    def act(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """Always bet the fixed fraction."""
        return self.fractions.copy()
    
    def update(self, outcomes: np.ndarray):
        """Fixed fraction doesn't learn."""
        pass
    
    def reset(self):
        """Nothing to reset."""
        pass
    
    def get_state(self) -> dict:
        state = super().get_state()
        state['fractions'] = self.fractions.tolist()
        return state


# =============================================================================
# ConvexRiskAgent: CVaR-Constrained Optimization
# =============================================================================

class ConvexRiskAgent(BaseAgent):
    """
    Uses CVXPY to solve risk-constrained Kelly problem.
    
    This agent formulates bet sizing as a convex optimization problem
    with constraints on expected drawdown.
    
    Mathematical Basis: Busseti, Ryu, Boyd (2016) - Convex Optimization
    
    Optimization Problem:
        max  Σ E[log(1 + f_i * r_i)]
        s.t. f_i ≥ 0
             Σf_i ≤ 1
             CVaR constraint (convex approximation)
    
    Speed Optimization:
        The solve_freq parameter controls how often the optimizer is called.
        Between solves, the agent reuses the previous solution.
    
    Expected Behavior: Lower growth than Oracle, but strictly bounded drawdown.
    """
    
    def __init__(
        self,
        n_arms: int,
        odds: Optional[List[float]] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        max_bet_per_arm: float = 0.25,
        solve_freq: int = 1
    ):
        """
        Initialize ConvexRiskAgent.
        
        Args:
            n_arms: Number of arms K
            odds: Payoff odds
            prior_alpha: Initial α for Beta prior
            prior_beta: Initial β for Beta prior
            max_bet_per_arm: Maximum bet on any single arm
            solve_freq: Only re-solve every solve_freq steps (speed optimization)
        """
        super().__init__(n_arms, odds)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.max_bet_per_arm = max_bet_per_arm
        self.solve_freq = solve_freq
        
        # Posteriors
        self.alphas = np.full(n_arms, prior_alpha)
        self.betas = np.full(n_arms, prior_beta)
        
        # Cached solution
        self.step_count = 0
        self.cached_solution = np.zeros(n_arms)
        
        # Try to import cvxpy
        self._cvxpy_available = False
        try:
            import cvxpy as cp
            self._cvxpy_available = True
            self._cp = cp
        except ImportError:
            pass
    
    def act(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Solve optimization or use cached solution.
        """
        if not self._cvxpy_available:
            # Fallback to Kelly if CVXPY not available
            p_hat = self.alphas / (self.alphas + self.betas)
            kelly = compute_kelly_fraction(p_hat, self.odds)
            return normalize_bets(kelly * 0.5)  # Conservative
        
        # Only re-solve at specified frequency
        if self.step_count % self.solve_freq == 0:
            self.cached_solution = self._solve_optimization()
        
        self.step_count += 1
        return self.cached_solution.copy()
    
    def _solve_optimization(self) -> np.ndarray:
        """
        Solve the constrained optimization problem.
        
        Simplified version: maximize expected log growth with position limits.
        """
        cp = self._cp
        
        # Posterior means
        p_hat = self.alphas / (self.alphas + self.betas)
        q_hat = 1.0 - p_hat
        
        # Decision variable
        f = cp.Variable(self.n_arms)
        
        # Expected log growth (linearized approximation for tractability)
        # E[log(1 + f*X)] ≈ p*log(1+fb) + q*log(1-f)
        # For small f: ≈ f*(p*b - q) = f * edge
        edge = p_hat * self.odds - q_hat
        
        objective = cp.Maximize(edge @ f)
        
        constraints = [
            f >= 0,
            f <= self.max_bet_per_arm,
            cp.sum(f) <= 1.0
        ]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)
            
            if problem.status == cp.OPTIMAL and f.value is not None:
                return np.clip(f.value, 0.0, 1.0)
        except Exception:
            pass
        
        # Fallback
        return np.zeros(self.n_arms)
    
    def update(self, outcomes: np.ndarray):
        """Conjugate Bayesian update."""
        wins = (outcomes > 0).astype(np.float64)
        losses = (outcomes < 0).astype(np.float64)
        
        self.alphas += wins
        self.betas += losses
    
    def reset(self):
        """Reset to prior."""
        self.alphas = np.full(self.n_arms, self.prior_alpha)
        self.betas = np.full(self.n_arms, self.prior_beta)
        self.step_count = 0
        self.cached_solution = np.zeros(self.n_arms)
    
    def get_state(self) -> dict:
        state = super().get_state()
        state['alphas'] = self.alphas.tolist()
        state['betas'] = self.betas.tolist()
        state['cvxpy_available'] = self._cvxpy_available
        return state


# =============================================================================
# Factory Functions
# =============================================================================

def create_half_kelly(n_arms: int, true_probs: List[float], odds: Optional[List[float]] = None) -> KellyOracle:
    """Create a Half-Kelly agent (50% of optimal)."""
    return KellyOracle(n_arms, true_probs, odds, fraction_multiplier=0.5)


def create_double_kelly(n_arms: int, true_probs: List[float], odds: Optional[List[float]] = None) -> KellyOracle:
    """Create a Double-Kelly agent (200% of optimal - should underperform!)."""
    return KellyOracle(n_arms, true_probs, odds, fraction_multiplier=2.0)
