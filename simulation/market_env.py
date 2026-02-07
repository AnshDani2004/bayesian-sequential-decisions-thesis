"""
MarketEnv: Multi-Armed Bandit Environment for Sequential Decision-Making.

This module implements the OpenAI Gym-like environment for simulating
repeated betting games with Bayesian agents.

Reference: theory/model_definition.tex (Section 3: The Environment)

Key Design Principles:
1. REPRODUCIBILITY: Accepts pregenerated_outcomes for Common Random Numbers (CRN)
2. VECTORIZATION: Efficient numpy operations for batch processing
3. RUIN DETECTION: Absorbing state when wealth falls below threshold

Wealth Dynamics:
    W_{t+1} = W_t * (1 + (1 - Σf_i) * r_f + Σf_i * X_{t,i})
    
    where X_{t,i} ∈ {+b_i, -1} is the realized return of arm i
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List


@dataclass
class StepResult:
    """Result of a single environment step."""
    outcomes: np.ndarray      # X_t: outcomes for each arm (+b or -1)
    reward: float             # log(W_{t+1} / W_t) - log growth
    done: bool                # True if ruined or horizon reached
    info: Dict[str, Any]      # Additional info (wealth, step, etc.)


class MarketEnv:
    """
    Multi-Armed Bandit environment with multiplicative wealth dynamics.
    
    This is the "physics engine" for the thesis simulation. It generates
    stochastic outcomes and tracks wealth evolution according to the
    Kelly betting framework.
    
    Features:
        - Supports K independent betting options (arms)
        - Multiplicative wealth dynamics (geometric growth)
        - Ruin detection (absorbing state)
        - Common Random Numbers via pregenerated_outcomes
        - Constraint validation (no shorting, no leverage > 100%)
    
    Example:
        >>> env = MarketEnv(n_arms=2, true_probs=[0.6, 0.4], horizon=1000)
        >>> env.reset(seed=42)
        >>> bets = np.array([0.2, 0.0])  # Bet 20% on arm 0
        >>> result = env.step(bets)
        >>> print(f"Wealth: {result.info['wealth']:.2f}")
    
    Mathematical Framework:
        At each step t:
        1. Agent commits bets f = [f_1, ..., f_K] (fractions of wealth)
        2. Environment samples outcomes X_t ~ Bernoulli(p) * (1 + b) - 1
        3. Wealth updates: W_{t+1} = W_t * (1 + portfolio_return)
        4. Reward = log(W_{t+1} / W_t) (log growth rate)
    """
    
    def __init__(
        self,
        n_arms: int,
        true_probs: List[float],
        horizon: int,
        odds: Optional[List[float]] = None,
        risk_free_rate: float = 0.0,
        ruin_threshold: float = 1e-6,
        pregenerated_outcomes: Optional[np.ndarray] = None
    ):
        """
        Initialize the MarketEnv.
        
        Args:
            n_arms: K, number of independent betting options
            true_probs: [p_1, ..., p_K], hidden win probabilities (unknown to agent)
            horizon: T, maximum number of time steps
            odds: [b_1, ..., b_K], payoff odds (default: 1:1 for all arms)
            risk_free_rate: r_f, return on uninvested wealth (default: 0)
            ruin_threshold: ε, wealth below which agent is ruined
            pregenerated_outcomes: (T, K) matrix of outcomes for CRN methodology
                                   If provided, step() uses these instead of sampling
        
        Raises:
            ValueError: If dimensions don't match or probabilities invalid
        """
        self.n_arms = n_arms
        self.true_probs = np.array(true_probs, dtype=np.float64)
        self.horizon = horizon
        self.odds = np.array(odds if odds is not None else [1.0] * n_arms)
        self.risk_free_rate = risk_free_rate
        self.ruin_threshold = ruin_threshold
        
        # Validate inputs
        self._validate_inputs()
        
        # CRN support: pregenerated outcomes matrix
        self.pregenerated_outcomes = pregenerated_outcomes
        self._using_pregenerated = pregenerated_outcomes is not None
        
        # State variables (initialized in reset())
        self.wealth: float = 1.0
        self.step_count: int = 0
        self.done: bool = False
        self.rng: np.random.Generator = np.random.default_rng()
        
        # History tracking
        self.wealth_history: List[float] = []
        self.outcome_history: List[np.ndarray] = []
        self.bet_history: List[np.ndarray] = []
    
    def _validate_inputs(self):
        """Validate constructor inputs."""
        if len(self.true_probs) != self.n_arms:
            raise ValueError(
                f"true_probs length ({len(self.true_probs)}) != n_arms ({self.n_arms})"
            )
        
        if not np.all((0 <= self.true_probs) & (self.true_probs <= 1)):
            raise ValueError("Probabilities must be in [0, 1]")
        
        if len(self.odds) != self.n_arms:
            raise ValueError(
                f"odds length ({len(self.odds)}) != n_arms ({self.n_arms})"
            )
        
        if not np.all(self.odds > 0):
            raise ValueError("Odds must be positive")
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
        
        Returns:
            Initial observation (zeros, no outcomes yet)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.wealth = 1.0
        self.step_count = 0
        self.done = False
        
        self.wealth_history = [self.wealth]
        self.outcome_history = []
        self.bet_history = []
        
        # Initial observation: no outcomes yet
        return np.zeros(self.n_arms)
    
    def step(self, bets: np.ndarray) -> StepResult:
        """
        Execute one time step.
        
        Args:
            bets: f ∈ R^K, fractions of wealth bet on each arm
                  Constraints: f_i ≥ 0, Σf_i ≤ 1
        
        Returns:
            StepResult containing:
                - outcomes: X_t ∈ {+b_i, -1}^K for each arm
                - reward: log(W_{t+1} / W_t)
                - done: True if ruined or horizon reached
                - info: dict with 'wealth', 'step', 'ruined'
        
        Raises:
            ValueError: If bet constraints violated
            RuntimeError: If called after episode is done
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")
        
        bets = np.asarray(bets, dtype=np.float64)
        self._validate_bets(bets)
        
        # Generate or retrieve outcomes
        if self._using_pregenerated:
            outcomes = self.pregenerated_outcomes[self.step_count, :]
        else:
            outcomes = self._sample_outcomes()
        
        # Calculate portfolio return
        # R_t = (1 - Σf_i) * r_f + Σf_i * X_{t,i}
        uninvested = 1.0 - np.sum(bets)
        portfolio_return = uninvested * self.risk_free_rate + np.sum(bets * outcomes)
        
        # Update wealth (multiplicative dynamics)
        new_wealth = self.wealth * (1.0 + portfolio_return)
        
        # Calculate log growth (reward)
        if new_wealth > 0:
            reward = np.log(new_wealth / self.wealth)
        else:
            reward = -np.inf
        
        # Update state
        self.wealth = max(new_wealth, 0.0)
        self.step_count += 1
        
        # Check termination conditions
        ruined = self.wealth < self.ruin_threshold
        horizon_reached = self.step_count >= self.horizon
        self.done = ruined or horizon_reached
        
        # Record history
        self.wealth_history.append(self.wealth)
        self.outcome_history.append(outcomes.copy())
        self.bet_history.append(bets.copy())
        
        return StepResult(
            outcomes=outcomes,
            reward=reward,
            done=self.done,
            info={
                'wealth': self.wealth,
                'step': self.step_count,
                'ruined': ruined,
                'horizon_reached': horizon_reached,
                'portfolio_return': portfolio_return,
            }
        )
    
    def _validate_bets(self, bets: np.ndarray):
        """Validate bet constraints."""
        if len(bets) != self.n_arms:
            raise ValueError(f"bets length ({len(bets)}) != n_arms ({self.n_arms})")
        
        if np.any(bets < 0):
            raise ValueError(f"Bets must be non-negative. Got: {bets}")
        
        if np.sum(bets) > 1.0 + 1e-9:  # Small tolerance for floating point
            raise ValueError(f"Total bets ({np.sum(bets):.4f}) exceed 1.0")
    
    def _sample_outcomes(self) -> np.ndarray:
        """
        Sample outcomes for each arm.
        
        Returns:
            X_t: Array where X_t[i] = +b_i (win) or -1 (loss)
        """
        # Sample Bernoulli outcomes
        wins = self.rng.random(self.n_arms) < self.true_probs
        
        # Convert to payoffs: win → +b, lose → -1
        outcomes = np.where(wins, self.odds, -1.0)
        
        return outcomes
    
    def get_optimal_kelly_fractions(self) -> np.ndarray:
        """
        Calculate optimal Kelly fractions (for oracle agent).
        
        Formula: f* = (bp - q) / b = p - q/b
        where q = 1 - p
        
        Returns:
            f*: Array of optimal betting fractions, clipped to [0, 1]
        """
        p = self.true_probs
        q = 1.0 - p
        b = self.odds
        
        # Kelly formula: f* = (bp - q) / b
        kelly = (b * p - q) / b
        
        # Clip to valid range [0, 1]
        return np.clip(kelly, 0.0, 1.0)
    
    def __repr__(self) -> str:
        return (
            f"MarketEnv(n_arms={self.n_arms}, "
            f"horizon={self.horizon}, "
            f"wealth={self.wealth:.4f}, "
            f"step={self.step_count})"
        )


def generate_pregenerated_outcomes(
    n_arms: int,
    horizon: int,
    true_probs: List[float],
    odds: Optional[List[float]] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Pre-generate outcomes matrix for Common Random Numbers (CRN).
    
    This allows multiple agents to be evaluated on the exact same
    market realizations, enabling fair "A/B testing" of strategies.
    
    Args:
        n_arms: Number of arms K
        horizon: Number of time steps T
        true_probs: Win probabilities [p_1, ..., p_K]
        odds: Payoff odds [b_1, ..., b_K] (default: 1:1)
        seed: Random seed for reproducibility
    
    Returns:
        outcomes: (T, K) matrix where outcomes[t, i] ∈ {+b_i, -1}
    
    Example:
        >>> outcomes = generate_pregenerated_outcomes(
        ...     n_arms=2, horizon=1000, true_probs=[0.6, 0.4], seed=42
        ... )
        >>> env1 = MarketEnv(2, [0.6, 0.4], 1000, pregenerated_outcomes=outcomes)
        >>> env2 = MarketEnv(2, [0.6, 0.4], 1000, pregenerated_outcomes=outcomes)
        >>> # env1 and env2 will see identical coin flips!
    """
    rng = np.random.default_rng(seed)
    
    probs = np.array(true_probs)
    payoffs = np.array(odds if odds is not None else [1.0] * n_arms)
    
    # Sample wins: shape (T, K)
    random_draws = rng.random((horizon, n_arms))
    wins = random_draws < probs  # Broadcasting: (T, K) < (K,)
    
    # Convert to payoffs
    outcomes = np.where(wins, payoffs, -1.0)
    
    return outcomes
