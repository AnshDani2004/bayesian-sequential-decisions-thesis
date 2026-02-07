"""
Student-t Environment for Stress Testing

Generates heavy-tailed returns for testing agent robustness.
Real financial markets exhibit fat tails (leptokurtosis).

Mathematical Basis:
r_t = μ + σ * (Z / sqrt(V/ν))
where Z ~ N(0,1), V ~ χ²_ν

Properties of Student-t(ν):
- ν = 3: Very heavy tails (kurtosis = ∞)
- ν = 5: Heavy tails (kurtosis = 9)
- ν → ∞: Approaches Normal

Expected Behavior:
- Naive Kelly will underestimate ≥3σ events
- Risk-Constrained should survive better
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StepResult:
    """Result from environment step."""
    return_t: float
    wealth: float
    done: bool
    info: dict


class StudentTEnvironment:
    """
    Market environment with fat-tailed (Student-t) returns.
    
    Supports:
    1. Stationary returns (single t-distribution)
    2. Regime-switching returns (different ν for bull/bear)
    
    Key Properties:
    - Heavy tails generate "Black Swan" events
    - Tests robustness of Kelly assumptions
    """
    
    def __init__(
        self,
        T: int,
        mu: float = 0.001,
        sigma: float = 0.01,
        nu: float = 3.0,
        regime_switching: bool = False,
        switch_point: Optional[int] = None,
        bull_params: Optional[dict] = None,
        bear_params: Optional[dict] = None,
        ruin_threshold: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize Student-t environment.
        
        Args:
            T: Horizon (number of steps)
            mu: Mean return
            sigma: Scale parameter
            nu: Degrees of freedom (lower = heavier tails)
            regime_switching: Enable bull/bear regimes
            switch_point: Step where regime switches
            bull_params: {'mu': x, 'sigma': y, 'nu': z}
            bear_params: {'mu': x, 'sigma': y, 'nu': z}
            ruin_threshold: Wealth below which agent is ruined
            seed: Random seed
        """
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.regime_switching = regime_switching
        self.switch_point = switch_point if switch_point else T // 2
        self.ruin_threshold = ruin_threshold
        
        # Regime parameters
        if regime_switching:
            self.bull_params = bull_params or {'mu': 0.02, 'sigma': 0.1, 'nu': 5.0}
            self.bear_params = bear_params or {'mu': -0.02, 'sigma': 0.2, 'nu': 3.0}
        else:
            self.bull_params = None
            self.bear_params = None
        
        # Random state
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        
        # State
        self.wealth = 1.0
        self.step_count = 0
        self.done = False
        
        # History
        self.wealth_history: List[float] = [1.0]
        self.return_history: List[float] = []
        self.regime_history: List[str] = []
    
    def _sample_student_t(self, mu: float, sigma: float, nu: float) -> float:
        """
        Sample from scaled Student-t distribution.
        
        r = μ + σ * (Z / sqrt(V/ν))
        where Z ~ N(0,1), V ~ χ²_ν
        """
        # Standard Student-t
        z = self.rng.standard_normal()
        v = self.rng.chisquare(nu)
        t = z / np.sqrt(v / nu)
        
        # Scale and shift
        return mu + sigma * t
    
    def _get_current_params(self) -> Tuple[float, float, float]:
        """Get parameters for current regime."""
        if not self.regime_switching:
            return self.mu, self.sigma, self.nu
        
        if self.step_count < self.switch_point:
            params = self.bull_params
            self.regime_history.append('bull')
        else:
            params = self.bear_params
            self.regime_history.append('bear')
        
        return params['mu'], params['sigma'], params['nu']
    
    def reset(self, seed: Optional[int] = None) -> float:
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed
        
        self.wealth = 1.0
        self.step_count = 0
        self.done = False
        
        self.wealth_history = [1.0]
        self.return_history = []
        self.regime_history = []
        
        return 0.0  # Initial observation
    
    def step(self, bet_fraction: float) -> StepResult:
        """
        Execute one step.
        
        Args:
            bet_fraction: Fraction of wealth to invest (0 ≤ f ≤ 1)
        
        Returns:
            StepResult with return, wealth, done, info
        """
        if self.done:
            raise RuntimeError("Episode already done")
        
        # Get regime parameters
        mu, sigma, nu = self._get_current_params()
        
        # Sample return
        r_t = self._sample_student_t(mu, sigma, nu)
        self.return_history.append(r_t)
        
        # Clip bet fraction
        f = np.clip(bet_fraction, 0.0, 1.0)
        
        # Portfolio return: r_p = f * r_t (simplified)
        portfolio_return = f * r_t
        
        # Update wealth
        new_wealth = self.wealth * (1.0 + portfolio_return)
        self.wealth = max(new_wealth, 1e-10)
        self.wealth_history.append(self.wealth)
        
        self.step_count += 1
        
        # Check termination
        ruined = self.wealth < self.ruin_threshold
        horizon_reached = self.step_count >= self.T
        self.done = ruined or horizon_reached
        
        info = {
            'regime': self.regime_history[-1] if self.regime_history else 'stationary',
            'return': r_t,
            'nu': nu,
            'ruined': ruined
        }
        
        return StepResult(
            return_t=r_t,
            wealth=self.wealth,
            done=self.done,
            info=info
        )
    
    def get_statistics(self) -> dict:
        """Get summary statistics of the episode."""
        if not self.return_history:
            return {}
        
        returns = np.array(self.return_history)
        
        # Compute kurtosis (should be > 3 for fat tails)
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        if std_r > 0:
            kurtosis = np.mean(((returns - mean_r) / std_r) ** 4)
        else:
            kurtosis = 0.0
        
        # Count extreme events (|r| > 3σ)
        extreme_events = np.sum(np.abs(returns - mean_r) > 3 * std_r)
        
        return {
            'mean_return': float(mean_r),
            'std_return': float(std_r),
            'kurtosis': float(kurtosis),
            'extreme_events': int(extreme_events),
            'final_wealth': self.wealth,
            'max_wealth': float(np.max(self.wealth_history)),
            'min_wealth': float(np.min(self.wealth_history)),
        }


class RegimeSwitchingStudentT(StudentTEnvironment):
    """
    Convenience class for regime-switching Student-t environment.
    
    Default setup:
    - Bull: μ=0.02, σ=0.1, ν=5 (moderate tails)
    - Bear: μ=-0.02, σ=0.2, ν=3 (heavy tails + crash)
    """
    
    def __init__(
        self,
        T: int = 1000,
        switch_point: int = 500,
        bull_mu: float = 0.02,
        bull_sigma: float = 0.1,
        bull_nu: float = 5.0,
        bear_mu: float = -0.02,
        bear_sigma: float = 0.2,
        bear_nu: float = 3.0,
        seed: Optional[int] = None
    ):
        super().__init__(
            T=T,
            regime_switching=True,
            switch_point=switch_point,
            bull_params={'mu': bull_mu, 'sigma': bull_sigma, 'nu': bull_nu},
            bear_params={'mu': bear_mu, 'sigma': bear_sigma, 'nu': bear_nu},
            seed=seed
        )


# =============================================================================
# Factory functions
# =============================================================================

def create_student_t_env(
    T: int = 1000,
    nu: float = 3.0,
    mu: float = 0.001,
    sigma: float = 0.01,
    seed: Optional[int] = None
) -> StudentTEnvironment:
    """Create stationary Student-t environment."""
    return StudentTEnvironment(T=T, mu=mu, sigma=sigma, nu=nu, seed=seed)


def create_regime_switching_student_t(
    T: int = 1000,
    switch_point: int = 500,
    seed: Optional[int] = None
) -> RegimeSwitchingStudentT:
    """Create regime-switching Student-t environment with defaults."""
    return RegimeSwitchingStudentT(T=T, switch_point=switch_point, seed=seed)
