"""
MarketEnvironment: The "Physics Engine" for Market Simulation.

This module implements the stochastic environment that generates returns
according to the mathematical framework in theory/model_definition.tex.

Key Design Principles:
1. NO LOOK-AHEAD: Returns r_t are only revealed after f_t is committed
2. REPRODUCIBILITY: All stochastic methods accept seeds
3. BATCH EFFICIENCY: Pre-generates entire return matrices for CRN methodology

Reference: theory/model_definition.tex (Section 3: The Environment)

Timing Convention (Shreve):
    - At time t: agent observes F_t = σ(r_1, ..., r_t)
    - Agent decides f_{t+1} based on F_t
    - At time t+1: r_{t+1} is revealed, wealth updates

Matrix Indexing Convention:
    - returns[i, t] corresponds to r_{t+1} in LaTeX notation
    - This is the return realized at the END of step t (0-indexed)
    - Equivalently: returns[i, 0] = r_1, returns[i, 1] = r_2, ...
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, NamedTuple
import warnings

from .distributions import (
    sample_markov_chain_batch,
    sample_regime_returns_batch,
    generate_random_samples,
)


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class RegimeConfig:
    """
    Configuration for a single market regime.
    
    Maps to θ_s = (μ_s, σ_s, ν_s) from model_definition.tex.
    
    Attributes:
        mu: Mean return μ (e.g., 0.05 for 5% annual)
        sigma: Volatility σ (e.g., 0.10 for 10% std dev)
        df: Degrees of freedom ν for Student-t (None for Gaussian)
        name: Human-readable name (e.g., "bull", "bear")
    """
    mu: float
    sigma: float
    df: Optional[float] = None  # None = Gaussian, 3 = Heavy-tailed Student-t
    name: str = "unnamed"
    
    @property
    def is_student_t(self) -> bool:
        """Returns True if this regime uses Student-t distribution."""
        return self.df is not None
    
    def to_array(self) -> np.ndarray:
        """Convert to [mu, sigma, df, is_student_t] array for Numba."""
        df = self.df if self.df is not None else 0.0
        is_t = 1.0 if self.is_student_t else 0.0
        return np.array([self.mu, self.sigma, df, is_t], dtype=np.float64)


class SimulationResult(NamedTuple):
    """
    Output of simulate_batch().
    
    Attributes:
        returns: Matrix of r_t values, shape (n_sims, t_steps)
                 returns[i, t] = r_{t+1} in LaTeX notation
        regimes: Matrix of S_t values, shape (n_sims, t_steps)
                 regimes[i, t] = hidden state at step t (for debugging)
    """
    returns: np.ndarray
    regimes: np.ndarray


# =============================================================================
# Task 2: The Environment Class
# =============================================================================

class MarketEnvironment:
    """
    The "Physics Engine" for market simulation.
    
    This class encapsulates the stochastic data generating process (DGP)
    specified in theory/model_definition.tex Section 3.
    
    Features:
        - Supports multiple regimes (e.g., bull/bear)
        - Hidden Markov Model (HMM) regime switching
        - Gaussian and Student-t return distributions
        - Batch generation for Common Random Numbers (CRN)
        - Full reproducibility via seeding
    
    Example:
        >>> config = {
        ...     'bull': {'mu': 0.05, 'sigma': 0.10},
        ...     'crisis': {'mu': -0.20, 'sigma': 0.40, 'df': 3}
        ... }
        >>> trans_matrix = np.array([[0.95, 0.05], [0.10, 0.90]])
        >>> env = MarketEnvironment(config, trans_matrix, seed=42)
        >>> result = env.simulate_batch(n_sims=1000, t_steps=252)
        >>> print(result.returns.shape)  # (1000, 252)
    
    Timing Contract (NO LOOK-AHEAD):
        The environment ONLY yields r_t AFTER the agent has committed to f_t.
        This is enforced by the batch generation paradigm: the agent never
        sees the return matrix during decision-making.
    """
    
    def __init__(
        self,
        regime_config: Dict[str, Dict[str, Any]],
        trans_matrix: np.ndarray,
        initial_regime: int = 0,
        seed: Optional[int] = None
    ):
        """
        Initialize the MarketEnvironment.
        
        Args:
            regime_config: Dictionary defining regimes. Each key is a regime name,
                          each value is a dict with keys:
                          - 'mu': Mean return
                          - 'sigma': Volatility
                          - 'df': Degrees of freedom (optional, for Student-t)
                          Example: {'bull': {'mu': 0.05, 'sigma': 0.1},
                                   'crisis': {'mu': -0.2, 'sigma': 0.4, 'df': 3}}
            
            trans_matrix: Transition matrix P of shape (M, M) where M is the
                         number of regimes. P[i,j] = P(S_{t+1} = j | S_t = i).
                         Rows must sum to 1.
            
            initial_regime: Starting regime S_0 (0-indexed)
            
            seed: Random seed for reproducibility
        
        Raises:
            ValueError: If trans_matrix rows don't sum to 1 or dimensions mismatch
        """
        # Parse regime configurations
        self.regime_names = list(regime_config.keys())
        self.n_regimes = len(self.regime_names)
        
        self.regimes = []
        for name, params in regime_config.items():
            self.regimes.append(RegimeConfig(
                mu=params['mu'],
                sigma=params['sigma'],
                df=params.get('df', None),
                name=name
            ))
        
        # Build regime parameter array for Numba
        self._regime_params = np.vstack([r.to_array() for r in self.regimes])
        
        # Validate and store transition matrix
        self.trans_matrix = np.asarray(trans_matrix, dtype=np.float64)
        self._validate_trans_matrix()
        
        self.initial_regime = initial_regime
        self.seed = seed
        
        # Determine df for chi-square sampling (use max df across regimes)
        dfs = [r.df for r in self.regimes if r.df is not None]
        self._df_for_chi2 = max(dfs) if dfs else 3.0  # Default to 3 if all Gaussian
    
    def _validate_trans_matrix(self):
        """Validate transition matrix properties."""
        if self.trans_matrix.shape != (self.n_regimes, self.n_regimes):
            raise ValueError(
                f"Transition matrix shape {self.trans_matrix.shape} does not match "
                f"number of regimes {self.n_regimes}"
            )
        
        row_sums = self.trans_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError(
                f"Transition matrix rows must sum to 1. Got row sums: {row_sums}"
            )
        
        if np.any(self.trans_matrix < 0):
            raise ValueError("Transition matrix cannot have negative entries")
    
    def simulate_batch(
        self,
        n_sims: int,
        t_steps: int,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Pre-generate the entire matrix of returns (N_sims × T_steps).
        
        This is the critical method for efficient simulation. It pre-generates
        all random numbers upfront, enabling:
        1. Common Random Numbers (CRN) methodology for fair agent comparison
        2. Vectorized/parallel computation via Numba
        3. Reproducible experiments
        
        Args:
            n_sims: Number of simulation paths N_sims (e.g., 10000)
            t_steps: Number of time steps T (e.g., 252 for 1 trading year)
            seed: Override seed for this batch (uses instance seed if None)
        
        Returns:
            SimulationResult containing:
                - returns: Matrix of shape (n_sims, t_steps)
                          returns[i, t] = r_{t+1} (return at end of step t)
                - regimes: Matrix of shape (n_sims, t_steps)
                          regimes[i, t] = S_{t+1} (hidden state at step t)
        
        Matrix Indexing Convention:
            Python index t (0-indexed) corresponds to LaTeX r_{t+1}
            - returns[i, 0] = r_1 (first return, revealed at time 1)
            - returns[i, T-1] = r_T (last return, revealed at time T)
        
        Example:
            >>> result = env.simulate_batch(n_sims=100, t_steps=50)
            >>> assert result.returns.shape == (100, 50)
            >>> assert result.regimes.shape == (100, 50)
        """
        # Use provided seed or instance seed
        effective_seed = seed if seed is not None else self.seed
        
        # Pre-generate all random samples
        uniform_samples, normal_samples, chi2_samples = generate_random_samples(
            n_sims=n_sims,
            n_steps=t_steps,
            df=self._df_for_chi2,
            seed=effective_seed
        )
        
        # Generate hidden state paths (Markov chain)
        regimes = sample_markov_chain_batch(
            n_sims=n_sims,
            n_steps=t_steps,
            trans_matrix=self.trans_matrix,
            initial_state=self.initial_regime,
            random_states=uniform_samples
        )
        
        # Generate returns based on regimes
        returns = sample_regime_returns_batch(
            n_sims=n_sims,
            n_steps=t_steps,
            states=regimes,
            regime_params=self._regime_params,
            normal_samples=normal_samples,
            chi2_samples=chi2_samples
        )
        
        return SimulationResult(returns=returns, regimes=regimes)
    
    def get_regime_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Return a summary of regime parameters.
        
        Returns:
            Dictionary mapping regime names to their parameters.
        """
        return {
            r.name: {
                'mu': r.mu,
                'sigma': r.sigma,
                'df': r.df,
                'distribution': 'Student-t' if r.is_student_t else 'Gaussian'
            }
            for r in self.regimes
        }
    
    def __repr__(self) -> str:
        regime_info = ", ".join(
            f"{r.name}(μ={r.mu:.3f}, σ={r.sigma:.3f})" 
            for r in self.regimes
        )
        return f"MarketEnvironment(regimes=[{regime_info}], seed={self.seed})"


# =============================================================================
# Factory Functions for Common Configurations
# =============================================================================

def create_bull_bear_environment(
    bull_mu: float = 0.08,
    bull_sigma: float = 0.15,
    bear_mu: float = -0.05,
    bear_sigma: float = 0.25,
    bear_df: float = 3.0,
    p_bull_to_bear: float = 0.05,
    p_bear_to_bull: float = 0.10,
    seed: Optional[int] = None
) -> MarketEnvironment:
    """
    Create a standard bull/bear market environment.
    
    Default parameters:
        - Bull: N(8%, 15%) - typical equity market
        - Bear: t_3(-5%, 25%) - crisis with heavy tails
        - P(bull→bear) = 5%, P(bear→bull) = 10%
    
    Args:
        bull_mu: Bull market mean return
        bull_sigma: Bull market volatility
        bear_mu: Bear market mean return  
        bear_sigma: Bear market volatility
        bear_df: Degrees of freedom for bear market (heavy tails)
        p_bull_to_bear: Transition probability from bull to bear
        p_bear_to_bull: Transition probability from bear to bull
        seed: Random seed
    
    Returns:
        Configured MarketEnvironment instance
    """
    config = {
        'bull': {'mu': bull_mu, 'sigma': bull_sigma},
        'bear': {'mu': bear_mu, 'sigma': bear_sigma, 'df': bear_df}
    }
    
    trans_matrix = np.array([
        [1 - p_bull_to_bear, p_bull_to_bear],
        [p_bear_to_bull, 1 - p_bear_to_bull]
    ])
    
    return MarketEnvironment(config, trans_matrix, initial_regime=0, seed=seed)


def create_single_regime_environment(
    mu: float = 0.08,
    sigma: float = 0.15,
    df: Optional[float] = None,
    seed: Optional[int] = None
) -> MarketEnvironment:
    """
    Create a simple single-regime environment (no regime switching).
    
    Args:
        mu: Mean return
        sigma: Volatility
        df: Degrees of freedom (None for Gaussian, 3 for heavy-tailed)
        seed: Random seed
    
    Returns:
        Configured MarketEnvironment with single regime
    """
    config = {
        'market': {'mu': mu, 'sigma': sigma, 'df': df}
    }
    
    trans_matrix = np.array([[1.0]])  # Always stay in same regime
    
    return MarketEnvironment(config, trans_matrix, initial_regime=0, seed=seed)
