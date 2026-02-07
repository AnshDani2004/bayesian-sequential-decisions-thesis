"""
Volatility-Augmented HMM for Regime Detection

This module implements a refined HMM that addresses the Phase 4 failure:
1. 2D observations: [return, rolling_volatility]
2. Log-space forward algorithm with logsumexp
3. CUSUM change-point detection for rapid regime switches
4. Constrained transition matrix (A_ii ≤ 0.95)

Mathematical Basis:
- Emission: b_j(y_t) = N(r_t; μ_j, σ_j) × Gamma(v_t; k_j, θ_j)
- Forward: ln α_t(j) = ln b_j(y_t) + logsumexp_i(ln α_{t-1}(i) + ln A_ij)
- CUSUM: S_t = max(0, S_{t-1} + (r_t - μ_bull) / σ_bull - k)

Reference: Busseti, Ryu, Boyd (2016) for risk-constrained optimization
"""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, gamma
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RegimeState:
    """Container for regime-specific parameters."""
    name: str
    mu: float          # Mean return
    sigma: float       # Return volatility
    vol_shape: float   # Gamma shape for volatility feature
    vol_scale: float   # Gamma scale for volatility feature


class VolAugmentedHMM:
    """
    Hidden Markov Model with volatility-augmented observations.
    
    Fixes Phase 4 failure through:
    1. 2D observation space: [return, rolling_volatility]
    2. Log-space forward algorithm (numerical stability)
    3. CUSUM detector for rapid change-point detection
    4. Constrained transition probabilities
    
    Expected Behavior:
    - Detect bear regime within 20 steps of true switch
    - P(Bear) should spike when volatility increases
    """
    
    def __init__(
        self,
        regime_params: Dict[str, dict],
        vol_window: int = 5,
        max_persistence: float = 0.95,
        cusum_threshold: float = 3.0,
        cusum_drift: float = 0.5
    ):
        """
        Initialize Vol-Augmented HMM.
        
        Args:
            regime_params: Dict of regime parameters
                {'bull': {'mu': 0.02, 'sigma': 0.1, 'vol_shape': 2.0, 'vol_scale': 0.05},
                 'bear': {'mu': -0.02, 'sigma': 0.2, 'vol_shape': 3.0, 'vol_scale': 0.1}}
            vol_window: Window size for rolling volatility
            max_persistence: Maximum A_ii (prevents sticky priors)
            cusum_threshold: CUSUM detection threshold
            cusum_drift: CUSUM drift parameter k
        """
        self.regime_names = list(regime_params.keys())
        self.n_regimes = len(self.regime_names)
        self.vol_window = vol_window
        self.max_persistence = max_persistence
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        
        # Parse regime parameters
        self.regimes: List[RegimeState] = []
        for name in self.regime_names:
            params = regime_params[name]
            self.regimes.append(RegimeState(
                name=name,
                mu=params.get('mu', 0.0),
                sigma=params.get('sigma', 0.1),
                vol_shape=params.get('vol_shape', 2.0),
                vol_scale=params.get('vol_scale', 0.05)
            ))
        
        # Build transition matrix (constrained)
        # Off-diagonal: (1 - max_persistence) / (n_regimes - 1)
        off_diag = (1.0 - max_persistence) / max(1, self.n_regimes - 1)
        self.log_trans = np.full((self.n_regimes, self.n_regimes), np.log(off_diag))
        np.fill_diagonal(self.log_trans, np.log(max_persistence))
        
        # State variables
        self.log_beliefs = np.log(np.ones(self.n_regimes) / self.n_regimes)
        self.return_history: List[float] = []
        self.vol_history: List[float] = []
        self.cusum_pos: float = 0.0  # CUSUM for positive shift
        self.cusum_neg: float = 0.0  # CUSUM for negative shift
        self.cusum_triggered: bool = False
        self.step_count: int = 0
        
        # Tracking
        self.belief_history: List[np.ndarray] = []
        self.cusum_history: List[Tuple[float, float]] = []
    
    def _compute_rolling_vol(self) -> float:
        """Compute rolling volatility from return history."""
        if len(self.return_history) < 2:
            return 0.05  # Default volatility
        
        window = self.return_history[-self.vol_window:]
        return float(np.std(window)) + 1e-6  # Avoid zero
    
    def _compute_log_emission(self, r_t: float, v_t: float) -> np.ndarray:
        """
        Compute log emission probabilities for each regime.
        
        Uses product of:
        - Gaussian likelihood for returns
        - Gamma likelihood for volatility
        
        Args:
            r_t: Current return
            v_t: Current rolling volatility
        
        Returns:
            log_probs: Shape (n_regimes,)
        """
        log_probs = np.zeros(self.n_regimes)
        
        for i, regime in enumerate(self.regimes):
            # Gaussian log-likelihood for return
            log_return = norm.logpdf(r_t, loc=regime.mu, scale=regime.sigma)
            
            # Gamma log-likelihood for volatility
            # Gamma(v; k, θ) where mean = k*θ, var = k*θ²
            log_vol = gamma.logpdf(v_t, a=regime.vol_shape, scale=regime.vol_scale)
            
            # Product of likelihoods
            log_probs[i] = log_return + log_vol
        
        return log_probs
    
    def _update_cusum(self, r_t: float):
        """
        Update CUSUM statistics for change-point detection.
        
        CUSUM detects shifts in mean by accumulating deviations.
        Triggers when cumulative sum exceeds threshold.
        
        S_t^+ = max(0, S_{t-1}^+ + (r_t - μ_bull)/σ_bull - k)
        S_t^- = max(0, S_{t-1}^- - (r_t - μ_bull)/σ_bull - k)
        """
        # Use bull regime as reference
        bull_idx = self.regime_names.index('bull') if 'bull' in self.regime_names else 0
        bull = self.regimes[bull_idx]
        
        # Standardized deviation
        z_t = (r_t - bull.mu) / bull.sigma
        
        # Update CUSUM
        self.cusum_neg = max(0.0, self.cusum_neg - z_t - self.cusum_drift)
        self.cusum_pos = max(0.0, self.cusum_pos + z_t - self.cusum_drift)
        
        # Check threshold
        if self.cusum_neg > self.cusum_threshold:
            self.cusum_triggered = True
        
        self.cusum_history.append((self.cusum_pos, self.cusum_neg))
    
    def update(self, r_t: float) -> np.ndarray:
        """
        Update regime beliefs with new observation.
        
        Implements log-space forward algorithm:
        ln α_t(j) = ln b_j(y_t) + logsumexp_i(ln α_{t-1}(i) + ln A_ij)
        
        Args:
            r_t: Current return observation
        
        Returns:
            beliefs: Probability of each regime (normalized)
        """
        # Store return
        self.return_history.append(r_t)
        
        # Compute rolling volatility
        v_t = self._compute_rolling_vol()
        self.vol_history.append(v_t)
        
        # Update CUSUM
        self._update_cusum(r_t)
        
        # Compute log emission probabilities
        log_emission = self._compute_log_emission(r_t, v_t)
        
        # If CUSUM triggered, boost bear likelihood
        if self.cusum_triggered:
            bear_idx = self.regime_names.index('bear') if 'bear' in self.regime_names else 1
            log_emission[bear_idx] += 2.0  # Additive boost in log-space
        
        # Forward step in log-space
        # ln α_t(j) = ln b_j(y_t) + logsumexp_i(ln α_{t-1}(i) + ln A_ij)
        new_log_beliefs = np.zeros(self.n_regimes)
        
        for j in range(self.n_regimes):
            # Transition from all states to state j
            log_transition = self.log_beliefs + self.log_trans[:, j]
            new_log_beliefs[j] = log_emission[j] + logsumexp(log_transition)
        
        # Normalize in log-space
        self.log_beliefs = new_log_beliefs - logsumexp(new_log_beliefs)
        
        # Store history
        beliefs = np.exp(self.log_beliefs)
        self.belief_history.append(beliefs.copy())
        self.step_count += 1
        
        return beliefs
    
    def get_beliefs(self) -> Dict[str, float]:
        """Get current regime probabilities as dict."""
        beliefs = np.exp(self.log_beliefs)
        return dict(zip(self.regime_names, beliefs.tolist()))
    
    def get_regime_probability(self, regime_name: str) -> float:
        """Get probability of specific regime."""
        if regime_name in self.regime_names:
            idx = self.regime_names.index(regime_name)
            return float(np.exp(self.log_beliefs[idx]))
        return 0.0
    
    def reset(self):
        """Reset to uniform beliefs."""
        self.log_beliefs = np.log(np.ones(self.n_regimes) / self.n_regimes)
        self.return_history = []
        self.vol_history = []
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.cusum_triggered = False
        self.step_count = 0
        self.belief_history = []
        self.cusum_history = []
    
    def detect_regime_switch(self, threshold: float = 0.5) -> Optional[str]:
        """
        Detect if regime has switched based on belief threshold.
        
        Returns regime name if P(regime) > threshold, else None.
        """
        beliefs = np.exp(self.log_beliefs)
        max_belief = np.max(beliefs)
        
        if max_belief > threshold:
            max_idx = np.argmax(beliefs)
            return self.regime_names[max_idx]
        return None


class VolAugmentedHMMKelly:
    """
    Kelly betting agent using Vol-Augmented HMM for regime detection.
    
    Combines:
    1. VolAugmentedHMM for regime inference
    2. Regime-specific Kelly fractions
    3. Dynamic leverage based on P(Bear)
    
    Key behavior:
    - In Bull: bet near-full Kelly
    - In Bear: reduce to 10% Kelly or go to cash
    """
    
    def __init__(
        self,
        n_arms: int,
        regime_params: Dict[str, dict],
        odds: Optional[List[float]] = None,
        vol_window: int = 5,
        bear_leverage: float = 0.1
    ):
        """
        Initialize Vol-Augmented HMM Kelly agent.
        
        Args:
            n_arms: Number of betting arms
            regime_params: HMM regime parameters
            odds: Betting odds
            vol_window: Volatility calculation window
            bear_leverage: Leverage multiplier when in bear regime
        """
        self.n_arms = n_arms
        self.odds = np.array(odds) if odds is not None else np.ones(n_arms)
        self.bear_leverage = bear_leverage
        
        # Create HMM
        self.hmm = VolAugmentedHMM(regime_params, vol_window=vol_window)
        
        # Track bets
        self.bet_history: List[np.ndarray] = []
    
    def act(self) -> np.ndarray:
        """
        Compute regime-weighted Kelly bet.
        
        f = P(Bull) * f_bull + P(Bear) * f_bear
        where f_bear = bear_leverage * f_bull
        """
        beliefs = self.hmm.get_beliefs()
        
        p_bull = beliefs.get('bull', 0.5)
        p_bear = beliefs.get('bear', 0.5)
        
        # Bull regime: assume favorable odds (μ > 0)
        # Bear regime: assume unfavorable (μ < 0), so bet less
        
        # Get regime parameters for expected return
        bull_mu = 0.02  # Default
        bear_mu = -0.02
        
        for regime in self.hmm.regimes:
            if regime.name == 'bull':
                bull_mu = regime.mu
            elif regime.name == 'bear':
                bear_mu = regime.mu
        
        # Kelly fraction based on regime-weighted expected return
        # In bear regime, cut leverage significantly
        weighted_mu = p_bull * bull_mu + p_bear * bear_mu
        
        # If weighted expected return is negative, go to cash
        if weighted_mu <= 0:
            leverage = 0.0
        else:
            # Scale by P(Bull) - more confident = more leverage
            leverage = p_bull * 0.5  # Max 50% leverage in full bull
        
        # Clip to safety bounds
        leverage = np.clip(leverage, 0.0, 0.5)
        
        # Distribute across arms
        bets = np.full(self.n_arms, leverage / self.n_arms)
        self.bet_history.append(bets.copy())
        
        return bets
    
    def update(self, r_t: float):
        """Update HMM with new return observation."""
        self.hmm.update(r_t)
    
    def reset(self):
        """Reset agent state."""
        self.hmm.reset()
        self.bet_history = []
    
    def get_state(self) -> dict:
        """Get current agent state."""
        return {
            'regime_beliefs': self.hmm.get_beliefs(),
            'cusum_triggered': self.hmm.cusum_triggered,
            'step_count': self.hmm.step_count
        }


# =============================================================================
# Factory function
# =============================================================================

def create_vol_augmented_hmm_kelly(
    n_arms: int = 1,
    bull_mu: float = 0.02,
    bull_sigma: float = 0.1,
    bear_mu: float = -0.02,
    bear_sigma: float = 0.2,
    odds: Optional[List[float]] = None
) -> VolAugmentedHMMKelly:
    """
    Factory for creating Vol-Augmented HMM Kelly agent with default params.
    """
    regime_params = {
        'bull': {
            'mu': bull_mu,
            'sigma': bull_sigma,
            'vol_shape': 2.0,
            'vol_scale': bull_sigma / 2
        },
        'bear': {
            'mu': bear_mu,
            'sigma': bear_sigma,
            'vol_shape': 4.0,
            'vol_scale': bear_sigma / 2
        }
    }
    
    return VolAugmentedHMMKelly(n_arms, regime_params, odds)
