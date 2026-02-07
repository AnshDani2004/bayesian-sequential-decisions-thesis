"""
Risk-Constrained Kelly Agent with CPPI-like Floor Protection

Implements the Busseti, Ryu, Boyd (2016) framework:
- Dynamic leverage based on distance to floor
- Never breach maximum drawdown constraint
- CPPI-like portfolio insurance

Mathematical Basis:
λ_t = min(1, (W_t - Floor) / (Estimated Risk))
f_constrained = λ_t * f_kelly

Key Behavior:
- As wealth approaches floor, leverage → 0
- As wealth grows, leverage → full Kelly
- Guarantees (in continuous time) wealth stays above floor
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass 
class DrawdownState:
    """Track drawdown metrics."""
    peak_wealth: float
    current_drawdown: float
    floor_wealth: float
    risk_capacity: float


class RiskConstrainedKelly:
    """
    Kelly betting with explicit drawdown constraints.
    
    Uses CPPI-like (Constant Proportion Portfolio Insurance) logic:
    - Floor = (1 - max_drawdown) * Peak_Wealth
    - Cushion = Current_Wealth - Floor
    - Leverage = min(1, m * Cushion / Current_Wealth)
    
    where m is a multiplier controlling aggressiveness.
    
    Expected Behavior:
    - Never breach max_drawdown on 95%+ of paths
    - Dynamically reduce leverage as drawdown increases
    - Recover leverage as wealth grows
    """
    
    def __init__(
        self,
        n_arms: int,
        true_probs: Optional[List[float]] = None,
        odds: Optional[List[float]] = None,
        max_drawdown: float = 0.20,
        cppi_multiplier: float = 3.0,
        min_leverage: float = 0.01,
        max_leverage: float = 0.50
    ):
        """
        Initialize Risk-Constrained Kelly agent.
        
        Args:
            n_arms: Number of betting arms
            true_probs: True win probabilities (if known)
            odds: Betting odds
            max_drawdown: Maximum allowed drawdown (e.g., 0.20 = 20%)
            cppi_multiplier: CPPI multiplier m (higher = more aggressive)
            min_leverage: Minimum leverage (avoid going to zero)
            max_leverage: Maximum leverage cap
        """
        self.n_arms = n_arms
        self.odds = np.array(odds) if odds is not None else np.ones(n_arms)
        self.max_drawdown = max_drawdown
        self.cppi_multiplier = cppi_multiplier
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        
        # If true probs known, compute base Kelly
        if true_probs is not None:
            self.true_probs = np.array(true_probs)
            self.base_kelly = self._compute_kelly(self.true_probs)
        else:
            self.true_probs = None
            self.base_kelly = np.full(n_arms, 0.1 / n_arms)  # Default
        
        # For Bayesian learning (if true_probs unknown)
        self.alphas = np.ones(n_arms)  # Beta prior
        self.betas = np.ones(n_arms)
        
        # Wealth tracking
        self.wealth = 1.0
        self.peak_wealth = 1.0
        self.floor_wealth = 1.0 - max_drawdown
        
        # History
        self.leverage_history: List[float] = []
        self.drawdown_history: List[float] = []
        self.wealth_history: List[float] = [1.0]
    
    def _compute_kelly(self, probs: np.ndarray) -> np.ndarray:
        """Compute Kelly fractions for given probabilities."""
        # Kelly: f* = (b*p - q) / b = p - q/b
        q = 1.0 - probs
        kelly = probs - q / self.odds
        kelly = np.clip(kelly, 0.0, 1.0)
        
        # Normalize to sum ≤ 1
        total = np.sum(kelly)
        if total > 1.0:
            kelly = kelly / total
        
        return kelly
    
    def _get_estimated_kelly(self) -> np.ndarray:
        """Get Kelly fractions (true or estimated)."""
        if self.true_probs is not None:
            return self.base_kelly.copy()
        else:
            # Use posterior mean
            p_hat = self.alphas / (self.alphas + self.betas)
            return self._compute_kelly(p_hat)
    
    def _compute_leverage_multiplier(self) -> float:
        """
        Compute CPPI-style leverage multiplier.
        
        λ = min(1, m * cushion / wealth)
        
        where cushion = wealth - floor
        """
        # Update floor based on peak
        self.floor_wealth = self.peak_wealth * (1.0 - self.max_drawdown)
        
        # Cushion: how far above floor
        cushion = self.wealth - self.floor_wealth
        
        # If below floor (shouldn't happen), emergency delever
        if cushion <= 0:
            return self.min_leverage
        
        # CPPI multiplier
        leverage = self.cppi_multiplier * cushion / self.wealth
        
        # Clip to bounds
        leverage = np.clip(leverage, self.min_leverage, self.max_leverage)
        
        return float(leverage)
    
    def act(self, observation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute risk-constrained bet.
        
        f_constrained = λ * f_kelly
        
        where λ decreases as we approach the floor.
        """
        # Get base Kelly fractions
        base_kelly = self._get_estimated_kelly()
        
        # Compute leverage multiplier
        leverage = self._compute_leverage_multiplier()
        self.leverage_history.append(leverage)
        
        # Scale Kelly by leverage
        constrained = base_kelly * leverage
        
        # Ensure sum ≤ 1
        total = np.sum(constrained)
        if total > 1.0:
            constrained = constrained / total
        
        return constrained
    
    def update(self, outcomes: np.ndarray, wealth: Optional[float] = None):
        """
        Update agent state after observing outcomes.
        
        Args:
            outcomes: Binary outcomes for each arm
            wealth: New wealth level (optional, for external tracking)
        """
        # Update Bayesian posteriors
        wins = (outcomes > 0).astype(int)
        self.alphas += wins
        self.betas += (1 - wins)
        
        # Update wealth tracking
        if wealth is not None:
            self.wealth = wealth
            self.wealth_history.append(wealth)
            
            # Update peak
            if wealth > self.peak_wealth:
                self.peak_wealth = wealth
            
            # Track drawdown
            drawdown = (self.peak_wealth - self.wealth) / self.peak_wealth
            self.drawdown_history.append(drawdown)
    
    def reset(self):
        """Reset agent state."""
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)
        self.wealth = 1.0
        self.peak_wealth = 1.0
        self.floor_wealth = 1.0 - self.max_drawdown
        self.leverage_history = []
        self.drawdown_history = []
        self.wealth_history = [1.0]
    
    def get_state(self) -> DrawdownState:
        """Get current drawdown state."""
        drawdown = (self.peak_wealth - self.wealth) / self.peak_wealth
        risk_capacity = self._compute_leverage_multiplier()
        
        return DrawdownState(
            peak_wealth=self.peak_wealth,
            current_drawdown=drawdown,
            floor_wealth=self.floor_wealth,
            risk_capacity=risk_capacity
        )
    
    def get_drawdown_metrics(self) -> dict:
        """Get summary of drawdown performance."""
        if not self.drawdown_history:
            return {'max_drawdown': 0.0, 'avg_drawdown': 0.0}
        
        return {
            'max_drawdown': float(np.max(self.drawdown_history)),
            'avg_drawdown': float(np.mean(self.drawdown_history)),
            'final_wealth': self.wealth,
            'breached_limit': float(np.max(self.drawdown_history)) > self.max_drawdown
        }


# =============================================================================
# Factory functions
# =============================================================================

def create_risk_constrained_kelly(
    n_arms: int = 1,
    true_probs: Optional[List[float]] = None,
    odds: Optional[List[float]] = None,
    max_drawdown: float = 0.20
) -> RiskConstrainedKelly:
    """Create a Risk-Constrained Kelly agent with default params."""
    return RiskConstrainedKelly(
        n_arms=n_arms,
        true_probs=true_probs,
        odds=odds,
        max_drawdown=max_drawdown
    )
