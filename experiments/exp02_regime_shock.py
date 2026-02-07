"""
Experiment 2: The Regime Shock (Adaptivity Test)

Scenario: 500 steps "Bull Market" (μ=0.05) followed by 500 steps "Bear Market" (μ=-0.05)

Compares:
- NaiveBayesKelly: The control (no regime awareness)
- HMMKellyAgent: Uses Hidden Markov Model to detect regime switch

Hypothesis: HMMKellyAgent will detect the bear market around t=520 and cut leverage,
saving its wealth, while NaiveBayesKelly continues betting and goes bust.

Output: 
1. Wealth evolution over time for both agents
2. Regime probability (P(Bear)) from HMMKellyAgent
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.market_env import MarketEnv
from simulation.agents import NaiveBayesKelly, HMMKellyAgent, KellyOracle


class RegimeShockEnvironment:
    """
    Custom environment for regime shock experiment.
    
    Generates returns from two different distributions:
    - Bull phase (t < switch_point): N(μ_bull, σ_bull)
    - Bear phase (t >= switch_point): N(μ_bear, σ_bear)
    """
    
    def __init__(
        self,
        T: int,
        switch_point: int,
        bull_params: dict,
        bear_params: dict,
        seed: int = None
    ):
        self.T = T
        self.switch_point = switch_point
        self.bull_params = bull_params
        self.bear_params = bear_params
        
        self.rng = np.random.default_rng(seed)
        
        # State
        self.wealth = 1.0
        self.step_count = 0
        self.done = False
        
        # History
        self.wealth_history = [1.0]
        self.return_history = []
        self.regime_history = []
    
    def reset(self, seed: int = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.wealth = 1.0
        self.step_count = 0
        self.done = False
        
        self.wealth_history = [1.0]
        self.return_history = []
        self.regime_history = []
        
        return np.array([0.0])
    
    def step(self, bet_fraction: float):
        """
        Execute one step.
        
        Args:
            bet_fraction: Fraction of wealth to bet on "risky asset"
        
        Returns:
            (return, done, info)
        """
        if self.done:
            raise RuntimeError("Episode done")
        
        # Determine regime
        if self.step_count < self.switch_point:
            regime = 'bull'
            params = self.bull_params
        else:
            regime = 'bear'
            params = self.bear_params
        
        self.regime_history.append(regime)
        
        # Sample return
        mu = params['mu']
        sigma = params['sigma']
        r = self.rng.normal(mu, sigma)
        
        self.return_history.append(r)
        
        # Clip bet to [0, 1]
        f = np.clip(bet_fraction, 0.0, 1.0)
        
        # Portfolio return: (1-f)*rf + f*r ≈ f*r for rf=0
        portfolio_return = f * r
        
        # Update wealth
        new_wealth = self.wealth * (1 + portfolio_return)
        self.wealth = max(new_wealth, 1e-10)
        
        self.wealth_history.append(self.wealth)
        self.step_count += 1
        
        # Check termination
        ruined = self.wealth < 1e-6
        horizon_reached = self.step_count >= self.T
        self.done = ruined or horizon_reached
        
        # Create outcome array for agent update (positive if r > 0, else negative)
        outcome = np.array([r])
        
        return outcome, self.done, {'regime': regime, 'return': r, 'wealth': self.wealth}


def run_experiment_2(
    n_runs: int = 50,
    T: int = 1000,
    switch_point: int = 500,
    output_dir: str = 'experiments/output'
):
    """
    Run Experiment 2: Regime Shock comparison.
    """
    print("=" * 60)
    print("EXPERIMENT 2: The Regime Shock (Adaptivity Test)")
    print("=" * 60)
    
    bull_params = {'mu': 0.02, 'sigma': 0.1}  # Bull: positive drift
    bear_params = {'mu': -0.02, 'sigma': 0.15}  # Bear: negative drift, higher vol
    
    print(f"\nScenario:")
    print(f"  Bull Market (t < {switch_point}): μ={bull_params['mu']}, σ={bull_params['sigma']}")
    print(f"  Bear Market (t >= {switch_point}): μ={bear_params['mu']}, σ={bear_params['sigma']}")
    print(f"  Runs: {n_runs}, Steps: {T}")
    
    # Define regime params for HMMKellyAgent
    regime_params = {
        'bull': bull_params,
        'bear': bear_params
    }
    
    # Storage for results
    results = {
        'NaiveBayes': {'wealth_trajectories': [], 'final_wealth': []},
        'HMMKelly': {'wealth_trajectories': [], 'final_wealth': [], 'bear_prob': []}
    }
    
    for run in range(n_runs):
        seed = 2000 + run
        
        # Run NaiveBayesKelly
        env = RegimeShockEnvironment(T, switch_point, bull_params, bear_params, seed)
        agent = NaiveBayesKelly(n_arms=1)
        
        env.reset()
        agent.reset()
        
        for t in range(T):
            bet = agent.act()[0]
            outcome, done, info = env.step(bet)
            agent.update(outcome)
            if done:
                break
        
        results['NaiveBayes']['wealth_trajectories'].append(env.wealth_history)
        results['NaiveBayes']['final_wealth'].append(env.wealth)
        
        # Run HMMKellyAgent
        env2 = RegimeShockEnvironment(T, switch_point, bull_params, bear_params, seed)
        agent2 = HMMKellyAgent(n_arms=1, regime_params=regime_params, ewma_alpha=0.05)
        
        env2.reset()
        agent2.reset()
        
        bear_probs = []
        for t in range(T):
            # Get bear probability before acting
            bear_prob = agent2.get_regime_probability('bear')
            bear_probs.append(bear_prob)
            
            bet = agent2.act()[0]
            outcome, done, info = env2.step(bet)
            agent2.update(outcome)
            if done:
                break
        
        results['HMMKelly']['wealth_trajectories'].append(env2.wealth_history)
        results['HMMKelly']['final_wealth'].append(env2.wealth)
        results['HMMKelly']['bear_prob'].append(bear_probs)
    
    # Statistics
    print("\nResults:")
    for name, data in results.items():
        if data['final_wealth']:
            median = np.median(data['final_wealth'])
            mean = np.mean(data['final_wealth'])
            survival_rate = np.mean([w > 0.01 for w in data['final_wealth']])
            print(f"  {name}: median={median:.4f}, mean={mean:.4f}, survival={survival_rate:.1%}")
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Average wealth trajectories
    ax1 = axes[0]
    
    # Pad trajectories to same length
    max_len = T + 1
    
    for name, color in [('NaiveBayes', 'blue'), ('HMMKelly', 'green')]:
        trajectories = results[name]['wealth_trajectories']
        
        # Pad trajectories
        padded = []
        for traj in trajectories:
            if len(traj) < max_len:
                traj = list(traj) + [traj[-1]] * (max_len - len(traj))
            padded.append(traj[:max_len])
        
        log_wealth = np.log10(np.array(padded) + 1e-10)
        avg = np.mean(log_wealth, axis=0)
        std = np.std(log_wealth, axis=0)
        
        steps = np.arange(max_len)
        ax1.plot(steps, avg, color=color, linewidth=2, label=name)
        ax1.fill_between(steps, avg - std, avg + std, color=color, alpha=0.2)
    
    # Mark regime switch
    ax1.axvline(x=switch_point, color='red', linestyle='--', linewidth=2, 
                label='Regime Switch')
    
    # Add regime labels
    ax1.text(switch_point / 2, ax1.get_ylim()[1] * 0.95, 'BULL', 
             ha='center', fontsize=12, color='green', fontweight='bold')
    ax1.text(switch_point + (T - switch_point) / 2, ax1.get_ylim()[1] * 0.95, 'BEAR', 
             ha='center', fontsize=12, color='red', fontweight='bold')
    
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Log10(Wealth)', fontsize=12)
    ax1.set_title('Wealth Evolution: NaiveBayes vs HMMKelly', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bear regime probability (HMMKelly)
    ax2 = axes[1]
    
    bear_probs = results['HMMKelly']['bear_prob']
    max_len_probs = T
    
    padded_probs = []
    for probs in bear_probs:
        if len(probs) < max_len_probs:
            probs = list(probs) + [probs[-1]] * (max_len_probs - len(probs))
        padded_probs.append(probs[:max_len_probs])
    
    avg_prob = np.mean(padded_probs, axis=0)
    std_prob = np.std(padded_probs, axis=0)
    
    steps = np.arange(max_len_probs)
    ax2.plot(steps, avg_prob, color='purple', linewidth=2, label='P(Bear)')
    ax2.fill_between(steps, avg_prob - std_prob, avg_prob + std_prob, 
                     color='purple', alpha=0.2)
    
    # Mark regime switch
    ax2.axvline(x=switch_point, color='red', linestyle='--', linewidth=2, 
                label='Actual Switch')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Time Step', fontsize=12)
    ax2.set_ylabel('P(Bear Regime)', fontsize=12)
    ax2.set_title('HMMKelly Regime Detection', fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / 'exp02_regime_shock.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    return results


if __name__ == "__main__":
    results = run_experiment_2(n_runs=50)
