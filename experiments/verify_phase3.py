"""
Phase 3 Verification Script: Sanity Checks for Agent-Environment Loop.

This script performs two critical scientific validations before proceeding
to novel experiments:

Experiment A: The Martingale Test (Stationarity)
    - Verifies E[μ_{t+1} | F_t] = μ_t for Bayesian posterior
    - Pass: Average posterior mean is flat at p_true

Experiment B: The Growth Hierarchy (Optimality)
    - Verifies Kelly optimality and overbet penalty
    - Pass: Oracle > HalfKelly, DoubleKelly ≈ 0

Reference: theory/model_definition.tex (Phase 2 Review)

CRITICAL: If these tests fail, DO NOT PROCEED. The "engine" of the thesis
(Bayesian updating or wealth dynamics) is flawed.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.market_env import MarketEnv, generate_pregenerated_outcomes
from simulation.agents import (
    KellyOracle,
    NaiveBayesKelly,
    ThompsonKellyAgent,
    FixedFraction,
    compute_kelly_fraction,
    create_half_kelly,
    create_double_kelly,
)


# =============================================================================
# Experiment A: The Martingale Convergence Test
# =============================================================================

def run_martingale_test(
    n_runs: int = 100,
    T: int = 500,
    p_true: float = 0.5,
    save_path: str = None
) -> dict:
    """
    Verify E[μ_{t+1} | F_t] = μ_t for Bayesian posterior.
    
    Setup: 1 Arm, p=0.5 (Fair coin)
    Agent: NaiveBayesKelly initialized with Beta(1,1)
    
    Success Criteria:
        - Individual trajectories should wander (random walk)
        - Average line should be FLAT at 0.5
        - No systematic drift up or down
    
    Args:
        n_runs: Number of independent trials (default: 100)
        T: Number of time steps per trial (default: 500)
        p_true: True win probability (default: 0.5)
        save_path: Path to save plot (None = display only)
    
    Returns:
        dict with results: 'passed', 'avg_final_mean', 'max_deviation'
    """
    print("=" * 60)
    print("EXPERIMENT A: Martingale Convergence Test")
    print("=" * 60)
    print(f"Setup: {n_runs} runs × {T} steps, p_true = {p_true}")
    
    all_trajectories = np.zeros((n_runs, T))
    
    for run in range(n_runs):
        # Create environment with fair coin
        env = MarketEnv(
            n_arms=1,
            true_probs=[p_true],
            horizon=T,
            odds=[1.0]
        )
        
        # Create agent with uniform prior
        agent = NaiveBayesKelly(n_arms=1, prior_alpha=1.0, prior_beta=1.0)
        
        env.reset(seed=run)  # Different seed each run
        
        for t in range(T):
            # Record posterior mean BEFORE observing outcome
            mu_t = agent.get_posterior_mean()[0]
            all_trajectories[run, t] = mu_t
            
            # Simulate one step (bet 0 to just observe)
            bets = np.array([0.0])  # Don't actually bet, just observe
            result = env.step(bets)
            
            # Update beliefs
            agent.update(result.outcomes)
    
    # Calculate statistics
    avg_trajectory = np.mean(all_trajectories, axis=0)
    final_avg = avg_trajectory[-1]
    max_deviation = np.max(np.abs(avg_trajectory - p_true))
    
    # Success criteria: average should stay close to p_true
    passed = max_deviation < 0.05  # Allow 5% deviation
    
    print(f"\nResults:")
    print(f"  Average final posterior mean: {final_avg:.4f}")
    print(f"  Expected value: {p_true:.4f}")
    print(f"  Max deviation from {p_true}: {max_deviation:.4f}")
    print(f"  STATUS: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot individual trajectories (light opacity)
    for run in range(min(50, n_runs)):  # Only plot first 50 for clarity
        ax.plot(all_trajectories[run], color='blue', alpha=0.1, linewidth=0.5)
    
    # Plot average trajectory (bold)
    ax.plot(avg_trajectory, color='red', linewidth=2, label='Average')
    
    # Plot reference line at p_true
    ax.axhline(y=p_true, color='green', linestyle='--', linewidth=2, 
               label=f'True p = {p_true}')
    
    ax.set_xlabel('Time Step t', fontsize=12)
    ax.set_ylabel('Posterior Mean μ_t', fontsize=12)
    ax.set_title('Martingale Convergence Test: E[μ_{t+1} | F_t] = μ_t', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add pass/fail annotation
    status_text = "✅ PASSED" if passed else "❌ FAILED"
    ax.text(0.02, 0.98, status_text, transform=ax.transAxes, 
            fontsize=14, verticalalignment='top',
            color='green' if passed else 'red',
            fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    
    plt.show()
    
    return {
        'passed': passed,
        'avg_final_mean': final_avg,
        'max_deviation': max_deviation,
        'all_trajectories': all_trajectories,
        'avg_trajectory': avg_trajectory
    }


# =============================================================================
# Experiment B: The Growth Hierarchy (Optimality)
# =============================================================================

def run_growth_hierarchy_test(
    n_runs: int = 100,
    T: int = 1000,
    p: float = 0.6,
    b: float = 1.0,
    save_path: str = None
) -> dict:
    """
    Verify Kelly optimality and the overbet penalty.
    
    Setup: 1 Arm, p=0.6, b=1.0 (Favorable coin)
    Agents: KellyOracle, HalfKelly, DoubleKelly
    
    Theoretical Predictions:
        - g(f*) = p*log(1+bf*) + q*log(1-f*) (maximum growth)
        - g(2f*) ≈ 0 (volatility drag cancels expected growth!)
        - g(0.5f*) > 0 but lower than optimal
    
    Success Criteria:
        - KellyOracle has highest terminal median wealth
        - DoubleKelly wealth converges to near zero
        - HalfKelly has lower growth but much lower variance
    
    Args:
        n_runs: Number of independent trials
        T: Number of time steps
        p: Win probability
        b: Odds (payoff on win)
        save_path: Path to save plot
    
    Returns:
        dict with results and trajectories
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Growth Rate Hierarchy Test")
    print("=" * 60)
    
    # Calculate optimal Kelly fraction
    q = 1 - p
    f_star = (b * p - q) / b
    
    print(f"Setup: {n_runs} runs × {T} steps")
    print(f"Game: p = {p}, b = {b} (edge = {p*b - q:.2f})")
    print(f"Optimal Kelly fraction: f* = {f_star:.4f}")
    print(f"  HalfKelly: f = {0.5*f_star:.4f}")
    print(f"  DoubleKelly: f = {2*f_star:.4f}")
    
    # Theoretical growth rates
    def theoretical_growth(f):
        if f <= 0 or f >= 1:
            return -np.inf
        return p * np.log(1 + b * f) + q * np.log(1 - f)
    
    g_star = theoretical_growth(f_star)
    g_half = theoretical_growth(0.5 * f_star)
    g_double = theoretical_growth(min(2 * f_star, 0.99))  # Cap at 0.99
    
    print(f"\nTheoretical Growth Rates:")
    print(f"  g(f*) = {g_star:.6f}")
    print(f"  g(0.5f*) = {g_half:.6f}")
    print(f"  g(2f*) = {g_double:.6f}")
    
    # Generate pregenerated outcomes for CRN
    outcomes = generate_pregenerated_outcomes(
        n_arms=1, horizon=T, true_probs=[p], odds=[b], seed=42
    )
    
    # Track wealth for each agent across runs
    agents_config = {
        'KellyOracle': lambda: KellyOracle(1, [p], [b], fraction_multiplier=1.0),
        'HalfKelly': lambda: KellyOracle(1, [p], [b], fraction_multiplier=0.5),
        'DoubleKelly': lambda: KellyOracle(1, [p], [b], fraction_multiplier=2.0),
    }
    
    results = {name: np.zeros((n_runs, T + 1)) for name in agents_config}
    
    for run in range(n_runs):
        # Generate different outcomes for each run
        run_outcomes = generate_pregenerated_outcomes(
            n_arms=1, horizon=T, true_probs=[p], odds=[b], seed=42 + run
        )
        
        for name, agent_factory in agents_config.items():
            env = MarketEnv(
                n_arms=1, true_probs=[p], horizon=T, odds=[b],
                pregenerated_outcomes=run_outcomes
            )
            agent = agent_factory()
            
            env.reset()
            results[name][run, 0] = env.wealth
            
            for t in range(T):
                bets = agent.act()
                result = env.step(bets)
                results[name][run, t + 1] = env.wealth
                
                if result.done and result.info.get('ruined', False):
                    # Fill remaining with ruin value
                    results[name][run, t + 1:] = env.wealth
                    break
    
    # Calculate statistics
    stats = {}
    for name, wealth_matrix in results.items():
        final_wealth = wealth_matrix[:, -1]
        log_wealth = np.log(wealth_matrix + 1e-10)
        avg_log_wealth = np.mean(log_wealth, axis=0)
        
        stats[name] = {
            'median_final': np.median(final_wealth),
            'mean_final': np.mean(final_wealth),
            'std_final': np.std(final_wealth),
            'ruin_rate': np.mean(final_wealth < 0.01),
            'empirical_growth': (avg_log_wealth[-1] - avg_log_wealth[0]) / T,
        }
    
    print(f"\nResults:")
    for name, s in stats.items():
        print(f"  {name}:")
        print(f"    Median final wealth: {s['median_final']:.4f}")
        print(f"    Empirical growth rate: {s['empirical_growth']:.6f}")
        print(f"    Ruin rate: {s['ruin_rate']:.1%}")
    
    # Check success criteria
    oracle_better = stats['KellyOracle']['empirical_growth'] > stats['HalfKelly']['empirical_growth']
    double_bad = stats['DoubleKelly']['empirical_growth'] < stats['HalfKelly']['empirical_growth']
    
    passed = oracle_better and double_bad
    
    print(f"\nValidation:")
    print(f"  Oracle > HalfKelly: {'✅' if oracle_better else '❌'}")
    print(f"  DoubleKelly underperforms: {'✅' if double_bad else '❌'}")
    print(f"  STATUS: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Log-wealth trajectories
    ax1 = axes[0]
    colors = {'KellyOracle': 'green', 'HalfKelly': 'blue', 'DoubleKelly': 'red'}
    
    for name, wealth_matrix in results.items():
        log_wealth = np.log(wealth_matrix + 1e-10)
        avg_log_wealth = np.mean(log_wealth, axis=0)
        std_log_wealth = np.std(log_wealth, axis=0)
        
        steps = np.arange(T + 1)
        ax1.plot(steps, avg_log_wealth, color=colors[name], linewidth=2, label=name)
        ax1.fill_between(steps, 
                         avg_log_wealth - std_log_wealth,
                         avg_log_wealth + std_log_wealth,
                         color=colors[name], alpha=0.2)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time Step', fontsize=12)
    ax1.set_ylabel('Log Wealth', fontsize=12)
    ax1.set_title('Growth Rate Comparison (Avg ± Std)', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final wealth distribution
    ax2 = axes[1]
    
    for name, wealth_matrix in results.items():
        final_log_wealth = np.log(wealth_matrix[:, -1] + 1e-10)
        ax2.hist(final_log_wealth, bins=30, alpha=0.5, 
                 color=colors[name], label=name, density=True)
    
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Final Log Wealth', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Final Wealth Distribution', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add status
    status_text = "✅ PASSED" if passed else "❌ FAILED"
    fig.suptitle(f'Growth Hierarchy Test - {status_text}', fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    
    plt.show()
    
    return {
        'passed': passed,
        'stats': stats,
        'wealth_trajectories': results,
        'theoretical_growths': {'optimal': g_star, 'half': g_half, 'double': g_double}
    }


# =============================================================================
# Main Execution
# =============================================================================

def run_all_verifications(output_dir: str = None):
    """Run all Phase 3 verification experiments."""
    
    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION SUITE")
    print("=" * 60)
    print("\nRunning all sanity checks before proceeding to experiments.\n")
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Experiment A: Martingale Test
    martingale_path = str(output_path / 'martingale_test.png') if output_dir else None
    result_a = run_martingale_test(n_runs=100, T=500, save_path=martingale_path)
    
    # Experiment B: Growth Hierarchy
    growth_path = str(output_path / 'growth_hierarchy.png') if output_dir else None
    result_b = run_growth_hierarchy_test(n_runs=100, T=1000, save_path=growth_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Experiment A (Martingale): {'✅ PASSED' if result_a['passed'] else '❌ FAILED'}")
    print(f"  Experiment B (Growth):     {'✅ PASSED' if result_b['passed'] else '❌ FAILED'}")
    
    all_passed = result_a['passed'] and result_b['passed']
    
    if all_passed:
        print("\n🎉 ALL VERIFICATIONS PASSED! Safe to proceed to experiments.")
    else:
        print("\n⚠️  SOME VERIFICATIONS FAILED. Review and fix before proceeding!")
    
    return {
        'martingale': result_a,
        'growth_hierarchy': result_b,
        'all_passed': all_passed
    }


if __name__ == "__main__":
    # Run verifications
    results = run_all_verifications(output_dir='experiments/output')
