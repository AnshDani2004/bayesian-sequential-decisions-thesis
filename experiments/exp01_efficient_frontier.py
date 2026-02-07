"""
Experiment 1: The Cost of Safety (Efficient Frontier)

Maps the trade-off between growth and risk by comparing:
- FractionalKelly with c ∈ [0.1, 0.2, ..., 2.0]
- ConvexRiskAgent with various max_bet constraints

Hypothesis: ConvexRiskAgent should dominate FractionalKelly
(i.e., higher wealth for same drawdown) because it adapts to the path.

Output: Scatter plot of Median Terminal Wealth vs Maximum Drawdown
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.market_env import MarketEnv, generate_pregenerated_outcomes
from simulation.agents import FractionalKelly, ConvexRiskAgent, KellyOracle


def compute_drawdown_stats(wealth_history: np.ndarray) -> dict:
    """
    Compute drawdown statistics from wealth history.
    
    Args:
        wealth_history: Array of wealth values over time
    
    Returns:
        dict with 'max_drawdown', 'avg_drawdown', 'final_wealth'
    """
    wealth = np.array(wealth_history)
    
    # Running maximum
    running_max = np.maximum.accumulate(wealth)
    
    # Drawdown at each point
    drawdowns = (running_max - wealth) / running_max
    
    return {
        'max_drawdown': np.max(drawdowns),
        'avg_drawdown': np.mean(drawdowns),
        'final_wealth': wealth[-1],
        'peak_wealth': np.max(wealth),
    }


def run_single_simulation(agent, T: int, outcomes: np.ndarray, 
                          true_probs: list, odds: list) -> dict:
    """Run a single simulation and return statistics."""
    n_arms = len(true_probs)
    
    env = MarketEnv(
        n_arms=n_arms,
        true_probs=true_probs,
        horizon=T,
        odds=odds,
        pregenerated_outcomes=outcomes
    )
    
    env.reset()
    agent.reset()
    
    for t in range(T):
        bets = agent.act()
        result = env.step(bets)
        agent.update(result.outcomes)
        
        if result.done and result.info.get('ruined', False):
            break
    
    return compute_drawdown_stats(env.wealth_history)


def run_experiment_1(
    n_runs: int = 50,
    T: int = 1000,
    true_prob: float = 0.55,
    odds: float = 1.0,
    output_dir: str = 'experiments/output'
):
    """
    Run Experiment 1: Efficient Frontier mapping.
    """
    print("=" * 60)
    print("EXPERIMENT 1: The Cost of Safety (Efficient Frontier)")
    print("=" * 60)
    
    n_arms = 1
    true_probs = [true_prob]
    odds_list = [odds]
    
    # Kelly fraction for reference
    q = 1 - true_prob
    f_star = (odds * true_prob - q) / odds
    print(f"\nSetup: p={true_prob}, b={odds}, f*={f_star:.4f}")
    print(f"Runs: {n_runs} × {T} steps each")
    
    # Define agent configurations
    fractional_c_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    convex_max_bet_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    results = {
        'FractionalKelly': [],
        'ConvexRisk': [],
        'Oracle': None
    }
    
    # Generate outcomes for CRN
    all_outcomes = [
        generate_pregenerated_outcomes(n_arms, T, true_probs, odds_list, seed=1000 + i)
        for i in range(n_runs)
    ]
    
    # Run Oracle for reference
    print("\nRunning KellyOracle (reference)...")
    oracle_stats = []
    for run_idx, outcomes in enumerate(all_outcomes):
        agent = KellyOracle(n_arms, true_probs, odds_list)
        stats = run_single_simulation(agent, T, outcomes, true_probs, odds_list)
        oracle_stats.append(stats)
    
    results['Oracle'] = {
        'label': 'Oracle (f*)',
        'median_wealth': np.median([s['final_wealth'] for s in oracle_stats]),
        'max_drawdown': np.median([s['max_drawdown'] for s in oracle_stats]),
    }
    print(f"  Oracle: median wealth={results['Oracle']['median_wealth']:.2f}, "
          f"drawdown={results['Oracle']['max_drawdown']:.2%}")
    
    # Run FractionalKelly variants
    print("\nRunning FractionalKelly variants...")
    for c in fractional_c_values:
        stats_list = []
        for run_idx, outcomes in enumerate(all_outcomes):
            agent = FractionalKelly(n_arms, true_probs, odds_list, fraction_multiplier=c)
            stats = run_single_simulation(agent, T, outcomes, true_probs, odds_list)
            stats_list.append(stats)
        
        median_wealth = np.median([s['final_wealth'] for s in stats_list])
        median_drawdown = np.median([s['max_drawdown'] for s in stats_list])
        
        results['FractionalKelly'].append({
            'c': c,
            'label': f'Frac-{c:.2f}',
            'median_wealth': median_wealth,
            'max_drawdown': median_drawdown,
        })
        print(f"  c={c:.2f}: wealth={median_wealth:.2f}, drawdown={median_drawdown:.2%}")
    
    # Run ConvexRisk variants
    print("\nRunning ConvexRisk variants...")
    for max_bet in convex_max_bet_values:
        stats_list = []
        for run_idx, outcomes in enumerate(all_outcomes):
            agent = ConvexRiskAgent(n_arms, odds_list, max_bet_per_arm=max_bet, solve_freq=10)
            stats = run_single_simulation(agent, T, outcomes, true_probs, odds_list)
            stats_list.append(stats)
        
        median_wealth = np.median([s['final_wealth'] for s in stats_list])
        median_drawdown = np.median([s['max_drawdown'] for s in stats_list])
        
        results['ConvexRisk'].append({
            'max_bet': max_bet,
            'label': f'Cvx-{max_bet:.2f}',
            'median_wealth': median_wealth,
            'max_drawdown': median_drawdown,
        })
        print(f"  max_bet={max_bet:.2f}: wealth={median_wealth:.2f}, drawdown={median_drawdown:.2%}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot FractionalKelly
    fk_x = [r['max_drawdown'] for r in results['FractionalKelly']]
    fk_y = [np.log10(r['median_wealth'] + 1) for r in results['FractionalKelly']]
    ax.scatter(fk_x, fk_y, s=100, c='blue', marker='o', label='FractionalKelly', alpha=0.7)
    
    # Connect with line
    sorted_fk = sorted(zip(fk_x, fk_y), key=lambda x: x[0])
    ax.plot([p[0] for p in sorted_fk], [p[1] for p in sorted_fk], 
            'b--', alpha=0.5, linewidth=1)
    
    # Annotate FractionalKelly points
    for r in results['FractionalKelly']:
        ax.annotate(f'c={r["c"]:.1f}', 
                   (r['max_drawdown'], np.log10(r['median_wealth'] + 1)),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    # Plot ConvexRisk
    cr_x = [r['max_drawdown'] for r in results['ConvexRisk']]
    cr_y = [np.log10(r['median_wealth'] + 1) for r in results['ConvexRisk']]
    ax.scatter(cr_x, cr_y, s=100, c='red', marker='s', label='ConvexRisk', alpha=0.7)
    
    # Connect with line
    sorted_cr = sorted(zip(cr_x, cr_y), key=lambda x: x[0])
    ax.plot([p[0] for p in sorted_cr], [p[1] for p in sorted_cr], 
            'r--', alpha=0.5, linewidth=1)
    
    # Annotate ConvexRisk points
    for r in results['ConvexRisk']:
        ax.annotate(f'β={r["max_bet"]:.2f}', 
                   (r['max_drawdown'], np.log10(r['median_wealth'] + 1)),
                   textcoords="offset points", xytext=(5, -10), fontsize=8)
    
    # Plot Oracle reference
    ax.scatter(results['Oracle']['max_drawdown'], 
               np.log10(results['Oracle']['median_wealth'] + 1),
               s=200, c='green', marker='*', label='Oracle', zorder=10)
    
    ax.set_xlabel('Maximum Drawdown (Median)', fontsize=12)
    ax.set_ylabel('Log10(Median Terminal Wealth)', fontsize=12)
    ax.set_title('Efficient Frontier: Growth vs. Risk', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / 'exp01_efficient_frontier.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    return results


if __name__ == "__main__":
    results = run_experiment_1(n_runs=50, T=1000)
