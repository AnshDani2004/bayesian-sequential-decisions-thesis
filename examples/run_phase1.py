#!/usr/bin/env python3
"""
Phase 1 Canonical Entrypoint: Baseline Kelly Experiment

This script runs the baseline Gaussian Kelly experiment and produces:
1. Terminal wealth distribution plot
2. Summary statistics CSV
3. Results JSON with full metadata

Usage:
    python examples/run_phase1.py              # Full run (100 sims)
    python examples/run_phase1.py --smoke      # Smoke test (10 sims)
    python examples/run_phase1.py --seed 42    # Specific seed
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from simulation.environment import create_single_regime_environment
from simulation.agents import KellyOracle, NaiveBayesKelly
from simulation.utils.metadata import (
    get_run_metadata, 
    save_with_metadata, 
    print_metadata
)


def run_simulation(
    agent,
    env,
    n_sims: int = 100,
    t_steps: int = 1000,
    seed: int = 42
) -> dict:
    """
    Run Monte Carlo simulation with Common Random Numbers.
    
    Args:
        agent: Betting agent
        env: Market environment
        n_sims: Number of Monte Carlo simulations
        t_steps: Time steps per simulation
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with simulation results
    """
    # Pre-generate all returns (Common Random Numbers)
    result = env.simulate_batch(n_sims=n_sims, t_steps=t_steps, seed=seed)
    returns = result.returns
    
    # Track wealth using log-space for numerical stability
    log_wealth = np.zeros((n_sims, t_steps + 1))
    log_wealth[:, 0] = 0.0  # log(1) = 0
    
    # Run simulations
    for sim in range(n_sims):
        agent.reset()
        for t in range(t_steps):
            # Get betting fraction
            f = agent.act()[0]
            r = returns[sim, t]
            
            # Log-wealth update (numerically stable)
            multiplier = 1 + f * r
            if multiplier <= 0:
                log_wealth[sim, t + 1:] = -np.inf  # Ruined
                break
            else:
                log_wealth[sim, t + 1] = log_wealth[sim, t] + np.log1p(f * r)
            
            # Update agent beliefs
            outcome = np.array([1.0 if r > 0 else -1.0])
            agent.update(outcome)
    
    # Convert back to wealth
    wealth = np.exp(log_wealth)
    terminal_wealth = wealth[:, -1]
    
    # Compute metrics
    peak_wealth = np.maximum.accumulate(wealth, axis=1)
    drawdowns = (peak_wealth - wealth) / np.maximum(peak_wealth, 1e-10)
    max_drawdowns = np.max(drawdowns, axis=1)
    
    # Compute CAGR (annualized assuming 252 trading days)
    valid_mask = terminal_wealth > 0
    cagr = np.zeros(n_sims)
    cagr[valid_mask] = (terminal_wealth[valid_mask] ** (252 / t_steps)) - 1
    
    return {
        'terminal_wealth': terminal_wealth.tolist(),
        'max_drawdown': max_drawdowns.tolist(),
        'cagr': cagr.tolist(),
        'ruin_count': int(np.sum(terminal_wealth <= 0)),
        'survival_rate': float(np.mean(terminal_wealth > 0)),
        'median_wealth': float(np.median(terminal_wealth[valid_mask])) if valid_mask.any() else 0,
        'mean_wealth': float(np.mean(terminal_wealth[valid_mask])) if valid_mask.any() else 0,
        'wealth_std': float(np.std(terminal_wealth[valid_mask])) if valid_mask.any() else 0,
        'median_max_dd': float(np.median(max_drawdowns)),
        'p95_max_dd': float(np.percentile(max_drawdowns, 95)),
    }


def create_wealth_distribution_plot(results: dict, output_path: str) -> None:
    """Create terminal wealth distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    wealth = np.array(results['terminal_wealth'])
    valid_wealth = wealth[wealth > 0]
    
    ax.hist(valid_wealth, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=1.0, color='red', linestyle='--', label='Initial Wealth')
    ax.axvline(x=np.median(valid_wealth), color='green', linestyle='-', 
               label=f'Median: {np.median(valid_wealth):.2f}')
    
    ax.set_xlabel('Terminal Wealth')
    ax.set_ylabel('Frequency')
    ax.set_title('Terminal Wealth Distribution (Gaussian Kelly)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1 baseline experiment')
    parser.add_argument('--smoke', action='store_true', help='Smoke test (10 sims)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n-sims', type=int, default=100, help='Number of simulations')
    parser.add_argument('--t-steps', type=int, default=1000, help='Time steps')
    args = parser.parse_args()
    
    # Configuration
    n_sims = 10 if args.smoke else args.n_sims
    t_steps = 100 if args.smoke else args.t_steps
    seed = args.seed
    
    config = {
        'n_sims': n_sims,
        't_steps': t_steps,
        'mu': 0.08,
        'sigma': 0.15,
        'agent': 'NaiveBayesKelly',
    }
    
    print("=" * 60)
    print("PHASE 1: BASELINE GAUSSIAN KELLY EXPERIMENT")
    print("=" * 60)
    
    # Print metadata
    metadata = get_run_metadata(seed=seed, config=config)
    print_metadata(metadata)
    
    print(f"\nConfiguration:")
    print(f"  Simulations: {n_sims}")
    print(f"  Time steps:  {t_steps}")
    print(f"  Seed:        {seed}")
    print()
    
    # Create environment and agent
    env = create_single_regime_environment(
        mu=config['mu'],
        sigma=config['sigma'],
        seed=seed
    )
    
    agent = NaiveBayesKelly(n_arms=1, odds=[1.0])
    
    print("Running simulations...")
    results = run_simulation(agent, env, n_sims, t_steps, seed)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Survival Rate:    {results['survival_rate']:.1%}")
    print(f"  Median Wealth:    {results['median_wealth']:.3f}")
    print(f"  Mean Wealth:      {results['mean_wealth']:.3f}")
    print(f"  Wealth Std:       {results['wealth_std']:.3f}")
    print(f"  Median Max DD:    {results['median_max_dd']:.1%}")
    print(f"  95th %ile Max DD: {results['p95_max_dd']:.1%}")
    print(f"  Ruin Count:       {results['ruin_count']}")
    print()
    
    # Save results
    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON with metadata
    json_path = results_dir / f'phase1_{timestamp}.json'
    save_with_metadata(results, str(json_path), seed=seed, config=config)
    print(f"  Saved: {json_path}")
    
    # Save plot
    plot_path = results_dir / f'phase1_wealth_dist_{timestamp}.png'
    create_wealth_distribution_plot(results, str(plot_path))
    
    print("\n✅ Phase 1 experiment complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
