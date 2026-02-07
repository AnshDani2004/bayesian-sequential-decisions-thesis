"""
Experiment 4: Student-t Survival Test

Stress tests agents under heavy-tailed (Student-t) returns.

Tests the hypothesis that:
- NaiveBayesKelly will fail due to underestimating tail risk
- RiskConstrainedKelly will survive via drawdown protection
- VolAugmentedHMM will detect volatility spikes and deleverage

Success Criteria:
- RiskConstrained survives 90%+ of paths at ν=3
- RiskConstrained never breaches 20% drawdown on 95% of paths
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.student_t_env import StudentTEnvironment, RegimeSwitchingStudentT
from simulation.risk_constrained import RiskConstrainedKelly
from simulation.hmm_refined import VolAugmentedHMMKelly
from simulation.agents import NaiveBayesKelly, KellyOracle


def run_single_trial(agent, env, agent_type: str) -> dict:
    """Run single trial and collect metrics."""
    env.reset()
    
    if hasattr(agent, 'reset'):
        agent.reset()
    
    max_drawdown = 0.0
    peak_wealth = 1.0
    
    for t in range(env.T):
        # Get bet
        if agent_type == 'hmm':
            bets = agent.act()
            bet = np.sum(bets)
        elif agent_type == 'risk_constrained':
            bets = agent.act()
            bet = np.sum(bets)
        else:
            bets = agent.act()
            bet = np.sum(bets)
        
        # Step environment
        result = env.step(bet)
        
        # Update agent
        if agent_type == 'hmm':
            agent.update(result.return_t)
        elif agent_type == 'risk_constrained':
            outcomes = np.array([1.0 if result.return_t > 0 else -1.0])
            agent.update(outcomes, wealth=result.wealth)
        else:
            outcomes = np.array([1.0 if result.return_t > 0 else -1.0])
            agent.update(outcomes)
        
        # Track drawdown
        if result.wealth > peak_wealth:
            peak_wealth = result.wealth
        current_dd = (peak_wealth - result.wealth) / peak_wealth
        max_drawdown = max(max_drawdown, current_dd)
        
        if result.done:
            break
    
    return {
        'final_wealth': result.wealth,
        'max_drawdown': max_drawdown,
        'ruined': result.wealth < 0.01,
        'steps_survived': t + 1
    }


def run_experiment_4(
    nu_values: list = [3, 4, 5, 10, 30],
    n_runs: int = 50,
    T: int = 1000,
    output_dir: str = 'experiments/output'
):
    """
    Run Student-t survival experiment.
    """
    print("=" * 60)
    print("EXPERIMENT 4: Student-t Survival Test")
    print("=" * 60)
    print(f"\nDegrees of freedom: {nu_values}")
    print(f"Runs per ν: {n_runs}")
    
    # Agent configurations
    agent_configs = {
        'NaiveBayes': {
            'create': lambda: NaiveBayesKelly(n_arms=1),
            'type': 'naive'
        },
        'RiskConstrained': {
            'create': lambda: RiskConstrainedKelly(
                n_arms=1, 
                true_probs=[0.55],  # Favorable
                max_drawdown=0.20
            ),
            'type': 'risk_constrained'
        },
    }
    
    results = {name: [] for name in agent_configs.keys()}
    
    for nu in nu_values:
        print(f"\nTesting ν = {nu}...")
        
        for name, config in agent_configs.items():
            survivals = []
            dd_breaches = []
            final_wealths = []
            
            for run in range(n_runs):
                seed = 4000 + run
                
                # Create environment with this ν
                env = StudentTEnvironment(
                    T=T,
                    mu=0.001,
                    sigma=0.02,
                    nu=nu,
                    seed=seed
                )
                
                # Create agent
                agent = config['create']()
                
                # Run trial
                trial = run_single_trial(agent, env, config['type'])
                
                survivals.append(not trial['ruined'])
                dd_breaches.append(trial['max_drawdown'] > 0.20)
                final_wealths.append(trial['final_wealth'])
            
            survival_rate = np.mean(survivals)
            dd_breach_rate = np.mean(dd_breaches)
            median_wealth = np.median(final_wealths)
            
            results[name].append({
                'nu': nu,
                'survival_rate': survival_rate,
                'dd_breach_rate': dd_breach_rate,
                'median_wealth': median_wealth
            })
            
            print(f"  {name}: survival={survival_rate:.1%}, DD breach={dd_breach_rate:.1%}")
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'NaiveBayes': 'blue', 'RiskConstrained': 'green'}
    
    # Plot 1: Survival Rate
    ax1 = axes[0]
    for name in agent_configs.keys():
        x = [r['nu'] for r in results[name]]
        y = [r['survival_rate'] for r in results[name]]
        ax1.plot(x, y, 'o-', color=colors[name], label=name, linewidth=2, markersize=8)
    
    ax1.axhline(y=0.90, color='red', linestyle='--', label='Target: 90%')
    ax1.set_xlabel('Degrees of Freedom (ν)', fontsize=12)
    ax1.set_ylabel('Survival Rate', fontsize=12)
    ax1.set_title('Survival vs Tail Heaviness', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Annotate tail heaviness
    ax1.text(3, 0.1, 'Heavy Tails\n(ν=3)', fontsize=10, ha='center')
    ax1.text(30, 0.1, 'Near-Normal\n(ν=30)', fontsize=10, ha='center')
    
    # Plot 2: Drawdown Breach Rate
    ax2 = axes[1]
    for name in agent_configs.keys():
        x = [r['nu'] for r in results[name]]
        y = [r['dd_breach_rate'] for r in results[name]]
        ax2.plot(x, y, 's-', color=colors[name], label=name, linewidth=2, markersize=8)
    
    ax2.axhline(y=0.05, color='red', linestyle='--', label='Target: 5%')
    ax2.set_xlabel('Degrees of Freedom (ν)', fontsize=12)
    ax2.set_ylabel('DD Breach Rate (>20%)', fontsize=12)
    ax2.set_title('Drawdown Control', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Plot 3: Median Wealth
    ax3 = axes[2]
    for name in agent_configs.keys():
        x = [r['nu'] for r in results[name]]
        y = [np.log10(r['median_wealth'] + 0.01) for r in results[name]]
        ax3.plot(x, y, '^-', color=colors[name], label=name, linewidth=2, markersize=8)
    
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Degrees of Freedom (ν)', fontsize=12)
    ax3.set_ylabel('Log10(Median Wealth)', fontsize=12)
    ax3.set_title('Terminal Wealth', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / 'exp04_student_t_survival.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Check success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)
    
    rc_nu3 = [r for r in results['RiskConstrained'] if r['nu'] == 3][0]
    
    if rc_nu3['survival_rate'] >= 0.90:
        print(f"✅ RiskConstrained survives {rc_nu3['survival_rate']:.1%} at ν=3 (target: 90%)")
    else:
        print(f"⚠️  RiskConstrained survives {rc_nu3['survival_rate']:.1%} at ν=3 (target: 90%)")
    
    if rc_nu3['dd_breach_rate'] <= 0.05:
        print(f"✅ RiskConstrained DD breach rate {rc_nu3['dd_breach_rate']:.1%} at ν=3 (target: ≤5%)")
    else:
        print(f"⚠️  RiskConstrained DD breach rate {rc_nu3['dd_breach_rate']:.1%} at ν=3 (target: ≤5%)")
    
    return results


if __name__ == "__main__":
    results = run_experiment_4()
