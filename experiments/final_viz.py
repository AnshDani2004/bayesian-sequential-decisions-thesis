"""
Final Visualization for Thesis Publication

Generates publication-quality figures (300 DPI) for SIURO submission:
- Figure 1: Efficient Frontier (Growth vs. Safety)
- Figure 2: Anatomy of a Crash (Time-Series)

Requirements:
- LaTeX labels (uses matplotlib's mathtext)
- Professional fonts
- Clear annotations
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathlib import Path
import sys

# Configure matplotlib for publication
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
})

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.risk_constrained import RiskConstrainedKelly
from simulation.student_t_env import StudentTEnvironment, RegimeSwitchingStudentT
from simulation.hmm_refined import VolAugmentedHMMKelly
from simulation.agents import NaiveBayesKelly, KellyOracle


def compute_metrics(wealth_history: list, T: int) -> dict:
    """Compute CAGR, max drawdown, etc."""
    wealth = np.array(wealth_history)
    
    # CAGR (annualized, assuming 252 trading days)
    final = wealth[-1]
    initial = wealth[0]
    cagr = (final / initial) ** (252 / T) - 1
    
    # Max drawdown
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / peak
    max_dd = np.max(drawdown)
    
    return {
        'cagr': cagr,
        'max_drawdown': max_dd,
        'final_wealth': final
    }


def generate_efficient_frontier(
    output_dir: str = 'experiments/output',
    n_runs: int = 50,
    T: int = 1000
):
    """
    Figure 1: Efficient Frontier
    
    Shows trade-off between growth (CAGR) and safety (Max Drawdown).
    """
    print("Generating Figure 1: Efficient Frontier...")
    
    # Drawdown targets for Risk-Constrained Kelly
    dd_targets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    
    # Kelly fraction multipliers for unconstrained
    kelly_fractions = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    
    # Results storage
    rc_results = []  # Risk-Constrained
    uc_results = []  # Unconstrained Kelly
    
    # Run Risk-Constrained Kelly
    print("  Running Risk-Constrained Kelly...")
    for target_dd in dd_targets:
        cagrs = []
        max_dds = []
        
        for run in range(n_runs):
            seed = 6000 + run
            env = StudentTEnvironment(T=T, mu=0.001, sigma=0.02, nu=5, seed=seed)
            agent = RiskConstrainedKelly(
                n_arms=1,
                true_probs=[0.55],
                max_drawdown=target_dd,
                cppi_multiplier=3.0
            )
            
            env.reset()
            agent.reset()
            
            for t in range(T):
                bets = agent.act()
                result = env.step(np.sum(bets))
                outcomes = np.array([1.0 if result.return_t > 0 else -1.0])
                agent.update(outcomes, wealth=result.wealth)
                if result.done:
                    break
            
            metrics = compute_metrics(env.wealth_history, T)
            cagrs.append(metrics['cagr'])
            max_dds.append(metrics['max_drawdown'])
        
        rc_results.append({
            'target_dd': target_dd,
            'cagr': np.median(cagrs),
            'max_dd': np.median(max_dds),
            'cagr_std': np.std(cagrs),
            'max_dd_std': np.std(max_dds)
        })
    
    # Run Unconstrained Kelly variants
    print("  Running Unconstrained Kelly variants...")
    for fraction in kelly_fractions:
        cagrs = []
        max_dds = []
        
        for run in range(n_runs):
            seed = 6000 + run
            env = StudentTEnvironment(T=T, mu=0.001, sigma=0.02, nu=5, seed=seed)
            
            # Simple fractional Kelly
            kelly_bet = fraction * 0.1  # Approximate f*
            
            env.reset()
            
            for t in range(T):
                result = env.step(kelly_bet)
                if result.done:
                    break
            
            metrics = compute_metrics(env.wealth_history, T)
            cagrs.append(metrics['cagr'])
            max_dds.append(metrics['max_drawdown'])
        
        uc_results.append({
            'fraction': fraction,
            'cagr': np.median(cagrs),
            'max_dd': np.median(max_dds)
        })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Unconstrained Kelly (faint grey)
    uc_x = [r['max_dd'] for r in uc_results]
    uc_y = [r['cagr'] * 100 for r in uc_results]  # Convert to percentage
    ax.plot(uc_x, uc_y, 'o--', color='grey', alpha=0.6, 
            label=r'Unconstrained Kelly ($c \cdot f^*$)', markersize=8)
    
    # Annotate Kelly fractions
    for r in uc_results:
        if r['fraction'] in [0.5, 1.0, 1.5]:
            ax.annotate(f"$c={r['fraction']}$", 
                       (r['max_dd'], r['cagr'] * 100),
                       textcoords="offset points", xytext=(5, 5), 
                       fontsize=9, color='grey')
    
    # Plot Risk-Constrained Kelly (bold red)
    rc_x = [r['max_dd'] for r in rc_results]
    rc_y = [r['cagr'] * 100 for r in rc_results]
    ax.plot(rc_x, rc_y, 's-', color='crimson', linewidth=2,
            label=r'Risk-Constrained Kelly ($\alpha$-floor)', markersize=8)
    
    # Annotate target drawdowns
    for r in rc_results:
        if r['target_dd'] in [0.10, 0.20, 0.30]:
            ax.annotate(f"$\\alpha={r['target_dd']:.2f}$",
                       (r['max_dd'], r['cagr'] * 100),
                       textcoords="offset points", xytext=(5, -12),
                       fontsize=9, color='crimson')

    
    # Create "Impossibility Region" (high growth, low risk)
    impossibility_x = [0, 0, 0.15, 0.3]
    impossibility_y = [100, 50, 30, 100]
    polygon = Polygon(list(zip(impossibility_x, impossibility_y)),
                     alpha=0.15, facecolor='blue', edgecolor='blue',
                     linestyle='--', linewidth=1.5)
    ax.add_patch(polygon)
    ax.text(0.05, 60, 'Impossibility\nRegion', fontsize=10, 
            color='blue', fontstyle='italic')
    
    ax.set_xlabel(r'Maximum Drawdown (%)', fontsize=12)
    ax.set_ylabel(r'CAGR (%)', fontsize=12)
    ax.set_title(r'Efficient Frontier: Growth vs. Safety', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-50, 100)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_path / 'fig1_efficient_frontier.png'
    plt.savefig(fig_path, dpi=300)
    print(f"  Saved: {fig_path}")
    
    # Also save as PDF for LaTeX
    pdf_path = output_path / 'fig1_efficient_frontier.pdf'
    plt.savefig(pdf_path, format='pdf')
    print(f"  Saved: {pdf_path}")
    
    plt.close()
    return rc_results, uc_results


def generate_crash_anatomy(
    output_dir: str = 'experiments/output',
    T: int = 200,
    switch_point: int = 100
):
    """
    Figure 2: Anatomy of a Crash
    
    Shows:
    - Top: Price/Wealth trajectories
    - Bottom: P(Bear) with detection point
    """
    print("\nGenerating Figure 2: Anatomy of a Crash...")
    
    seed = 42  # Fixed seed for reproducibility
    
    # Create environment
    env = RegimeSwitchingStudentT(
        T=T,
        switch_point=switch_point,
        bull_mu=0.02,
        bull_sigma=0.1,
        bull_nu=5.0,
        bear_mu=-0.03,
        bear_sigma=0.25,
        bear_nu=3.0,
        seed=seed
    )
    
    # Create agents
    naive_agent = NaiveBayesKelly(n_arms=1)
    
    regime_params = {
        'bull': {'mu': 0.02, 'sigma': 0.1, 'vol_shape': 2.0, 'vol_scale': 0.05},
        'bear': {'mu': -0.03, 'sigma': 0.25, 'vol_shape': 4.0, 'vol_scale': 0.12}
    }
    hmm_agent = VolAugmentedHMMKelly(n_arms=1, regime_params=regime_params)
    
    # Run NaiveBayes
    env.reset(seed=seed)
    naive_agent.reset()
    naive_wealth = [1.0]
    
    for t in range(T):
        bets = naive_agent.act()
        result = env.step(np.sum(bets))
        outcomes = np.array([1.0 if result.return_t > 0 else -1.0])
        naive_agent.update(outcomes)
        naive_wealth.append(result.wealth)
        if result.done:
            naive_wealth.extend([naive_wealth[-1]] * (T - t))
            break
    
    # Run HMM agent
    env.reset(seed=seed)
    hmm_agent.reset()
    hmm_wealth = [1.0]
    bear_probs = [0.5]
    
    for t in range(T):
        bets = hmm_agent.act()
        result = env.step(np.sum(bets))
        hmm_agent.update(result.return_t)
        hmm_wealth.append(result.wealth)
        bear_probs.append(hmm_agent.hmm.get_regime_probability('bear'))
        if result.done:
            hmm_wealth.extend([hmm_wealth[-1]] * (T - t))
            bear_probs.extend([bear_probs[-1]] * (T - t))
            break
    
    # Generate price trajectory (using env returns)
    env.reset(seed=seed)
    price = [100.0]
    for t in range(T):
        result = env.step(0.0)  # Don't bet, just observe
        price.append(price[-1] * (1 + result.return_t))
    
    # Find detection point (first time P(Bear) > 0.5 after switch)
    detection_step = None
    for t in range(switch_point, len(bear_probs)):
        if bear_probs[t] > 0.5:
            detection_step = t
            break
    
    detection_lag = detection_step - switch_point if detection_step else T - switch_point
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    steps = np.arange(T + 1)
    
    # === Top subplot: Price and Wealth ===
    ax1.semilogy(steps, price[:T+1], 'k-', linewidth=1.5, label='Asset Price', alpha=0.7)
    ax1.semilogy(steps, naive_wealth[:T+1], 'b-', linewidth=2, 
                 label='NaiveBayes Wealth')
    ax1.semilogy(steps, hmm_wealth[:T+1], 'g-', linewidth=2,
                 label='VolAugmented-HMM Wealth')
    
    # Mark regime switch
    ax1.axvline(x=switch_point, color='red', linestyle='--', linewidth=2, 
                label='Regime Switch')
    
    # Add regime labels
    ax1.fill_between([0, switch_point], [0.01, 0.01], [1e6, 1e6], 
                     alpha=0.1, color='green')
    ax1.fill_between([switch_point, T], [0.01, 0.01], [1e6, 1e6], 
                     alpha=0.1, color='red')
    ax1.text(switch_point / 2, max(price) * 0.5, 'BULL', fontsize=12, 
             ha='center', color='green', fontweight='bold')
    ax1.text(switch_point + (T - switch_point) / 2, max(price) * 0.5, 'BEAR', 
             fontsize=12, ha='center', color='red', fontweight='bold')
    
    ax1.set_ylabel(r'Value (Log Scale)', fontsize=12)
    ax1.set_title(r'Anatomy of a Market Crash: Agent Survival', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T)
    
    # === Bottom subplot: P(Bear) ===
    ax2.fill_between(steps[:len(bear_probs)], 0, bear_probs[:T+1], 
                     color='purple', alpha=0.4, label=r'$P(S_t = \mathrm{Bear} | y_{1:t})$')
    ax2.plot(steps[:len(bear_probs)], bear_probs[:T+1], 'purple', linewidth=1.5)
    
    # Detection threshold
    ax2.axhline(y=0.5, color='grey', linestyle=':', alpha=0.7)
    ax2.text(5, 0.53, r'Detection Threshold ($\gamma = 0.5$)', fontsize=9, color='grey')
    
    # Mark detection point
    if detection_step:
        ax2.axvline(x=detection_step, color='green', linestyle='-', linewidth=2)
        ax2.annotate(f'Detection\n(lag = {detection_lag} steps)',
                    xy=(detection_step, 0.5),
                    xytext=(detection_step + 15, 0.7),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='green'),
                    color='green')
    
    # Mark regime switch
    ax2.axvline(x=switch_point, color='red', linestyle='--', linewidth=2)
    
    ax2.set_xlabel(r'Time Step $t$', fontsize=12)
    ax2.set_ylabel(r'$P(\mathrm{Bear} | \mathcal{F}_t)$', fontsize=12)
    ax2.set_title(r'HMM Posterior Regime Probability', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, T)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig_path = output_path / 'fig2_crash_anatomy.png'
    plt.savefig(fig_path, dpi=300)
    print(f"  Saved: {fig_path}")
    
    pdf_path = output_path / 'fig2_crash_anatomy.pdf'
    plt.savefig(pdf_path, format='pdf')
    print(f"  Saved: {pdf_path}")
    
    print(f"  Detection lag: {detection_lag} steps")
    
    plt.close()
    return detection_lag


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 6: Production-Quality Visualization")
    print("=" * 60)
    
    generate_efficient_frontier()
    generate_crash_anatomy()
    
    print("\n✅ All figures generated successfully!")
