"""
Experiment 3: HMM Sensitivity Sweep

Maps the relationship between:
- Transition persistence (A_ii) 
- Detection lag (steps to detect regime switch)
- False positive rate (spurious switches)

Goal: Find optimal A_ii that balances:
- Fast detection (low lag)
- Stability (low false positives)

Validates the Phase 5 fix: VolAugmentedHMM should detect switches faster
than the original EWMA-based HMM.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.hmm_refined import VolAugmentedHMM
from simulation.student_t_env import RegimeSwitchingStudentT


def measure_detection_lag(
    hmm: VolAugmentedHMM,
    env: RegimeSwitchingStudentT,
    detection_threshold: float = 0.5
) -> dict:
    """
    Measure how many steps it takes to detect the regime switch.
    
    Args:
        hmm: HMM agent
        env: Regime-switching environment
        detection_threshold: P(Bear) threshold for detection
    
    Returns:
        dict with detection_lag, false_positive_rate, etc.
    """
    env.reset()
    hmm.reset()
    
    switch_point = env.switch_point
    detection_step = None
    false_positives = 0
    
    # Track bear probability over time
    bear_probs = []
    
    for t in range(env.T):
        # Get current regime
        true_regime = 'bull' if t < switch_point else 'bear'
        
        # Sample return (bet 0 to just observe)
        result = env.step(0.0)
        r_t = result.return_t
        
        # Update HMM
        hmm.update(r_t)
        p_bear = hmm.get_regime_probability('bear')
        bear_probs.append(p_bear)
        
        # Check for detection
        if true_regime == 'bull' and p_bear > detection_threshold:
            false_positives += 1
        
        if true_regime == 'bear' and detection_step is None:
            if p_bear > detection_threshold:
                detection_step = t - switch_point
        
        if result.done:
            break
    
    # If never detected
    if detection_step is None:
        detection_step = env.T - switch_point  # Max lag
    
    return {
        'detection_lag': detection_step,
        'false_positives': false_positives,
        'bear_probs': bear_probs,
        'switch_point': switch_point
    }


def run_sensitivity_sweep(
    persistence_values: list = [0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99],
    n_runs: int = 30,
    T: int = 1000,
    switch_point: int = 500,
    output_dir: str = 'experiments/output'
):
    """
    Sweep over transition persistence values.
    """
    print("=" * 60)
    print("EXPERIMENT 3: HMM Sensitivity Sweep")
    print("=" * 60)
    print(f"\nPersistence values: {persistence_values}")
    print(f"Runs per value: {n_runs}")
    
    results = []
    
    for persistence in persistence_values:
        print(f"\nTesting A_ii = {persistence:.2f}...")
        
        lags = []
        fps = []
        
        for run in range(n_runs):
            seed = 3000 + run
            
            # Create environment
            env = RegimeSwitchingStudentT(
                T=T,
                switch_point=switch_point,
                seed=seed
            )
            
            # Create HMM with specific persistence
            regime_params = {
                'bull': {'mu': 0.02, 'sigma': 0.1, 'vol_shape': 2.0, 'vol_scale': 0.05},
                'bear': {'mu': -0.02, 'sigma': 0.2, 'vol_shape': 4.0, 'vol_scale': 0.1}
            }
            hmm = VolAugmentedHMM(regime_params, max_persistence=persistence)
            
            # Measure
            metrics = measure_detection_lag(hmm, env)
            lags.append(metrics['detection_lag'])
            fps.append(metrics['false_positives'])
        
        avg_lag = np.mean(lags)
        avg_fp = np.mean(fps)
        
        results.append({
            'persistence': persistence,
            'avg_detection_lag': avg_lag,
            'std_detection_lag': np.std(lags),
            'avg_false_positives': avg_fp,
        })
        
        print(f"  Avg lag: {avg_lag:.1f} steps, Avg FP: {avg_fp:.1f}")
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = [r['persistence'] for r in results]
    lag_means = [r['avg_detection_lag'] for r in results]
    lag_stds = [r['std_detection_lag'] for r in results]
    fp_means = [r['avg_false_positives'] for r in results]
    
    # Plot 1: Detection Lag
    ax1.errorbar(x, lag_means, yerr=lag_stds, fmt='o-', capsize=5, color='blue')
    ax1.axhline(y=20, color='green', linestyle='--', label='Target: 20 steps')
    ax1.set_xlabel('Transition Persistence (A_ii)', fontsize=12)
    ax1.set_ylabel('Detection Lag (steps)', fontsize=12)
    ax1.set_title('Detection Speed vs Persistence', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: False Positives
    ax2.bar(x, fp_means, width=0.015, color='red', alpha=0.7)
    ax2.set_xlabel('Transition Persistence (A_ii)', fontsize=12)
    ax2.set_ylabel('False Positives (avg)', fontsize=12)
    ax2.set_title('Stability vs Persistence', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / 'exp03_hmm_sensitivity.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Find optimal
    for r in results:
        if r['avg_detection_lag'] <= 20:
            print(f"\n✅ SUCCESS: A_ii = {r['persistence']:.2f} achieves detection in {r['avg_detection_lag']:.1f} steps")
            break
    else:
        print("\n⚠️  No persistence value achieved target detection lag")
    
    return results


if __name__ == "__main__":
    results = run_sensitivity_sweep()
