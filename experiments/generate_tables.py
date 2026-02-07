"""
Generate LaTeX Tables for Thesis

Creates booktabs-formatted LaTeX tables:
- Table 1: Comparative Performance (Student-t World)
- Table 2: HMM Sensitivity Analysis

Output: tables.tex (can be directly \input{} in LaTeX)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.risk_constrained import RiskConstrainedKelly
from simulation.student_t_env import StudentTEnvironment
from simulation.hmm_refined import VolAugmentedHMMKelly, VolAugmentedHMM
from simulation.agents import NaiveBayesKelly, FractionalKelly


def run_comparative_performance(n_runs: int = 100, T: int = 1000, nu: float = 3.0):
    """
    Table 1: Comparative Performance in Student-t World
    
    Columns: Agent, Median Wealth, IQR, Max DD (95%), Ruin Prob
    """
    print("Generating Table 1: Comparative Performance...")
    
    agent_configs = {
        'Buy \\& Hold': {
            'create': lambda: None,
            'bet': 1.0  # Always 100% invested
        },
        'Full Kelly ($f^*$)': {
            'create': lambda: None,
            'bet': 0.10  # Approximate optimal Kelly
        },
        'Fractional Kelly ($0.5 f^*$)': {
            'create': lambda: None,
            'bet': 0.05
        },
        'Risk-Constrained Kelly': {
            'create': lambda: RiskConstrainedKelly(
                n_arms=1, true_probs=[0.55], max_drawdown=0.20
            ),
            'type': 'risk_constrained'
        },
        'Vol-Augmented HMM': {
            'create': lambda: VolAugmentedHMMKelly(
                n_arms=1,
                regime_params={
                    'bull': {'mu': 0.02, 'sigma': 0.1, 'vol_shape': 2.0, 'vol_scale': 0.05},
                    'bear': {'mu': -0.02, 'sigma': 0.2, 'vol_shape': 4.0, 'vol_scale': 0.1}
                }
            ),
            'type': 'hmm'
        }
    }
    
    results = []
    
    for name, config in agent_configs.items():
        print(f"  Running {name}...")
        
        final_wealths = []
        max_drawdowns = []
        ruined = []
        
        for run in range(n_runs):
            seed = 7000 + run
            env = StudentTEnvironment(T=T, mu=0.001, sigma=0.02, nu=nu, seed=seed)
            env.reset()
            
            if 'type' not in config:
                # Simple fixed-bet agent
                bet = config['bet']
                peak = 1.0
                max_dd = 0.0
                
                for t in range(T):
                    result = env.step(bet)
                    if result.wealth > peak:
                        peak = result.wealth
                    dd = (peak - result.wealth) / peak
                    max_dd = max(max_dd, dd)
                    if result.done:
                        break
                
                final_wealths.append(result.wealth)
                max_drawdowns.append(max_dd)
                ruined.append(result.wealth < 0.01)
            
            elif config['type'] == 'risk_constrained':
                agent = config['create']()
                agent.reset()
                
                for t in range(T):
                    bets = agent.act()
                    result = env.step(np.sum(bets))
                    outcomes = np.array([1.0 if result.return_t > 0 else -1.0])
                    agent.update(outcomes, wealth=result.wealth)
                    if result.done:
                        break
                
                metrics = agent.get_drawdown_metrics()
                final_wealths.append(result.wealth)
                max_drawdowns.append(metrics['max_drawdown'])
                ruined.append(result.wealth < 0.01)
            
            elif config['type'] == 'hmm':
                agent = config['create']()
                agent.reset()
                peak = 1.0
                max_dd = 0.0
                
                for t in range(T):
                    bets = agent.act()
                    result = env.step(np.sum(bets))
                    agent.update(result.return_t)
                    
                    if result.wealth > peak:
                        peak = result.wealth
                    dd = (peak - result.wealth) / peak
                    max_dd = max(max_dd, dd)
                    
                    if result.done:
                        break
                
                final_wealths.append(result.wealth)
                max_drawdowns.append(max_dd)
                ruined.append(result.wealth < 0.01)
        
        # Compute statistics
        wealth_arr = np.array(final_wealths)
        dd_arr = np.array(max_drawdowns)
        
        results.append({
            'Agent': name,
            'Median Wealth': np.median(wealth_arr),
            'IQR': np.percentile(wealth_arr, 75) - np.percentile(wealth_arr, 25),
            'Max DD (95\\%)': np.percentile(dd_arr, 95),
            'Ruin Prob': np.mean(ruined)
        })
    
    df = pd.DataFrame(results)
    return df


def run_hmm_sensitivity(n_runs: int = 50, T: int = 1000, switch_point: int = 500):
    """
    Table 2: HMM Sensitivity Analysis
    
    Columns: Persistence, Mean Detection Lag, False Positive Rate
    """
    print("\nGenerating Table 2: HMM Sensitivity Analysis...")
    
    from simulation.student_t_env import RegimeSwitchingStudentT
    
    persistence_values = [0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    
    results = []
    
    for persistence in persistence_values:
        print(f"  Testing $A_{{ii}} = {persistence}$...")
        
        lags = []
        fps = []
        
        for run in range(n_runs):
            seed = 8000 + run
            
            env = RegimeSwitchingStudentT(T=T, switch_point=switch_point, seed=seed)
            
            regime_params = {
                'bull': {'mu': 0.02, 'sigma': 0.1, 'vol_shape': 2.0, 'vol_scale': 0.05},
                'bear': {'mu': -0.02, 'sigma': 0.2, 'vol_shape': 4.0, 'vol_scale': 0.1}
            }
            hmm = VolAugmentedHMM(regime_params, max_persistence=persistence)
            
            env.reset()
            hmm.reset()
            
            detection_step = None
            fp_count = 0
            
            for t in range(T):
                result = env.step(0.0)
                hmm.update(result.return_t)
                p_bear = hmm.get_regime_probability('bear')
                
                if t < switch_point and p_bear > 0.5:
                    fp_count += 1
                
                if t >= switch_point and detection_step is None and p_bear > 0.5:
                    detection_step = t - switch_point
                
                if result.done:
                    break
            
            lag = detection_step if detection_step is not None else T - switch_point
            lags.append(lag)
            fps.append(fp_count)
        
        results.append({
            '$A_{ii}$': f'{persistence:.2f}',
            'Mean Lag (steps)': np.mean(lags),
            'FP Rate (\\%)': 100 * np.mean([f > 0 for f in fps])
        })
    
    df = pd.DataFrame(results)
    return df


def generate_latex_tables(output_dir: str = 'experiments/output'):
    """Generate both tables and save to tables.tex"""
    
    # Run experiments
    df1 = run_comparative_performance()
    df2 = run_hmm_sensitivity()
    
    # Generate LaTeX
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    latex_content = r"""% Auto-generated LaTeX tables
% Copy-paste into thesis document

% Table 1: Comparative Performance
\begin{table}[htbp]
\centering
\caption{Comparative Agent Performance in Student-t Environment ($\nu=3$, $T=1000$, $n=100$ runs)}
\label{tab:comparative_performance}
"""
    
    # Convert df1 to LaTeX
    latex1 = df1.to_latex(
        index=False,
        float_format="%.2f",
        escape=False,
        column_format='l' + 'r' * (len(df1.columns) - 1)
    )
    # Add booktabs
    latex1 = latex1.replace('\\toprule', '\\toprule')
    latex1 = latex1.replace('\\midrule', '\\midrule')
    latex1 = latex1.replace('\\bottomrule', '\\bottomrule')
    
    latex_content += latex1
    latex_content += r"""
\end{table}

% Table 2: HMM Sensitivity Analysis
\begin{table}[htbp]
\centering
\caption{HMM Sensitivity: Detection Speed vs. Stability Trade-off}
\label{tab:hmm_sensitivity}
"""
    
    # Convert df2 to LaTeX
    latex2 = df2.to_latex(
        index=False,
        float_format="%.1f",
        escape=False,
        column_format='c' * len(df2.columns)
    )
    
    latex_content += latex2
    latex_content += r"""
\end{table}
"""
    
    # Save
    table_path = output_path / 'tables.tex'
    with open(table_path, 'w') as f:
        f.write(latex_content)
    
    print(f"\n✅ Tables saved to: {table_path}")
    
    # Also print for reference
    print("\n" + "=" * 60)
    print("TABLE 1: Comparative Performance")
    print("=" * 60)
    print(df1.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("TABLE 2: HMM Sensitivity")
    print("=" * 60)
    print(df2.to_string(index=False))
    
    return df1, df2


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 6: LaTeX Table Generation")
    print("=" * 60)
    
    generate_latex_tables()
