# Phase 4 Experimental Design

## Overview

This document outlines the experimental design for Phase 4 of the thesis, testing the performance of Bayesian agents in adversarial market conditions.

---

## Experiment 1: The Cost of Safety (Efficient Frontier)

### Hypothesis
ConvexRiskAgent should **dominate** FractionalKelly—achieving higher wealth for the same level of drawdown risk—because it adapts to the realized path rather than using a fixed fraction.

### Setup
- **Market**: Single arm, p=0.55, b=1.0 (edge = 0.10)
- **Optimal Kelly**: f* = 0.10
- **Runs**: 50 × 1000 steps with CRN

### Agents Tested
| Agent | Parameter Range | Purpose |
|-------|-----------------|---------|
| FractionalKelly | c ∈ [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] | Map growth-risk tradeoff |
| ConvexRisk | max_bet ∈ [0.05, 0.1, 0.15, 0.2, 0.25, 0.3] | Path-adaptive constraints |

### Key Results

| c (Kelly Mult.) | Median Wealth | Max Drawdown |
|-----------------|---------------|--------------|
| 0.10 | 2.72 | 14.6% |
| 0.50 | 54.82 | 57.9% |
| **1.00 (Full)** | **248.41** | **86.6%** |
| 1.50 | 90.25 | 97.5% |
| 2.00 | 2.45 | 99.8% |

> **Key Insight**: The overbet penalty is severe. At c=2.0 (double Kelly), median wealth collapses from 248 to 2.45—nearly back to starting capital despite 1000 favorable bets.

### Plot
![Efficient Frontier](file:///Users/ansh/Developer/bayesian-sequential-decisions-thesis/experiments/output/exp01_efficient_frontier.png)

---

## Experiment 2: The Regime Shock (Adaptivity Test)

### Hypothesis
HMMKellyAgent will detect the bear market around t=520 and cut leverage, preserving wealth, while NaiveBayesKelly continues betting and suffers losses.

### Setup
- **Bull Phase** (t < 500): μ=0.02, σ=0.1 (positive drift)
- **Bear Phase** (t ≥ 500): μ=-0.02, σ=0.15 (negative drift, higher volatility)
- **Runs**: 50 × 1000 steps

### Agents Tested
| Agent | Mechanism | Expected Behavior |
|-------|-----------|-------------------|
| NaiveBayesKelly | Plug-in estimator | Slow to adapt, vulnerable |
| HMMKellyAgent | EWMA + regime inference | Should detect switch |

### Results

| Agent | Median Wealth | Survival Rate |
|-------|---------------|---------------|
| NaiveBayesKelly | 1.91 | 100% |
| HMMKellyAgent | 0.15 | 96% |

### Interpretation

> [!WARNING]
> HMMKelly underperformed NaiveBayes in this configuration.

The HMMKellyAgent requires parameter tuning:
1. **EWMA alpha too aggressive**: α=0.05 may cause false regime switches
2. **Regime parameters mismatch**: The binary outcome model doesn't translate cleanly to Gaussian returns
3. **Kelly formula mismatch**: Using continuous Kelly (μ/σ²) for binary outcomes

### Plot
![Regime Shock](file:///Users/ansh/Developer/bayesian-sequential-decisions-thesis/experiments/output/exp02_regime_shock.png)

---

## Next Steps

1. **Tune HMMKellyAgent**: Adjust EWMA decay, transition probabilities
2. **Add ThompsonKelly to Exp 2**: Compare learning-based adaptation
3. **Longer bear phase**: Test with more severe regime (μ=-0.05)
