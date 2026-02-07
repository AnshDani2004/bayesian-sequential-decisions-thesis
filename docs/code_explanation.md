# Phase 2: Environment Module Documentation

This document explains how the simulation code maps to the mathematical framework in [`theory/model_definition.tex`](file:///Users/ansh/Developer/bayesian-sequential-decisions-thesis/theory/model_definition.tex).

---

## 📋 Module Overview

| **File** | **Purpose** |
|----------|-------------|
| `simulation/distributions.py` | Numba-optimized random number generators |
| `simulation/environment.py` | `MarketEnvironment` class ("physics engine") |
| `tests/test_environment.py` | Unit tests for all environment functionality |

---

## 🔗 Mapping `simulate_batch` to Probability Space $(\Omega, \mathbb{P})$

### Mathematical Framework (from LaTeX)

The filtered probability space $(\Omega, \mathcal{F}, \{\mathcal{F}_t\}, \mathbb{P})$ defines:
- $\Omega$: Sample space of all possible market trajectories
- $\mathcal{F}_t = \sigma(r_1, \ldots, r_t)$: Information at time $t$
- $\mathbb{P}$: Probability measure over trajectories

### Code Implementation

```python
result = env.simulate_batch(n_sims=N, t_steps=T)
# result.returns: shape (N, T) matrix of returns
# result.regimes: shape (N, T) matrix of hidden states
```

**Interpretation**:
- Each row `result.returns[i, :]` is a **sample from $\Omega$** — one complete market trajectory
- The `n_sims` dimension represents **Monte Carlo sampling** from $\mathbb{P}$
- The `t_steps` dimension represents **time evolution** $t = 1, 2, \ldots, T$

---

## ⏱️ Matrix Indexing Convention

> **CRITICAL**: The matrix index `[i, t]` corresponds to $r_{t+1}$ in LaTeX notation.

### Why This Convention?

In the Shreve timing convention:
- At time $t$: Agent observes $\mathcal{F}_t = \sigma(r_1, \ldots, r_t)$
- Agent decides $f_{t+1}$ based on $\mathcal{F}_t$
- At time $t+1$: Return $r_{t+1}$ is realized

### Python ↔ LaTeX Mapping

| **Python** | **LaTeX** | **Meaning** |
|------------|-----------|-------------|
| `returns[i, 0]` | $r_1$ | First return (revealed at time 1) |
| `returns[i, 1]` | $r_2$ | Second return (revealed at time 2) |
| `returns[i, t]` | $r_{t+1}$ | Return realized at the **end** of step $t$ |
| `returns[i, T-1]` | $r_T$ | Last return (revealed at time $T$) |

### Example

```python
# Simulate 100 paths, each with 50 time steps
result = env.simulate_batch(n_sims=100, t_steps=50)

# Path i's trajectory:
#   r_1 = result.returns[i, 0]   # Return at end of step 0
#   r_2 = result.returns[i, 1]   # Return at end of step 1
#   ...
#   r_50 = result.returns[i, 49] # Return at end of step 49
```

---

## 🧮 Regime Switching (HMM)

### Mathematical Framework

Hidden state process $S_t \in \{1, \ldots, M\}$ with:
- Transition matrix: $P_{ij} = \mathbb{P}(S_{t+1} = j \mid S_t = i)$
- Regime-dependent returns: $r_t \mid S_t = s \sim p(r \mid \theta_s)$

### Code Implementation

```python
env = MarketEnvironment(
    regime_config={
        'bull': {'mu': 0.05, 'sigma': 0.10},           # Gaussian
        'bear': {'mu': -0.02, 'sigma': 0.25, 'df': 3}  # Student-t
    },
    trans_matrix=np.array([
        [0.95, 0.05],  # P(bull→bull)=0.95, P(bull→bear)=0.05
        [0.10, 0.90]   # P(bear→bull)=0.10, P(bear→bear)=0.90
    ])
)
```

---

## 🚫 No Look-Ahead Guarantee

The environment enforces the **predictability constraint** $f_t \in \mathcal{F}_{t-1}$:

1. **Batch Generation**: All returns are pre-generated before any agent runs
2. **Separate Arrays**: Returns stored separately from agent state
3. **Design Pattern**: Agent receives historical returns only, never future

```python
# CORRECT usage in agent loop:
for t in range(T):
    # Agent has access to: returns[:, :t] (history only)
    f_next = agent.decide(returns[i, :t])
    
    # Return r_{t+1} realized AFTER decision
    r_realized = returns[i, t]
    wealth = update_wealth(wealth, f_next, r_realized)
```

---

## 🎲 Reproducibility via Seeding

Every stochastic method accepts a `seed` parameter:

```python
# Instance-level seed
env = MarketEnvironment(config, trans_matrix, seed=42)

# Per-batch override
result1 = env.simulate_batch(100, 50, seed=123)
result2 = env.simulate_batch(100, 50, seed=123)

assert np.array_equal(result1.returns, result2.returns)  # ✅ Identical
```

---

## ⚡ Numba Optimization

Core loops are JIT-compiled for high performance:

| **Function** | **Speedup** | **Parallelization** |
|--------------|-------------|---------------------|
| `sample_markov_chain` | ~50x | Single-threaded |
| `sample_student_t` | ~30x | Single-threaded |
| `sample_markov_chain_batch` | ~100x | `prange` parallel |
| `sample_regime_returns_batch` | ~100x | `prange` parallel |

**Performance**: Can simulate $10^5$+ paths in seconds.

---

## 📊 Common Random Numbers (CRN)

The batch design enables **Common Random Numbers** methodology:

```python
# Same random draws for fair comparison
result = env.simulate_batch(n_sims=10000, t_steps=252, seed=42)

# Agent A runs on this data
wealth_a = simulate_agent_a(result.returns)

# Agent B runs on SAME data
wealth_b = simulate_agent_b(result.returns)

# Valid comparison: variance due to strategy, not luck
```

---

## ✅ Unit Test Summary

| **Test Class** | **Verifies** |
|----------------|--------------|
| `TestFatTails` | Kurtosis > 3 for Student-t($\nu$=3) |
| `TestRegimeSwitching` | Average run length matches $1/(1-P_{ii})$ |
| `TestDimensions` | Output shapes exactly `(n_sims, t_steps)` |
| `TestReproducibility` | Same seed → identical results |
| `TestValidation` | Catches invalid transition matrices |
| `TestMeanVariance` | Empirical $\mu$, $\sigma$ match config |

Run tests:
```bash
cd tests && pytest test_environment.py -v
```
