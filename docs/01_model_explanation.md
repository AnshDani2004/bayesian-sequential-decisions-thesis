# Model Explanation: From LaTeX to Python

This document translates the formal mathematical notation from [`theory/model_definition.tex`](file:///Users/ansh/Developer/bayesian-sequential-decisions-thesis/theory/model_definition.tex) into plain English and shows how each concept maps to Python code.

---

## 📋 Notation Table

| **LaTeX Symbol** | **Python Variable** | **Plain English Meaning** |
|------------------|---------------------|---------------------------|
| $\Omega$ | `omega` | Sample space (all possible random outcomes) |
| $\mathcal{F}_t$ | `filtration_t` or `history_t` | Information available to the agent at time $t$ |
| $f_t$ | `betting_fraction_t` | Fraction of wealth bet at time $t$ |
| $r_t$ | `return_t` | Return (gain/loss) at time $t$ |
| $W_t$ | `wealth_t` | Total wealth at time $t$ |
| $\mu$ | `mean_return` or `mu` | Expected return (mean) |
| $\sigma^2$ | `variance` or `sigma_sq` | Variance of returns |
| $\nu$ | `degrees_of_freedom` or `nu` | Degrees of freedom for Student-t distribution |
| $D_t$ | `drawdown_t` | Maximum loss from peak wealth |
| $\text{CVaR}_{\alpha}$ | `cvar_alpha` | Conditional Value at Risk (average loss in worst $\alpha$ cases) |
| $K$ | `num_arms` | Number of bandit arms (investment options) |
| $\hat{\mu}_{i,t}$ | `posterior_mean[i, t]` | Bayesian estimate of arm $i$'s mean at time $t$ |

---

## 🧠 Concept Walkthrough

### 1. Why is $f_t$ "Predictable" (i.e., $\mathcal{F}_t$-measurable)?

**The Problem: Look-Ahead Bias**

Imagine you're deciding how much to bet on a coin flip. If you could see the outcome *before* betting, you'd always bet big on heads and nothing on tails. This is **cheating** and doesn't work in the real world.

**The Mathematical Fix**

We require that $f_t \in \mathcal{F}_t$, meaning:
- **At time $t$**, you can only use information from times $0, 1, \ldots, t-1$ (the past).
- You **cannot** use $r_t$ (the current return) because it hasn't happened yet.

**In Python Code**

```python
def make_decision(history_t):
    """
    Compute betting fraction at time t.
    
    Args:
        history_t: List of past returns [r_0, r_1, ..., r_{t-1}]
    
    Returns:
        f_t: Betting fraction (must NOT depend on r_t!)
    """
    # ✅ CORRECT: Using only past returns
    mean_estimate = np.mean(history_t)
    variance_estimate = np.var(history_t)
    f_t = mean_estimate / variance_estimate  # Kelly fraction
    
    # ❌ WRONG: Using current return (not available yet!)
    # f_t = some_function(r_t)  # This would be cheating!
    
    return f_t
```

**Intuition**

Think of $\mathcal{F}_t$ as a "knowledge cutoff." At time $t$, you're making a bet for the *next* outcome. You can use everything you've learned so far, but you can't peek into the future.

---

### 2. Why Student-t Distribution for "Heavy Tails"?

**The Normal Distribution's Weakness**

In finance, the Gaussian (Normal) distribution $\mathcal{N}(\mu, \sigma^2)$ assumes:
- Returns are symmetric around the mean
- Extreme events (e.g., market crashes) are **vanishingly rare**

For example, under $\mathcal{N}(0, 1)$:
- A 3-sigma event (3 standard deviations from mean) happens ~0.3% of the time
- A 5-sigma event happens once in 3.5 million draws

But in reality, crashes happen far more often than this!

**The Student-t Distribution**

The Student-t distribution with $\nu = 3$ degrees of freedom has:
- **Fatter tails**: Extreme events occur more frequently
- **Infinite 4th moment**: Variance of variance is undefined → estimation is unreliable

**Visual Comparison**

```
         Normal vs. Student-t (ν=3)
         
  Height
    │       Normal (thin tails)
    │         /‾‾‾\
    │        /     \
    │ ______/       \______
    │                       Student-t (fat tails)
    │              /‾‾‾‾‾\
    │             /       \
    │ ___________/         \_________
    └─────────────────────────────── Returns
              -3σ  μ  +3σ
```

With Student-t, you see many more observations in the tails (far from the mean).

**In Python**

```python
import scipy.stats as stats

# Normal distribution
normal_returns = stats.norm.rvs(loc=0.08, scale=0.15, size=10000)

# Student-t distribution (ν=3)
student_t_returns = stats.t.rvs(df=3, loc=0.08, scale=0.15, size=10000)

# Compare worst 1% of outcomes
print(f"Normal worst 1%: {np.percentile(normal_returns, 1):.2f}")
print(f"Student-t worst 1%: {np.percentile(student_t_returns, 1):.2f}")
# Student-t will show MUCH worse losses!
```

**Why This Matters for Kelly**

The Kelly Criterion assumes you know $\mu$ and $\sigma^2$ precisely. But under Student-t:
- Sample variance is **noisy** (high estimation error)
- Rare but severe losses can cause bankruptcy

This is why we need **risk constraints** (CVaR, drawdown limits) to survive in practice.

---

### 3. The Wealth Update Equation

**LaTeX**: $W_t = W_{t-1}(1 + f_t r_t)$

**Intuition**

If you have \$100 and bet 50% ($f_t = 0.5$) on an investment that returns 10% ($r_t = 0.10$):
$$
W_t = 100 \times (1 + 0.5 \times 0.10) = 100 \times 1.05 = \$105
$$

If the investment loses 10% ($r_t = -0.10$):
$$
W_t = 100 \times (1 + 0.5 \times (-0.10)) = 100 \times 0.95 = \$95
$$

**In Python**

```python
def update_wealth(W_prev, f_t, r_t):
    """
    Update wealth based on betting fraction and realized return.
    
    Args:
        W_prev: Wealth at time t-1
        f_t: Betting fraction (0 ≤ f_t ≤ 1)
        r_t: Return at time t (can be negative)
    
    Returns:
        W_t: New wealth at time t
    """
    return W_prev * (1 + f_t * r_t)

# Example
wealth_0 = 100
betting_fraction = 0.5
return_realized = 0.10

wealth_1 = update_wealth(wealth_0, betting_fraction, return_realized)
print(f"New wealth: ${wealth_1:.2f}")  # $105.00
```

---

### 4. The Kelly Objective: Why Logarithms?

**LaTeX**: $\max_{f_t} \mathbb{E}[\log(1 + f_t r_t)]$

**The Geometric Mean Trick**

Kelly maximizes the **geometric** growth rate, not the arithmetic mean. Here's why:

Suppose you have two investments:
- **Strategy A**: +50% one day, -50% the next (arithmetic mean = 0%)
  - Start: \$100 → \$150 → \$75 (end: -25%!)
- **Strategy B**: +20% one day, -10% the next (arithmetic mean = +5%)
  - Start: \$100 → \$120 → \$108 (end: +8%)

The logarithm captures the **compounding effect**:
$$
\log(W_2 / W_0) = \log(W_1 / W_0) + \log(W_2 / W_1)
$$

This is why Kelly focuses on $\mathbb{E}[\log(1 + f r)]$ rather than $\mathbb{E}[f r]$.

**In Python**

```python
def kelly_objective(f, returns_sample):
    """
    Compute expected log-wealth growth for a given betting fraction.
    
    Args:
        f: Betting fraction
        returns_sample: Array of historical returns
    
    Returns:
        Expected log growth
    """
    log_growth = np.mean(np.log(1 + f * returns_sample))
    return log_growth

# Find optimal f
from scipy.optimize import minimize_scalar

returns = np.random.normal(0.08, 0.15, 10000)
result = minimize_scalar(lambda f: -kelly_objective(f, returns), bounds=(0, 2))
f_optimal = result.x
print(f"Optimal Kelly fraction: {f_optimal:.4f}")
```

---

### 5. CVaR: The "Average Loss in Bad Times"

**LaTeX**: $\text{CVaR}_{0.05}(X) = \mathbb{E}[X \mid X \leq \text{VaR}_{0.05}(X)]$

**Intuition**

- **VaR (Value at Risk)**: The 5th percentile of returns. "In the worst 5% of cases, I lose at least this much."
- **CVaR (Conditional VaR)**: The *average* loss when you're in that worst 5%.

**Example**

Suppose your portfolio returns over 100 days are:
```
[-30%, -25%, -20%, -15%, -10%, ..., +5%, +10%, +15%, +20%, +25%]
```

- $\text{VaR}_{0.05}$: The 5th percentile = -25%
- $\text{CVaR}_{0.05}$: Average of [-30%, -25%] = -27.5%

**Why CVaR > VaR?**

CVaR is more conservative because it accounts for *how bad* things can get, not just the threshold.

**In Python**

```python
def compute_cvar(returns, alpha=0.05):
    """
    Compute alpha-CVaR (Conditional Value at Risk).
    
    Args:
        returns: Array of portfolio returns
        alpha: Risk level (e.g., 0.05 for 5%)
    
    Returns:
        CVaR value (negative number = loss)
    """
    var_threshold = np.percentile(returns, alpha * 100)
    cvar = np.mean(returns[returns <= var_threshold])
    return cvar

# Example
portfolio_returns = np.random.normal(0.08, 0.15, 10000)
cvar_5 = compute_cvar(portfolio_returns, alpha=0.05)
print(f"CVaR (5%): {cvar_5:.2%}")  # e.g., -18.5%
```

---

## 🔗 Connecting the Dots: From Math to Code

### The Full Simulation Loop

```python
# Initialize
W_0 = 100.0  # Initial wealth
T = 10000    # Number of time steps
history = []

# Environment parameters
mu = 0.08          # True mean return
sigma = 0.15       # True volatility
nu = 3             # Degrees of freedom (heavy tails)

# Simulation
wealth = W_0
for t in range(T):
    # 1. Compute betting fraction (F_t-measurable!)
    if len(history) > 0:
        mu_hat = np.mean([r for _, r in history])  # Estimate mean
        sigma_hat_sq = np.var([r for _, r in history])  # Estimate variance
        f_t = mu_hat / sigma_hat_sq if sigma_hat_sq > 0 else 0
    else:
        f_t = 0.1  # Initial guess
    
    # 2. Observe return (F_{t+1}-measurable)
    r_t = stats.t.rvs(df=nu, loc=mu, scale=sigma) / 100  # Student-t return
    
    # 3. Update wealth
    wealth = wealth * (1 + f_t * r_t)
    
    # 4. Record history
    history.append((f_t, r_t))
    
    # 5. Stop if bankrupt
    if wealth <= 0:
        print(f"Bankruptcy at t={t}")
        break

print(f"Final wealth: ${wealth:.2f}")
```

---

## ✅ Key Takeaways

1. **Filtration $\mathcal{F}_t$** = "What you know at time $t$" → Use only `history[:t]` in Python
2. **Predictability** = No look-ahead bias → `f_t` cannot depend on `r_t`
3. **Student-t** = Heavy tails → More realistic for financial crashes
4. **Kelly Criterion** = Maximize $\mathbb{E}[\log(1 + fr)]$ → Optimal long-run growth
5. **CVaR** = Average loss in worst cases → More conservative than VaR

---

## 🚀 Next Steps

Now that the mathematical framework is defined:
1. Implement `BanditEnvironment` class with Student-t support
2. Create `Agent` base class enforcing the predictability constraint
3. Test Kelly formula in both Gaussian and Student-t regimes
4. Add CVaR constraints using `cvxpy`

All Python variable names will now strictly match the LaTeX notation! 🎓
