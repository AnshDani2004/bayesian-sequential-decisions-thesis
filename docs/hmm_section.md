# HMM Regime Switching Section

### 6. Hidden Markov Models (HMM) & Regime Switching

**Motivation**: Markets aren't stationary—they switch between "bull" (high returns) and "bear" (low/negative returns) regimes.

**The Challenge**: You can't directly observe which regime you're in—you must infer it from the returns.

**HMM Structure**

```
Hidden States:  S_1 → S_2 → S_3 → S_4 → ...  (not observable)
                 ↓     ↓     ↓     ↓
Observations:   r_1   r_2   r_3   r_4   ...  (observable returns)
```

**Components**:

1. **State Space**: $S_t \in \{1, 2, \ldots, M\}$ (e.g., $M=2$ for Bull/Bear)
2. **Transition Matrix** $P$:
   ```
   P[i,j] = P(S_{t+1}=j | S_t=i)
   
   Example (Bull/Bear):
   P = [[0.95, 0.05],   # From Bull: 95% stay Bull, 5% → Bear
        [0.10, 0.90]]   # From Bear: 10% → Bull, 90% stay Bear
   ```

3. **Regime-Dependent Returns**:
   - Bull regime ($S_t = 1$): $r_t \sim \mathcal{N}(0.12, 0.15^2)$ (high mean)
   - Bear regime ($S_t = 2$): $r_t \sim \mathcal{N}(-0.05, 0.25^2)$ (negative mean, higher volatility)

**Bayesian Belief Propagation**

At each time $t$, maintain a **belief state**: $\pi_t(i) = \mathbb{P}(S_t = i \mid r_1, \ldots, r_t)$

**Update Formula** (Forward Algorithm):
$$\pi_t(i) = \frac{\mathcal{L}(r_t \mid S_t = i) \sum_{j=1}^{M} P_{ji} \pi_{t-1}(j)}{\text{normalizer}}$$

**Two-Step Process**:
1. **Prediction**: Use transition matrix to propagate belief forward
2. **Update**: Use likelihood of observed return to refine belief

**In Python**

```python
import numpy as np
from scipy.stats import norm

def update_belief(pi_prev, r_t, P, theta):
    """
    Update belief state using Bayes' Rule (Forward Algorithm).
    
    Args:
        pi_prev: Previous belief [pi_{t-1}(1), ..., pi_{t-1}(M)]
        r_t: Observed return at time t
        P: Transition matrix (M x M)
        theta: List of regime parameters [(mu_1, sigma_1), ...]
    
    Returns:
        pi_t: Updated belief
    """
    M = len(pi_prev)
    
    # Step 1: Prediction (propagate using transition matrix)
    pi_predict = P.T @ pi_prev  # P.T because P[i,j] = P(S_{t+1}=j|S_t=i)
    
    # Step 2: Compute likelihoods
    likelihoods = np.array([
        norm.pdf(r_t, loc=theta[i][0], scale=theta[i][1])
        for i in range(M)
    ])
    
    # Step 3: Bayesian update
    pi_t_unnormalized = likelihoods * pi_predict
    
    # Step 4: Normalize
    pi_t = pi_t_unnormalized / np.sum(pi_t_unnormalized)
    
    return pi_t
```

**Example**: Regime Detection

```python
M = 2  # Two regimes
P = np.array([[0.95, 0.05], [0.10, 0.90]])
theta = [(0.12, 0.15), (-0.05, 0.25)]  # Bull and Bear params

# Initial belief (uniform)
pi_0 = np.array([0.5, 0.5])

# Observe large positive return
r_1 = 0.15  # Suggests Bull market
pi_1 = update_belief(pi_0, r_1, P, theta)
print(f"After r_1={r_1:.2f}: P(Bull)={pi_1[0]:.2%}, P(Bear)={pi_1[1]:.2%}")
# Output: P(Bull) ≈ 90%, P(Bear) ≈ 10%
```

**Regime-Weighted Kelly Allocation**

Instead of a single Kelly formula, blend across regimes:
$$f_{t+1} = \sum_{i=1}^{M} \pi_t(i) \cdot f^*_i, \quad \text{where } f^*_i = \frac{\mu_i}{\sigma_i^2}$$

This allows the agent to hedge against regime uncertainty!
