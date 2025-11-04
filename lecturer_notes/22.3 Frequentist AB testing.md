# AB Testing: Frequentist Approach - Comprehensive Practice Guide

## Table of Contents
1. [Conceptual Questions](#conceptual-questions)
2. [Calculation Problems](#calculation-problems)
3. [Practical Implementation Tasks](#practical-implementation-tasks)
4. [Real-World Case Studies](#real-world-case-studies)
5. [Solutions](#solutions)

---

## Conceptual Questions

### Q1: Fundamentals of Hypothesis Testing
**a)** What is the null hypothesis (H₀) in an AB test comparing conversion rates?
**b)** What is the alternative hypothesis (H₁)?
**c)** What does it mean to reject the null hypothesis?
**d)** Explain the difference between one-tailed and two-tailed tests.

### Q2: Type I and Type II Errors
**a)** Define Type I error (α) and provide an example in the context of AB testing.
**b)** Define Type II error (β) and provide an example in the context of AB testing.
**c)** What is statistical power and how does it relate to Type II error?
**d)** If you decrease α, what happens to β? Why?

### Q3: P-values
**a)** What does a p-value represent?
**b)** If p-value = 0.03 and α = 0.05, what decision do you make?
**c)** Why is p-value < 0.05 not the same as "95% probability that the result is real"?
**d)** Can a p-value tell you the size of an effect? Explain.

### Q4: Sample Size and Power
**a)** Why does increasing sample size increase statistical power?
**b)** What factors determine the required sample size for an AB test?
**c)** If you want to detect a smaller effect, do you need more or fewer samples?
**d)** What happens if you run a test with insufficient sample size?

### Q5: Multiple Testing Problem
**a)** What is the multiple testing problem?
**b)** If you run 20 AB tests at α = 0.05, how many false positives would you expect by chance?
**c)** What is the Bonferroni correction and how does it work?
**d)** What are alternative corrections to Bonferroni?

---

## Calculation Problems

### Problem 1: Basic Z-Test for Proportions
You're testing a new website design:
- **Control Group (A):** 1,000 visitors, 120 conversions
- **Treatment Group (B):** 1,000 visitors, 145 conversions

**Tasks:**
1. Calculate the conversion rates for both groups
2. State the null and alternative hypotheses
3. Calculate the pooled proportion
4. Calculate the standard error
5. Calculate the z-statistic
6. Find the p-value (two-tailed test)
7. Make a decision at α = 0.05
8. Calculate the 95% confidence interval for the difference

### Problem 2: Sample Size Calculation
You want to detect a lift of 10% from a baseline conversion rate of 5%.

**Tasks:**
1. Calculate the minimum sample size needed per group for:
   - α = 0.05
   - Power = 0.80
   - Two-tailed test
2. How would the sample size change if you wanted power = 0.90?
3. What if the baseline conversion rate was 1% instead?

### Problem 3: T-Test for Continuous Metrics
You're testing the effect of a new feature on average order value:
- **Control Group (A):** n=500, mean=$45.20, std=$12.30
- **Treatment Group (B):** n=500, mean=$48.50, std=$13.10

**Tasks:**
1. State the hypotheses
2. Calculate the pooled standard deviation
3. Calculate the t-statistic
4. Find the degrees of freedom
5. Find the p-value
6. Make a decision at α = 0.05
7. Calculate Cohen's d (effect size)

### Problem 4: Chi-Square Test
You're testing three different call-to-action buttons:

|          | Clicked | Not Clicked | Total |
|----------|---------|-------------|-------|
| Button A | 85      | 415         | 500   |
| Button B | 102     | 398         | 500   |
| Button C | 78      | 422         | 500   |

**Tasks:**
1. Calculate expected frequencies
2. Calculate the chi-square statistic
3. Find the degrees of freedom
4. Find the p-value
5. Make a decision at α = 0.05

### Problem 5: Multiple Testing Correction
You run 10 AB tests simultaneously with the following p-values:
```
Test 1: 0.032
Test 2: 0.089
Test 3: 0.012
Test 4: 0.156
Test 5: 0.045
Test 6: 0.203
Test 7: 0.008
Test 8: 0.091
Test 9: 0.067
Test 10: 0.178
```

**Tasks:**
1. Which tests would be significant at α = 0.05 without correction?
2. Apply Bonferroni correction. Which tests remain significant?
3. Apply Benjamini-Hochberg procedure (FDR = 0.05). Which tests remain significant?

---

## Practical Implementation Tasks

### Task 1: Implement Z-Test from Scratch
Write Python code to:
1. Generate simulated AB test data
2. Implement a function that performs a two-proportion z-test
3. Calculate p-value without using statsmodels
4. Verify your implementation against scipy.stats

### Task 2: Sample Size Calculator
Build a function that:
1. Takes baseline conversion rate, minimum detectable effect, α, and power as inputs
2. Returns required sample size per group
3. Visualizes how sample size changes with different parameters

### Task 3: Sequential Testing Simulator
Create a simulation that:
1. Generates data for a true effect size of 2%
2. Tests the data at multiple points (peeking)
3. Shows how peeking inflates Type I error rate
4. Compare with proper sequential testing boundaries

### Task 4: Power Analysis
Write code to:
1. Calculate statistical power for different sample sizes
2. Create a power curve plot
3. Determine optimal sample size given constraints

### Task 5: AB Test Dashboard
Build a function that outputs:
1. Summary statistics for both groups
2. Confidence intervals
3. P-value and statistical significance
4. Effect size (absolute and relative)
5. Recommendation (which variant to choose)

---

## Real-World Case Studies

### Case Study 1: E-commerce Checkout Flow
**Background:** An e-commerce site wants to test a one-page checkout (B) vs. multi-step checkout (A).

**Data:**
- Control (A): 5,200 users, 728 completed purchases
- Treatment (B): 5,000 users, 750 completed purchases

**Questions:**
1. Is the difference statistically significant?
2. What is the practical significance (lift)?
3. Should they roll out variant B?
4. What would the expected impact be on 1 million monthly users?

### Case Study 2: Email Campaign
**Background:** Testing subject line variations for an email campaign.

**Data:**
- Subject A "Sale Inside": 10,000 sent, 1,240 opens
- Subject B "50% Off Today": 10,000 sent, 1,380 opens

**Questions:**
1. Perform the hypothesis test
2. Calculate confidence interval for the difference
3. If the company sends 2 million emails per month, what's the expected increase in opens?
4. What would you recommend?

### Case Study 3: Mobile App Feature
**Background:** Testing whether adding a "favorites" feature increases user engagement (sessions per user per week).

**Data:**
- Control: 1,200 users, mean=4.2 sessions, std=2.1
- Treatment: 1,200 users, mean=4.6 sessions, std=2.3

**Questions:**
1. Is this difference statistically significant?
2. Calculate effect size
3. Is this practically significant?
4. What are the business implications?

### Case Study 4: Early Stopping Dilemma
**Background:** You planned for 10,000 samples per group. After 3,000 samples:
- Control: 3,000 users, 180 conversions (6.0%)
- Treatment: 3,000 users, 225 conversions (7.5%)
- P-value: 0.012

**Questions:**
1. Should you stop the test early?
2. What are the risks of stopping?
3. How would you adjust the analysis?
4. What would you recommend?

---

## Solutions

### Conceptual Questions - Solutions

#### Q1: Fundamentals of Hypothesis Testing
**a)** H₀: pₐ = pᵦ (no difference in conversion rates between variants)
**b)** H₁: pₐ ≠ pᵦ (there is a difference) for two-tailed; pᵦ > pₐ for one-tailed
**c)** Rejecting H₀ means we have sufficient evidence that there is a real difference between the groups, not just random chance
**d)**
- **Two-tailed:** Tests if there's ANY difference (B could be better OR worse than A)
- **One-tailed:** Tests if B is specifically better (or worse) than A
- Use two-tailed when direction is uncertain; one-tailed when you only care about improvement

#### Q2: Type I and Type II Errors
**a)** Type I error (False Positive): Concluding there's a difference when there isn't. Example: Declaring variant B better when it's actually the same as A, wasting resources on a pointless change.

**b)** Type II error (False Negative): Missing a real difference. Example: Failing to detect that variant B is actually better, missing an opportunity for improvement.

**c)** Statistical power = 1 - β. It's the probability of correctly detecting a real effect when it exists. Higher power means lower chance of Type II error.

**d)** As α decreases, β increases (power decreases). More stringent criteria for significance means we're more likely to miss real effects. There's a tradeoff between false positives and false negatives.

#### Q3: P-values
**a)** P-value is the probability of observing data as extreme as (or more extreme than) what we observed, assuming the null hypothesis is true.

**b)** Since p-value (0.03) < α (0.05), we reject the null hypothesis and conclude the difference is statistically significant.

**c)** P-value is calculated assuming H₀ is true—it's P(data|H₀), not P(H₀|data). It tells us about data extremeness, not the probability that our hypothesis is correct.

**d)** No. P-value only tells us about statistical significance (likelihood of observing data if no effect exists). A tiny effect can have a small p-value with large sample size, while a large effect can have a large p-value with small sample size.

#### Q4: Sample Size and Power
**a)** Larger samples reduce sampling variability (standard error), making it easier to detect a true signal from noise. The signal-to-noise ratio improves.

**b)** Required sample size depends on:
- Baseline conversion rate
- Minimum detectable effect (MDE)
- Significance level (α)
- Desired power (1-β)
- Variance of the metric

**c)** More samples. Smaller effects are harder to distinguish from random noise, so you need more data to reliably detect them.

**d)** Underpowered tests have:
- High risk of missing real effects (Type II error)
- Unstable estimates (high variance)
- Reduced credibility
- Wasted resources on inconclusive results

#### Q5: Multiple Testing Problem
**a)** When conducting multiple hypothesis tests, the probability of at least one false positive increases. Running 20 tests at α=0.05 gives ~64% chance of at least one false positive.

**b)** Expected false positives = 20 × 0.05 = 1 false positive on average

**c)** Bonferroni correction: Divide α by number of tests. For 20 tests: use α = 0.05/20 = 0.0025. This controls family-wise error rate (FWER) but is very conservative.

**d)** Alternatives:
- **Holm-Bonferroni:** Less conservative step-down procedure
- **Benjamini-Hochberg:** Controls False Discovery Rate (FDR), less conservative
- **Šidák correction:** Similar to Bonferroni but accounts for independence
- **Bootstrapping/permutation tests**

---

### Calculation Problems - Solutions

#### Problem 1: Basic Z-Test for Proportions

**1. Conversion rates:**
- pₐ = 120/1000 = 0.12 (12%)
- pᵦ = 145/1000 = 0.145 (14.5%)

**2. Hypotheses:**
- H₀: pₐ = pᵦ
- H₁: pₐ ≠ pᵦ (two-tailed)

**3. Pooled proportion:**
```
p̂ = (120 + 145) / (1000 + 1000) = 265/2000 = 0.1325
```

**4. Standard error:**
```
SE = √[p̂(1-p̂)(1/nₐ + 1/nᵦ)]
SE = √[0.1325 × 0.8675 × (1/1000 + 1/1000)]
SE = √[0.1149 × 0.002]
SE = √0.0002298
SE ≈ 0.01516
```

**5. Z-statistic:**
```
z = (pᵦ - pₐ) / SE
z = (0.145 - 0.12) / 0.01516
z = 0.025 / 0.01516
z ≈ 1.649
```

**6. P-value (two-tailed):**
```
P-value = 2 × P(Z > 1.649)
P-value = 2 × 0.0495
P-value ≈ 0.099
```

**7. Decision:**
Since p-value (0.099) > α (0.05), we **fail to reject** the null hypothesis. The difference is not statistically significant at the 5% level.

**8. 95% Confidence Interval:**
```
CI = (pᵦ - pₐ) ± 1.96 × SE
CI = 0.025 ± 1.96 × 0.01516
CI = 0.025 ± 0.0297
CI = (-0.0047, 0.0547) or (-0.47%, 5.47%)
```
Since the interval contains 0, it confirms our failure to reject H₀.

---

#### Problem 2: Sample Size Calculation

**1. Sample size for α=0.05, power=0.80:**

Using the formula for proportions:
```
n = [Z₁₋α/₂√(2p̄(1-p̄)) + Z₁₋β√(pₐ(1-pₐ) + pᵦ(1-pᵦ))]² / (pᵦ - pₐ)²

Where:
- pₐ = 0.05 (baseline)
- pᵦ = 0.055 (10% relative lift)
- p̄ = (pₐ + pᵦ)/2 = 0.0525
- Z₁₋α/₂ = 1.96 (for α=0.05, two-tailed)
- Z₁₋β = 0.84 (for power=0.80)

n = [1.96√(2×0.0525×0.9475) + 0.84√(0.05×0.95 + 0.055×0.945)]² / (0.005)²
n = [1.96√0.0995 + 0.84√0.0995]² / 0.000025
n = [1.96×0.3154 + 0.84×0.3154]² / 0.000025
n = [0.6182 + 0.2649]² / 0.000025
n = [0.8831]² / 0.000025
n = 0.7799 / 0.000025
n ≈ 31,196 per group
```

**2. For power = 0.90:**
```
Z₁₋β = 1.28
n ≈ 41,600 per group
```
Higher power requires more samples (about 33% increase).

**3. For baseline = 1% (10% relative lift to 1.1%):**
```
n ≈ 95,800 per group
```
Lower baseline rates require much larger samples for the same relative effect.

---

#### Problem 3: T-Test for Continuous Metrics

**1. Hypotheses:**
- H₀: μₐ = μᵦ
- H₁: μₐ ≠ μᵦ

**2. Pooled standard deviation:**
```
s_p = √[((nₐ-1)sₐ² + (nᵦ-1)sᵦ²) / (nₐ + nᵦ - 2)]
s_p = √[((499×12.3²) + (499×13.1²)) / 998]
s_p = √[(499×151.29 + 499×171.61) / 998]
s_p = √[(75,493.71 + 85,633.39) / 998]
s_p = √[161,127.1 / 998]
s_p ≈ 12.71
```

**3. T-statistic:**
```
SE = s_p × √(1/nₐ + 1/nᵦ)
SE = 12.71 × √(1/500 + 1/500)
SE = 12.71 × √0.004
SE = 12.71 × 0.0632
SE ≈ 0.803

t = (x̄ᵦ - x̄ₐ) / SE
t = (48.50 - 45.20) / 0.803
t = 3.30 / 0.803
t ≈ 4.11
```

**4. Degrees of freedom:**
```
df = nₐ + nᵦ - 2 = 500 + 500 - 2 = 998
```

**5. P-value:**
For t = 4.11 with df = 998, p-value < 0.0001 (highly significant)

**6. Decision:**
Since p-value < 0.05, we **reject the null hypothesis**. The treatment significantly increases average order value.

**7. Cohen's d (effect size):**
```
d = (x̄ᵦ - x̄ₐ) / s_p
d = (48.50 - 45.20) / 12.71
d = 3.30 / 12.71
d ≈ 0.26
```
This is a **small to medium** effect size by Cohen's conventions (0.2 = small, 0.5 = medium, 0.8 = large).

---

#### Problem 4: Chi-Square Test

**1. Expected frequencies:**
```
Total clicks = 85 + 102 + 78 = 265
Total non-clicks = 415 + 398 + 422 = 1235
Total observations = 1500

Expected clicks per button = 265/3 ≈ 88.33
Expected non-clicks per button = 1235/3 ≈ 411.67
```

|          | Clicked (O) | Expected (E) | Not Clicked (O) | Expected (E) |
|----------|-------------|--------------|-----------------|--------------|
| Button A | 85          | 88.33        | 415             | 411.67       |
| Button B | 102         | 88.33        | 398             | 411.67       |
| Button C | 78          | 88.33        | 422             | 411.67       |

**2. Chi-square statistic:**
```
χ² = Σ[(O - E)² / E]

For Button A clicks: (85 - 88.33)²/88.33 = 0.125
For Button A non-clicks: (415 - 411.67)²/411.67 = 0.027
For Button B clicks: (102 - 88.33)²/88.33 = 2.115
For Button B non-clicks: (398 - 411.67)²/411.67 = 0.454
For Button C clicks: (78 - 88.33)²/88.33 = 1.208
For Button C non-clicks: (422 - 411.67)²/411.67 = 0.259

χ² = 0.125 + 0.027 + 2.115 + 0.454 + 1.208 + 0.259
χ² ≈ 4.188
```

**3. Degrees of freedom:**
```
df = (rows - 1) × (columns - 1) = (2 - 1) × (3 - 1) = 2
```

**4. P-value:**
For χ² = 4.188 with df = 2, p-value ≈ 0.123

**5. Decision:**
Since p-value (0.123) > α (0.05), we **fail to reject** the null hypothesis. There's no statistically significant difference between the three buttons.

---

#### Problem 5: Multiple Testing Correction

**1. Without correction (α = 0.05):**
Significant tests: 1 (0.032), 3 (0.012), 5 (0.045), 7 (0.008)
**4 tests** would be declared significant.

**2. Bonferroni correction:**
```
Adjusted α = 0.05/10 = 0.005
```
Only tests with p < 0.005 are significant:
- Test 7 (0.008) is NOT significant
- Test 3 (0.012) is NOT significant

**No tests remain significant** with Bonferroni correction.

Wait, let me recalculate:
- Test 7: p = 0.008 > 0.005 (not significant)

Actually, if we had a test with p < 0.005, it would be significant. But none of our tests meet this criterion.

**3. Benjamini-Hochberg procedure (FDR = 0.05):**

Steps:
1. Order p-values from smallest to largest
2. Find largest i where p(i) ≤ (i/m) × Q

Ordered p-values:
```
i   Test  p-value  (i/10)×0.05
1   7     0.008    0.005
2   3     0.012    0.010
3   1     0.032    0.015
4   5     0.045    0.020
5   9     0.067    0.025
6   2     0.089    0.030
7   8     0.091    0.035
8   4     0.156    0.040
9   10    0.178    0.045
10  6     0.203    0.050
```

Checking from bottom up:
- Test 3: p=0.012 ≤ 0.010? NO
- Test 7: p=0.008 ≤ 0.005? NO

With strict BH procedure, **no tests remain significant**.

However, this seems conservative. Let me recalculate more carefully:
- i=1: 0.008 ≤ 0.005? NO
- So no tests pass BH at FDR=0.05

**Note:** In practice, these results suggest the effects may be weaker than initially thought when properly accounting for multiple comparisons.

---

### Practical Implementation Tasks - Solutions

#### Task 1: Implement Z-Test from Scratch

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def generate_ab_test_data(n_control, n_treatment, p_control, p_treatment, seed=42):
    """Generate simulated AB test data"""
    np.random.seed(seed)
    control = np.random.binomial(1, p_control, n_control)
    treatment = np.random.binomial(1, p_treatment, n_treatment)
    return control, treatment

def two_proportion_z_test(conversions_a, n_a, conversions_b, n_b, alternative='two-sided'):
    """
    Perform two-proportion z-test

    Parameters:
    -----------
    conversions_a : int - number of conversions in control
    n_a : int - sample size of control
    conversions_b : int - number of conversions in treatment
    n_b : int - sample size of treatment
    alternative : str - 'two-sided', 'greater', or 'less'

    Returns:
    --------
    dict with test results
    """
    # Calculate proportions
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b

    # Pooled proportion
    p_pool = (conversions_a + conversions_b) / (n_a + n_b)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))

    # Z-statistic
    z_stat = (p_b - p_a) / se

    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    # Confidence interval (95%)
    se_diff = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
    ci_lower = (p_b - p_a) - 1.96 * se_diff
    ci_upper = (p_b - p_a) + 1.96 * se_diff

    return {
        'p_control': p_a,
        'p_treatment': p_b,
        'difference': p_b - p_a,
        'relative_lift': (p_b - p_a) / p_a if p_a > 0 else np.inf,
        'z_statistic': z_stat,
        'p_value': p_value,
        'ci_95': (ci_lower, ci_upper),
        'significant': p_value < 0.05
    }

# Example usage
control, treatment = generate_ab_test_data(1000, 1000, 0.12, 0.145)

conversions_a = control.sum()
conversions_b = treatment.sum()

# Our implementation
result = two_proportion_z_test(conversions_a, 1000, conversions_b, 1000)

print("=== Custom Implementation ===")
print(f"Control rate: {result['p_control']:.4f}")
print(f"Treatment rate: {result['p_treatment']:.4f}")
print(f"Difference: {result['difference']:.4f}")
print(f"Relative lift: {result['relative_lift']:.2%}")
print(f"Z-statistic: {result['z_statistic']:.4f}")
print(f"P-value: {result['p_value']:.4f}")
print(f"95% CI: ({result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f})")
print(f"Significant: {result['significant']}")

# Verify with statsmodels
from statsmodels.stats.proportion import proportions_ztest

counts = np.array([conversions_b, conversions_a])
nobs = np.array([1000, 1000])
z_stat_sm, p_value_sm = proportions_ztest(counts, nobs)

print("\n=== Statsmodels Verification ===")
print(f"Z-statistic: {z_stat_sm:.4f}")
print(f"P-value: {p_value_sm:.4f}")
print(f"\nMatch: Z={abs(result['z_statistic'] - z_stat_sm) < 0.01}, "
      f"P={abs(result['p_value'] - p_value_sm) < 0.01}")
```

---

#### Task 2: Sample Size Calculator

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.80, alternative='two-sided'):
    """
    Calculate required sample size for AB test

    Parameters:
    -----------
    baseline_rate : float - baseline conversion rate (e.g., 0.05 for 5%)
    mde : float - minimum detectable effect (relative, e.g., 0.10 for 10% lift)
    alpha : float - significance level
    power : float - statistical power (1 - beta)
    alternative : str - 'two-sided' or 'one-sided'

    Returns:
    --------
    int - required sample size per group
    """
    # Calculate treatment rate
    treatment_rate = baseline_rate * (1 + mde)

    # Average rate
    p_avg = (baseline_rate + treatment_rate) / 2

    # Z-scores
    if alternative == 'two-sided':
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)

    z_beta = stats.norm.ppf(power)

    # Effect size
    effect = treatment_rate - baseline_rate

    # Sample size formula
    numerator = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
                 z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) +
                                 treatment_rate * (1 - treatment_rate)))**2
    denominator = effect**2

    n = numerator / denominator

    return int(np.ceil(n))

def visualize_sample_size_sensitivity(baseline_rate=0.05):
    """Create visualizations showing how sample size changes with parameters"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sample size vs MDE
    mde_range = np.linspace(0.05, 0.50, 50)
    sample_sizes_mde = [calculate_sample_size(baseline_rate, mde) for mde in mde_range]

    axes[0, 0].plot(mde_range * 100, sample_sizes_mde, linewidth=2)
    axes[0, 0].set_xlabel('Minimum Detectable Effect (%)', fontsize=12)
    axes[0, 0].set_ylabel('Sample Size per Group', fontsize=12)
    axes[0, 0].set_title(f'Sample Size vs MDE (baseline={baseline_rate:.1%})', fontsize=13)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    # 2. Sample size vs Power
    power_range = np.linspace(0.50, 0.99, 50)
    sample_sizes_power = [calculate_sample_size(baseline_rate, 0.10, power=p)
                          for p in power_range]

    axes[0, 1].plot(power_range * 100, sample_sizes_power, linewidth=2, color='green')
    axes[0, 1].set_xlabel('Statistical Power (%)', fontsize=12)
    axes[0, 1].set_ylabel('Sample Size per Group', fontsize=12)
    axes[0, 1].set_title(f'Sample Size vs Power (MDE=10%)', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Sample size vs Baseline Rate
    baseline_range = np.linspace(0.01, 0.20, 50)
    sample_sizes_baseline = [calculate_sample_size(b, 0.10) for b in baseline_range]

    axes[1, 0].plot(baseline_range * 100, sample_sizes_baseline, linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Baseline Conversion Rate (%)', fontsize=12)
    axes[1, 0].set_ylabel('Sample Size per Group', fontsize=12)
    axes[1, 0].set_title('Sample Size vs Baseline Rate (MDE=10%)', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Sample size heatmap (MDE vs Baseline)
    mde_grid = np.linspace(0.05, 0.30, 30)
    baseline_grid = np.linspace(0.01, 0.15, 30)

    sample_size_matrix = np.zeros((len(baseline_grid), len(mde_grid)))
    for i, b in enumerate(baseline_grid):
        for j, m in enumerate(mde_grid):
            sample_size_matrix[i, j] = calculate_sample_size(b, m)

    im = axes[1, 1].imshow(sample_size_matrix, aspect='auto', origin='lower',
                           cmap='viridis', extent=[mde_grid[0]*100, mde_grid[-1]*100,
                                                   baseline_grid[0]*100, baseline_grid[-1]*100])
    axes[1, 1].set_xlabel('Minimum Detectable Effect (%)', fontsize=12)
    axes[1, 1].set_ylabel('Baseline Rate (%)', fontsize=12)
    axes[1, 1].set_title('Sample Size Heatmap', fontsize=13)
    plt.colorbar(im, ax=axes[1, 1], label='Sample Size per Group')

    plt.tight_layout()
    plt.savefig('sample_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
print("=== Sample Size Examples ===\n")

# Example 1: E-commerce conversion
n1 = calculate_sample_size(baseline_rate=0.05, mde=0.10, alpha=0.05, power=0.80)
print(f"E-commerce (5% baseline, 10% lift, 80% power): {n1:,} per group")

# Example 2: Higher power
n2 = calculate_sample_size(baseline_rate=0.05, mde=0.10, alpha=0.05, power=0.90)
print(f"Same but with 90% power: {n2:,} per group ({(n2-n1)/n1:.1%} increase)")

# Example 3: Lower baseline
n3 = calculate_sample_size(baseline_rate=0.01, mde=0.10, alpha=0.05, power=0.80)
print(f"Lower baseline (1% baseline, 10% lift): {n3:,} per group")

# Example 4: Smaller effect
n4 = calculate_sample_size(baseline_rate=0.05, mde=0.05, alpha=0.05, power=0.80)
print(f"Smaller effect (5% baseline, 5% lift): {n4:,} per group")

# Create visualizations
visualize_sample_size_sensitivity()
```

---

#### Task 3: Sequential Testing Simulator

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulate_peeking_problem(n_total=10000, n_peeks=20, true_effect=0.02,
                             alpha=0.05, n_simulations=1000):
    """
    Simulate the problem with peeking at AB test results

    Parameters:
    -----------
    n_total : int - total planned sample size per group
    n_peeks : int - number of times we peek at results
    true_effect : float - true effect size (0 means no effect)
    alpha : float - significance level
    n_simulations : int - number of simulations to run

    Returns:
    --------
    dict with simulation results
    """
    baseline = 0.10
    treatment_rate = baseline + true_effect

    peek_points = np.linspace(500, n_total, n_peeks, dtype=int)

    # Track false positive rate at each peek
    stopped_early = 0
    stopped_at_peek = np.zeros(n_peeks)
    final_results = []

    for sim in range(n_simulations):
        for peek_idx, n in enumerate(peek_points):
            # Generate data up to this point
            control = np.random.binomial(1, baseline, n)
            treatment = np.random.binomial(1, treatment_rate, n)

            # Perform test
            conv_a = control.sum()
            conv_b = treatment.sum()

            p_a = conv_a / n
            p_b = conv_b / n
            p_pool = (conv_a + conv_b) / (2 * n)
            se = np.sqrt(p_pool * (1 - p_pool) * (2 / n))
            z = (p_b - p_a) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            # Check if significant
            if p_value < alpha:
                stopped_early += 1
                stopped_at_peek[peek_idx] += 1
                final_results.append({
                    'stopped': True,
                    'peek': peek_idx,
                    'n': n,
                    'p_value': p_value
                })
                break
        else:
            # Didn't stop early, use final result
            final_results.append({
                'stopped': False,
                'peek': n_peeks - 1,
                'n': n_total,
                'p_value': p_value
            })

    false_positive_rate = stopped_early / n_simulations

    return {
        'false_positive_rate': false_positive_rate,
        'stopped_at_peek': stopped_at_peek / n_simulations,
        'peek_points': peek_points,
        'expected_fpr': alpha,
        'inflation_factor': false_positive_rate / alpha
    }

def plot_peeking_results(results):
    """Visualize the peeking problem"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: False positive rate inflation
    axes[0].axhline(y=results['expected_fpr'], color='red', linestyle='--',
                    linewidth=2, label=f'Expected FPR (α={results["expected_fpr"]:.2f})')
    axes[0].axhline(y=results['false_positive_rate'], color='blue', linestyle='-',
                    linewidth=2, label=f'Actual FPR with peeking ({results["false_positive_rate"]:.3f})')
    axes[0].set_ylim(0, max(results['false_positive_rate'], results['expected_fpr']) * 1.2)
    axes[0].set_ylabel('False Positive Rate', fontsize=12)
    axes[0].set_title(f'Type I Error Inflation\n' +
                     f'Inflation Factor: {results["inflation_factor"]:.2f}x', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks([])

    # Plot 2: When tests stopped
    axes[1].bar(range(len(results['stopped_at_peek'])), results['stopped_at_peek'],
                color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Peek Number', fontsize=12)
    axes[1].set_ylabel('Proportion Stopped', fontsize=12)
    axes[1].set_title('Distribution of Early Stops', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('peeking_problem.png', dpi=300, bbox_inches='tight')
    plt.show()

# Simulate with NO true effect (null is true)
print("=== Simulating Peeking Problem ===\n")
print("Testing with NO true effect (H0 is true)...")

results_no_effect = simulate_peeking_problem(
    n_total=10000,
    n_peeks=20,
    true_effect=0.0,  # No effect!
    alpha=0.05,
    n_simulations=1000
)

print(f"Expected false positive rate: {results_no_effect['expected_fpr']:.3f}")
print(f"Actual false positive rate with peeking: {results_no_effect['false_positive_rate']:.3f}")
print(f"Inflation factor: {results_no_effect['inflation_factor']:.2f}x")
print(f"\nConclusion: Peeking at results {20} times inflates Type I error by "
      f"{(results_no_effect['inflation_factor']-1)*100:.1f}%!")

plot_peeking_results(results_no_effect)

# Proper sequential testing with O'Brien-Fleming boundaries
def obrien_fleming_boundary(alpha, n_peeks):
    """Calculate O'Brien-Fleming spending function boundaries"""
    information_fractions = np.linspace(1/n_peeks, 1, n_peeks)

    # O'Brien-Fleming spending function
    boundaries = []
    for frac in information_fractions:
        # Approximate O'Brien-Fleming boundary
        z_boundary = stats.norm.ppf(1 - alpha/2) / np.sqrt(frac)
        alpha_boundary = 2 * (1 - stats.norm.cdf(z_boundary))
        boundaries.append(alpha_boundary)

    return np.array(boundaries)

print("\n=== Proper Sequential Testing ===")
alpha = 0.05
n_peeks = 5
boundaries = obrien_fleming_boundary(alpha, n_peeks)

print(f"\nO'Brien-Fleming boundaries for {n_peeks} looks:")
for i, bound in enumerate(boundaries, 1):
    print(f"  Look {i}: α = {bound:.6f}")
```

---

#### Task 4: Power Analysis

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_power(n, baseline_rate, treatment_rate, alpha=0.05, alternative='two-sided'):
    """
    Calculate statistical power for given parameters

    Parameters:
    -----------
    n : int - sample size per group
    baseline_rate : float - control conversion rate
    treatment_rate : float - treatment conversion rate
    alpha : float - significance level
    alternative : str - 'two-sided' or 'one-sided'

    Returns:
    --------
    float - statistical power
    """
    # Effect size
    effect = treatment_rate - baseline_rate

    # Pooled proportion under alternative
    p_avg = (baseline_rate + treatment_rate) / 2

    # Standard error under null
    se_null = np.sqrt(2 * p_avg * (1 - p_avg) / n)

    # Standard error under alternative
    se_alt = np.sqrt((baseline_rate * (1 - baseline_rate) +
                      treatment_rate * (1 - treatment_rate)) / n)

    # Critical value
    if alternative == 'two-sided':
        z_crit = stats.norm.ppf(1 - alpha/2)
    else:
        z_crit = stats.norm.ppf(1 - alpha)

    # Non-centrality parameter
    ncp = effect / se_alt

    # Power calculation
    if alternative == 'two-sided':
        power = (1 - stats.norm.cdf(z_crit - ncp) +
                 stats.norm.cdf(-z_crit - ncp))
    else:
        power = 1 - stats.norm.cdf(z_crit - ncp)

    return power

def create_power_curves(baseline_rate=0.05, alpha=0.05):
    """Create comprehensive power analysis visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Power curve for different effect sizes
    sample_sizes = np.arange(100, 10000, 100)
    effects = [0.005, 0.01, 0.015, 0.02]  # Absolute effects

    for effect in effects:
        treatment_rate = baseline_rate + effect
        powers = [calculate_power(n, baseline_rate, treatment_rate, alpha)
                 for n in sample_sizes]
        relative_lift = (effect / baseline_rate) * 100
        axes[0, 0].plot(sample_sizes, powers, linewidth=2,
                       label=f'{relative_lift:.0f}% lift ({effect:.3f})')

    axes[0, 0].axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='80% power')
    axes[0, 0].set_xlabel('Sample Size per Group', fontsize=12)
    axes[0, 0].set_ylabel('Statistical Power', fontsize=12)
    axes[0, 0].set_title(f'Power Curves (baseline={baseline_rate:.1%}, α={alpha})', fontsize=13)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # 2. Power vs Effect Size for fixed sample size
    n_fixed = 5000
    relative_lifts = np.linspace(0.02, 0.30, 50)
    powers = []

    for lift in relative_lifts:
        treatment_rate = baseline_rate * (1 + lift)
        power = calculate_power(n_fixed, baseline_rate, treatment_rate, alpha)
        powers.append(power)

    axes[0, 1].plot(relative_lifts * 100, powers, linewidth=2, color='green')
    axes[0, 1].axhline(y=0.80, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Relative Lift (%)', fontsize=12)
    axes[0, 1].set_ylabel('Statistical Power', fontsize=12)
    axes[0, 1].set_title(f'Power vs Effect Size (n={n_fixed:,} per group)', fontsize=13)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # 3. Required sample size for 80% power
    relative_lifts = np.linspace(0.05, 0.50, 50)
    required_n = []

    for lift in relative_lifts:
        treatment_rate = baseline_rate * (1 + lift)
        # Binary search for required n
        n_low, n_high = 100, 100000
        while n_high - n_low > 10:
            n_mid = (n_low + n_high) // 2
            power = calculate_power(n_mid, baseline_rate, treatment_rate, alpha)
            if power < 0.80:
                n_low = n_mid
            else:
                n_high = n_mid
        required_n.append(n_high)

    axes[1, 0].plot(relative_lifts * 100, required_n, linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Relative Lift (%)', fontsize=12)
    axes[1, 0].set_ylabel('Required Sample Size per Group', fontsize=12)
    axes[1, 0].set_title('Sample Size for 80% Power', fontsize=13)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # 4. Power contour plot (sample size vs effect)
    sample_range = np.logspace(2.5, 4.5, 50)  # 316 to 31622
    lift_range = np.linspace(0.05, 0.40, 50)

    power_matrix = np.zeros((len(lift_range), len(sample_range)))
    for i, lift in enumerate(lift_range):
        for j, n in enumerate(sample_range):
            treatment_rate = baseline_rate * (1 + lift)
            power_matrix[i, j] = calculate_power(int(n), baseline_rate,
                                                treatment_rate, alpha)

    contour = axes[1, 1].contourf(sample_range, lift_range * 100, power_matrix,
                                  levels=20, cmap='RdYlGn')
    axes[1, 1].contour(sample_range, lift_range * 100, power_matrix,
                      levels=[0.8], colors='black', linewidths=2)
    axes[1, 1].set_xlabel('Sample Size per Group', fontsize=12)
    axes[1, 1].set_ylabel('Relative Lift (%)', fontsize=12)
    axes[1, 1].set_title('Power Heatmap (black line = 80% power)', fontsize=13)
    axes[1, 1].set_xscale('log')
    plt.colorbar(contour, ax=axes[1, 1], label='Power')

    plt.tight_layout()
    plt.savefig('power_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example calculations
print("=== Power Analysis Examples ===\n")

# Example 1: What's our power with current sample?
n = 5000
baseline = 0.05
treatment = 0.055  # 10% lift
power = calculate_power(n, baseline, treatment)
print(f"With n={n:,} per group, baseline={baseline:.1%}, treatment={treatment:.1%}:")
print(f"  Power = {power:.3f} ({power*100:.1f}%)")

# Example 2: How does doubling sample size help?
power_double = calculate_power(n*2, baseline, treatment)
print(f"\nDoubling to n={n*2:,}:")
print(f"  Power = {power_double:.3f} ({power_double*100:.1f}%)")
print(f"  Improvement: {(power_double - power)*100:.1f} percentage points")

# Example 3: Power for different effect sizes
print(f"\nPower for different lifts (n={n:,}):")
for lift_pct in [5, 10, 15, 20]:
    lift = lift_pct / 100
    treatment_rate = baseline * (1 + lift)
    pwr = calculate_power(n, baseline, treatment_rate)
    print(f"  {lift_pct}% lift: Power = {pwr:.3f}")

# Create comprehensive visualizations
create_power_curves(baseline_rate=0.05, alpha=0.05)
```

---

#### Task 5: AB Test Dashboard

```python
import numpy as np
from scipy import stats
import pandas as pd

def ab_test_dashboard(conversions_a, n_a, conversions_b, n_b,
                     metric_name="Conversion Rate", alpha=0.05):
    """
    Comprehensive AB test analysis dashboard

    Parameters:
    -----------
    conversions_a : int - conversions in control
    n_a : int - sample size in control
    conversions_b : int - conversions in treatment
    n_b : int - sample size in treatment
    metric_name : str - name of the metric being tested
    alpha : float - significance level

    Returns:
    --------
    dict with comprehensive results and prints formatted report
    """
    # Calculate basic statistics
    p_a = conversions_a / n_a
    p_b = conversions_b / n_b
    diff_abs = p_b - p_a
    diff_rel = (diff_abs / p_a) * 100 if p_a > 0 else np.inf

    # Pooled proportion for z-test
    p_pool = (conversions_a + conversions_b) / (n_a + n_b)
    se_pooled = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z_stat = diff_abs / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval for difference
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_lower = diff_abs - 1.96 * se_diff
    ci_upper = diff_abs + 1.96 * se_diff

    # Confidence intervals for individual rates
    ci_a_lower = p_a - 1.96 * np.sqrt(p_a * (1 - p_a) / n_a)
    ci_a_upper = p_a + 1.96 * np.sqrt(p_a * (1 - p_a) / n_a)
    ci_b_lower = p_b - 1.96 * np.sqrt(p_b * (1 - p_b) / n_b)
    ci_b_upper = p_b + 1.96 * np.sqrt(p_b * (1 - p_b) / n_b)

    # Statistical significance
    is_significant = p_value < alpha

    # Effect size (Cohen's h)
    cohens_h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))

    # Practical significance (typically >1-2% relative lift is meaningful)
    is_practically_significant = abs(diff_rel) > 5

    # Power analysis (post-hoc)
    # This estimates what our power was given the observed effect
    effect = p_b - p_a
    se_alt = np.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) /
                     ((n_a + n_b) / 2))
    ncp = effect / se_alt
    z_crit = stats.norm.ppf(1 - alpha/2)
    post_hoc_power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)

    # Determine recommendation
    if is_significant and diff_rel > 0:
        if is_practically_significant:
            recommendation = "✅ IMPLEMENT VARIANT B"
            reason = "Statistically and practically significant improvement"
        else:
            recommendation = "⚠️  CONSIDER IMPLEMENTATION"
            reason = "Statistically significant but small practical effect"
    elif is_significant and diff_rel < 0:
        recommendation = "❌ DO NOT IMPLEMENT"
        reason = "Variant B performs significantly worse"
    else:
        recommendation = "⏸️  NO CLEAR WINNER"
        reason = "Difference is not statistically significant"

    # Print formatted report
    print("=" * 70)
    print(f"{'AB TEST ANALYSIS REPORT':^70}")
    print("=" * 70)

    print(f"\n📊 METRIC: {metric_name}")
    print(f"🎯 SIGNIFICANCE LEVEL: α = {alpha}")

    print(f"\n{'VARIANT A (CONTROL)':-^70}")
    print(f"  Sample Size:       {n_a:>10,}")
    print(f"  Conversions:       {conversions_a:>10,}")
    print(f"  Conversion Rate:   {p_a:>10.4f}  ({p_a*100:.2f}%)")
    print(f"  95% CI:            [{ci_a_lower:.4f}, {ci_a_upper:.4f}]")

    print(f"\n{'VARIANT B (TREATMENT)':-^70}")
    print(f"  Sample Size:       {n_b:>10,}")
    print(f"  Conversions:       {conversions_b:>10,}")
    print(f"  Conversion Rate:   {p_b:>10.4f}  ({p_b*100:.2f}%)")
    print(f"  95% CI:            [{ci_b_lower:.4f}, {ci_b_upper:.4f}]")

    print(f"\n{'DIFFERENCE ANALYSIS':-^70}")
    print(f"  Absolute Difference:  {diff_abs:>10.4f}  ({diff_abs*100:+.2f} ppts)")
    print(f"  Relative Lift:        {diff_rel:>10.2f}%")
    print(f"  95% CI:               [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Relative CI:          [{ci_lower/p_a*100:+.1f}%, {ci_upper/p_a*100:+.1f}%]")

    print(f"\n{'STATISTICAL TEST':-^70}")
    print(f"  Test:              Two-proportion z-test (two-tailed)")
    print(f"  Z-statistic:       {z_stat:>10.4f}")
    print(f"  P-value:           {p_value:>10.6f}")
    print(f"  Significant:       {'✅ YES' if is_significant else '❌ NO'}")

    print(f"\n{'EFFECT SIZE':-^70}")
    print(f"  Cohen's h:         {cohens_h:>10.4f}", end="")
    if abs(cohens_h) < 0.2:
        print("  (Small)")
    elif abs(cohens_h) < 0.5:
        print("  (Medium)")
    else:
        print("  (Large)")

    print(f"\n{'POWER ANALYSIS':-^70}")
    print(f"  Post-hoc Power:    {post_hoc_power:>10.3f}  ({post_hoc_power*100:.1f}%)")
    if post_hoc_power < 0.8:
        print(f"  ⚠️  WARNING: Test may be underpowered (< 80%)")

    print(f"\n{'RECOMMENDATION':-^70}")
    print(f"\n  {recommendation}")
    print(f"  {reason}\n")

    # Business impact estimation
    if is_significant and n_a > 1000:
        print(f"{'PROJECTED IMPACT':-^70}")
        print(f"  Per 10,000 users: {diff_abs * 10000:+.0f} additional conversions")
        print(f"  Per 100,000 users: {diff_abs * 100000:+.0f} additional conversions")
        print(f"  Per 1,000,000 users: {diff_abs * 1000000:+.0f} additional conversions\n")

    print("=" * 70)

    # Return dictionary with all results
    return {
        'control': {
            'n': n_a,
            'conversions': conversions_a,
            'rate': p_a,
            'ci': (ci_a_lower, ci_a_upper)
        },
        'treatment': {
            'n': n_b,
            'conversions': conversions_b,
            'rate': p_b,
            'ci': (ci_b_lower, ci_b_upper)
        },
        'difference': {
            'absolute': diff_abs,
            'relative': diff_rel,
            'ci': (ci_lower, ci_upper)
        },
        'test': {
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant': is_significant,
            'alpha': alpha
        },
        'effect_size': {
            'cohens_h': cohens_h
        },
        'power': post_hoc_power,
        'recommendation': recommendation
    }

# Example usage
print("\n" + "="*70)
print("EXAMPLE 1: Successful Test")
print("="*70)

result1 = ab_test_dashboard(
    conversions_a=728,
    n_a=5200,
    conversions_b=750,
    n_b=5000,
    metric_name="Checkout Completion Rate"
)

print("\n" + "="*70)
print("EXAMPLE 2: Inconclusive Test")
print("="*70)

result2 = ab_test_dashboard(
    conversions_a=120,
    n_a=1000,
    conversions_b=145,
    n_b=1000,
    metric_name="Button Click Rate"
)

print("\n" + "="*70)
print("EXAMPLE 3: Negative Result")
print("="*70)

result3 = ab_test_dashboard(
    conversions_a=1240,
    n_a=10000,
    conversions_b=1100,
    n_b=10000,
    metric_name="Email Open Rate"
)
```

---

### Real-World Case Studies - Solutions

#### Case Study 1: E-commerce Checkout Flow

```python
# Data
n_a, conv_a = 5200, 728
n_b, conv_b = 5000, 750

result = ab_test_dashboard(conv_a, n_a, conv_b, n_b,
                           metric_name="Checkout Completion")

# Additional analysis
p_a = conv_a / n_a
p_b = conv_b / n_b

print("\n📈 BUSINESS IMPACT ANALYSIS")
print("=" * 70)
print(f"Current conversion rate (A): {p_a:.2%}")
print(f"New conversion rate (B): {p_b:.2%}")
print(f"Relative improvement: {(p_b/p_a - 1)*100:+.1f}%")
print(f"\nWith 1M monthly users:")
print(f"  Current conversions: {1000000 * p_a:,.0f}")
print(f"  New conversions: {1000000 * p_b:,.0f}")
print(f"  Additional conversions: {1000000 * (p_b - p_a):+,.0f}")
```

**Answer:**
- Difference is statistically significant (p < 0.05)
- Practical lift of ~7.4%
- Should implement variant B
- Expected ~8,000 additional conversions per month with 1M users

---

#### Case Study 2: Email Campaign

```python
result = ab_test_dashboard(1240, 10000, 1380, 10000,
                           metric_name="Email Open Rate")
```

**Answer:**
- Highly significant (p < 0.001)
- 11.3% relative lift in open rate
- Strong recommendation to use Subject B
- Expected ~280,000 additional opens per month with 2M emails

---

#### Case Study 3: Mobile App Feature

```python
# For continuous metric, use t-test
from scipy import stats

n_a, mean_a, std_a = 1200, 4.2, 2.1
n_b, mean_b, std_b = 1200, 4.6, 2.3

# Pooled standard deviation
s_p = np.sqrt(((n_a-1)*std_a**2 + (n_b-1)*std_b**2) / (n_a + n_b - 2))

# T-test
t_stat, p_value = stats.ttest_ind_from_stats(
    mean_a, std_a, n_a,
    mean_b, std_b, n_b
)

# Cohen's d
cohens_d = (mean_b - mean_a) / s_p

print("Mobile App Feature Test")
print("=" * 70)
print(f"Control: {mean_a:.2f} sessions (SD={std_a:.2f})")
print(f"Treatment: {mean_b:.2f} sessions (SD={std_b:.2f})")
print(f"Difference: {mean_b - mean_a:.2f} sessions ({(mean_b/mean_a-1)*100:+.1f}%)")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Cohen's d: {cohens_d:.4f}")
print(f"Significant: {'YES' if p_value < 0.05 else 'NO'}")
```

**Answer:**
- Statistically significant (p < 0.001)
- Effect size is small (Cohen's d ≈ 0.18)
- 9.5% relative increase in engagement
- Practically significant for a feature with low implementation cost

---

#### Case Study 4: Early Stopping Dilemma

**Answer:**

1. **Should you stop early?**
   - NO - Stopping based on significance while peeking inflates Type I error
   - The p-value of 0.012 seems impressive, but with only 30% of planned data

2. **Risks of stopping:**
   - True alpha may be closer to 0.10-0.15 due to peeking
   - Effect size estimates are unstable with smaller samples
   - Could be observing random variation that will regress to mean

3. **How to adjust:**
   - Use sequential testing methods (O'Brien-Fleming, Pocock boundaries)
   - Apply alpha spending functions
   - Or simply wait for planned sample size

4. **Recommendation:**
   - Continue to full sample size
   - If must stop early, apply appropriate correction
   - Consider implementing sequential testing framework for future tests

---

## Additional Practice Problems

### Problem 6: Novelty Effect
You run a test for 2 weeks:
- Week 1: Treatment is significantly better (p=0.01)
- Week 2: No significant difference (p=0.45)

What happened? How should you analyze this?

### Problem 7: Network Effects
Testing a social feature where user value depends on how many friends use it. Standard AB testing assumptions violated. What do you do?

### Problem 8: Cost-Benefit Analysis
Variant B increases conversion by 2% (significant) but costs $10,000/month to maintain. Each conversion worth $5. Monthly users: 500,000. Worth it?

---

## Summary Checklist

✅ **Before Test:**
- [ ] Calculate required sample size
- [ ] Define success metrics
- [ ] Set alpha and power
- [ ] Determine test duration
- [ ] Plan for multiple testing if needed

✅ **During Test:**
- [ ] Monitor for data quality issues
- [ ] Don't peek (or use proper sequential methods)
- [ ] Check for SRM (Sample Ratio Mismatch)

✅ **After Test:**
- [ ] Calculate p-value and confidence intervals
- [ ] Assess practical significance
- [ ] Check effect size
- [ ] Consider business impact
- [ ] Make clear recommendation

---

## Key Formulas Reference

### Z-Test for Proportions
```
z = (p₂ - p₁) / SE
SE = √[p̂(1-p̂)(1/n₁ + 1/n₂)]
p̂ = (x₁ + x₂) / (n₁ + n₂)
```

### Sample Size
```
n = [Z₁₋α/₂√(2p̄(1-p̄)) + Z₁₋β√(p₁(1-p₁) + p₂(1-p₂))]² / (p₂-p₁)²
```

### Confidence Interval
```
CI = (p₂ - p₁) ± Z₁₋α/₂ × √[p₁(1-p₁)/n₁ + p₂(1-p₂)/n₂]
```

### Effect Sizes
```
Cohen's h = 2(arcsin(√p₂) - arcsin(√p₁))
Cohen's d = (μ₂ - μ₁) / σ_pooled
```

---

**End of Practice Guide**
