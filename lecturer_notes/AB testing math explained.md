# A/B Testing: Mathematical Foundation Explained from Scratch

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Probability Theory](#basic-probability-theory)
3. [Statistical Distributions](#statistical-distributions)
4. [Hypothesis Testing Fundamentals](#hypothesis-testing-fundamentals)
5. [A/B Testing Framework](#ab-testing-framework)
6. [Sample Size Calculation](#sample-size-calculation)
7. [Statistical Power Analysis](#statistical-power-analysis)
8. [Common Test Statistics](#common-test-statistics)
9. [Practical Examples](#practical-examples)

---

## Introduction

A/B testing is a statistical method used to compare two versions (A and B) of something to determine which performs better. This document explains the mathematical foundation from first principles.

---

## Basic Probability Theory

### Random Variables

A **random variable** is a variable whose value is determined by chance. We denote random variables with capital letters (e.g., X, Y).

**Types:**
- **Discrete**: Takes countable values (e.g., number of clicks)
- **Continuous**: Takes any value in a range (e.g., time on page)

### Expected Value (Mean)

The **expected value** E[X] is the average value of a random variable over many trials.

**For discrete variables:**
```
E[X] = Σ x · P(X = x)
```

**For continuous variables:**
```
E[X] = ∫ x · f(x) dx
```

### Variance

**Variance** measures the spread of a distribution:
```
Var(X) = E[(X - μ)²] = E[X²] - (E[X])²
```

**Standard Deviation:**
```
σ = √Var(X)
```

### Bernoulli Distribution

A **Bernoulli trial** has two outcomes: success (1) or failure (0).

If P(success) = p, then:
- E[X] = p
- Var(X) = p(1-p)

**Example:** A click on a button is a Bernoulli trial.

---

## Statistical Distributions

### Binomial Distribution

When we have **n independent Bernoulli trials**, the total number of successes follows a **Binomial distribution** B(n, p).

**Probability mass function:**
```
P(X = k) = C(n,k) · p^k · (1-p)^(n-k)

where C(n,k) = n! / (k!(n-k)!)
```

**Properties:**
- E[X] = np
- Var(X) = np(1-p)

### Normal Distribution

The **Normal (Gaussian) distribution** N(μ, σ²) is characterized by:

**Probability density function:**
```
f(x) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))
```

**Properties:**
- Mean = μ
- Variance = σ²
- Bell-shaped, symmetric around μ
- About 68% of values within μ ± σ
- About 95% of values within μ ± 2σ
- About 99.7% of values within μ ± 3σ

### Central Limit Theorem (CLT)

**The most important theorem in statistics:**

For a large enough sample size n, the distribution of the sample mean approaches a Normal distribution, regardless of the original distribution.

```
X̄ ~ N(μ, σ²/n)   as n → ∞
```

**Why this matters for A/B testing:**
- Even if individual conversions are Bernoulli (0 or 1), the average conversion rate follows a Normal distribution with enough samples
- This allows us to use powerful Normal-based statistical tests

---

## Hypothesis Testing Fundamentals

### The Framework

1. **Null Hypothesis (H₀)**: The default assumption (no difference between A and B)
2. **Alternative Hypothesis (H₁)**: What we want to detect (there is a difference)

### Types of Tests

**Two-tailed test:**
```
H₀: μ_A = μ_B
H₁: μ_A ≠ μ_B
```

**One-tailed test (right-tailed):**
```
H₀: μ_A ≤ μ_B
H₁: μ_A > μ_B
```

### Errors in Hypothesis Testing

| Decision | H₀ True | H₀ False |
|----------|---------|----------|
| Reject H₀ | **Type I Error (α)** | Correct (Power) |
| Fail to reject H₀ | Correct | **Type II Error (β)** |

**Type I Error (False Positive):**
- Probability = α (significance level)
- Rejecting H₀ when it's actually true
- Typical value: α = 0.05 (5%)

**Type II Error (False Negative):**
- Probability = β
- Failing to reject H₀ when it's actually false
- Power = 1 - β (typically want 80% power)

### P-value

The **p-value** is the probability of observing a test statistic at least as extreme as the one calculated, assuming H₀ is true.

**Decision rule:**
- If p-value < α: Reject H₀ (statistically significant)
- If p-value ≥ α: Fail to reject H₀ (not significant)

### Confidence Intervals

A **95% confidence interval** means: if we repeated the experiment many times, 95% of the intervals would contain the true parameter value.

**For a proportion:**
```
CI = p̂ ± z_(α/2) · √(p̂(1-p̂)/n)
```

where z_(α/2) is the critical value (1.96 for 95% confidence).

---

## A/B Testing Framework

### Setup

- **Control Group (A)**: Original version
- **Treatment Group (B)**: New version
- **Metric**: What we're measuring (conversion rate, revenue, etc.)

### Conversion Rate Example

**Group A:**
- n_A users
- X_A conversions
- Conversion rate: p̂_A = X_A / n_A

**Group B:**
- n_B users
- X_B conversions
- Conversion rate: p̂_B = X_B / n_B

### Null Hypothesis

```
H₀: p_A = p_B  (no difference in conversion rates)
H₁: p_A ≠ p_B  (there is a difference)
```

---

## Sample Size Calculation

### Why Sample Size Matters

Sample size determines:
1. **Power**: Ability to detect a true effect
2. **Precision**: Width of confidence intervals
3. **Cost**: More samples = more time/money

### Formula for Comparing Two Proportions

To detect a minimum effect size δ with significance α and power (1-β):

```
n = (z_(α/2) + z_β)² · [p_A(1-p_A) + p_B(1-p_B)] / δ²

where:
- δ = |p_B - p_A| (minimum detectable effect)
- p_A = baseline conversion rate
- p_B = p_A + δ
- z_(α/2) = critical value for significance (1.96 for α=0.05, two-tailed)
- z_β = critical value for power (0.84 for 80% power)
```

### Simplified Formula (Equal Sample Sizes)

When p_A ≈ p_B ≈ p (pooled proportion):

```
n ≈ 2 · (z_(α/2) + z_β)² · p(1-p) / δ²
```

### Example Calculation

**Given:**
- Baseline conversion rate: p_A = 0.10 (10%)
- Minimum detectable effect: δ = 0.02 (2 percentage points)
- Significance level: α = 0.05
- Desired power: 1 - β = 0.80

**Calculate:**
```
p_B = 0.10 + 0.02 = 0.12
z_(0.025) = 1.96
z_(0.20) = 0.84

n = (1.96 + 0.84)² · [0.10(0.90) + 0.12(0.88)] / (0.02)²
n = (2.80)² · [0.09 + 0.1056] / 0.0004
n = 7.84 · 0.1956 / 0.0004
n ≈ 3,834 per group
```

**Total sample size needed: ~7,668 users**

---

## Statistical Power Analysis

### What is Power?

**Statistical power** = P(reject H₀ | H₁ is true)

It's the probability of detecting an effect when it actually exists.

```
Power = 1 - β
```

### Factors Affecting Power

1. **Effect size (δ)**: Larger effects are easier to detect
2. **Sample size (n)**: More data = more power
3. **Significance level (α)**: Lower α = lower power
4. **Variance (σ²)**: Less noise = more power

### Power Relationship

```
Power ↑ when:
- Effect size ↑
- Sample size ↑
- Significance level ↑ (but more false positives)
- Variance ↓
```

### Minimum Detectable Effect (MDE)

The **smallest effect** we can reliably detect given:
- Sample size n
- Significance α
- Power (1-β)

```
MDE = (z_(α/2) + z_β) · √[p̂(1-p̂) · (1/n_A + 1/n_B)]
```

---

## Common Test Statistics

### Z-Test for Proportions

**When to use:** Large samples (np > 5 and n(1-p) > 5)

**Test statistic:**
```
z = (p̂_B - p̂_A) / SE

where SE = √[p̂_pooled · (1 - p̂_pooled) · (1/n_A + 1/n_B)]

p̂_pooled = (X_A + X_B) / (n_A + n_B)
```

**Under H₀**, z follows a standard Normal distribution N(0,1).

**Decision:**
- Two-tailed: Reject H₀ if |z| > z_(α/2) (typically 1.96)
- One-tailed: Reject H₀ if z > z_α (typically 1.645)

### T-Test for Means

**When to use:** Comparing means of continuous metrics

**Test statistic:**
```
t = (X̄_B - X̄_A) / SE

where SE = √(s²_A/n_A + s²_B/n_B)
```

**Degrees of freedom** (Welch's approximation):
```
df = (s²_A/n_A + s²_B/n_B)² / [(s²_A/n_A)²/(n_A-1) + (s²_B/n_B)²/(n_B-1)]
```

### Chi-Square Test

**When to use:** Testing independence in categorical data

**Test statistic:**
```
χ² = Σ (Observed - Expected)² / Expected
```

---

## Practical Examples

### Example 1: Button Color Test

**Scenario:** Testing if a red button converts better than a blue button.

**Data:**
- Blue button (A): 1,000 visitors, 100 conversions → p̂_A = 0.10
- Red button (B): 1,000 visitors, 130 conversions → p̂_B = 0.13

**Step 1: Set up hypotheses**
```
H₀: p_A = p_B
H₁: p_A ≠ p_B
α = 0.05
```

**Step 2: Calculate pooled proportion**
```
p̂_pooled = (100 + 130) / (1000 + 1000) = 230/2000 = 0.115
```

**Step 3: Calculate standard error**
```
SE = √[0.115 · 0.885 · (1/1000 + 1/1000)]
SE = √[0.101775 · 0.002]
SE = √0.00020355
SE ≈ 0.01427
```

**Step 4: Calculate z-statistic**
```
z = (0.13 - 0.10) / 0.01427
z = 0.03 / 0.01427
z ≈ 2.10
```

**Step 5: Find p-value**
For two-tailed test with z = 2.10:
```
p-value = 2 · P(Z > 2.10) ≈ 2 · 0.0179 ≈ 0.036
```

**Step 6: Conclusion**
Since p-value (0.036) < α (0.05), we **reject H₀**.

**Result:** The red button has a statistically significant higher conversion rate.

**Confidence interval for difference:**
```
(p̂_B - p̂_A) ± z_(α/2) · SE
0.03 ± 1.96 · 0.01427
0.03 ± 0.028
[0.002, 0.058] or [0.2%, 5.8%]
```

---

### Example 2: Sample Size Planning

**Scenario:** Planning a test for a checkout page redesign.

**Given:**
- Current conversion rate: 5%
- Want to detect a 1 percentage point increase (to 6%)
- Significance: α = 0.05
- Power: 80%

**Calculate:**
```
p_A = 0.05, p_B = 0.06, δ = 0.01
z_(0.025) = 1.96, z_(0.20) = 0.84

n = (1.96 + 0.84)² · [0.05(0.95) + 0.06(0.94)] / (0.01)²
n = 7.84 · [0.0475 + 0.0564] / 0.0001
n = 7.84 · 0.1039 / 0.0001
n ≈ 8,146 per group
```

**Total needed: 16,292 visitors**

If you get 1,000 visitors/day:
- Time needed ≈ 16.3 days

---

### Example 3: Multiple Metrics

**Scenario:** Testing impact on both conversion rate and revenue.

**Problem:** Multiple comparisons increase false positive risk.

**Bonferroni Correction:**
If testing k hypotheses, use α/k for each test.

For 2 metrics with α = 0.05:
```
Use α_adjusted = 0.05/2 = 0.025 for each test
```

**Alternative:** Use a primary metric (conversion) and secondary metrics (revenue) for context only.

---

## Key Formulas Summary

| Concept | Formula |
|---------|---------|
| **Sample mean** | X̄ = (Σ x_i) / n |
| **Sample variance** | s² = Σ(x_i - X̄)² / (n-1) |
| **Standard error (proportion)** | SE = √[p̂(1-p̂)/n] |
| **Standard error (mean)** | SE = s/√n |
| **Z-statistic (proportions)** | z = (p̂_B - p̂_A) / SE_pooled |
| **T-statistic (means)** | t = (X̄_B - X̄_A) / SE |
| **Confidence interval** | estimate ± critical_value · SE |
| **Sample size (proportions)** | n = (z_(α/2) + z_β)² · [p_A(1-p_A) + p_B(1-p_B)] / δ² |
| **Power** | 1 - β = P(reject H₀ \| H₁ true) |
| **Effect size (Cohen's h)** | h = 2·arcsin(√p_B) - 2·arcsin(√p_A) |

---

## Critical Values Reference

**For α = 0.05:**

| Test Type | Critical Value |
|-----------|----------------|
| Two-tailed z-test | ±1.96 |
| One-tailed z-test | 1.645 |
| Two-tailed (α=0.01) | ±2.576 |

**Statistical Power (z_β):**

| Power | z_β |
|-------|-----|
| 80% | 0.842 |
| 90% | 1.282 |
| 95% | 1.645 |

---

## Common Pitfalls

### 1. Peeking Problem
**Issue:** Checking results before reaching target sample size increases false positives.

**Solution:** Use sequential testing methods or commit to fixed sample size.

### 2. Multiple Testing
**Issue:** Testing multiple variations/metrics inflates Type I error.

**Solution:** Bonferroni correction or control False Discovery Rate (FDR).

### 3. Sample Ratio Mismatch
**Issue:** Unequal split ratios when expecting 50/50.

**Solution:** Check assignment mechanism, may indicate bugs.

### 4. Novelty Effects
**Issue:** Users react differently to changes initially.

**Solution:** Run test longer to account for adaptation.

### 5. Selection Bias
**Issue:** Non-random assignment to groups.

**Solution:** Ensure proper randomization mechanism.

---

## Advanced Topics

### Sequential Testing

**SPRT (Sequential Probability Ratio Test):**
Allows continuous monitoring with controlled error rates.

### Bayesian A/B Testing

Uses prior distributions and calculates probability that B beats A:
```
P(p_B > p_A | data)
```

### Multi-Armed Bandits

Balances exploration (testing) with exploitation (using best variant).

### Stratified Sampling

Ensure proportional representation across segments.

---

## Mathematical Intuition: Why A/B Testing Works

This section explains the deep mathematical reasoning behind A/B testing and builds intuition for why these methods are reliable.

### 1. The Foundation: Law of Large Numbers

**The Core Idea:**
If you flip a fair coin 10 times, you might get 7 heads. But if you flip it 10,000 times, you'll get very close to 50% heads.

**Mathematical Statement:**
As sample size n → ∞, the sample mean X̄ converges to the true population mean μ:

```
X̄_n → μ   as n → ∞
```

**Why This Matters for A/B Testing:**
- With few users, we might observe 20% conversion by pure chance even if true rate is 10%
- With many users, our observed rate gets closer to the true rate
- This is why we need sufficient sample size - small samples are unreliable

**Intuitive Example:**
Imagine version B truly has 15% conversion rate:
- With 10 users: We might observe 0%, 10%, 20%, or 30% (high variance)
- With 1,000 users: We'll likely observe 13%-17% (closer to truth)
- With 10,000 users: We'll almost certainly observe 14.5%-15.5% (very close)

---

### 2. Central Limit Theorem: The Magic of Normality

**The Core Idea:**
No matter what your data looks like originally, averages of large samples always follow a bell curve (Normal distribution).

**Why This is Profound:**
Individual conversions are binary (0 or 1) - very non-Normal! But the *average* conversion rate becomes Normal with enough data.

**Visual Intuition:**

```
Individual users:        |  |     |        |  |  (just 0s and 1s)
Average of 10 users:     | || ||| || |     (starts to smooth)
Average of 100 users:    |||||||||||||     (becoming bell-shaped)
Average of 1000 users:      .-""--.       (nearly perfect bell curve)
                          .'        '.
                         /            \
                        |              |
```

**Mathematical Beauty:**
```
For binary outcomes (Bernoulli):
- Each individual: X_i ∈ {0, 1}
- Sample mean: X̄ = (X₁ + X₂ + ... + Xₙ)/n
- Distribution: X̄ ~ N(p, p(1-p)/n)  for large n
```

**Why This Enables Testing:**
Because averages are Normal, we can:
1. Calculate exact probabilities using z-scores
2. Construct confidence intervals with known coverage
3. Compare two averages using well-understood distributions

---

### 3. Variance Reduction Through Averaging

**The Core Idea:**
Individual measurements are noisy, but averages are much more precise.

**Mathematical Relationship:**
```
Var(individual) = σ²
Var(average of n) = σ²/n
```

Notice: Variance decreases by factor of n!

**Standard Error:**
```
SE = σ/√n
```

The standard error shrinks with square root of sample size.

**Practical Implications:**

| Sample Size | SE relative to individual σ | Precision Gain |
|-------------|----------------------------|----------------|
| n = 1 | σ | Baseline |
| n = 100 | σ/10 | 10x more precise |
| n = 10,000 | σ/100 | 100x more precise |

**Why This Matters:**
To double precision, you need 4x sample size:
```
SE_new = σ/√(4n) = (σ/√n)/2 = SE_old/2
```

**Intuitive Understanding:**
Think of measurement errors canceling out:
- Some users convert by luck → balanced by users who don't convert by bad luck
- More data → more cancellation → more precision
- The √n relationship is the "cost" of this cancellation

---

### 4. Why We Use Standard Error (Not Standard Deviation)

This confuses many people. Let's clarify:

**Standard Deviation (σ):**
- Measures spread of *individual* observations
- "How much do individual users vary?"
- For binary: σ = √(p(1-p))
- Example: With p=0.1, σ ≈ 0.3 (30 percentage points!)

**Standard Error (SE):**
- Measures uncertainty in the *average*
- "How precisely do we know the true mean?"
- SE = σ/√n
- Example: With p=0.1 and n=1000, SE ≈ 0.009 (0.9 percentage points)

**Key Insight:**
We care about precision of our *estimate* (the mean), not variability of individuals.

**Analogy:**
- σ: "People's heights vary by 6 inches" (individual variation)
- SE: "The average height is 5'9" ± 0.1 inches" (precision of estimate)

---

### 5. The Logic of Hypothesis Testing

**The Philosophical Question:**
We observe a difference. Is it real or just random chance?

**The Mathematical Approach:**

1. **Assume no real difference** (null hypothesis H₀)
2. **Calculate**: "If there's no real difference, how likely is this big a difference by chance?"
3. **Decide**: If very unlikely (p < 0.05), reject the assumption

**The Proof by Contradiction:**
```
1. Assume: H₀ is true (no difference)
2. Observe: Large difference in data
3. Calculate: P(observing this | H₀) = 0.01 (very small)
4. Conclude: Our assumption was probably wrong
5. Therefore: There is likely a real difference
```

**Why 5% (α = 0.05)?**
This is a convention balancing two risks:
- Too high (e.g., 50%): We'll claim differences that don't exist
- Too low (e.g., 0.1%): We'll miss real differences

5% means: "I'm willing to be wrong 1 in 20 times to detect real effects"

---

### 6. P-values: The Most Misunderstood Concept

**What p-value IS:**
The probability of seeing a result this extreme *if there's truly no difference*.

**What p-value is NOT:**
- NOT the probability that H₀ is true
- NOT the probability you made a mistake
- NOT the size of the effect

**Correct Interpretation:**
"If version B were truly no better than A, we'd see a difference this large or larger only 3% of the time by random chance."

**Wrong Interpretation:**
"There's a 3% chance B is the same as A." ❌

**Intuitive Example:**

Imagine a fair coin (truly 50/50):
- Flip 10 times, get 8 heads
- P-value ≈ 0.11 (not unusual, could happen 11% of time)
- Conclusion: Don't reject fairness

- Flip 100 times, get 80 heads
- P-value < 0.001 (extremely unlikely if fair)
- Conclusion: Probably not a fair coin

The p-value quantifies "how surprised should we be?"

---

### 7. Confidence Intervals: A Better Way to Think

**What It Really Means:**

95% CI = [0.12, 0.18] means:

"If we repeated this experiment 100 times, about 95 of the intervals we calculate would contain the true conversion rate."

**Why This is More Intuitive:**

Instead of: "p = 0.03" (confusing!)

We get: "Conversion rate is between 12% and 18%" (actionable!)

**The Mathematical Construction:**

```
CI = estimate ± (critical value) × (standard error)
CI = p̂ ± 1.96 × SE
```

**Why 1.96?**
For Normal distribution:
- 95% of values fall within 1.96 standard deviations of the mean
- This comes from the area under the Normal curve

**Visual Intuition:**
```
                    95% of probability
         |<----------------------------->|
              .---------.---------.
            .'           |           '.
           /             |             \
          /              |              \
    -----|--------------[###############]|-----
         |              ↑               ↑      |
      -3σ          -1.96σ           +1.96σ   +3σ

The shaded area [###] represents 95% of all values
```

---

### 8. Why Two Variances are Better Than One

**The Pooled Variance Question:**

When comparing A and B, we have two sample variances. Should we combine them?

**Yes, under H₀:**
```
p̂_pooled = (X_A + X_B) / (n_A + n_B)
```

**Why?**
Under null hypothesis, both groups have the *same* true rate. Pooling gives us:
- More data to estimate this common variance
- More statistical power
- Better standard error estimate

**The Math:**
```
SE_pooled = √[p̂_pooled(1-p̂_pooled) · (1/n_A + 1/n_B)]
```

This is smaller (more precise) than using separate variances.

**Intuition:**
If A and B truly have the same conversion rate p, why estimate it twice (less precisely) when we could estimate it once with all the data (more precisely)?

---

### 9. Sample Size: The Square Root Barrier

**The Frustrating Reality:**

To detect a difference half as small, you need 4x the data!

**Why This Happens:**

Recall: SE = σ/√n

To make SE half as big:
```
σ/√n_new = (σ/√n_old)/2
√n_new = 2√n_old
n_new = 4n_old
```

**Practical Implications:**

| Want to detect | Sample size multiplier |
|----------------|----------------------|
| Half the effect | 4x more data |
| 1/3 the effect | 9x more data |
| 1/4 the effect | 16x more data |

**The Economic Trade-off:**

This is why we ask:
- "What's the minimum improvement worth detecting?" (MDE)
- Not: "Can we detect any improvement at all?" (infinite data needed)

**Intuitive Understanding:**

Small signals are buried in noise. To see them clearly:
- Either make the signal bigger (increase effect size)
- Or reduce the noise (collect more data)

Noise reduction follows √n law - the "diminishing returns" of statistics.

---

### 10. Statistical Power: The Flip Side of the Coin

**The Two Types of Errors:**

Remember the confusion matrix:

|  | H₀ True | H₀ False |
|--|---------|----------|
| Reject H₀ | Type I (α) | ✓ Power |
| Keep H₀ | ✓ | Type II (β) |

**Why Power Matters:**

Power = P(detecting effect when it exists)

**The Trade-offs:**

Imagine a criminal trial:
- Type I error: Convict innocent person (α)
- Type II error: Free guilty person (β)
- Power: Convict guilty person (1-β)

In A/B testing:
- Type I: Launch bad change (false positive)
- Type II: Miss good change (false negative)
- Power: Launch good change correctly

**Why 80% Power?**

This is conventional, meaning:
- If there's a real effect, we'll detect it 80% of the time
- We'll miss it 20% of the time (β = 0.2)

**The Power Equation:**

```
Power increases with:
↑ Larger effect size (easier to see)
↑ Larger sample size (less noise)
↑ Higher α (but more false positives!)
↓ Lower variance (cleaner signal)
```

**Intuition:**

Power is like the strength of your microscope:
- High power: See small bacteria (small effects)
- Low power: Only see large parasites (large effects)

More data = better microscope!

---

### 11. Why Randomization is Sacred

**The Fundamental Problem:**

People differ. How do we know differences are from our treatment, not from different people?

**The Elegant Solution:**

Random assignment makes groups *probabilistically identical* before treatment.

**Mathematical Guarantee:**

With random assignment:
```
E[X_A - X_B | before treatment] = 0
```

Any difference observed must come from:
1. The treatment effect (what we want)
2. Random chance (what we account for with statistics)

**What Randomization Prevents:**

Without randomization:
- Maybe version B shown to returning users (higher conversion)
- Maybe version B shown during weekends (different behavior)
- Maybe version B shown to mobile users (lower conversion)

**The Beautiful Consequence:**

Because groups are identical on average:
- All confounders are balanced
- All unknown factors are balanced
- Even things we didn't think to measure are balanced!

This is why randomization is the "gold standard" - it handles all biases simultaneously.

---

### 12. The Complete Picture: Putting It All Together

**The Chain of Reasoning:**

1. **Law of Large Numbers** → Sample means approach true means
2. **Central Limit Theorem** → Sample means are Normally distributed
3. **Normal Distribution** → We know exact probabilities
4. **Randomization** → Groups start identical
5. **Difference in means** → Measures treatment effect
6. **Standard error** → Quantifies uncertainty
7. **Hypothesis test** → Evaluates evidence
8. **P-value** → Measures surprise if no effect
9. **Confidence interval** → Bounds the true effect
10. **Power analysis** → Ensures adequate sample size

**The Remarkable Synthesis:**

A/B testing combines:
- **Probability theory** (randomness is predictable in aggregate)
- **Sampling theory** (small samples reveal population truth)
- **Decision theory** (balance different types of errors)
- **Experimental design** (isolate causal effects)

**Why It Works:**

The mathematics doesn't eliminate uncertainty - it *quantifies* it.

We go from:
- "Version B seems better" (subjective impression)

To:
- "Version B increases conversion by 2-5 percentage points with 95% confidence, p = 0.03" (rigorous conclusion)

**The Philosophical Foundation:**

We can never be 100% certain (unless we test everyone forever). But we can:
- Quantify our uncertainty
- Control our error rates
- Make optimal decisions given the data

This is the power of statistical thinking: turning uncertainty into actionable knowledge.

---

## Conclusion

A/B testing relies on:
1. **Probability theory** - understanding randomness
2. **Statistical distributions** - modeling data
3. **Hypothesis testing** - making decisions under uncertainty
4. **Sample size calculations** - planning adequate experiments
5. **Careful interpretation** - avoiding common pitfalls

The mathematics ensures we make data-driven decisions with quantified confidence levels, minimizing the risk of false conclusions.

---

## Practical Implementation: Working with Real Data

This section demonstrates how to conduct A/B tests in practice using Python, R, and manual calculations.

### Example Dataset: E-commerce Button Test

**Scenario:** Testing two checkout button colors on an e-commerce website.

**Data:**
```
Group A (Blue Button):
- Users: 2,500
- Conversions: 312
- Conversion Rate: 12.48%

Group B (Red Button):
- Users: 2,480
- Conversions: 348
- Conversion Rate: 14.03%
```

---

### Method 1: Manual Calculation (Step-by-Step)

**Step 1: Calculate sample statistics**

```
n_A = 2500
X_A = 312
p̂_A = 312/2500 = 0.1248

n_B = 2480
X_B = 348
p̂_B = 348/2480 = 0.1403

Observed difference: p̂_B - p̂_A = 0.1403 - 0.1248 = 0.0155 (1.55%)
```

**Step 2: Calculate pooled proportion**

```
p̂_pooled = (X_A + X_B) / (n_A + n_B)
p̂_pooled = (312 + 348) / (2500 + 2480)
p̂_pooled = 660 / 4980
p̂_pooled = 0.1325
```

**Step 3: Calculate standard error**

```
SE = √[p̂_pooled · (1 - p̂_pooled) · (1/n_A + 1/n_B)]
SE = √[0.1325 · 0.8675 · (1/2500 + 1/2480)]
SE = √[0.1149 · 0.0008048]
SE = √0.00009247
SE = 0.00962
```

**Step 4: Calculate z-statistic**

```
z = (p̂_B - p̂_A) / SE
z = 0.0155 / 0.00962
z = 1.611
```

**Step 5: Calculate p-value**

For two-tailed test:
```
P(Z > 1.611) ≈ 0.0536
p-value = 2 × 0.0536 = 0.1072
```

**Step 6: Conclusion**

```
p-value (0.1072) > α (0.05)
→ Fail to reject H₀
→ The difference is NOT statistically significant at 5% level
```

**Step 7: Confidence interval**

```
95% CI = (p̂_B - p̂_A) ± 1.96 · SE
95% CI = 0.0155 ± 1.96 · 0.00962
95% CI = 0.0155 ± 0.0188
95% CI = [-0.0033, 0.0343]
```

**Interpretation:**
- The red button shows 1.55% higher conversion
- This could be due to chance (p = 0.11)
- True difference likely between -0.33% and +3.43%
- Need more data for conclusive result

---

### Method 2: Python Implementation

#### Using scipy.stats

```python
import numpy as np
from scipy import stats

# Data
n_A = 2500
X_A = 312
n_B = 2480
X_B = 348

# Calculate proportions
p_A = X_A / n_A
p_B = X_B / n_B

print(f"Conversion Rate A: {p_A:.4f} ({p_A*100:.2f}%)")
print(f"Conversion Rate B: {p_B:.4f} ({p_B*100:.2f}%)")
print(f"Absolute Difference: {(p_B - p_A):.4f} ({(p_B - p_A)*100:.2f}%)")
print(f"Relative Lift: {((p_B - p_A) / p_A * 100):.2f}%\n")

# Pooled proportion
p_pooled = (X_A + X_B) / (n_A + n_B)

# Standard error
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_A + 1/n_B))

# Z-statistic
z_stat = (p_B - p_A) / se

# P-value (two-tailed)
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"Pooled proportion: {p_pooled:.4f}")
print(f"Standard Error: {se:.6f}")
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}\n")

# Confidence interval
ci_margin = 1.96 * se
ci_lower = (p_B - p_A) - ci_margin
ci_upper = (p_B - p_A) + ci_margin

print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"95% CI (percentage points): [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]\n")

# Decision
alpha = 0.05
if p_value < alpha:
    print(f"✓ SIGNIFICANT: Reject H₀ (p = {p_value:.4f} < {alpha})")
    print("The difference is statistically significant.")
else:
    print(f"✗ NOT SIGNIFICANT: Fail to reject H₀ (p = {p_value:.4f} >= {alpha})")
    print("The difference is not statistically significant.")
```

**Output:**
```
Conversion Rate A: 0.1248 (12.48%)
Conversion Rate B: 0.1403 (14.03%)
Absolute Difference: 0.0155 (1.55%)
Relative Lift: 12.42%

Pooled proportion: 0.1325
Standard Error: 0.009617
Z-statistic: 1.6112
P-value: 0.1072

95% Confidence Interval: [-0.0033, 0.0343]
95% CI (percentage points): [-0.33%, 3.43%]

✗ NOT SIGNIFICANT: Fail to reject H₀ (p = 0.1072 >= 0.05)
The difference is not statistically significant.
```

#### Using statsmodels (more robust)

```python
from statsmodels.stats.proportion import proportions_ztest

# Data
counts = np.array([X_B, X_A])  # Note: B first for positive z
nobs = np.array([n_B, n_A])

# Two-tailed test
z_stat, p_value = proportions_ztest(counts, nobs, alternative='two-sided')

print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically significant")
else:
    print("Result: Not statistically significant")
```

#### Complete Analysis Function

```python
def ab_test_analysis(n_A, X_A, n_B, X_B, alpha=0.05):
    """
    Complete A/B test analysis

    Parameters:
    -----------
    n_A, n_B : int
        Sample sizes for groups A and B
    X_A, X_B : int
        Number of conversions in groups A and B
    alpha : float
        Significance level (default 0.05)

    Returns:
    --------
    dict : Results dictionary with all metrics
    """
    # Proportions
    p_A = X_A / n_A
    p_B = X_B / n_B
    diff = p_B - p_A

    # Pooled proportion and SE
    p_pooled = (X_A + X_B) / (n_A + n_B)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_A + 1/n_B))

    # Test statistic and p-value
    z_stat = diff / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval
    z_critical = stats.norm.ppf(1 - alpha/2)
    ci_margin = z_critical * se
    ci = (diff - ci_margin, diff + ci_margin)

    # Relative lift
    relative_lift = (diff / p_A) * 100 if p_A > 0 else np.inf

    # Statistical power (post-hoc)
    effect_size = diff
    se_separate = np.sqrt(p_A*(1-p_A)/n_A + p_B*(1-p_B)/n_B)
    noncentrality = effect_size / se_separate
    power = 1 - stats.norm.cdf(z_critical - noncentrality)

    results = {
        'p_A': p_A,
        'p_B': p_B,
        'difference': diff,
        'relative_lift': relative_lift,
        'z_statistic': z_stat,
        'p_value': p_value,
        'ci_95': ci,
        'significant': p_value < alpha,
        'power': power,
        'sample_size_A': n_A,
        'sample_size_B': n_B
    }

    return results

# Run analysis
results = ab_test_analysis(n_A, X_A, n_B, X_B)

# Pretty print results
print("="*60)
print("A/B TEST RESULTS")
print("="*60)
print(f"Group A: {results['p_A']:.2%} conversion ({X_A}/{n_A})")
print(f"Group B: {results['p_B']:.2%} conversion ({X_B}/{n_B})")
print(f"\nAbsolute Difference: {results['difference']:.2%}")
print(f"Relative Lift: {results['relative_lift']:.2f}%")
print(f"\nZ-statistic: {results['z_statistic']:.4f}")
print(f"P-value: {results['p_value']:.4f}")
print(f"95% CI: [{results['ci_95'][0]:.2%}, {results['ci_95'][1]:.2%}]")
print(f"\nStatistical Power: {results['power']:.2%}")
print(f"\nSignificant at α=0.05? {results['significant']}")
print("="*60)
```

---

### Method 3: R Implementation

```r
# Data
n_A <- 2500
X_A <- 312
n_B <- 2480
X_B <- 348

# Proportions
p_A <- X_A / n_A
p_B <- X_B / n_B

cat("Conversion Rate A:", sprintf("%.2f%%", p_A * 100), "\n")
cat("Conversion Rate B:", sprintf("%.2f%%", p_B * 100), "\n")
cat("Difference:", sprintf("%.2f%%", (p_B - p_A) * 100), "\n\n")

# Perform proportions test
prop_test <- prop.test(
  x = c(X_A, X_B),
  n = c(n_A, n_B),
  alternative = "two.sided",
  conf.level = 0.95,
  correct = FALSE  # Don't use continuity correction for consistency
)

print(prop_test)

# Extract results
cat("\n--- SUMMARY ---\n")
cat("Chi-squared statistic:", prop_test$statistic, "\n")
cat("P-value:", prop_test$p.value, "\n")
cat("95% CI:", prop_test$conf.int[1], "to", prop_test$conf.int[2], "\n")

if (prop_test$p.value < 0.05) {
  cat("Result: Statistically significant at α = 0.05\n")
} else {
  cat("Result: Not statistically significant at α = 0.05\n")
}
```

**Alternative using z-test manually in R:**

```r
# Manual z-test in R
ab_test_z <- function(n_A, X_A, n_B, X_B, alpha = 0.05) {
  # Proportions
  p_A <- X_A / n_A
  p_B <- X_B / n_B
  diff <- p_B - p_A

  # Pooled proportion
  p_pooled <- (X_A + X_B) / (n_A + n_B)

  # Standard error
  se <- sqrt(p_pooled * (1 - p_pooled) * (1/n_A + 1/n_B))

  # Z-statistic
  z_stat <- diff / se

  # P-value (two-tailed)
  p_value <- 2 * (1 - pnorm(abs(z_stat)))

  # Confidence interval
  z_crit <- qnorm(1 - alpha/2)
  ci_lower <- diff - z_crit * se
  ci_upper <- diff + z_crit * se

  # Return results
  list(
    p_A = p_A,
    p_B = p_B,
    difference = diff,
    relative_lift = (diff / p_A) * 100,
    z_statistic = z_stat,
    p_value = p_value,
    ci_95 = c(ci_lower, ci_upper),
    significant = p_value < alpha
  )
}

# Run test
results <- ab_test_z(n_A, X_A, n_B, X_B)

# Print results
cat("\n=== A/B TEST RESULTS ===\n")
cat(sprintf("Group A: %.2f%% (%d/%d)\n", results$p_A * 100, X_A, n_A))
cat(sprintf("Group B: %.2f%% (%d/%d)\n", results$p_B * 100, X_B, n_B))
cat(sprintf("\nAbsolute Difference: %.2f%%\n", results$difference * 100))
cat(sprintf("Relative Lift: %.2f%%\n", results$relative_lift))
cat(sprintf("\nZ-statistic: %.4f\n", results$z_statistic))
cat(sprintf("P-value: %.4f\n", results$p_value))
cat(sprintf("95%% CI: [%.2f%%, %.2f%%]\n",
            results$ci_95[1] * 100, results$ci_95[2] * 100))
cat(sprintf("\nSignificant? %s\n", ifelse(results$significant, "YES", "NO")))
```

---

### Method 4: Sample Size Planning (Before Running Test)

**Scenario:** Planning a test before collecting data.

```python
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportion_effectsize

# Parameters
p_A = 0.12  # Baseline conversion rate
p_B = 0.14  # Desired conversion rate (target)
alpha = 0.05  # Significance level
power = 0.80  # Desired power

# Effect size (Cohen's h)
effect_size = proportion_effectsize(p_A, p_B)

print(f"Baseline conversion: {p_A:.1%}")
print(f"Target conversion: {p_B:.1%}")
print(f"Absolute difference: {(p_B - p_A):.1%}")
print(f"Relative lift: {((p_B - p_A) / p_A * 100):.1f}%")
print(f"\nEffect size (Cohen's h): {effect_size:.4f}\n")

# Calculate required sample size per group
n_per_group = zt_ind_solve_power(
    effect_size=effect_size,
    alpha=alpha,
    power=power,
    ratio=1.0,  # Equal sample sizes
    alternative='two-sided'
)

print(f"Required sample size per group: {int(np.ceil(n_per_group))}")
print(f"Total sample size needed: {int(np.ceil(n_per_group * 2))}\n")

# Time estimation
visitors_per_day = 500
days_needed = np.ceil(n_per_group * 2 / visitors_per_day)
print(f"If you get {visitors_per_day} visitors/day:")
print(f"Test duration: {int(days_needed)} days")
```

**Output:**
```
Baseline conversion: 12.0%
Target conversion: 14.0%
Absolute difference: 2.0%
Relative lift: 16.7%

Effect size (Cohen's h): 0.0581

Required sample size per group: 4674
Total sample size needed: 9348

If you get 500 visitors/day:
Test duration: 19 days
```

---

### Method 5: Sequential Testing (Monitoring During Test)

**Important:** Checking results repeatedly increases false positive rate. Here's a safer approach:

```python
def sequential_analysis(results_so_far, max_samples, alpha=0.05):
    """
    Sequential analysis with alpha spending function
    Uses O'Brien-Fleming boundaries
    """
    n_A_current, X_A_current = results_so_far['A']
    n_B_current, X_B_current = results_so_far['B']

    # Information fraction
    info_fraction = (n_A_current + n_B_current) / (2 * max_samples)

    # O'Brien-Fleming boundary adjustment
    z_boundary = stats.norm.ppf(1 - alpha/2) / np.sqrt(info_fraction)

    # Current z-statistic
    p_A = X_A_current / n_A_current
    p_B = X_B_current / n_B_current
    p_pooled = (X_A_current + X_B_current) / (n_A_current + n_B_current)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_A_current + 1/n_B_current))
    z_current = (p_B - p_A) / se

    # Decision
    decision = "CONTINUE"
    if abs(z_current) >= z_boundary:
        decision = "STOP - Significant"
    elif info_fraction >= 1.0:
        decision = "STOP - Max sample reached"

    return {
        'info_fraction': info_fraction,
        'z_boundary': z_boundary,
        'z_current': z_current,
        'decision': decision,
        'samples_used': n_A_current + n_B_current,
        'samples_remaining': 2 * max_samples - (n_A_current + n_B_current)
    }

# Example: Check at 50% completion
max_n = 5000  # Max sample size per group
current_results = {
    'A': (2500, 312),  # (n, conversions)
    'B': (2480, 348)
}

seq_result = sequential_analysis(current_results, max_n)
print(f"Information fraction: {seq_result['info_fraction']:.1%}")
print(f"Z-boundary: ±{seq_result['z_boundary']:.3f}")
print(f"Current Z: {seq_result['z_current']:.3f}")
print(f"Decision: {seq_result['decision']}")
```

---

### Method 6: Bayesian A/B Testing (Alternative Approach)

```python
from scipy.stats import beta
import matplotlib.pyplot as plt

# Data
n_A, X_A = 2500, 312
n_B, X_B = 2480, 348

# Prior (uniform)
alpha_prior, beta_prior = 1, 1

# Posterior distributions
# Beta(α + successes, β + failures)
posterior_A = beta(alpha_prior + X_A, beta_prior + (n_A - X_A))
posterior_B = beta(alpha_prior + X_B, beta_prior + (n_B - X_B))

# Sample from posteriors
n_samples = 100000
samples_A = posterior_A.rvs(n_samples)
samples_B = posterior_B.rvs(n_samples)

# Probability that B > A
prob_B_better = (samples_B > samples_A).mean()

# Expected loss
loss_if_choose_A = (samples_B - samples_A).clip(min=0).mean()
loss_if_choose_B = (samples_A - samples_B).clip(min=0).mean()

print("=== BAYESIAN A/B TEST ===\n")
print(f"Posterior A: Beta({alpha_prior + X_A}, {beta_prior + (n_A - X_A)})")
print(f"Posterior B: Beta({alpha_prior + X_B}, {beta_prior + (n_B - X_B)})\n")
print(f"P(B > A) = {prob_B_better:.1%}")
print(f"\nExpected Loss:")
print(f"  If choose A: {loss_if_choose_A:.4f} ({loss_if_choose_A*100:.2f}%)")
print(f"  If choose B: {loss_if_choose_B:.4f} ({loss_if_choose_B*100:.2f}%)\n")

if prob_B_better > 0.95:
    print("Decision: Choose B (high confidence)")
elif prob_B_better < 0.05:
    print("Decision: Choose A (high confidence)")
else:
    print("Decision: Inconclusive - need more data")
```

---

### Example 2: Continuous Metric (Average Order Value)

**Data:**
```
Group A (Current Checkout):
- Users: 1,000
- Mean AOV: $45.30
- Std Dev: $12.50

Group B (New Checkout):
- Users: 1,020
- Mean AOV: $47.80
- Std Dev: $13.20
```

**Python Implementation:**

```python
from scipy.stats import ttest_ind_from_stats

# Data
n_A = 1000
mean_A = 45.30
std_A = 12.50

n_B = 1020
mean_B = 47.80
std_B = 13.20

# Welch's t-test (unequal variances)
t_stat, p_value = ttest_ind_from_stats(
    mean1=mean_A, std1=std_A, nobs1=n_A,
    mean2=mean_B, std2=std_B, nobs2=n_B,
    equal_var=False  # Welch's test
)

# Effect size (Cohen's d)
pooled_std = np.sqrt(((n_A - 1) * std_A**2 + (n_B - 1) * std_B**2) / (n_A + n_B - 2))
cohens_d = (mean_B - mean_A) / pooled_std

# Confidence interval for difference
se_diff = np.sqrt(std_A**2/n_A + std_B**2/n_B)
diff = mean_B - mean_A
ci_margin = 1.96 * se_diff
ci = (diff - ci_margin, diff + ci_margin)

print("=== T-TEST FOR CONTINUOUS METRIC ===\n")
print(f"Group A: ${mean_A:.2f} ± ${std_A:.2f} (n={n_A})")
print(f"Group B: ${mean_B:.2f} ± ${std_B:.2f} (n={n_B})")
print(f"\nDifference: ${diff:.2f}")
print(f"Relative change: {(diff/mean_A)*100:.2f}%")
print(f"\nT-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Cohen's d: {cohens_d:.4f}")
print(f"95% CI: [${ci[0]:.2f}, ${ci[1]:.2f}]")
print(f"\nSignificant? {p_value < 0.05}")
```

---

### Summary: Practical Workflow

**1. Before the test (Planning):**
```python
# Define metrics and calculate required sample size
# Set significance level and power
# Estimate test duration
```

**2. During the test (Monitoring):**
```python
# Use sequential methods if checking early
# Watch for sample ratio mismatch
# Check data quality
```

**3. After the test (Analysis):**
```python
# Calculate test statistic and p-value
# Compute confidence intervals
# Check statistical significance AND practical significance
# Make decision based on complete picture
```

**4. Decision Framework:**
```
if p_value < 0.05 AND effect_size > minimum_practical_difference:
    → Implement change
elif p_value >= 0.05:
    → No evidence of difference, keep control
else:
    → Statistically significant but too small to matter
    → Decision based on cost/benefit
```

---

## Further Reading

- **Books:**
  - "Statistical Inference" by Casella & Berger
  - "Trustworthy Online Controlled Experiments" by Kohavi et al.

- **Online Resources:**
  - Evan Miller's A/B testing tools
  - Google's Optimizely statistics guide
  - Microsoft's ExP platform papers

- **Python Libraries:**
  - `scipy.stats` - Statistical tests
  - `statsmodels` - Advanced statistical models
  - `abracadabra` - A/B testing framework
  - `pymc` - Bayesian A/B testing

- **R Packages:**
  - `stats` (base) - Proportion and t-tests
  - `pwr` - Power analysis
  - `bayesAB` - Bayesian A/B testing

---

## Practice Exercises: Solve by Hand

Test your understanding with these exercises. Work through them manually before checking the solutions.

### Exercise 1: Basic Proportions Test (Easy)

**Scenario:** Testing two landing page designs.

**Data:**
- Design A: 400 visitors, 48 conversions
- Design B: 420 visitors, 63 conversions

**Tasks:**
1. Calculate the conversion rate for each design
2. Calculate the pooled proportion
3. Calculate the standard error
4. Calculate the z-statistic
5. Find the p-value (use z-table or approximation)
6. Calculate 95% confidence interval for the difference
7. State your conclusion at α = 0.05

---

### Exercise 2: Sample Size Calculation (Medium)

**Scenario:** Planning a new A/B test.

**Given:**
- Current conversion rate: 8%
- Minimum detectable effect: 1.5 percentage points (want to detect 9.5%)
- Significance level: α = 0.05
- Desired power: 80%

**Tasks:**
1. Calculate the effect size δ
2. Find critical values: z_(α/2) and z_β
3. Calculate required sample size per group using the formula
4. If you get 200 visitors per day, how long should the test run?

---

### Exercise 3: Hypothesis Testing (Medium)

**Scenario:** Email subject line test.

**Data:**
- Subject A: 1,200 emails sent, 156 opens
- Subject B: 1,250 emails sent, 188 opens

**Tasks:**
1. State the null and alternative hypotheses
2. Calculate the test statistic
3. Determine if significant at α = 0.01 (note: stricter threshold)
4. Calculate the relative lift from A to B
5. Interpret the practical significance

---

### Exercise 4: Continuous Metric Analysis (Hard)

**Scenario:** Testing average session duration (in minutes).

**Data:**
- Version A: n=500, mean=4.2 min, std dev=1.8 min
- Version B: n=520, mean=4.7 min, std dev=1.9 min

**Tasks:**
1. Calculate the difference in means
2. Calculate the standard error of the difference
3. Calculate the t-statistic
4. Estimate degrees of freedom (use simplified: smaller n - 1)
5. With df ≈ 499, critical value ≈ 1.96, is this significant at α = 0.05?
6. Calculate 95% confidence interval for the difference

---

### Exercise 5: Power Analysis (Hard)

**Scenario:** Post-test power calculation.

**Data from completed test:**
- Group A: 800 users, 12% conversion
- Group B: 800 users, 15% conversion
- The test showed p = 0.08 (not significant at α = 0.05)

**Tasks:**
1. What was the observed effect size?
2. Calculate the standard error
3. Calculate what power this test had to detect the observed 3% difference
4. How many samples would have been needed for 80% power?

---

### Exercise 6: Confidence Intervals (Medium)

**Scenario:** Interpreting test results.

**Data:**
- Control: 600 users, 18% conversion
- Variant: 650 users, 21% conversion

**Tasks:**
1. Calculate the difference in conversion rates
2. Calculate 90% confidence interval (z = 1.645)
3. Calculate 95% confidence interval (z = 1.96)
4. Calculate 99% confidence interval (z = 2.576)
5. Explain how confidence level affects interval width

---

### Exercise 7: Multi-Variant Test (Hard)

**Scenario:** Testing three button colors (A, B, C).

**Data:**
- Button A: 500 users, 60 conversions (12.0%)
- Button B: 480 users, 67 conversions (13.96%)
- Button C: 490 users, 54 conversions (11.02%)

**Tasks:**
1. Why can't we just do three pairwise tests at α = 0.05?
2. Calculate Bonferroni-corrected α for three comparisons
3. Test A vs B with corrected α
4. State which button you would choose and why

---

## Solutions to Practice Exercises

### Solution 1: Basic Proportions Test

**Step 1: Conversion rates**
```
p̂_A = 48/400 = 0.12 (12%)
p̂_B = 63/420 = 0.15 (15%)
Difference = 0.15 - 0.12 = 0.03 (3 percentage points)
```

**Step 2: Pooled proportion**
```
p̂_pooled = (48 + 63)/(400 + 420) = 111/820 = 0.1354
```

**Step 3: Standard error**
```
SE = √[p̂_pooled × (1 - p̂_pooled) × (1/n_A + 1/n_B)]
SE = √[0.1354 × 0.8646 × (1/400 + 1/420)]
SE = √[0.1171 × 0.00488]
SE = √0.000571
SE = 0.0239
```

**Step 4: Z-statistic**
```
z = (p̂_B - p̂_A)/SE
z = (0.15 - 0.12)/0.0239
z = 0.03/0.0239
z = 1.255
```

**Step 5: P-value**
```
From z-table: P(Z > 1.255) ≈ 0.105
Two-tailed: p-value = 2 × 0.105 = 0.21
```

**Step 6: 95% Confidence interval**
```
CI = (p̂_B - p̂_A) ± 1.96 × SE
CI = 0.03 ± 1.96 × 0.0239
CI = 0.03 ± 0.047
CI = [-0.017, 0.077] or [-1.7%, 7.7%]
```

**Step 7: Conclusion**
```
p-value (0.21) > α (0.05) → Fail to reject H₀
The difference is NOT statistically significant.
Design B shows 3% higher conversion, but this could easily be due to chance.
The confidence interval includes 0, confirming no significant difference.
```

---

### Solution 2: Sample Size Calculation

**Step 1: Effect size**
```
δ = |p_B - p_A| = |0.095 - 0.08| = 0.015 (1.5 percentage points)
p_A = 0.08
p_B = 0.095
```

**Step 2: Critical values**
```
z_(α/2) = z_(0.025) = 1.96  (for 95% confidence, two-tailed)
z_β = z_(0.20) = 0.84        (for 80% power)
```

**Step 3: Sample size calculation**
```
n = (z_(α/2) + z_β)² × [p_A(1-p_A) + p_B(1-p_B)] / δ²

n = (1.96 + 0.84)² × [0.08(0.92) + 0.095(0.905)] / (0.015)²
n = (2.80)² × [0.0736 + 0.086] / 0.000225
n = 7.84 × 0.1596 / 0.000225
n = 1.251 / 0.000225
n = 5,560 per group
```

**Step 4: Test duration**
```
Total sample needed = 2 × 5,560 = 11,120 users
Daily visitors = 200
Duration = 11,120 / 200 = 55.6 days ≈ 56 days (about 8 weeks)
```

**Note:** To detect a smaller effect (1.5% instead of 2%), we need significantly more data!

---

### Solution 3: Hypothesis Testing

**Step 1: Hypotheses**
```
H₀: p_A = p_B (no difference in open rates)
H₁: p_A ≠ p_B (there is a difference)
α = 0.01
```

**Step 2: Calculate test statistic**
```
p̂_A = 156/1200 = 0.13 (13%)
p̂_B = 188/1250 = 0.1504 (15.04%)
Difference = 0.0204 (2.04 percentage points)

Pooled proportion:
p̂_pooled = (156 + 188)/(1200 + 1250) = 344/2450 = 0.1404

Standard error:
SE = √[0.1404 × 0.8596 × (1/1200 + 1/1250)]
SE = √[0.1207 × 0.001633]
SE = √0.000197
SE = 0.0140

Z-statistic:
z = 0.0204/0.0140 = 1.457
```

**Step 3: Significance at α = 0.01**
```
For α = 0.01 (two-tailed), critical value = ±2.576
|z| = 1.457 < 2.576

p-value ≈ 2 × P(Z > 1.457) ≈ 2 × 0.0726 ≈ 0.145

p-value (0.145) > α (0.01) → Fail to reject H₀
NOT significant at the stricter α = 0.01 level
```

**Step 4: Relative lift**
```
Relative lift = (p̂_B - p̂_A)/p̂_A × 100%
Relative lift = (0.1504 - 0.13)/0.13 × 100%
Relative lift = 0.0204/0.13 × 100% = 15.7%
```

**Step 5: Practical interpretation**
```
Subject B shows a 2.04 percentage point increase (15.7% relative lift).
However, with p = 0.145, we cannot rule out chance.
At the stricter α = 0.01 threshold, we need more evidence.
Consider running test longer or using Bayesian methods.
```

---

### Solution 4: Continuous Metric Analysis

**Step 1: Difference in means**
```
X̄_B - X̄_A = 4.7 - 4.2 = 0.5 minutes (30 seconds increase)
```

**Step 2: Standard error of difference**
```
SE_diff = √(s²_A/n_A + s²_B/n_B)
SE_diff = √(1.8²/500 + 1.9²/520)
SE_diff = √(3.24/500 + 3.61/520)
SE_diff = √(0.00648 + 0.00694)
SE_diff = √0.01342
SE_diff = 0.1159 minutes
```

**Step 3: T-statistic**
```
t = (X̄_B - X̄_A)/SE_diff
t = 0.5/0.1159
t = 4.314
```

**Step 4: Degrees of freedom**
```
Simplified approach: df ≈ min(n_A, n_B) - 1 = 500 - 1 = 499
With large df, t-distribution ≈ normal distribution
```

**Step 5: Significance test**
```
|t| = 4.314 > 1.96 (critical value)
This is HIGHLY significant (p < 0.001)

More precise: P(t > 4.314) with df=499 ≈ 0.00001
Two-tailed p-value ≈ 0.00002
```

**Step 6: 95% Confidence interval**
```
CI = (X̄_B - X̄_A) ± 1.96 × SE_diff
CI = 0.5 ± 1.96 × 0.1159
CI = 0.5 ± 0.227
CI = [0.273, 0.727] minutes
CI = [16.4, 43.6] seconds

Conclusion: Version B increases session duration by 0.5 minutes
with 95% confidence between 16 and 44 seconds.
```

---

### Solution 5: Power Analysis

**Step 1: Observed effect size**
```
p_A = 0.12, p_B = 0.15
δ = 0.15 - 0.12 = 0.03 (3 percentage points)
```

**Step 2: Standard error**
```
p̂_pooled = (0.12 × 800 + 0.15 × 800)/(800 + 800)
p̂_pooled = (96 + 120)/1600 = 216/1600 = 0.135

SE = √[0.135 × 0.865 × (1/800 + 1/800)]
SE = √[0.1168 × 0.0025]
SE = √0.000292
SE = 0.0171
```

**Step 3: Calculate actual power**
```
z_observed = 0.03/0.0171 = 1.754
(This gave p = 0.08, slightly above 0.05)

For significance at α = 0.05, we need z > 1.96
We observed z = 1.754

Power = P(reject H₀ | H₁ true with δ = 0.03)
      = P(Z > 1.96 - (0.03/SE))
      = P(Z > 1.96 - 1.754)
      = P(Z > 0.206)
      ≈ 0.418 or 41.8% power

The test was UNDERPOWERED! Only ~42% chance of detecting the effect.
```

**Step 4: Sample size for 80% power**
```
n = (z_(α/2) + z_β)² × [p_A(1-p_A) + p_B(1-p_B)] / δ²
n = (1.96 + 0.84)² × [0.12(0.88) + 0.15(0.85)] / (0.03)²
n = 7.84 × [0.1056 + 0.1275] / 0.0009
n = 7.84 × 0.2331 / 0.0009
n = 1.827 / 0.0009
n ≈ 2,030 per group

Needed: 2,030 × 2 = 4,060 total
Actually had: 800 × 2 = 1,600 total
We were 60% short on sample size!
```

**Lesson:** This is why pre-test power analysis is crucial!

---

### Solution 6: Confidence Intervals

**Step 1: Difference in conversion rates**
```
p̂_A = 0.18 (18%)
p̂_B = 0.21 (21%)
Difference = 0.03 (3 percentage points)

Pooled proportion:
p̂_pooled = (0.18 × 600 + 0.21 × 650)/(600 + 650)
p̂_pooled = (108 + 136.5)/1250 = 244.5/1250 = 0.1956

Standard error:
SE = √[0.1956 × 0.8044 × (1/600 + 1/650)]
SE = √[0.1573 × 0.00321]
SE = √0.000505
SE = 0.0225
```

**Step 2: 90% Confidence interval**
```
90% CI = 0.03 ± 1.645 × 0.0225
90% CI = 0.03 ± 0.037
90% CI = [-0.007, 0.067] or [-0.7%, 6.7%]
```

**Step 3: 95% Confidence interval**
```
95% CI = 0.03 ± 1.96 × 0.0225
95% CI = 0.03 ± 0.044
95% CI = [-0.014, 0.074] or [-1.4%, 7.4%]
```

**Step 4: 99% Confidence interval**
```
99% CI = 0.03 ± 2.576 × 0.0225
99% CI = 0.03 ± 0.058
99% CI = [-0.028, 0.088] or [-2.8%, 8.8%]
```

**Step 5: Interpretation**
```
Confidence Level | Width | Contains 0?
90%              | 7.4%  | Yes
95%              | 8.8%  | Yes
99%              | 11.6% | Yes

As confidence increases:
- Interval gets wider (more uncertainty captured)
- All intervals contain 0 → not significant at any level
- Trade-off: higher confidence = less precision

Higher confidence means we're more certain the interval
contains the true value, but the interval must be wider.
```

---

### Solution 7: Multi-Variant Test

**Step 1: Why not three pairwise tests?**
```
Problem: Multiple comparisons inflate Type I error rate

If we do 3 tests at α = 0.05 each:
P(at least one false positive) = 1 - (1 - 0.05)³
                                = 1 - 0.95³
                                = 1 - 0.857
                                = 0.143 (14.3%)

Instead of 5% false positive rate, we have 14.3%!
This is called "multiple testing problem"
```

**Step 2: Bonferroni correction**
```
Number of comparisons: 3 (A vs B, A vs C, B vs C)
Corrected α = 0.05/3 = 0.0167 per test

This maintains overall Type I error at ~5%
```

**Step 3: Test A vs B with α = 0.0167**
```
p̂_A = 60/500 = 0.12
p̂_B = 67/480 = 0.1396
Difference = 0.0196 (1.96 percentage points)

Pooled proportion:
p̂_pooled = (60 + 67)/(500 + 480) = 127/980 = 0.1296

Standard error:
SE = √[0.1296 × 0.8704 × (1/500 + 1/480)]
SE = √[0.1128 × 0.00408]
SE = √0.000460
SE = 0.0214

Z-statistic:
z = 0.0196/0.0214 = 0.916

Critical value for α = 0.0167 (two-tailed): z = ±2.39
|z| = 0.916 < 2.39 → NOT significant

p-value ≈ 2 × P(Z > 0.916) ≈ 2 × 0.18 ≈ 0.36
```

**Step 4: Decision**
```
Test Results Summary:
- Button A: 12.0%
- Button B: 13.96% (highest, but not significantly different from A)
- Button C: 11.02% (lowest)

None of the differences are significant after Bonferroni correction.

Recommendation:
1. If forced to choose: Button B (highest conversion)
2. Better approach: Run longer test or
3. Use Bayesian methods for better multi-variant analysis
4. Consider practical significance: Is 2% lift worth implementation cost?
```

---

## Key Takeaways from Exercises

1. **Calculation Practice:** Hand calculations reinforce understanding of formulas
2. **Sample Size Matters:** Small differences need large samples
3. **Significance vs Magnitude:** Statistically significant ≠ practically important
4. **Power Analysis:** Always check if test was adequately powered
5. **Confidence Intervals:** More informative than p-values alone
6. **Multiple Testing:** Beware of inflated error rates with multiple comparisons
7. **Effect Size:** Cohen's d and relative lift matter for interpretation

---

*This document provides the mathematical foundation for A/B testing. Practice with real data to build intuition alongside theoretical understanding.*
