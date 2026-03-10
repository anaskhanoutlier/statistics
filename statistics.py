

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PART A: PROBABILITY DISTRIBUTIONS
# ─────────────────────────────────────────

def probability_distributions_analysis():
    """
    Analyze and visualize key probability distributions used in statistics.
    Covers: Normal, Binomial, Poisson, Exponential, Chi-squared, t-distribution
    """
    print("\n" + "═"*60)
    print("  PART A: Probability Distributions")
    print("═"*60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Probability Distributions — Statistical Analysis\n"
                 "Statistics Minor | AMU | Python · SciPy · NumPy",
                 fontsize=12, fontweight='bold')

    x = np.linspace(-4, 4, 300)

    # ── 1. Normal Distribution — Effect of μ and σ ──
    ax = axes[0, 0]
    params = [(0, 1, 'μ=0, σ=1', 'royalblue'),
              (1, 1, 'μ=1, σ=1', 'tomato'),
              (0, 0.5, 'μ=0, σ=0.5', 'green'),
              (0, 2, 'μ=0, σ=2', 'purple')]
    for mu, sigma, label, color in params:
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color=color, linewidth=2, label=label)
    ax.set_title("Normal Distribution N(μ, σ²)", fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)

    # ── 2. Binomial Distribution ──
    ax = axes[0, 1]
    n_val = 20
    for p, color, label in [(0.3, 'royalblue', 'p=0.3'),
                             (0.5, 'tomato', 'p=0.5'),
                             (0.7, 'green', 'p=0.7')]:
        k = np.arange(0, n_val + 1)
        pmf = stats.binom.pmf(k, n_val, p)
        ax.plot(k, pmf, 'o-', color=color, linewidth=1.5, markersize=4, label=label)
    ax.set_title(f"Binomial Distribution B(n={n_val}, p)", fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel("k"); ax.set_ylabel("P(X=k)")
    ax.grid(True, alpha=0.3)

    # ── 3. Poisson Distribution ──
    ax = axes[0, 2]
    k_pois = np.arange(0, 20)
    for lam, color in [(2, 'royalblue'), (5, 'tomato'), (10, 'green')]:
        ax.plot(k_pois, stats.poisson.pmf(k_pois, lam), 'o-',
                color=color, linewidth=1.5, markersize=4, label=f'λ={lam}')
    ax.set_title("Poisson Distribution P(λ)", fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel("k"); ax.set_ylabel("P(X=k)")
    ax.grid(True, alpha=0.3)

    # ── 4. Central Limit Theorem Demo ──
    ax = axes[1, 0]
    np.random.seed(42)
    population = stats.expon.rvs(scale=2, size=10000)
    for n_samp, color in [(5, 'lightblue'), (30, 'steelblue'), (100, 'navy')]:
        sample_means = [np.mean(np.random.choice(population, n_samp)) for _ in range(2000)]
        ax.hist(sample_means, bins=40, alpha=0.5, color=color,
                density=True, label=f'n={n_samp}')
    x_norm = np.linspace(min(sample_means)-1, max(sample_means)+1, 100)
    ax.plot(x_norm, stats.norm.pdf(x_norm, np.mean(population), np.std(population)/np.sqrt(100)),
            'r-', linewidth=2, label='Normal (CLT)')
    ax.set_title("Central Limit Theorem\n(Exponential Population)", fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel("Sample Mean")

    # ── 5. Chi-squared Distribution ──
    ax = axes[1, 1]
    x_chi = np.linspace(0, 30, 300)
    for df, color in [(1, 'blue'), (3, 'green'), (5, 'orange'), (10, 'red')]:
        ax.plot(x_chi, stats.chi2.pdf(x_chi, df), color=color,
                linewidth=2, label=f'df={df}')
    ax.set_ylim(0, 0.5)
    ax.set_title("Chi-squared Distribution χ²(df)", fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)

    # ── 6. t-distribution vs Normal ──
    ax = axes[1, 2]
    x_t = np.linspace(-4, 4, 300)
    ax.plot(x_t, stats.norm.pdf(x_t), 'k-', linewidth=2.5, label='Normal', zorder=5)
    for df, color in [(1, 'red'), (3, 'orange'), (10, 'green'), (30, 'blue')]:
        ax.plot(x_t, stats.t.pdf(x_t, df), '--', color=color,
                linewidth=1.5, label=f't(df={df})')
    ax.set_title("t-distribution vs Normal\n(Heavier Tails)", fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("project9a_distributions.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Print key statistics
    print("\n  Key Distribution Properties:")
    print(f"  Normal N(0,1)    — Mean: 0, Std: 1, Skewness: 0")
    print(f"  Binomial B(20,.5)— Mean: 10, Std: {np.sqrt(20*0.5*0.5):.4f}")
    print(f"  Poisson P(5)     — Mean: 5, Std: {np.sqrt(5):.4f}")

    # CLT verification
    np.random.seed(0)
    pop = np.random.exponential(3, 10000)
    means_n30 = [np.mean(np.random.choice(pop, 30)) for _ in range(5000)]
    k2, p = stats.normaltest(means_n30)
    print(f"\n  CLT Verification (n=30 sample means from Exponential):")
    print(f"  k² statistic = {k2:.4f}, p-value = {p:.4f}")
    print(f"  → {'Sample means are approximately Normal ✓' if p > 0.05 else 'Reject normality'}")


# ─────────────────────────────────────────
# PART B: HYPOTHESIS TESTING
# ─────────────────────────────────────────

def hypothesis_testing_suite():
    """
    Comprehensive hypothesis testing:
    - One-sample t-test
    - Two-sample t-test  
    - Paired t-test
    - Chi-square test of independence
    - ANOVA (F-test)
    - Mann-Whitney U (non-parametric)
    """
    print("\n" + "═"*60)
    print("  PART B: Hypothesis Testing Suite")
    print("═"*60)

    np.random.seed(42)
    alpha = 0.05

    def print_result(test_name, statistic, p_value, alpha=0.05):
        result = "REJECT H₀" if p_value < alpha else "FAIL TO REJECT H₀"
        print(f"\n  ── {test_name} ──")
        print(f"  Statistic = {statistic:.4f}  |  p-value = {p_value:.4f}")
        print(f"  α = {alpha}  →  [{result}]")
        if p_value < alpha:
            print(f"  ✅ Statistically significant result.")
        else:
            print(f"  ➡ Not statistically significant.")

    # 1. One-sample t-test
    # H₀: Mean height of students = 165 cm
    heights = np.random.normal(168, 8, 50)
    t_stat, p_val = stats.ttest_1samp(heights, 165)
    print_result("One-Sample t-Test: H₀: μ = 165 cm", t_stat, p_val)
    print(f"  Sample Mean: {heights.mean():.2f} cm, Std: {heights.std():.2f}")

    # 2. Two-sample Independent t-test
    # H₀: No difference in test scores between two groups
    group_a = np.random.normal(72, 10, 40)
    group_b = np.random.normal(78, 12, 40)
    t_stat2, p_val2 = stats.ttest_ind(group_a, group_b)
    print_result("Two-Sample t-Test: Group A vs Group B Scores", t_stat2, p_val2)
    print(f"  Group A: {group_a.mean():.2f} ± {group_a.std():.2f}")
    print(f"  Group B: {group_b.mean():.2f} ± {group_b.std():.2f}")

    # 3. Paired t-test (Before vs After)
    before = np.random.normal(70, 8, 30)
    after  = before + np.random.normal(3, 4, 30)  # Slight improvement
    t_stat3, p_val3 = stats.ttest_rel(before, after)
    print_result("Paired t-Test: Before vs After Treatment", t_stat3, p_val3)
    print(f"  Mean difference: {(after - before).mean():.2f}")

    # 4. Chi-square Test of Independence
    # H₀: Gender and subject preference are independent
    observed = np.array([[45, 35, 20],  # Male: Maths, Science, Arts
                          [30, 25, 45]]) # Female: Maths, Science, Arts
    chi2, p_chi, dof, expected = stats.chi2_contingency(observed)
    print_result(f"Chi-square Test (df={dof}): Gender vs Subject Preference", chi2, p_chi)
    print(f"  Observed:\n{observed}")

    # 5. One-way ANOVA
    # H₀: Mean yields are equal across 4 fertilizer groups
    g1 = np.random.normal(60, 5, 20)
    g2 = np.random.normal(65, 5, 20)
    g3 = np.random.normal(70, 5, 20)
    g4 = np.random.normal(63, 5, 20)
    f_stat, p_anova = stats.f_oneway(g1, g2, g3, g4)
    print_result("One-way ANOVA: Fertilizer Effect on Crop Yield", f_stat, p_anova)

    # 6. Mann-Whitney U (non-parametric)
    income_rural  = np.random.exponential(20000, 50)
    income_urban  = np.random.exponential(35000, 50)
    u_stat, p_mw  = stats.mannwhitneyu(income_rural, income_urban, alternative='two-sided')
    print_result("Mann-Whitney U: Rural vs Urban Income (Non-parametric)", u_stat, p_mw)

    # 7. Shapiro-Wilk Normality Test
    print("\n  ── Shapiro-Wilk Normality Test ──")
    for name, sample in [('Normal sample', np.random.normal(0, 1, 50)),
                          ('Skewed sample', np.random.exponential(2, 50))]:
        w, p = stats.shapiro(sample)
        print(f"  {name:<20}: W={w:.4f}, p={p:.4f} → "
              f"{'Normal ✓' if p > 0.05 else 'Not Normal'}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Hypothesis Testing Suite — Statistical Analysis\n"
                 "Statistics Minor | AMU | Python · SciPy · Statsmodels",
                 fontsize=12, fontweight='bold')

    # t-distribution with rejection region
    ax = axes[0, 0]
    x_t = np.linspace(-4, 4, 300)
    ax.plot(x_t, stats.t.pdf(x_t, 49), 'b-', linewidth=2)
    t_crit = stats.t.ppf(0.975, 49)
    ax.fill_between(x_t, stats.t.pdf(x_t, 49),
                    where=np.abs(x_t) >= t_crit, color='red', alpha=0.4, label=f'Rejection (α=0.05)')
    ax.axvline(t_stat, color='darkred', linestyle='--', linewidth=2, label=f't = {t_stat:.3f}')
    ax.set_title("One-Sample t-Test\nRejection Region", fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Before-After Paired
    ax = axes[0, 1]
    diff = after - before
    ax.hist(diff, bins=20, color='steelblue', edgecolor='white', density=True, alpha=0.7)
    x_d = np.linspace(diff.min(), diff.max(), 100)
    ax.plot(x_d, stats.norm.pdf(x_d, diff.mean(), diff.std()), 'r-', linewidth=2)
    ax.axvline(0, color='black', linestyle='--', linewidth=1.5, label='H₀: diff=0')
    ax.axvline(diff.mean(), color='blue', linestyle='--', linewidth=1.5, label=f'Mean={diff.mean():.2f}')
    ax.set_title("Paired t-Test\nDistribution of Differences", fontweight='bold')
    ax.legend(fontsize=7)

    # ANOVA groups boxplot
    ax = axes[0, 2]
    anova_data = pd.DataFrame({
        'Yield': np.concatenate([g1, g2, g3, g4]),
        'Group': ['G1']*20 + ['G2']*20 + ['G3']*20 + ['G4']*20
    })
    sns.boxplot(data=anova_data, x='Group', y='Yield', palette='Set2', ax=ax)
    ax.set_title(f"One-Way ANOVA\nF={f_stat:.2f}, p={p_anova:.4f}", fontweight='bold')

    # Chi-square distribution
    ax = axes[1, 0]
    x_chi = np.linspace(0, 20, 300)
    dof2 = (observed.shape[0]-1) * (observed.shape[1]-1)
    ax.plot(x_chi, stats.chi2.pdf(x_chi, dof2), 'b-', linewidth=2)
    chi_crit = stats.chi2.ppf(0.95, dof2)
    ax.fill_between(x_chi, stats.chi2.pdf(x_chi, dof2),
                    where=x_chi >= chi_crit, color='red', alpha=0.4, label=f'α=0.05')
    ax.axvline(chi2, color='darkred', linestyle='--', linewidth=2, label=f'χ²={chi2:.2f}')
    ax.set_title(f"Chi-square Test (df={dof2})\nIndependence", fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Power analysis
    ax = axes[1, 1]
    effect_sizes = np.linspace(0.1, 1.5, 100)
    ns = [10, 20, 50, 100]
    for n_pow, color in zip(ns, ['blue', 'green', 'orange', 'red']):
        power = [stats.norm.cdf(-1.96 + e * np.sqrt(n_pow)) +
                 stats.norm.cdf(-1.96 - e * np.sqrt(n_pow))
                 for e in effect_sizes]
        ax.plot(effect_sizes, power, color=color, linewidth=2, label=f'n={n_pow}')
    ax.axhline(0.8, color='black', linestyle='--', linewidth=1.5, label='Power=0.8')
    ax.set_xlabel("Effect Size (Cohen's d)")
    ax.set_ylabel("Statistical Power")
    ax.set_title("Power Analysis\n(Two-tailed, α=0.05)", fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # QQ Plot for normality
    ax = axes[1, 2]
    (osm, osr), (slope, intercept, r) = stats.probplot(heights)
    ax.scatter(osm, osr, color='steelblue', s=20, alpha=0.7)
    ax.plot(osm, slope * np.array(osm) + intercept, 'r-', linewidth=2)
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"Q-Q Plot (Normality Check)\nR² = {r**2:.4f}", fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("project9b_hypothesis_testing.png", dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────
# PART C: MULTIPLE REGRESSION & CORRELATION
# ─────────────────────────────────────────

def regression_and_correlation():
    """
    Full regression analysis with statsmodels:
    - Pearson, Spearman, Kendall correlations
    - Multiple Linear Regression (OLS)
    - Regression diagnostics
    """
    print("\n" + "═"*60)
    print("  PART C: Regression & Correlation Analysis")
    print("═"*60)

    np.random.seed(0)
    n = 200
    rainfall   = np.random.uniform(200, 1500, n)
    temperature = np.random.uniform(15, 45, n)
    soil_ph    = np.random.uniform(5, 8, n)
    fertilizer = np.random.uniform(20, 200, n)
    crop_yield = (0.8 * rainfall/100 + 0.5 * fertilizer/10
                  - 0.3 * (temperature - 25)**2 / 10
                  + 2 * soil_ph + np.random.normal(0, 5, n))

    df = pd.DataFrame({'rainfall': rainfall, 'temperature': temperature,
                       'soil_ph': soil_ph, 'fertilizer': fertilizer,
                       'crop_yield': crop_yield})

    # Correlation analysis
    print("\n  Correlation Coefficients (with crop_yield):")
    print(f"  {'Variable':<15} {'Pearson':>8} {'Spearman':>10} {'Kendall':>8}")
    print(f"  {'─'*45}")
    for col in ['rainfall', 'temperature', 'soil_ph', 'fertilizer']:
        r_p, _ = stats.pearsonr(df[col], df['crop_yield'])
        r_s, _ = stats.spearmanr(df[col], df['crop_yield'])
        r_k, _ = stats.kendalltau(df[col], df['crop_yield'])
        print(f"  {col:<15} {r_p:>8.4f} {r_s:>10.4f} {r_k:>8.4f}")

    # OLS Regression (statsmodels)
    X = sm.add_constant(df[['rainfall', 'temperature', 'soil_ph', 'fertilizer']])
    model = sm.OLS(df['crop_yield'], X).fit()
    print("\n  OLS Regression Summary:")
    print(model.summary())

    # Breusch-Pagan Homoscedasticity test
    _, bp_pval, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"\n  Breusch-Pagan Test (Homoscedasticity): p = {bp_pval:.4f}")
    print(f"  → {'Homoscedastic (assumption met) ✓' if bp_pval > 0.05 else 'Heteroscedastic'}")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Regression & Correlation Analysis — Crop Yield Data\n"
                 "Statistics Minor | AMU | Python · Statsmodels · SciPy",
                 fontsize=12, fontweight='bold')

    # Correlation heatmap
    ax = axes[0, 0]
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                linewidths=0.5, annot_kws={'size': 8}, vmin=-1, vmax=1, center=0)
    ax.set_title("Correlation Matrix", fontweight='bold')

    # Scatter matrix (pair plot subset)
    ax = axes[0, 1]
    ax.scatter(df['rainfall'], df['crop_yield'], alpha=0.3, s=10, color='steelblue')
    m, b, r, p, se = stats.linregress(df['rainfall'], df['crop_yield'])
    x_line = np.linspace(df['rainfall'].min(), df['rainfall'].max(), 100)
    ax.plot(x_line, m*x_line+b, 'r-', linewidth=2, label=f'r={r:.3f}')
    ax.set_xlabel("Rainfall (mm)"); ax.set_ylabel("Crop Yield")
    ax.set_title("Rainfall vs Crop Yield", fontweight='bold')
    ax.legend()

    # Predicted vs Actual
    ax = axes[0, 2]
    y_pred_ols = model.fittedvalues
    ax.scatter(y_pred_ols, df['crop_yield'], alpha=0.3, s=10, color='green')
    lims = [min(y_pred_ols.min(), df['crop_yield'].min()),
            max(y_pred_ols.max(), df['crop_yield'].max())]
    ax.plot(lims, lims, 'k--', linewidth=1.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Actual vs Predicted\nR²={model.rsquared:.4f}", fontweight='bold')

    # Residuals
    ax = axes[1, 0]
    ax.scatter(y_pred_ols, model.resid, alpha=0.3, s=10, color='purple')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Fitted Values"); ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot", fontweight='bold')

    # Residual histogram
    ax = axes[1, 1]
    ax.hist(model.resid, bins=25, color='steelblue', edgecolor='white', density=True, alpha=0.7)
    xn = np.linspace(model.resid.min(), model.resid.max(), 100)
    ax.plot(xn, stats.norm.pdf(xn, model.resid.mean(), model.resid.std()), 'r-', linewidth=2)
    ax.set_title("Residual Distribution\n(Should be Normal)", fontweight='bold')

    # Coefficient plot with confidence intervals
    ax = axes[1, 2]
    coefs = model.params[1:]
    conf  = model.conf_int().iloc[1:]
    y_pos = range(len(coefs))
    ax.barh(y_pos, coefs, xerr=[coefs - conf[0], conf[1] - coefs],
            color='steelblue', alpha=0.7, ecolor='black', capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coefs.index, fontsize=8)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title("Regression Coefficients\n(95% CI)", fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig("project9c_regression.png", dpi=150, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 65)
    print("PROJECT 9: Statistics — Distributions, Testing & Regression")
    print("BS Geography + Statistics (Minor) | AMU | Python · SciPy")
    print("=" * 65)

    probability_distributions_analysis()
    hypothesis_testing_suite()
    regression_and_correlation()

    print("\n✅ All statistical analyses complete.")
    print("   Plots saved: project9a, project9b, project9c")


if __name__ == "__main__":
    main()
