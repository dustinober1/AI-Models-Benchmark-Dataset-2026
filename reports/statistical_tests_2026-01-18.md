# Statistical Tests Report
**Generated:** 2026-01-18

## Methodology
### Statistical Methods
This analysis uses **non-parametric statistical tests** appropriate for the
highly right-skewed distributions identified in Phase 1 (skewness 2.34-9.63).

**Non-parametric tests:**
- **Mann-Whitney U test:** Two-group comparison (alternative to independent t-test)
- **Kruskal-Wallis test:** Three+ group comparison (alternative to one-way ANOVA)
- **Bootstrap confidence intervals:** BCa method with 9,999 resamples

**Why non-parametric?**
- No assumption of normality
- Robust to outliers
- Appropriate for skewed distributions
- Rank-based methods (Spearman, Mann-Whitney U, Kruskal-Wallis)

**Multiple testing correction:**
- Benjamini-Hochberg FDR correction applied to pairwise comparisons
- Controls false discovery rate while maintaining power
- Significance threshold: α = 0.05

**Bootstrap methodology:**
- BCa (bias-corrected and accelerated) method
- n_resamples = 9,999 for accuracy
- Confidence level = 95%
- Fallback to percentile method if BCa fails

## Regional Comparison Results
### Kruskal-Wallis Test (US vs China vs Europe)

#### Intelligence Index
- **H statistic:** 0.87
- **p-value:** 0.6471
- **Significant:** No
- **Groups:** US, China, Europe
- **Sample sizes:** [74, 40, 12]

#### Price Usd
- **H statistic:** 2.28
- **p-value:** 0.3192
- **Significant:** No
- **Groups:** US, China, Europe
- **Sample sizes:** [74, 40, 12]

#### Speed(Median Token/S)
- **H statistic:** 10.09
- **p-value:** 0.0064
- **Significant:** Yes
- **Groups:** US, China, Europe
- **Sample sizes:** [74, 40, 12]

### Pairwise Mann-Whitney U Tests (with FDR correction)

**US vs China:**
- **Mann-Whitney U:** 1885.00
- **Raw p-value:** 0.0160
- **FDR-adjusted p-value:** 0.0240
- **Significant (after FDR):** Yes
- **Effect size (r):** -0.274

**US vs Europe:**
- **Mann-Whitney U:** 343.50
- **Raw p-value:** 0.2120
- **FDR-adjusted p-value:** 0.2120
- **Significant (after FDR):** No
- **Effect size (r):** 0.226

**China vs Europe:**
- **Mann-Whitney U:** 101.50
- **Raw p-value:** 0.0027
- **FDR-adjusted p-value:** 0.0080
- **Significant (after FDR):** Yes
- **Effect size (r):** 0.577

## Bootstrap Confidence Intervals
### Uncertainty Quantification for Key Metrics

All confidence intervals computed using BCa method with 9,999 resamples.

#### Intelligence Index
**Mean:**
- Point estimate: 21.81
- 95% CI: [20.31, 23.50]
- Standard error: 0.8050
- Sample size: n=181

**Median:**
- Point estimate: 20.00
- 95% CI: [nan, nan]
- Standard error: 1.1722

#### price_usd
**Mean:**
- Point estimate: 1.00
- 95% CI: [0.79, 1.29]
- Standard error: 0.1252
- Sample size: n=181

**Median:**
- Point estimate: 0.32
- 95% CI: [0.25, 0.45]
- Standard error: 0.0576

#### Speed(median token/s)
**Mean:**
- Point estimate: 90.72
- 95% CI: [78.75, 105.39]
- Standard error: 6.7999
- Sample size: n=181

**Median:**
- Point estimate: 72.00
- 95% CI: [51.00, 81.00]
- Standard error: 7.0210

## Significant Findings
### Tests with p < 0.05 (after FDR correction)

**Speed(Median Token/S) (Regional):**
- Kruskal-Wallis test: H=10.09, p=0.0064
- US, China, Europe show significant differences

**US vs China:**
- Mann-Whitney U: 1885.00, p_adj=0.0240
- Effect size: r=-0.274

**China vs Europe:**
- Mann-Whitney U: 101.50, p_adj=0.0080
- Effect size: r=0.577

## Null Findings
### Tests with p >= 0.05 (no significant difference)

*This section reports all non-significant findings to avoid publication bias.*

**Intelligence Index (Regional):**
- Kruskal-Wallis test: H=0.87, p=0.6471
- No significant differences between US, China, Europe

**Price Usd (Regional):**
- Kruskal-Wallis test: H=2.28, p=0.3192
- No significant differences between US, China, Europe

**US vs Europe:**
- Mann-Whitney U: 343.50, p_adj=0.2120
- No significant difference detected

## Limitations and Recommendations

### Limitations
- Sample sizes vary by region (some regions have few providers)
- Cross-sectional data (2026 snapshot, not longitudinal)
- String columns (Speed, Latency) required casting to Float64
- Some providers classified as 'Other' due to unknown region
- Non-parametric tests have lower power than parametric alternatives

### Recommendations
- Use bootstrap CIs for uncertainty quantification in all analyses
- Apply FDR correction when performing multiple tests
- Report both significant and null findings to avoid publication bias
- Consider log transformation for highly skewed variables
- Increase sample size for regional comparisons if possible
