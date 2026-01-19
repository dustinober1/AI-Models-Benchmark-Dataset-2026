# Correlation Analysis Report

**Generated:** 2026-01-18 19:10:19
**Analysis:** Spearman correlation with FDR correction

---

## Methodology

### Statistical Approach
- **Correlation Method:** Spearman rank correlation (non-parametric)
  - Rationale: All numerical variables are right-skewed per Phase 1 distribution analysis
  - Spearman is rank-based, does not assume normality
  - Robust to outliers compared to Pearson correlation
- **Multiple Testing Correction:** Benjamini-Hochberg FDR (False Discovery Rate)
  - Rationale: More powerful than Bonferroni for multiple comparisons
  - Controls expected proportion of false discoveries among rejected hypotheses
  - Adjusted p-values reported alongside raw p-values
- **Significance Threshold:** p_adjusted < 0.05

### Data
- **Sample Size:** 181 models with valid intelligence scores
- **Variables Analyzed:** 5 numerical metrics
- **Tests Performed:** 10 pairwise correlations

---

## Correlation Summary

### All Pairwise Correlations

| Variable 1 | Variable 2 | Correlation | Raw p-value | Adjusted p-value | Significant | Interpretation |
|-----------|-----------|-------------|-------------|------------------|-------------|----------------|
| Intelligence Index | Price (USD) | 0.590 | 0.0000 | 0.0000 | Yes | Moderate positive correlation, statistically significant |
| Intelligence Index | Speed (median token/s) | 0.261 | 0.0004 | 0.0004 | Yes | Weak positive correlation, statistically significant |
| Intelligence Index | Latency (First Answer Chunk /s) | 0.444 | 0.0000 | 0.0000 | Yes | Moderate positive correlation, statistically significant |
| Intelligence Index | Context Window | 0.542 | 0.0000 | 0.0000 | Yes | Moderate positive correlation, statistically significant |
| Price (USD) | Speed (median token/s) | 0.291 | 0.0001 | 0.0001 | Yes | Weak positive correlation, statistically significant |
| Price (USD) | Latency (First Answer Chunk /s) | 0.554 | 0.0000 | 0.0000 | Yes | Moderate positive correlation, statistically significant |
| Price (USD) | Context Window | 0.387 | 0.0000 | 0.0000 | Yes | Weak positive correlation, statistically significant |
| Speed (median token/s) | Latency (First Answer Chunk /s) | 0.531 | 0.0000 | 0.0000 | Yes | Moderate positive correlation, statistically significant |
| Speed (median token/s) | Context Window | 0.381 | 0.0000 | 0.0000 | Yes | Weak positive correlation, statistically significant |
| Latency (First Answer Chunk /s) | Context Window | 0.333 | 0.0000 | 0.0000 | Yes | Weak positive correlation, statistically significant |

---

## Significant Findings (FDR-corrected)

**Total Significant Correlations:** 10 of 10

Statistically significant correlations after FDR correction:

#### Intelligence Index vs Price (USD)
- **Correlation:** 0.590 (positive moderate)
- **Raw p-value:** 0.0000
- **Adjusted p-value:** 0.0000
- **Interpretation:** Moderate positive correlation, statistically significant

#### Intelligence Index vs Speed (median token/s)
- **Correlation:** 0.261 (positive weak)
- **Raw p-value:** 0.0004
- **Adjusted p-value:** 0.0004
- **Interpretation:** Weak positive correlation, statistically significant

#### Intelligence Index vs Latency (First Answer Chunk /s)
- **Correlation:** 0.444 (positive moderate)
- **Raw p-value:** 0.0000
- **Adjusted p-value:** 0.0000
- **Interpretation:** Moderate positive correlation, statistically significant

#### Intelligence Index vs Context Window
- **Correlation:** 0.542 (positive moderate)
- **Raw p-value:** 0.0000
- **Adjusted p-value:** 0.0000
- **Interpretation:** Moderate positive correlation, statistically significant

#### Price (USD) vs Speed (median token/s)
- **Correlation:** 0.291 (positive weak)
- **Raw p-value:** 0.0001
- **Adjusted p-value:** 0.0001
- **Interpretation:** Weak positive correlation, statistically significant

#### Price (USD) vs Latency (First Answer Chunk /s)
- **Correlation:** 0.554 (positive moderate)
- **Raw p-value:** 0.0000
- **Adjusted p-value:** 0.0000
- **Interpretation:** Moderate positive correlation, statistically significant

#### Price (USD) vs Context Window
- **Correlation:** 0.387 (positive weak)
- **Raw p-value:** 0.0000
- **Adjusted p-value:** 0.0000
- **Interpretation:** Weak positive correlation, statistically significant

#### Speed (median token/s) vs Latency (First Answer Chunk /s)
- **Correlation:** 0.531 (positive moderate)
- **Raw p-value:** 0.0000
- **Adjusted p-value:** 0.0000
- **Interpretation:** Moderate positive correlation, statistically significant

#### Speed (median token/s) vs Context Window
- **Correlation:** 0.381 (positive weak)
- **Raw p-value:** 0.0000
- **Adjusted p-value:** 0.0000
- **Interpretation:** Weak positive correlation, statistically significant

#### Latency (First Answer Chunk /s) vs Context Window
- **Correlation:** 0.333 (positive weak)
- **Raw p-value:** 0.0000
- **Adjusted p-value:** 0.0000
- **Interpretation:** Weak positive correlation, statistically significant

---

## Null Findings (STAT-11)

**Total Non-Significant Correlations:** 0 of 10

Correlations that did NOT reach statistical significance after FDR correction:
(These are reported to avoid publication bias)

*All correlations were statistically significant.*

---

## Context Window by Intelligence Tier (STAT-05)

### Overview

This section analyzes how context window capacity scales with model intelligence.
- **Spearman Correlation:** 0.542
- **p-value:** 0.0000
- **Interpretation:** Moderate positive correlation, statistically significant

### Context Window Statistics by Intelligence Quartile

| Intelligence Tier | Count | Mean Context Window | Median Context Window | Std Dev | Min | Max |
|-------------------|-------|---------------------|----------------------|---------|-----|-----|
| Q1 (Low) | 52 | 331,365 | 128,000 | 1,379,817 | 4,000 | 10,000,000 |
| Q2 (Mid-Low) | 44 | 285,523 | 256,000 | 276,778 | 33,000 | 1,000,000 |
| Q3 (Mid-High) | 42 | 383,429 | 256,000 | 456,581 | 128,000 | 2,000,000 |
| Q4 (High) | 43 | 490,256 | 256,000 | 461,849 | 128,000 | 2,000,000 |

### Interpretation

The intelligence quartile analysis reveals how context capacity scales with model intelligence:
- **Moderate relationship:** Clear positive trend between intelligence and context window.
- Higher intelligence models generally have larger context windows.

### Visualization

See figure: `reports/figures/context_window_by_intelligence_tier.png`
- Box plot shows context window distribution for each intelligence quartile
- Q1 = Lowest 25% intelligence, Q4 = Highest 25% intelligence

---

## Conclusions

### Key Insights

1. **10 significant correlations** identified after FDR correction
2. **0 null findings** reported (transparency per STAT-11)
3. **Non-parametric approach validated:** Spearman correlation appropriate for skewed distributions

### Methodology Strengths
- Used Spearman correlation (robust to non-normality and outliers)
- Applied FDR correction (controls false discovery rate)
- Reported both significant and null findings (avoids publication bias)
- Analyzed context window by intelligence tier (STAT-05 requirement)

### Limitations
- Cross-sectional data (single time point)
- Correlation does not imply causation
- Sample size limited to models with valid intelligence scores
- Some variables have extreme skewness (e.g., Context Window)

---

## Figures

1. **Correlation Heatmap:** `reports/figures/correlation_heatmap.png`
   - Hierarchical clustering groups correlated variables
   - Color scale: Blue (negative) to Red (positive)
   - Annotations show correlation coefficients

2. **Context Window by Intelligence Tier:** `reports/figures/context_window_by_intelligence_tier.png`
   - Box plot showing context window distribution by intelligence quartile
   - Q1 (Low) to Q4 (High) intelligence tiers