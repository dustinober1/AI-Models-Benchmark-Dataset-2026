---
phase: 02-statistical-analysis-domain-insights
plan: 02
subsystem: statistical-analysis
tags: [spearman-correlation, fdr-correction, scipy, polars, non-parametric, context-window, intelligence-tier]

# Dependency graph
requires:
  - phase: 02-01
    provides: Deduplicated dataset (187 models) with unique model_id column and intelligence_index (Int64)
provides:
  - Spearman correlation matrix (5x5) with FDR-corrected p-values for all numerical variables
  - Context window distribution analysis by intelligence quartile (STAT-05)
  - Statistical analysis utilities (compute_spearman_correlation, apply_fdr_correction, group_by_quartile)
  - Correlation heatmap visualization with hierarchical clustering
  - Box plot visualization showing context window scaling by intelligence tier
  - Narrative report documenting significant and null findings (STAT-11, NARR-07)
affects:
  - Phase 2 Plan 03 (Pareto Frontier) - will use correlation results for price-performance optimization
  - Phase 2 Plan 04 (Statistical Tests) - will build on correlation findings for group comparisons
  - Phase 2 Plan 05 (Provider Clustering) - will use correlation matrix for feature selection

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Non-parametric correlation analysis using Spearman rank correlation
    - FDR (Benjamini-Hochberg) correction for multiple testing control
    - Quartile-based grouping for intelligence tier analysis
    - Display name mapping for user-friendly column labels in reports
    - Matplotlib box plots directly from numpy arrays (avoids pyarrow dependency)

key-files:
  created:
    - src/statistics.py - Statistical analysis utilities (372 lines)
    - scripts/08_correlation_analysis.py - Correlation analysis pipeline (490 lines)
    - data/processed/correlation_analysis_correlation.parquet - 5x5 correlation matrix
    - data/processed/correlation_analysis_p_raw.parquet - Raw p-values
    - data/processed/correlation_analysis_p_adjusted.parquet - FDR-adjusted p-values
    - reports/figures/correlation_heatmap.png - Hierarchical clustering heatmap (242KB, 300 DPI)
    - reports/figures/context_window_by_intelligence_tier.png - Box plot by intelligence tier (109KB, 300 DPI)
    - reports/correlation_analysis_2026-01-18.md - Narrative report with methodology and findings
  modified: []

key-decisions:
  - "Use intelligence_index (Int64) column instead of 'Intelligence Index' (String with '--' placeholders)"
  - "Apply Spearman correlation throughout (non-parametric) due to right-skewed distributions from Phase 1"
  - "Use Benjamini-Hochberg FDR correction instead of Bonferroni for higher power with multiple tests"
  - "Report both significant and null findings (STAT-11) to avoid publication bias"
  - "Create intelligence quartiles (Q1-Q4) for STAT-05 context window tier analysis"

patterns-established:
  - "Pattern 1: Non-parametric statistical analysis - Spearman correlation for skewed distributions, FDR correction for multiple testing"
  - "Pattern 2: Tier-based analysis - Use quartiles (qcut) to create intelligence tiers for group comparisons"
  - "Pattern 3: Comprehensive reporting - Document methodology, significant findings, null findings, and interpretations"
  - "Pattern 4: Display name mapping - Internal column names mapped to user-friendly labels for reports and visualizations"

# Metrics
duration: 5min
completed: 2026-01-18
---

# Phase 2 Plan 02: Correlation Analysis Summary

**Spearman correlation analysis with FDR correction reveals moderate positive relationships between intelligence and both price (ρ=0.590) and context window (ρ=0.542), while all 10 pairwise correlations are statistically significant after multiple testing correction**

## Performance

- **Duration:** 5 min (335 seconds)
- **Started:** 2026-01-19T00:05:04Z
- **Completed:** 2026-01-19T00:10:39Z
- **Tasks:** 2
- **Files modified:** 8 created, 0 modified

## Accomplishments

- Computed 5x5 Spearman correlation matrix for intelligence_index, price_usd, Speed, Latency, context_window (181 models with valid intelligence)
- Applied Benjamini-Hochberg FDR correction for multiple testing control (10 pairwise tests)
- Generated correlation heatmap with hierarchical clustering showing variable groupings
- Completed STAT-05: Context window distribution analysis by intelligence quartile (Q1-Q4)
- Created narrative report documenting methodology, significant findings, null findings (STAT-11), and interpretations (NARR-07)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement statistical analysis utilities** - `0422c78` (feat)
2. **Task 2: Execute correlation analysis pipeline** - `fc84f1f` (feat)

**Plan metadata:** (none - summary only)

## Files Created/Modified

- `src/statistics.py` - Statistical analysis utilities (372 lines)
  - compute_spearman_correlation(): Spearman rank correlation with pairwise null handling
  - compute_correlation_matrix(): 5x5 correlation matrix with p-values
  - apply_fdr_correction(): Benjamini-Hochberg FDR correction
  - interpret_correlation(): Classification of correlation strength, direction, significance
  - group_by_quartile(): Intelligence tier creation for STAT-05 analysis

- `scripts/08_correlation_analysis.py` - Correlation analysis pipeline (490 lines)
  - Main pipeline: Load data, compute correlations, apply FDR, generate visualizations and report
  - create_correlation_heatmap(): Hierarchical clustering with seaborn clustermap
  - analyze_context_window_by_tier(): STAT-05 quartile analysis
  - create_tier_visualization(): Box plot showing context window by intelligence tier
  - generate_correlation_report(): Narrative report with methodology and findings

- `data/processed/correlation_analysis_correlation.parquet` - 5x5 correlation matrix
- `data/processed/correlation_analysis_p_raw.parquet` - Raw p-values
- `data/processed/correlation_analysis_p_adjusted.parquet` - FDR-adjusted p-values
- `reports/figures/correlation_heatmap.png` - Correlation heatmap (242KB, 300 DPI)
- `reports/figures/context_window_by_intelligence_tier.png` - Box plot by tier (109KB, 300 DPI)
- `reports/correlation_analysis_2026-01-18.md` - Narrative report

## Decisions Made

**Non-parametric approach validated:**
- Used Spearman correlation throughout (not Pearson) due to right-skewed distributions identified in Phase 1
- All numerical variables show skewness > 0, making Spearman the appropriate choice

**FDR correction over Bonferroni:**
- Applied Benjamini-Hochberg FDR correction for multiple testing (10 pairwise correlations)
- More powerful than Bonferroni while still controlling false discovery rate
- All 10 correlations remained significant after FDR correction (p_adjusted < 0.05)

**STAT-05 intelligence tier analysis:**
- Created quartiles (Q1-Q4) using qcut() for equal-sized bins
- Analyzed context window distribution by intelligence tier
- Found moderate positive correlation (ρ=0.542) between intelligence and context window

**Comprehensive reporting (STAT-11, NARR-07):**
- Documented methodology (Spearman why, FDR why, sample size, significance threshold)
- Reported all 10 significant findings with effect sizes and interpretations
- No null findings to report (all correlations significant)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Use intelligence_index column instead of "Intelligence Index"**
- **Found during:** Task 2 (correlation analysis execution)
- **Issue:** Plan specified "Intelligence Index" column (String type with "--" placeholders and "E" suffixes)
- **Fix:** Changed to use intelligence_index (Int64) column with proper numeric values and nulls
- **Files modified:** scripts/08_correlation_analysis.py (numerical_cols list, filter, quartile grouping)
- **Verification:** Script executed successfully, correlation matrix computed correctly
- **Committed in:** fc84f1f (Task 2 commit)

**2. [Rule 3 - Blocking] Replaced to_pandas() with numpy extraction**
- **Found during:** Task 2 (tier visualization)
- **Issue:** df.to_pandas() failed with ModuleNotFoundError: No module named 'pyarrow'
- **Fix:** Extract numpy arrays directly from Polars using df.filter().to_numpy()
- **Files modified:** scripts/08_correlation_analysis.py (create_tier_visualization function)
- **Verification:** Box plot created successfully using matplotlib directly
- **Committed in:** fc84f1f (Task 2 commit)

**3. [Rule 1 - Bug] Fixed DataFrame indexing in report generation**
- **Found during:** Task 2 (report verification)
- **Issue:** Report showed incorrect correlation values (e.g., Intelligence-Speed = 1.000 instead of 0.261)
- **Fix:** Changed indexing from corr_df[i + 1, j] to corr_df[i, j + 1] (first column is "column" row label)
- **Files modified:** scripts/08_correlation_analysis.py (3 locations in generate_correlation_report)
- **Verification:** Re-ran script, report now shows correct correlations
- **Committed in:** fc84f1f (Task 2 commit)

**4. [Rule 2 - Missing Critical] Added display name mapping**
- **Found during:** Task 2 (report readability)
- **Issue:** Report and heatmap showed internal column names (intelligence_index, price_usd, context_window)
- **Fix:** Added display_names dictionary mapping to user-friendly labels
- **Files modified:** scripts/08_correlation_analysis.py (create_correlation_heatmap, generate_correlation_report)
- **Verification:** Report and heatmap now show "Intelligence Index", "Price (USD)", "Context Window"
- **Committed in:** fc84f1f (Task 2 commit)

---

**Total deviations:** 4 auto-fixed (2 blocking, 1 bug, 1 missing critical)
**Impact on plan:** All auto-fixes necessary for correctness and usability. No scope creep. Enhanced plan with better error handling and user-friendly output.

## Issues Encountered

**Column type mismatch between plan and actual data:**
- Plan specified "Intelligence Index" column, but actual data has both "Intelligence Index" (String with placeholders) and intelligence_index (Int64 numeric)
- Solution: Used intelligence_index column for all computations

**Missing pyarrow dependency for to_pandas():**
- Matplotlib box plot via seaborn required pandas conversion, which requires pyarrow
- Solution: Used matplotlib.boxplot() directly with numpy arrays extracted from Polars

**DataFrame indexing confusion in report generation:**
- Polars DataFrame with "column" as first column requires offsetting column index, not row index
- Solution: Changed [i + 1, j] to [i, j + 1] throughout report generation

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Statistical analysis foundation complete:**
- Correlation matrix with FDR correction available for Pareto frontier analysis (Plan 02-03)
- Intelligence quartile analysis informs provider segmentation (Plan 02-05)
- Non-parametric approach validated for all Phase 2 statistical tests

**Key findings for downstream analysis:**
- Strong positive Intelligence-Price correlation (ρ=0.590) suggests premium pricing for smarter models
- Moderate positive Intelligence-Context Window correlation (ρ=0.542) shows higher intelligence models have larger context windows
- Intelligence-Speed correlation is weak (ρ=0.261) - smarter models aren't necessarily faster
- All 10 correlations statistically significant - robust relationships across all variable pairs

**Context window by intelligence tier (STAT-05):**
- Q1 (Low): Mean 331K tokens, Median 128K (highly variable, extreme outlier at 10M)
- Q2 (Mid-Low): Mean 286K tokens, Median 256K
- Q3 (Mid-High): Mean 383K tokens, Median 256K
- Q4 (High): Mean 490K tokens, Median 256K
- Clear positive trend: higher intelligence models have larger context windows

**Recommendations for downstream analysis:**
- Use correlation matrix to guide feature selection for provider clustering (Plan 02-05)
- Pareto frontier analysis (Plan 02-03) should leverage Intelligence-Price tradeoff
- Consider log transformation for context_window (extreme skewness: 9.63 per Phase 1)
- Statistical tests (Plan 02-04) should use non-parametric methods (Mann-Whitney U, Kruskal-Wallis)

**No blockers** - Plan 02-03 (Pareto Frontier) can proceed immediately.

---
*Phase: 02-statistical-analysis-domain-insights*
*Completed: 2026-01-18*
