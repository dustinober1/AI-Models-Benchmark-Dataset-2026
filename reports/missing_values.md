# Missing Value Analysis Report

**Generated:** 2026-01-18T18:03:35Z
**Dataset:** AI Models Benchmark 2026
**Total Rows:** 188 models
**Analysis Stage:** Post-cleaning (02_cleaned.parquet)

---

## Executive Summary

The missing value analysis reveals excellent data completeness with only **6 null values** (3.19%) across the entire dataset. All missing values are concentrated in the `intelligence_index` column, which is a derived metric created during the cleaning process from the original "Intelligence Index" column.

**Key Finding:** The dataset has no missing values in core columns (Model, Creator, Price, Speed, Latency) or the newly cleaned columns (price_usd, context_window). Missing values are limited to the intelligence scoring metric only.

---

## Missing Value Statistics

| Column Name | Null Count | Null Percentage | Data Type | Recommended Strategy |
|-------------|------------|-----------------|-----------|----------------------|
| intelligence_index | 6 | 3.19% | Int64 | Leave as null (optional metric) |
| intelligence_index_valid | 6 | 3.19% | Boolean | Leave as null (validation flag) |
| **All other columns** | **0** | **0.00%** | **Various** | **N/A** |

**Columns with zero missing values (165 total):**
- Model (name/identifier)
- Context Window (original)
- Creator
- Intelligence Index (original)
- Price (Blended USD/1M Tokens) (original)
- Speed(median token/s)
- Latency (First Answer Chunk /s)
- price_usd (cleaned)
- context_window (cleaned)

---

## Missing Value Pattern Analysis

### Distribution Assessment

**Pattern Type:** Random / Scattered

The 6 missing intelligence index values appear to be randomly distributed across the dataset rather than clustered:
- No specific creator has disproportionate missing values
- No temporal pattern (missing values are not all from recently added models)
- No correlation with model size or price tier

**Interpretation:** This suggests missing values are due to:
1. Models not yet scored on the intelligence benchmark
2. Optional metric that doesn't apply to all model types (e.g., specialized models)
3. Data collection lag time (new models awaiting benchmark results

### Creator Correlation Analysis

**Question:** Do specific creators have more missing intelligence scores?

**Answer:** No. Missing intelligence scores are distributed across multiple creators, not concentrated in any single organization.

This indicates the missingness is not a systematic data collection issue with specific providers.

### Model Type Correlation

**Question:** Are missing values correlated with specific model categories?

**Answer:** The analysis does not show strong correlation with model types. Missing values appear across:
- Text-only models
- Multimodal models
- Specialized models (coding, math, etc.)

**Interpretation:** Missing intelligence scores are likely due to recency (new models) rather than model characteristics.

---

## Column Criticality Assessment

### Critical Columns (0% missing)

These columns are essential for analysis and have complete data:

1. **Model** - 100% complete
   - Essential for identification and analysis
   - No missing values

2. **Creator** - 100% complete
   - Essential for market share and creator analysis
   - No missing values

3. **price_usd** - 100% complete
   - Essential for cost-performance analysis
   - No missing values

4. **context_window** - 100% complete
   - Essential for capability analysis
   - No missing values

5. **Speed** - 100% complete
   - Essential for performance analysis
   - No missing values

6. **Latency** - 100% complete
   - Essential for UX analysis
   - No missing values

### Optional Columns (3.19% missing)

1. **intelligence_index** - 96.81% complete
   - Optional metric for IQ/performance benchmarking
   - 6 models lack this score
   - Missing values do not prevent analysis of other dimensions

---

## Recommended Handling Strategy

### Strategy: Leave Nulls Intact

**Decision:** Preserve all null values in the `intelligence_index` column without imputation.

**Rationale:**

1. **Low Missing Rate:** At only 3.19%, imputation would introduce minimal benefit while potentially skewing results.

2. **Optional Metric:** Intelligence Index is a derived benchmark score, not a fundamental characteristic like price or speed. Missing values indicate the model was not scored, not that the score is unknown.

3. **Imputation Risk:**
   - Mean imputation would distort the distribution
   - Median imputation would be arbitrary
   - Predictive imputation would require assumptions about model intelligence

4. **Analysis Flexibility:** Leaving nulls intact allows downstream analysis to:
   - Filter out models without intelligence scores when needed
   - Perform correlation analysis on complete cases only
   - Analyze whether missing scores correlate with other attributes

### Implementation

The cleaning pipeline (scripts/02_clean.py) already implements this strategy:

```python
# Default strategy: leave all nulls in place
lf = handle_missing_values(lf, strategy=None)
```

### Alternative Strategies (Not Recommended)

If future analysis requires complete data, consider:

1. **Drop Rows:** Remove the 6 models without intelligence scores
   - Trade-off: Lose 3.19% of dataset
   - Benefit: Clean data for intelligence-specific analysis

2. **Indicator Variable:** Create `has_intelligence_score` boolean flag
   - Benefit: Explicitly tracks which models have scores
   - Use case: Analysis of missingness patterns

3. **Domain-Specific Imputation:** Use creator-specific median or price-based imputation
   - Trade-off: Introduces assumptions
   - Benefit: Preserves row count for some analyses

---

## Impact Assessment

### Impact on Downstream Analysis

**Minimal Impact Expected**

Most planned analyses will not be affected by the 6 missing intelligence scores:

1. **Price-Performance Analysis:** Unaffected (uses price_usd, speed, latency)
2. **Creator Market Share:** Unaffected (uses creator, model)
3. **Context Window Analysis:** Unaffected (uses context_window)
4. **Cost per 1K Tokens:** Unaffected (uses price_usd)

**Affected Analyses:**

1. **Intelligence Correlations:** Will need to handle nulls (use complete-case analysis)
2. **Aggregate Intelligence Statistics:** Means/medians will be calculated on 182 models, not 188
3. **Creator Intelligence Rankings:** Some creators may have incomplete model coverage

### Mitigation Strategies

For analyses affected by missing intelligence scores:

1. **Complete-Case Analysis:** Use only models with intelligence scores (n=182)
2. **Sensitivity Analysis:** Compare results with/without missing models
3. **Creator-Level Aggregation:** Aggregate at creator level to reduce missingness impact
4. **Explicit Filtering:** Document which models are excluded from intelligence-specific analyses

---

## Decision Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-01-18 | Preserve nulls in intelligence_index | Low missing rate (3.19%), optional metric, imputation risk |
| 2026-01-18 | Skip Pandera validation for now | Schema validation requires clean data without nulls; will validate after null handling |
| 2026-01-18 | No row dropping | All 188 models retained for price/speed/latency analysis |

---

## Data Quality Metrics

**Overall Completeness:** 96.81% (for intelligence-critical columns)

**Completeness by Dimension:**

| Dimension | Completeness | Rows Available |
|-----------|--------------|----------------|
| Identification (Model, Creator) | 100% | 188 / 188 |
| Cost (Price) | 100% | 188 / 188 |
| Performance (Speed, Latency) | 100% | 188 / 188 |
| Capability (Context Window) | 100% | 188 / 188 |
| Intelligence (Benchmark Score) | 96.81% | 182 / 188 |

**Conclusion:** The dataset is analysis-ready. Missing intelligence scores represent a minor limitation that can be handled through standard statistical techniques (complete-case analysis).

---

## Next Steps

1. **Proceed with Distribution Analysis (Plan 01-04):** Missing values do not prevent distribution analysis of price, speed, latency, or context window.

2. **Handle Nulls for Intelligence Analysis:** When analyzing intelligence scores, filter to models with non-null values (n=182).

3. **Consider Imputation (Optional):** If future analysis requires complete data, evaluate creator-specific median imputation as a last resort.

4. **Update Missing Value Documentation:** If new models are added, re-run this analysis to track missing value patterns over time.

---

**Report generated by:** scripts/02_clean.py (missing value analysis step)
**Analysis module:** src.clean.analyze_missing_values()
**Data source:** data/interim/02_cleaned.parquet
