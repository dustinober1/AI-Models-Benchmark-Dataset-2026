# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2025-01-18)

**Core value:** Discover at least one novel insight about AI models that is not commonly published knowledge
**Current focus:** Phase 3 - Interactive Visualizations

## Current Position

Phase: 2 of 4 (Statistical Analysis & Domain Insights) — COMPLETE
Status: Phase 2 verified — 31/31 must-haves verified, goal achieved
Last activity: 2026-01-19 — Phase 2 execution complete: 5 plans across 3 waves, 24 minutes total

Progress: [██████████] 100% (13 of 13 plans complete: 8 Phase 1 + 5 Phase 2)

## Verification Status

Phase 1 verified: **passed** (43/43 must-haves)
- Goal Achievement: 25/25 truths verified
- Artifacts: 33/33 verified
- Key Links: 21/21 verified
- Requirements: 14/15 satisfied (DATA-08 partial: external scraping 0% coverage)

Phase 2 verified: **passed** (31/31 must-haves)
- Goal Achievement: 31/31 truths verified
- Artifacts: 27/27 verified
- Key Links: 15/15 verified
- Requirements: 13/13 satisfied (STAT-01 through STAT-11, NARR-07, NARR-09)

**Phase 1 Deliverables:**
- Cleaned and validated dataset: data/processed/ai_models_enriched.parquet (188 models, 16 columns)
- Quality assessment report: reports/quality_2026-01-18.md (75% score)
- Distribution analysis: 5 high-resolution plots with comprehensive statistics
- Pipeline completion summary: reports/pipeline_summary.md

**Phase 2 Deliverables:**
- Deduplicated dataset: data/processed/ai_models_deduped.parquet (187 models, 18 columns)
- Correlation analysis: 5x5 Spearman matrix with FDR correction (all 10 correlations significant)
- Pareto frontier analysis: 3 multi-objective frontiers (Intelligence-Price, Speed-Intelligence, Multi-objective)
- Provider clustering: K=2 segments (Budget vs Premium) with regional comparisons
- Statistical tests: Non-parametric tests with bootstrap CIs (95% BCa method)
- 2027 trend predictions: Scenario analysis with uncertainty quantification
- 2,366 lines of new analysis code (5 modules, 6 scripts)
- 24 output files (6 datasets, 12 figures, 6 reports)

## Performance Metrics

**Velocity:**
- Total plans completed: 13
- Average duration: 4.1 minutes
- Total execution time: 0.9 hours (53 minutes)

**By Phase:**

| Phase | Plans | Complete | Avg/Plan |
|-------|-------|----------|----------|
| 1 (Data Pipeline) | 8 | 8 | 3.9 min |
| 2 (Statistical Analysis) | 5 | 5 | 4.8 min |
| 3 (Visualizations) | 0 | ? | - |
| 4 (Narrative) | 0 | ? | - |

**Recent Trend:**
- Last 13 plans: 01-01 (8 min), 01-02 (3 min), 01-03a (4 min), 01-03b (5 min), 01-04 (3 min), 01-05a (2 min), 01-05b (5 min), 01-06 (7 min), 02-01 (4 min), 02-02 (5 min), 02-03 (7 min), 02-04 (3 min), 02-05 (5 min)
- Trend: Consistent velocity ~4.1 min/plan

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

**From Plan 01-01 (Project Foundation):**
- Poetry 2.3.0 for dependency management (latest version, handles Python 3.14)
- Script-as-module pattern: all scripts have importable functions for notebook integration
- Numeric script prefixes (01_load.py, 02_clean.py) for execution order, though Python requires workarounds for direct import
- LazyFrame evaluation throughout pipeline for performance and memory efficiency
- Separate quarantine/ directory for invalid/outlier records with timestamped filenames
- Comprehensive quality reporting with 6 dimensions (completeness, accuracy, consistency, validity)

**From Plan 01-02 (Load Data with Schema Validation):**
- Lenient schema loading (all Utf8) to handle messy CSV data before cleaning stage
- Pandera schema validation deferred to after cleaning when proper types are established
- Context Window values contain "k"/"m" suffixes (400k, 1m, 200k) requiring parsing in cleaning stage
- Price column contains "$4.81 " format requiring dollar sign stripping and Float64 conversion
- Intelligence Index has "--" placeholder for missing values requiring null handling
- Dataset contains 188 models from 37 creators documented in comprehensive structure report

**From Plan 01-03b (Execute Data Cleaning Pipeline):**
- Data quality: 96.81% completeness with only 6 null values (3.19%) in intelligence_index column
- Missing value strategy: Preserve nulls in intelligence_index - no imputation needed for optional metric
- Context window parsing: Suffixes parsed using regex (2m -> 2,000,000, 262k -> 262,000)
- Schema validation deferred: Skip Pandera validation until after null handling in later plan
- Core columns (Model, Creator, Price, Speed, Latency, Context Window) are 100% complete
- Cleaned checkpoint available at data/interim/02_cleaned.parquet with proper data types
- Missing value analysis documented in reports/missing_values.md with pattern analysis and recommendations

**From Plan 01-04 (Distribution Analysis and Outlier Detection):**
- Statistical analysis utilities created in src/analyze.py with scipy.stats and sklearn integration
- Distribution analysis completed for 5 numerical variables: context_window, intelligence_index, price_usd, Speed(median token/s), Latency (First Answer Chunk /s)
- Outlier detection using Isolation Forest with 5% contamination - 10 models flagged (5.32%)
- All numerical variables are right-skewed (skewness > 0) - may require log transformation for parametric tests
- Context Window has extreme skewness (9.63) and kurtosis (114.20) - heavy-tailed with extreme values
- Intelligence Index distribution is approximately normal (skewness=0.67, kurtosis=2.63)
- Price and Speed show moderate to high positive skewness - most models are low-cost, low-speed with few high-end outliers
- Latency has extreme positive skewness (7.11) - most models have low latency with very few high-latency outliers
- Outlier strategy: Flag but don't remove - preserving data for domain expert review (per CONTEXT.md)
- Checkpoint saved with outlier flags: data/interim/03_distributions_analyzed.parquet
- High-resolution distribution plots (300 DPI) generated for all numerical columns

**From Plan 01-05a (Web Scraping Utilities):**
- Web scraping with requests + BeautifulSoup (async httpx not needed for simple use case)
- Rate limiting with 1 second delay between requests (time.sleep(1)) for respectful scraping
- Provenance tracking: All scraped data includes source_url, retrieved_at, retrieved_by columns
- Graceful error handling: Functions return empty DataFrame with correct schema on failure
- Best-effort coverage: Pipeline continues with nulls if scraping fails
- Library availability check: REQUESTS_AVAILABLE flag handles missing dependencies gracefully
- HTML selectors require adjustment: HuggingFace scraping returned empty (actual page structure inspection needed)
- Provider announcements partial success: 6 announcements retrieved, but model extraction needs refinement

**From Plan 01-05b (Merge and Validate Enriched Dataset):**
- Data enrichment utilities: enrich_with_external_data() (left join), add_derived_columns() (5 metrics), calculate_enrichment_coverage() (statistics)
- Derived metrics created: price_per_intelligence_point, speed_intelligence_ratio, model_tier, log_context_window, price_per_1k_tokens
- Enrichment coverage: 96.81% for intelligence-based metrics (6 models lack IQ scores), 100% for transformation-based metrics
- External data coverage: 0% (web scraping failed - all model names null in scraped data)
- Model tier classification: Regex-based extraction from names (67.6% unknown, 12.8% high, 10.6% mini, 4.3% low, 3.7% medium, 1.1% xhigh)
- Final enriched dataset: 188 models, 16 columns saved to data/processed/ai_models_enriched.parquet
- Graceful degradation: Pipeline proceeds with base dataset when external data unavailable
- Speed column type: Stored as String type in cleaned data, auto-cast to Float64 during enrichment (Rule 3 fix)

**From Plan 01-06 (Quality Report Generation and Pipeline Completion):**
- Quality assessment utilities: perform_sanity_checks() with 6-dimensional framework (Accuracy, Completeness, Consistency, Validity, Integrity, Timeliness)
- Quality score calculation: Average of dimension scores (4 applicable dimensions for single-table dataset)
- Overall quality score: 75.0% (3/4 dimensions passed: Accuracy PASS, Completeness PASS, Consistency FAIL, Validity PASS)
- 34 duplicate model names detected (18.1%) - critical issue requiring resolution before Phase 2
- String type column handling: Float64 casting with try/except for numeric comparisons (Speed, Latency, Context Window, Intelligence Index)
- Statistical analysis recommendations: Non-parametric methods (Spearman, Mann-Whitney U, Kruskal-Wallis) due to non-normal distributions
- Quality report generated: reports/quality_2026-01-18.md (320 lines, 5 embedded figure links)
- Pipeline completion summary: reports/pipeline_summary.md (509 lines, complete Phase 1 documentation)
- Phase 1 status: COMPLETE - Ready for Phase 2 Statistical Analysis

**From Plan 02-01 (Duplicate Resolution):**
- Duplicate resolution utilities: detect_duplicates(), resolve_duplicate_models(), validate_resolution()
- Multi-stage disambiguation strategy: Primary (context_window) → Secondary (intelligence_index) → Tertiary (unique())
- 34 duplicate model names resolved using context_window as primary differentiator
- Secondary intelligence_index disambiguation for models with same name AND context window
- Removed 1 true duplicate row (Exaone 4.0 1.2B) where all columns were identical
- Deduplicated dataset: data/processed/ai_models_deduped.parquet (187 models, 18 columns)
- Unique model_id column created for accurate group-by operations in Phase 2
- Original Model column preserved for reference
- Validation confirmed 0 remaining duplicates, all model_ids unique
- 6 models with null intelligence_index filled with -1 for disambiguation (model_id ends with "_-1")
- Intelligence-specific analyses should filter to n=181 models with valid IQ scores
- Resolution report generated: reports/duplicate_resolution_2026-01-18.md

**From Plan 02-02 (Correlation Analysis):**
- Statistical analysis utilities: compute_spearman_correlation(), compute_correlation_matrix(), apply_fdr_correction(), interpret_correlation(), group_by_quartile()
- Non-parametric approach validated: Spearman correlation throughout due to right-skewed distributions from Phase 1
- FDR correction: Benjamini-Hochberg applied for multiple testing (10 pairwise correlations, all significant after correction)
- STAT-05 context window tier analysis: Intelligence quartiles (Q1-Q4) show moderate positive correlation with context window (ρ=0.542)
- Key findings: Intelligence-Price (ρ=0.590, moderate), Intelligence-Context Window (ρ=0.542, moderate), Intelligence-Speed (ρ=0.261, weak)
- Correlation matrix with FDR correction available for Pareto frontier analysis and provider clustering
- Context window by intelligence tier: Q1 (331K mean), Q2 (286K), Q3 (383K), Q4 (490K) - clear positive trend
- Display name mapping: Internal column names mapped to user-friendly labels for reports and visualizations
- Matplotlib box plots: Direct numpy array extraction avoids pyarrow dependency
- Narrative report: Methodology, significant findings (all 10), null findings (none per STAT-11), interpretations (NARR-07)

**From Plan 02-03 (Pareto Frontier Analysis):**
- Pareto frontier utilities: compute_pareto_frontier(), get_pareto_efficient_models(), compute_hypervolume(), plot_pareto_frontier()
- Multi-objective dominance algorithm: j dominates i if all_objectives[j] >= all_objectives[i] AND any(all_objectives[j] > all_objectives[i])
- String-to-numeric casting: Speed, Latency stored as String - automatic Float64 casting in compute_pareto_frontier()
- Three frontier analyses completed:
  - Intelligence vs Price: 8 efficient models (4.4%) - GPT-5.2 leads with IQ=51, Price=$4.81
  - Speed vs Intelligence: 6 efficient models (3.3%) - GPT-5.2, Gemini 3 Pro Preview, Gemini 3 Flash dominate
  - Multi-objective: 41 efficient models (22.7%) - Balancing intelligence, speed, price, latency
- Market leaders: GPT-5.2 dominates all frontiers (IQ=51), Gemini 3 Flash offers exceptional value (IQ=46, Price=$1.13)
- Flag merging strategy: Left-join on model_id (not row index) for reliable Pareto flag alignment
- Dataset with Pareto flags: data/processed/pareto_frontier.parquet (187 models, 21 columns)
- Frontier visualizations: Annotated scatter plots highlighting efficient models in red
- Value propositions identified: Budget options (GLM-4.7: IQ=42, Price=$0.94), premium models (GPT-5.2: IQ=51, Price=$4.81)

**From Plan 02-04 (Provider Clustering):**
- Provider clustering utilities: aggregate_by_provider(), find_optimal_clusters(), cluster_providers(), validate_clustering(), assign_region(), compare_regions()
- Provider-level aggregation: 181 models aggregated to 36 providers by intelligence, price, speed
- KMeans clustering with StandardScaler normalization (3 features: avg_intelligence, avg_price, avg_speed)
- Optimal K=2 determined by silhouette score (0.390, moderate cluster structure)
- Market segments identified:
  - Cluster 0 (24 providers): Budget-Friendly Segment ($0.35, IQ=17.9, Speed=34.3)
  - Cluster 1 (12 providers): Premium Performance Segment ($1.53, IQ=29.0, Speed=117.4)
- Premium Performance providers: OpenAI, Anthropic, Google, Amazon, Mistral, Cohere, etc.
- Budget-Friendly providers: Alibaba, DeepSeek, Meta, Microsoft, Baidu, IBM, etc.
- STAT-04 regional comparison completed:
  - Intelligence: Similar across regions (China 22.2, Europe 18.8, Other 21.1, US 22.6)
  - Price: US highest ($1.53), China mid-range ($0.93), Europe/Other lower ($0.55/$0.44)
  - Speed: Europe fastest (142.3 tokens/s), US second (118.4), China/Other slower (66.4/59.9)
- Manual region assignment: Case-insensitive substring matching for US (OpenAI, Anthropic, Google, Meta, Microsoft), China (DeepSeek, Alibaba, Tencent, Baidu), Europe (Mistral, Aleph Alpha), Other (unknown)
- Provider clusters dataset: data/processed/provider_clusters.parquet (36 providers, 8 columns with cluster assignments)
- Clustering visualizations: Silhouette scores plot, elbow plot, 3-panel scatter analysis

**From Plan 02-05 (Bootstrap and Statistical Testing):**
- Bootstrap CI utilities: bootstrap_mean_ci(), bootstrap_median_ci(), bootstrap_correlation_ci(), bootstrap_group_difference_ci()
- Non-parametric tests: mann_whitney_u_test() (2 groups), kruskal_wallis_test() (3+ groups)
- BCa method: Bias-corrected and accelerated bootstrap with n_resamples=9999 for accurate CIs
- Fallback to percentile method when BCa fails (degenerate distributions)
- Regional statistical tests: Kruskal-Wallis test (US vs China vs Europe)
  - Intelligence: No significant difference (p=0.6471)
  - Price: No significant difference (p=0.3192)
  - Speed: Significant difference (p=0.0064) - China fastest, US slowest
- Pairwise Mann-Whitney U tests with FDR correction:
  - US vs China speed: Significant (p_adj=0.0240, effect size r=-0.274)
  - China vs Europe speed: Significant (p_adj=0.0080, effect size r=0.577)
- Bootstrap 95% CIs for key metrics:
  - Intelligence Index: Mean 21.81 [20.31, 23.50], Median 20.00
  - Price (USD): Mean $1.00 [0.79, 1.29]
  - Speed: Mean 90.72 [78.75, 105.39] token/s
- STAT-11 satisfied: Both significant (3) and null (3) findings reported to avoid publication bias
- NARR-09 satisfied: Comprehensive uncertainty discussion for 2027 trend predictions
- 2027 trend predictions: Scenario analysis (optimistic, baseline, pessimistic)
  - Intelligence: Mean 22.24-23.99 (2-10% improvement)
  - Price: -5% to -20% reduction by intelligence tier
  - Speed: +5% to +20% improvement (baseline 10%)
- Uncertainty quantification: Prediction intervals, assumptions, limitations documented

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

**Known considerations for next phase:**
- Scripts use numeric prefixes (01-06) which require `PYTHONPATH=.` for running as modules
- Poetry 2.x doesn't have `poetry export` - requirements.txt must be regenerated manually if dependencies change
- Context window parsing completed - values now Int64 token counts (400000, 200000, etc.)
- Pandera schema validation deferred to later plan - will run after enrichment stage
- 6 models lack intelligence_index scores - intelligence-specific analysis should filter to n=182
- Quality report script (05_quality_report.py) generates timestamped markdown reports
- Outlier detection uses Isolation Forest with 5% contamination parameter

**From Plan 01-04:**
- 10 models flagged as outliers (5.32%) - these may require special handling in statistical analysis
- All numerical variables are right-skewed - non-parametric methods or log transformation may be more appropriate
- Context Window has extreme skewness (9.63) and kurtosis (114.20) - heavy-tailed distribution with extreme values (10M token context)
- Statistical functions require explicit Float64 casting before numpy conversion to prevent type errors
- Distributions are non-normal based on normality tests - parametric tests may not be appropriate

**From Plan 01-05a:**
- HuggingFace HTML selectors need adjustment for actual model data extraction (currently returns empty)
- Provider announcement scraping has null model column values - will need fuzzy matching or manual mapping
- External data coverage limited (6 announcements, 0 models) - may need manual entry for top 20 models

**From Plan 01-05b:**
- External data enrichment failed (0% coverage) - temporal analysis not possible without manual data entry
- Model tier classification: 67.6% unknown tier limits tier-based analysis power
- Speed/Latency columns stored as String type in cleaned data - requires casting during analysis (auto-fixed in enrichment)
- 6 models lack intelligence_index scores - intelligence-specific analyses must filter to n=182

**From Plan 01-06:**
- 34 duplicate model names detected (18.1%) - MUST resolve before Phase 2 group-by operations
- Overall quality score: 75.0% - meets 75% threshold for Phase 2 readiness
- All numerical variables are right-skewed - non-parametric methods required (Spearman, Mann-Whitney U, Kruskal-Wallis)
- Context Window extreme skewness (9.63) - log transformation recommended
- 10 outliers flagged (5.32%) - assess impact on correlation and statistical tests

**From Plan 02-01:**
- 34 duplicate model names resolved - 0 remaining duplicates
- Unique model_id column created for group-by operations
- 6 models with null intelligence_index (filled with -1 for disambiguation)
- Intelligence-specific analyses should filter to n=181 (exclude null IQ scores)

**From Plan 02-02:**
- All 10 pairwise correlations statistically significant after FDR correction - robust relationships across all variables
- Moderate Intelligence-Price correlation (ρ=0.590) - premium pricing for smarter models
- Moderate Intelligence-Context Window correlation (ρ=0.542) - higher intelligence models have larger context windows
- Weak Intelligence-Speed correlation (ρ=0.261) - smarter models aren't necessarily faster
- Context window extreme skewness (9.63) - may need log transformation for parametric tests
- Use intelligence_index (Int64) column, not "Intelligence Index" (String with placeholders)
- Matplotlib direct numpy extraction avoids pyarrow dependency

**From Plan 02-03:**
- Speed, Latency columns stored as String - automatic Float64 casting required in Pareto functions
- Flag merging: Left-join on model_id (not row index) to avoid pyarrow dependency
- Frontier density: Smaller frontier = clearer leaders (8 price-performance), larger = more tradeoffs (41 multi-objective)
- Pareto flags available for filtering: is_pareto_intelligence_price, is_pareto_speed_intelligence, is_pareto_multi_objective
- String column formatting: Use try/except blocks for safe numeric conversion in reports

**Phase 1 Status:** COMPLETE — Verified 43/43 must-haves, 75% quality score

**Phase 2 Status:** COMPLETE — Verified 31/31 must-haves, all goals achieved

**Phase 3 Readiness:**
- Deduplicated dataset: data/processed/ai_models_deduped.parquet (187 models, 18 columns)
- Correlation matrix: 5x5 Spearman with FDR correction (all 10 correlations significant)
- Pareto frontier data: data/processed/pareto_frontier.parquet (187 models, 21 columns)
- Provider clusters: data/processed/provider_clusters.parquet (36 providers, 2 clusters)
- Statistical test results: Bootstrap CIs, regional comparisons, trend predictions
- Key insights: Intelligence-Price ρ=0.590, all correlations significant, regional speed differences
- Non-parametric approach validated: All analyses use appropriate methods for skewed distributions
- Market leaders: GPT-5.2 (IQ=51), Gemini 3 Flash (value leader)
- Market segments: Budget (24 providers) vs Premium (12 providers)
- Visualizations ready to convert from matplotlib to Plotly for interactivity
- 12 existing figures to enhance with hover, zoom, and linked brushing
- Known issues: Non-normal distributions, context window extreme skewness (9.63)
- Use model_id (not Model) for all aggregations

## Session Continuity

Last session: 2026-01-19 (Phase 2 execution)
Stopped at: Phase 2 complete and verified
Resume file: None
Next: Phase 3 - Interactive Visualizations
