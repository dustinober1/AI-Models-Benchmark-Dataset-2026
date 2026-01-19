---
phase: 02-statistical-analysis-domain-insights
plan: 01
subsystem: data-quality
tags: [polars, deduplication, data-cleaning, context-window, intelligence-index]

# Dependency graph
requires:
  - phase: 01-data-pipeline-quality-assessment
    provides: Enriched dataset with parsed context_window and intelligence_index columns
provides:
  - Deduplicated dataset with unique model_id column (187 models)
  - Duplicate resolution utilities (detect, resolve, validate)
  - Resolution documentation with before/after statistics
affects:
  - Phase 2 statistical analysis (correlation, clustering, hypothesis testing)
  - All group-by operations now use model_id instead of Model

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Context window disambiguation for duplicate model names
    - Secondary intelligence index disambiguation
    - True duplicate removal with unique() keep="first"
    - Validation assertions for data quality checks

key-files:
  created:
    - src/deduplicate.py - Duplicate resolution utilities (detect_duplicates, resolve_duplicate_models, validate_resolution)
    - scripts/07_duplicate_resolution.py - Executable duplicate resolution pipeline
    - data/processed/ai_models_deduped.parquet - Deduplicated dataset (187 models, 18 columns)
    - reports/duplicate_resolution_2026-01-18.md - Resolution documentation
  modified: []

key-decisions:
  - "Use context_window (Int64 from Phase 1) instead of Context Window (str) for disambiguation"
  - "Add secondary intelligence_index disambiguation when context_window alone insufficient"
  - "Remove true duplicate rows (identical in all columns) rather than aggregating"
  - "Preserve original Model column for reference while using model_id for operations"

patterns-established:
  - "Pattern 1: Multi-stage disambiguation - Primary (context_window) → Secondary (intelligence_index) → Tertiary (unique())"
  - "Pattern 2: Validation-first approach - assert remaining_duplicates == 0 before proceeding"
  - "Pattern 3: Resolution source tracking - document how each model_id was created"

# Metrics
duration: 4min
completed: 2026-01-18
---

# Phase 2 Plan 01: Duplicate Resolution Summary

**Context window and intelligence index disambiguation resolves 34 duplicate model names (18.1%), creating 187 unique model_ids for accurate group-by operations in Phase 2 statistical analysis**

## Performance

- **Duration:** 4 min (290 seconds)
- **Started:** 2026-01-18T23:57:27Z
- **Completed:** 2026-01-19T00:02:17Z
- **Tasks:** 2
- **Files modified:** 4 created, 0 modified

## Accomplishments

- Resolved 34 duplicate model names (18.1% of original 188 models)
- Created 187 unique model_ids using context_window as primary differentiator
- Removed 1 true duplicate row (identical in all columns)
- Validated to 0 remaining duplicates
- Generated comprehensive resolution report with before/after statistics

## Task Commits

Each task was committed atomically:

1. **Task 1: Create duplicate resolution utilities** - `02624a4` (feat)
2. **Task 2: Execute duplicate resolution pipeline** - `27d1696` (feat)

**Plan metadata:** (none - summary only)

## Files Created/Modified

- `src/deduplicate.py` - Duplicate resolution utilities (detect_duplicates, resolve_duplicate_models, validate_resolution)
- `scripts/07_duplicate_resolution.py` - Executable pipeline with comprehensive reporting
- `data/processed/ai_models_deduped.parquet` - Deduplicated dataset (187 models, 18 columns with model_id)
- `reports/duplicate_resolution_2026-01-18.md` - Resolution documentation with statistics

## Decisions Made

**Enhanced disambiguation strategy (beyond RESEARCH.md example):**
- RESEARCH.md suggested using only `model_name + "_" + context_window` for disambiguation
- Actual data had 34 models with same name AND context window but different attributes (e.g., Speed: 177 vs 171, Intelligence Index: 25 vs 14)
- Added secondary intelligence_index disambiguation when context_window alone insufficient
- Removed 1 true duplicate row (Exaone 4.0 1.2B) where all columns were identical

**Column type fix (Rule 3 - Blocking):**
- RESEARCH.md example used `Context Window` (str) column
- Phase 1 cleaning created `context_window` (Int64) column with parsed token counts
- Fixed code to use `context_window` instead of `Context Window` for correct concatenation

**Null handling for intelligence_index:**
- 6 models have null Intelligence Index (3.19%)
- Filled with -1 for disambiguation purposes (model_id ends with "_-1")
- Intelligence-specific analyses should filter to n=181 models with valid IQ scores

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Use context_window column instead of Context Window**
- **Found during:** Task 1 (resolve_duplicate_models function)
- **Issue:** RESEARCH.md example used Context Window (str) but actual data has context_window (Int64)
- **Fix:** Changed pl.col("Context Window").cast(pl.Int64) to pl.col("context_window")
- **Files modified:** src/deduplicate.py
- **Verification:** Successfully concatenated model_id with integer context window values
- **Committed in:** 02624a4 (Task 1 commit)

**2. [Rule 1 - Bug] Added secondary intelligence_index disambiguation**
- **Found during:** Task 1 verification
- **Issue:** 34 models still duplicated after using context_window alone (same name, same context, different Speed/IQ)
- **Fix:** Added intelligence_index as secondary differentiator when context_window alone insufficient
- **Files modified:** src/deduplicate.py (resolve_duplicate_models function)
- **Verification:** Validation passed with 0 remaining duplicates, 187 unique model_ids
- **Committed in:** 02624a4 (Task 1 commit)

**3. [Rule 2 - Missing Critical] Added true duplicate removal**
- **Found during:** Task 1 verification
- **Issue:** 1 model still duplicated after context_window + intelligence_index disambiguation (Exaone 4.0 1.2B - identical in all columns)
- **Fix:** Added df.unique(subset=["model_id"], keep="first") to remove true duplicates
- **Files modified:** src/deduplicate.py (resolve_duplicate_models function)
- **Verification:** Final validation passed, 1 row removed (188 → 187 models)
- **Committed in:** 02624a4 (Task 1 commit)

---

**Total deviations:** 3 auto-fixed (1 blocking, 1 bug, 1 missing critical)
**Impact on plan:** All auto-fixes necessary for correct duplicate resolution. No scope creep. Enhanced the RESEARCH.md strategy to handle real-world data complexity beyond the simple example.

## Issues Encountered

**Column type mismatch between RESEARCH.md example and actual data:**
- RESEARCH.md showed using `Context Window` (str) column directly
- Phase 1 cleaning created `context_window` (Int64) column with parsed values
- Solution: Used `context_window` column instead, which provides better disambiguation (exact token counts vs "k"/"m" suffixes)

**Data complexity beyond RESEARCH.md example:**
- RESEARCH.md example assumed context_window alone would resolve duplicates
- Actual data had models with same name, same context window, but different Speed/Intelligence Index
- Solution: Added secondary intelligence_index disambiguation and tertiary unique() removal

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Dataset ready for Phase 2 statistical analysis:**
- `data/processed/ai_models_deduped.parquet` - 187 models with unique model_id
- No duplicate model names remain (validated)
- model_id column for accurate group-by operations
- Original Model column preserved for reference

**Recommendations for downstream analysis:**
- Use `model_id` for all group-by operations (not `Model`)
- Use Spearman correlation (non-parametric) for skewed distributions
- Filter to n=181 models with valid intelligence_index for IQ-specific analyses
- Consider log-transformation for context_window (extreme skewness: 9.63)
- 10 outliers flagged in Phase 1 (5.32%) - assess impact on correlations

**Known limitations:**
- 6 models lack intelligence_index scores (3.19%) - filled with -1 for disambiguation
- 1 true duplicate removed (Exaone 4.0 1.2B) - was data entry error
- External data enrichment failed (0% coverage) - temporal analysis not possible

**No blockers** - Plan 02-02 (Correlation Analysis) can proceed immediately.

---
*Phase: 02-statistical-analysis-domain-insights*
*Completed: 2026-01-18*
