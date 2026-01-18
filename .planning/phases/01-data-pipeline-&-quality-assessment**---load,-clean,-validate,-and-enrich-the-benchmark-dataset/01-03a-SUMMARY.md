---
phase: 01-data-pipeline
plan: 03a
subsystem: data-cleaning
tags: [polars, lazyframe, data-cleaning, type-conversion, missing-values]

# Dependency graph
requires:
  - phase: 01-data-pipeline
    provides: Data loading functions with lenient schema (src/load.py)
provides:
  - Data cleaning utilities for price strings, intelligence index, and missing values
  - Reusable functions with type hints and comprehensive docstrings
  - Foundation for cleaning pipeline execution in plan 03b
affects: [01-03b-cleaning-pipeline, 01-04-outlier-detection, 01-05-quality-report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - LazyFrame string manipulation with str.strip_chars() and str.replace()
    - Type casting with strict=False for graceful error handling
    - Configurable missing value strategies via dictionary parameter
    - Range validation with boolean flag columns

key-files:
  created: [src/clean.py]
  modified: []

key-decisions:
  - "Polars API: str.strip() → str.strip_chars() for string whitespace removal"
  - "Polars API: str.replace() requires literal=True for non-regex replacements"
  - "Error handling: Use strict=False casting to preserve null values instead of failing"
  - "Validation approach: Add flag columns instead of dropping invalid rows"

patterns-established:
  - "Cleaning function pattern: Accept LazyFrame, return LazyFrame with new columns"
  - "Validation pattern: Add *_valid or *_out_of_range flag columns"
  - "Missing value pattern: Return statistics dict, print summary, leave nulls by default"

# Metrics
duration: 4min
completed: 2026-01-18
---

# Phase 01: Data Pipeline & Quality Assessment - Plan 03a Summary

**Data cleaning utilities for price strings ($4.81 → Float64), intelligence index validation [0,100], and configurable missing value strategies using Polars LazyFrame API**

## Performance

- **Duration:** 4 min
- **Started:** 2026-01-18T23:00:14Z
- **Completed:** 2026-01-18T23:04:56Z
- **Tasks:** 3 (all auto, no checkpoints)
- **Files modified:** 1 created (src/clean.py)

## Accomplishments

- **Price column cleaning**: Transforms messy price strings ("$4.81 ") to Float64 using string manipulation chain (strip_chars, replace dollar sign, remove spaces, cast to Float64)
- **Intelligence index validation**: Extracts numeric values from strings, validates range [0, 100], adds validation flag column for out-of-range detection
- **Missing value analysis**: Comprehensive null statistics (count, percentage, presence) with console logging for data quality assessment
- **Missing value handling**: Configurable strategies (drop, forward_fill, backward_fill, mean, median, zero, leave) via dictionary parameter

## Task Commits

**Note:** Work was completed as part of plan 01-03b execution. The src/clean.py file was created and committed in `d094aa7 feat(01-03b): execute data cleaning pipeline and create checkpoint`.

Plan 01-03a requirements verification completed:
- clean_price_column function exists and handles "$4.81 " format correctly ✓
- clean_intelligence_index function validates range [0, 100] ✓
- analyze_missing_values function calculates null percentages ✓
- handle_missing_values function supports drop, fill, and leave strategies ✓
- All functions have comprehensive docstrings and type hints ✓
- Functions can be imported without errors ✓

## Files Created/Modified

- `src/clean.py` (363 lines) - Data cleaning utilities with 4 functions:
  - `clean_price_column(lf: pl.LazyFrame) -> pl.LazyFrame` - Extract numeric values from price strings
  - `clean_intelligence_index(lf: pl.LazyFrame) -> pl.LazyFrame` - Validate intelligence scores [0, 100]
  - `analyze_missing_values(df: pl.DataFrame) -> Dict[str, Dict[str, float]]` - Calculate null statistics
  - `handle_missing_values(lf: pl.LazyFrame, strategy: Optional[Dict[str, str]] = None) -> pl.LazyFrame` - Apply missing value strategies

## Decisions Made

**Polars API Compatibility:**
- Used `str.strip_chars()` instead of `str.strip()` for whitespace removal (Polars 1.x API)
- Added `literal=True` parameter to `str.replace()` for non-regex string replacements
- Used `strict=False` in `cast()` operations to preserve null values instead of raising errors

**Error Handling Strategy:**
- Flag problematic rows with boolean columns instead of dropping them immediately
- Preserve original columns for traceability and debugging
- Return null values for unconvertible data rather than failing the pipeline

**Missing Value Approach:**
- Default strategy is `None` (leave nulls in place) per CONTEXT.md guidance
- Support multiple strategies per column via dictionary configuration
- Document null patterns with comprehensive statistics (count, percentage, presence)

## Deviations from Plan

### Implementation Status

**Work completed as part of plan 01-03b:**
- The src/clean.py file was created during execution of plan 01-03b (commit d094aa7)
- This was a deviation from the original plan sequence (01-03a should have created src/clean.py before 01-03b executed the pipeline)
- All requirements for plan 01-03a are met and verified

**No code deviations:** Implementation follows plan specifications exactly:
- All 4 required functions implemented with correct signatures
- Comprehensive docstrings with examples and notes
- Type hints for all parameters and return values
- Polars LazyFrame API used throughout
- String manipulation follows pattern: strip_chars → replace → cast

## Issues Encountered

**Polars API compatibility during initial implementation:**
- Issue: `str.strip()` method doesn't exist in Polars 1.x
- Resolution: Updated to use `str.strip_chars()` for whitespace removal
- Issue: `str.replace()` requires explicit `literal=True` for non-regex replacements
- Resolution: Added `literal=True` parameter to all `str.replace()` calls
- Impact: Minor API corrections, no functional changes

## User Setup Required

None - no external service configuration required. All functionality uses existing Polars library and project dependencies.

## Next Phase Readiness

**Ready for plan 01-03b (Cleaning Pipeline Execution):**
- All cleaning functions implemented and tested
- Functions can be imported and executed by scripts/02_clean.py
- Error handling preserves data integrity (nulls instead of failures)
- Comprehensive documentation for pipeline integration

**Ready for downstream analysis:**
- Price column cleaned to Float64 for numerical analysis
- Intelligence index validated for statistical calculations
- Missing value patterns documented for quality assessment
- Configurable strategies available for future data handling

**No blockers or concerns:**
- All functions verified with test data
- Type hints enable IDE autocomplete and error checking
- Docstrings provide usage examples and notes
- LazyFrame API ensures efficient processing

---
*Phase: 01-data-pipeline*
*Plan: 03a*
*Completed: 2026-01-18*
