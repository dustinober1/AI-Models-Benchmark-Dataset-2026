---
phase: 01-data-pipeline
plan: 03a
type: execute
wave: 3
depends_on: [01-02]
files_modified:
  - src/clean.py
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Price column cleaning function handles messy strings ($4.81) and converts to Float64"
    - "Intelligence index cleaning function validates range [0, 100] and extracts numeric values"
    - "Missing value analysis function identifies and documents all null patterns"
    - "Missing value handling function supports multiple strategies (drop, fill, leave)"
  artifacts:
    - path: "src/clean.py"
      provides: "Data cleaning utilities with type hints and docstrings"
      exports: ["clean_price_column", "clean_intelligence_index", "analyze_missing_values", "handle_missing_values"]
      min_lines: 100
  key_links:
    - from: "src/clean.py"
      to: "polars"
      via: "string manipulation and type casting"
      pattern: "str\\.(strip|replace|extract)"
    - from: "src/clean.py"
      to: "scripts/02_clean.py"
      via: "function imports in plan 03b"
      pattern: "from src\\.clean import"
---

<objective>
Implement data cleaning functions for transforming messy values into analysis-ready format.

Purpose: Create reusable cleaning utilities that handle messy price strings, validate intelligence scores, analyze missing values, and apply configurable handling strategies - all as foundation for the cleaning pipeline execution.
Output: src/clean.py with comprehensive cleaning functions that can be imported and executed by plan 03b.
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/01-data-pipeline-&-quality-assessment**---load,-clean,-validate,-and-enrich-the-benchmark-dataset/01-CONTEXT.md
@.planning/phases/01-data-pipeline-&-quality-assessment**---load,-clean,-validate,-and-enrich-the-benchmark-dataset/01-RESEARCH.md
</context>

<tasks>

<task type="auto">
  <name>Implement price column cleaning function</name>
  <files>src/clean.py</files>
  <action>
    Create src/clean.py with data cleaning utilities:

    Import polars as pl

    Define `clean_price_column(lf: pl.LazyFrame) -> pl.LazyFrame` function:
    - Extract "Price (Blended USD/1M Tokens)" column
    - Apply transformations in sequence:
      * .str.strip() - remove leading/trailing whitespace
      * .str.replace("$", "") - remove dollar sign
      * .str.replace(" ", "") - remove spaces
      * .str.replace(",", "") - remove commas (if present in large values)
      * .cast(pl.Float64) - convert to numeric
      * .alias("price_usd") - rename to clean column name
    - Return LazyFrame with new price_usd column
    - Handle any conversion errors by flagging rows (do not drop yet)

    Reference RESEARCH.md Pattern 2 "Clean price column" example
    Add detailed comments explaining each transformation step
    Add docstring with input/output examples ("$4.81 " -> 4.81)

    Note: According to CONTEXT.md, use Claude's discretion for handling conversion errors - flag problematic rows but continue processing
  </action>
  <verify>
    `python -c "from src.clean import clean_price_column; print('Function imported')"` confirms function exists
  </verify>
  <done>
    clean_price_column function exists that extracts numeric values from messy price strings and creates Float64 price_usd column
  </done>
</task>

<task type="auto">
  <name>Implement intelligence index cleaning function</name>
  <files>src/clean.py</files>
  <action>
    Add to src/clean.py:

    Define `clean_intelligence_index(lf: pl.LazyFrame) -> pl.LazyFrame` function:
    - Extract "Intelligence Index" column (currently Int64)
    - Check for any non-numeric values or suffixes
    - If data is clean (already Int64), pass through with validation
    - If contains strings like "41\nE", extract numeric part using regex:
      * .str.extract(r"^(\d+)") - extract leading digits
      * .cast(pl.Int64) - convert to integer
    - Validate values are in range [0, 100]
    - Flag any values outside this range
    - Return LazyFrame with validated intelligence_index column

    Add comments explaining why this cleaning step exists (handle quoted multi-line values)
    Add docstring with examples of problematic input and cleaned output
  </action>
  <verify>
    `python -c "from src.clean import clean_intelligence_index; print('Function imported')"` confirms function exists
  </verify>
  <done>
    clean_intelligence_index function exists that validates and extracts numeric intelligence scores, flagging values outside valid range
  </done>
</task>

<task type="auto">
  <name>Implement missing value analysis and handling functions</name>
  <files>src/clean.py</files>
  <action>
    Add to src/clean.py:

    Define `analyze_missing_values(df: pl.DataFrame) -> dict` function:
    - For each column, calculate:
      * null_count: number of null values
      * null_percentage: (null_count / total_rows) * 100
    - Return dict with column names as keys and missing stats as values
    - Print summary to console with verbose logging
    - Identify columns with any missing values

    Define `handle_missing_values(lf: pl.LazyFrame, strategy: dict = None) -> pl.LazyFrame` function:
    - Accept strategy dict mapping column names to handling methods:
      * "drop" - remove rows with nulls in this column
      * "forward_fill" - fill with previous value
      * "backward_fill" - fill with next value
      * "mean" - fill with column mean (numeric only)
      * "median" - fill with column median (numeric only)
      * "zero" - fill with 0 (numeric only)
      * None - leave as null
    - Default strategy: Leave all nulls in place (will be analyzed in plan 04)
    - Document which columns have nulls and recommended handling
    - Return LazyFrame with nulls handled according to strategy

    Reference DATA-05 requirement for missing value analysis
    Add comprehensive docstrings explaining missing value strategies

    Note: According to CONTEXT.md, continue with nulls in enrichment columns - do not drop rows with missing enrichment data
  </action>
  <verify>
    `python -c "from src.clean import analyze_missing_values, handle_missing_values; print('Functions imported')"` confirms functions exist
  </verify>
  <done>
    Missing value analysis function identifies all null values, handling function applies strategies, and both document null patterns for reporting
  </done>
</task>

</tasks>

<verification>
- [ ] src/clean.py exists with all 4 cleaning functions
- [ ] clean_price_column function handles "$4.81 " format correctly
- [ ] clean_intelligence_index function validates range [0, 100]
- [ ] analyze_missing_values function calculates null percentages
- [ ] handle_missing_values function supports drop, fill, and leave strategies
- [ ] All functions have comprehensive docstrings and type hints
- [ ] Functions can be imported without errors
</verification>

<success_criteria>
All data cleaning functions are implemented in src/clean.py with proper error handling, comprehensive documentation, and support for the messy data patterns identified during loading (price strings, intelligence suffixes, missing values).
</success_criteria>

<output>
After completion, create `.planning/phases/01-data-pipeline-&-quality-assessment**---load,-clean,-validate,-and-enrich-the-benchmark-dataset/01-03a-SUMMARY.md`
</output>
