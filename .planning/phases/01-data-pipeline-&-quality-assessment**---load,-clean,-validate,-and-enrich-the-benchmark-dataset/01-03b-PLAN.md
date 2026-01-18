---
phase: 01-data-pipeline
plan: 03b
type: execute
wave: 3
depends_on: [01-03a]
files_modified:
  - data/interim/02_cleaned.parquet
  - scripts/02_clean.py
  - reports/missing_values.md
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Cleaning pipeline executes all functions from src/clean.py sequentially"
    - "Price column is converted from messy strings ($4.81) to Float64 values"
    - "Intelligence Index is cleaned and validated within [0, 100] range"
    - "Missing values are analyzed and documented with handling strategy"
    - "Cleaned data is checkpointed for downstream analysis"
  artifacts:
    - path: "data/interim/02_cleaned.parquet"
      provides: "Cleaned dataset with proper data types"
      format: "parquet"
    - path: "scripts/02_clean.py"
      provides: "Cleaning pipeline execution script"
      exports: ["main"]
      min_lines: 50
    - path: "reports/missing_values.md"
      provides: "Missing value analysis report"
      format: "markdown"
  key_links:
    - from: "scripts/02_clean.py"
      to: "data/interim/01_loaded.parquet"
      via: "load checkpoint"
      pattern: "read_parquet.*01_loaded"
    - from: "scripts/02_clean.py"
      to: "src/clean.py"
      via: "cleaning function imports"
      pattern: "from src\\.clean import"
    - from: "scripts/02_clean.py"
      to: "data/interim/02_cleaned.parquet"
      via: "sink_parquet checkpoint"
      pattern: "sink_parquet.*02_cleaned"
---

<objective>
Execute the data cleaning pipeline to transform raw data into analysis-ready format.

Purpose: Run all cleaning functions from plan 03a sequentially, validate results, document missing value patterns, and create a cleaned checkpoint that serves as foundation for distribution analysis and enrichment.
Output: Cleaned dataset saved as parquet checkpoint and missing value analysis report.
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
@.planning/phases/01-data-pipeline-&-quality-assessment**---load,-clean,-validate,-and-enrich-the-benchmark-dataset/01-03a-PLAN.md
</context>

<tasks>

<task type="auto">
  <name>Execute cleaning pipeline and create checkpoint</name>
  <files>scripts/02_clean.py, data/interim/02_cleaned.parquet</files>
  <action>
    Update scripts/02_clean.py to execute full cleaning pipeline:

    Import functions from src.clean and src.utils
    Import polars as pl

    Pipeline steps:
    1. Load checkpoint from data/interim/01_loaded.parquet
    2. Apply clean_price_column() to create price_usd
    3. Apply clean_intelligence_index() to validate intelligence scores
    4. Analyze missing values with analyze_missing_values()
    5. Apply handle_missing_values() with default strategy (leave nulls)
    6. Collect LazyFrame to materialize cleaned data
    7. Validate cleaned data with Pandera schema from src.validate
    8. Save to data/interim/02_cleaned.parquet using sink_parquet
    9. Print cleaning summary:
       * Rows before/after cleaning
       * Price conversion success rate
       * Intelligence index validation results
       * Missing value counts by column
       * Any warnings or errors encountered

    Add verbose logging at each step using src.utils.setup_logging()
    Document all cleaning decisions in comments
    Handle errors gracefully: if conversion fails, flag row but continue

    Reference RESEARCH.md Pattern 2 "LazyFrame Pipeline with Checkpointing"
  </action>
  <verify>
    `python scripts/02_clean.py` runs successfully
    `test -f data/interim/02_cleaned.parquet` confirms checkpoint created
    `python -c "import polars as pl; df = pl.read_parquet('data/interim/02_cleaned.parquet'); print(df.columns)"` shows price_usd column exists
    `python -c "import polars as pl; df = pl.read_parquet('data/interim/02_cleaned.parquet'); print(df['price_usd'].dtype)"` shows Float64
  </verify>
  <done>
    Cleaning pipeline executes successfully, price column is converted to Float64, intelligence index is validated, missing values are analyzed, and cleaned data is checkpointed to parquet
  </done>
</task>

<task type="auto">
  <name>Generate missing value analysis report</name>
  <files>reports/missing_values.md</files>
  <action>
    Create reports/missing_values.md with missing value analysis:

    Include:
    - Summary table with columns:
      * Column name
      * Null count
      * Null percentage
      * Recommended handling strategy
    - Analysis of missing value patterns:
      * Are missing values random or clustered?
      * Do they correlate with specific creators or model types?
      * Which columns are critical vs optional for analysis?
    - Recommended handling strategy for each column
    - Impact assessment: How will missing values affect downstream analysis?
    - Decision log: Which strategy was chosen and why

    Use analyze_missing_values() output to populate table
    Add narrative interpretation of findings
    Include timestamp and generation metadata

    Reference DATA-05 requirement for missing value documentation
  </action>
  <verify>
    `test -f reports/missing_values.md` confirms report exists
    `grep -q "Column name" reports/missing_values.md` confirms table structure
  </verify>
  <done>
    Missing value analysis report exists with complete statistics, pattern analysis, and recommended handling strategies
  </done>
</task>

</tasks>

<verification>
- [ ] scripts/02_clean.py imports all functions from src.clean
- [ ] Running scripts/02_clean.py completes without errors
- [ ] data/interim/02_cleaned.parquet exists with price_usd as Float64
- [ ] Console output shows cleaning summary and missing value analysis
- [ ] reports/missing_values.md exists with complete analysis table
- [ ] All cleaning decisions are documented in code comments
</verification>

<success_criteria>
Data cleaning pipeline executes successfully transforming messy values into clean types, missing values are analyzed and documented, and pristine data is checkpointed as foundation for distribution analysis and enrichment.
</success_criteria>

<output>
After completion, create `.planning/phases/01-data-pipeline-&-quality-assessment**---load,-clean,-validate,-and-enrich-the-benchmark-dataset/01-03b-SUMMARY.md`
</output>
