---
phase: 01-data-pipeline
plan: 05b
type: execute
wave: 5
depends_on: [01-05a]
files_modified:
  - data/processed/ai_models_enriched.parquet
  - src/enrich.py
  - scripts/06_enrich_external.py
  - reports/enrichment_coverage.md
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Data enrichment utilities implement left join to preserve all models"
    - "Derived columns are created for analysis (price per IQ, model tier, etc.)"
    - "Coverage statistics are calculated for all enrichment columns"
    - "Dataset is enriched and saved as final analysis-ready file"
    - "Coverage report documents match rates and recommendations"
  artifacts:
    - path: "data/processed/ai_models_enriched.parquet"
      provides: "Final enriched dataset for analysis"
      format: "parquet"
    - path: "src/enrich.py"
      provides: "Data enrichment utilities"
      exports: ["enrich_with_external_data", "add_derived_columns", "calculate_enrichment_coverage"]
      min_lines: 80
    - path: "reports/enrichment_coverage.md"
      provides: "Enrichment coverage analysis report"
      format: "markdown"
  key_links:
    - from: "scripts/06_enrich_external.py"
      to: "data/interim/02_cleaned.parquet"
      via: "load cleaned dataset"
      pattern: "read_parquet.*02_cleaned"
    - from: "scripts/06_enrich_external.py"
      to: "data/external/"
      via: "load scraped external data"
      pattern: "read_parquet.*data/external"
    - from: "scripts/06_enrich_external.py"
      to: "src/enrich.py"
      via: "enrichment function calls"
      pattern: "from src\\.enrich import.*(enrich|derive|coverage)"
    - from: "scripts/06_enrich_external.py"
      to: "data/processed/"
      via: "save final enriched dataset"
      pattern: "sink_parquet.*ai_models_enriched"
---

<objective>
Enrich the dataset with external data and derived analysis columns.

Purpose: Combine cleaned base data with scraped external metadata via left join (preserving all models), create derived analysis metrics, document enrichment coverage, and produce the final analysis-ready dataset for Phase 2.
Output: Enriched dataset with external metadata and derived columns, saved to data/processed/ with coverage report.
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
@.planning/phases/01-data-pipeline-&-quality-assessment**---load,-clean,-validate,-and-enrich-the-benchmark-dataset/01-05a-PLAN.md
</context>

<tasks>

<task type="auto">
  <name>Implement data enrichment utilities</name>
  <files>src/enrich.py</files>
  <action>
    Add to src/enrich.py:

    Define `enrich_with_external_data(base_df: pl.DataFrame, external_df: pl.DataFrame, join_key: str = "model") -> pl.DataFrame` function:
    - Validate join_key exists in both DataFrames
    - Perform left join: base_df.join(external_df, on=join_key, how="left")
    - Left join ensures all original models are kept (nulls if no match)
    - Add enrichment metadata columns:
      * enriched_at: datetime.now()
      * enrichment_source: external_df source column or "external_scraping"
      * coverage_rate: (non-null enrichment rows / total rows) calculated as percentage
    - Return enriched DataFrame

    Define `add_derived_columns(df: pl.DataFrame) -> pl.DataFrame` function:
    - Create derived metrics for analysis:
      * price_per_intelligence_point: price_usd / intelligence_index (handle division by zero)
      * speed_intelligence_ratio: speed / intelligence_index
      * model_tier: extract from model name using regex (xhigh, high, medium, low, mini)
      * log_context_window: log10(context_window) for better visualization
      * price_per_1k_tokens: price_usd / 1000 (more intuitive scale)
    - Handle edge cases (null values, division by zero)
    - Return DataFrame with new columns

    Define `calculate_enrichment_coverage(df: pl.DataFrame, enrichment_columns: list[str]) -> dict` function:
    - For each enrichment column, calculate:
      * non_null_count: number of rows with data
      * null_count: number of rows with nulls
      * coverage_percentage: (non_null_count / total_rows) * 100
    - Return dict with coverage statistics
    - Print summary to console

    Add comprehensive docstrings for each function
    Add type hints for all parameters and returns
  </action>
  <verify>
    `python -c "from src.enrich import enrich_with_external_data, add_derived_columns, calculate_enrichment_coverage; print('Functions imported')"` confirms functions exist
  </verify>
  <done>
    Enrichment utilities exist that join external data via left join, add derived analysis columns, and calculate coverage statistics for documentation
  </done>
</task>

<task type="auto">
  <name>Execute enrichment pipeline and create final dataset</name>
  <files>data/processed/ai_models_enriched.parquet</files>
  <action>
    Add to scripts/06_enrich_external.py after scraping:

    Enrichment pipeline:
    1. Load cleaned dataset from data/interim/02_cleaned.parquet
    2. Load external data from data/external/ (if exists)
    3. If external data exists:
       * Standardize model names (handle case sensitivity, special characters)
       * Call enrich_with_external_data() with left join on "Model" column
       * Print "Enriched dataset with external data"
    4. If external data doesn't exist or is empty:
       * Print "Proceeding without external enrichment (coverage would be 0%)"
       * Continue with base dataset only
    5. Call add_derived_columns() to create analysis metrics
    6. Call calculate_enrichment_coverage() to document coverage rate
    7. Print enrichment summary:
       * Coverage percentage for each enrichment column
       * Total rows before and after enrichment (should be same - left join)
       * Number of models with release dates, benchmark scores, etc.
    8. Create data/processed/ directory if not exists
    9. Save final enriched dataset to data/processed/ai_models_enriched.parquet
    10. Print final summary:
        * "Final enriched dataset saved to data/processed/ai_models_enriched.parquet"
        * "Shape: {rows} x {columns}"
        * "Ready for Phase 2: Statistical Analysis"

    Use verbose logging throughout
    Document any data quality issues discovered during enrichment
    Handle model name mismatches gracefully (some models won't have external data)

    Reference RESEARCH.md "Pitfall 5: External Data Enrichment Provenance Loss" for proper metadata tracking
  </action>
  <verify>
    `python scripts/06_enrich_external.py` runs successfully
    `test -f data/processed/ai_models_enriched.parquet` confirms final dataset exists
    `python -c "import polars as pl; df = pl.read_parquet('data/processed/ai_models_enriched.parquet'); print(df.shape)"` shows shape (188, >=10) with enrichment columns
    `python -c "import polars as pl; df = pl.read_parquet('data/processed/ai_models_enriched.parquet'); print(df.columns)"` shows derived columns like price_per_intelligence_point
  </verify>
  <done>
    Enrichment pipeline completes successfully, external data is joined with provenance tracking, derived columns are created, coverage is documented, and final enriched dataset is saved to data/processed/
  </done>
</task>

<task type="auto">
  <name>Generate enrichment coverage report</name>
  <files>reports/enrichment_coverage.md</files>
  <action>
    Create reports/enrichment_coverage.md with enrichment analysis:

    Include:
    - Data sources used:
      * HuggingFace Open LLM Leaderboard
      * Provider announcement pages (list which ones)
      * Any other sources
    - Coverage statistics table:
      * Enrichment column
      * Rows with data
      * Rows with nulls
      * Coverage percentage
    - Model name matching analysis:
      * How many models matched external sources
      * Common mismatch patterns (case, spacing, abbreviations)
      * Examples of unmatched models
    - Data quality assessment:
      * Reliability of each external source
      * Known issues or limitations
    - Recommendations:
      * Manual data entry opportunities
      * Additional sources to consider
      * Update schedule (how often to refresh)

    Use calculate_enrichment_coverage() output to populate table
    Add narrative interpretation of findings
    Include timestamp and generation metadata

    Reference DATA-08 requirement for external data enrichment documentation
  </action>
  <verify>
    `test -f reports/enrichment_coverage.md` confirms report exists
    `grep -q "Coverage" reports/enrichment_coverage.md` confirms coverage analysis
  </verify>
  <done>
    Enrichment coverage report exists documenting data sources, match rates, coverage percentages, and recommendations for improvement
  </done>
</task>

</tasks>

<verification>
- [ ] src/enrich.py exists with all enrichment functions
- [ ] enrich_with_external_data function performs left join with provenance tracking
- [ ] add_derived_columns function creates analysis metrics
- [ ] calculate_enrichment_coverage function calculates coverage statistics
- [ ] scripts/06_enrich_external.py executes complete pipeline
- [ ] data/processed/ai_models_enriched.parquet exists with enrichment columns
- [ ] Final dataset has derived columns (price_per_intelligence_point, model_tier, etc.)
- [ ] reports/enrichment_coverage.md exists with coverage statistics
</verification>

<success_criteria>
Dataset is enriched with external data via left join (preserving all models), derived analysis columns are created, coverage is documented, final enriched dataset is saved for Phase 2, and coverage report provides actionable insights.
</success_criteria>

<output>
After completion, create `.planning/phases/01-data-pipeline-&-quality-assessment**---load,-clean,-validate,-and-enrich-the-benchmark-dataset/01-05b-SUMMARY.md`
</output>
