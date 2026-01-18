---
phase: 01-data-pipeline
plan: 05a
type: execute
wave: 4
depends_on: [01-03b]
files_modified:
  - data/external/huggingface_models.parquet
  - src/enrich.py
  - scripts/06_enrich_external.py
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Web scraping utilities fetch model metadata from HuggingFace and provider sources"
    - "Scraping implements rate limiting and error handling for respectful data collection"
    - "Provenance is tracked with source_url, retrieved_at, and retrieved_by columns"
    - "Raw scraped data is saved to data/external/ for reproducibility"
  artifacts:
    - path: "data/external/huggingface_models.parquet"
      provides: "Scraped external data with provenance"
      format: "parquet"
    - path: "src/enrich.py"
      provides: "Web scraping utilities with rate limiting"
      exports: ["scrape_huggingface_models", "scrape_provider_announcements"]
      min_lines: 80
    - path: "scripts/06_enrich_external.py"
      provides: "Scraping execution script"
      exports: ["main"]
      min_lines: 50
  key_links:
    - from: "scripts/06_enrich_external.py"
      to: "src/enrich.py"
      via: "scraping function imports"
      pattern: "from src\\.enrich import.*scrape"
    - from: "src/enrich.py"
      to: "https://huggingface.co/open-llm-leaderboard"
      via: "web scraping with requests"
      pattern: "requests\\.get.*huggingface"
    - from: "scripts/06_enrich_external.py"
      to: "data/external/"
      via: "save scraped data"
      pattern: "sink_parquet.*data/external"
---

<objective>
Implement web scraping utilities and execute external data collection from HuggingFace and provider sources.

Purpose: Create reusable scraping functions with rate limiting and error handling, then execute them to fetch model metadata (release dates, benchmarks, provider info) with full provenance tracking for reproducibility.
Output: Scraped external data saved to data/external/ with provenance metadata.
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
  <name>Implement web scraping utilities</name>
  <files>src/enrich.py</files>
  <action>
    Create src/enrich.py with external data scraping utilities:

    Import requests, BeautifulSoup, datetime, time, polars as pl

    Define `scrape_huggingface_models() -> pl.DataFrame` function:
    - Set base_url = "https://huggingface.co/open-llm-leaderboard"
    - Initialize empty list models_data = []
    - Try-except block for error handling:
      * Send GET request with User-Agent header to avoid blocking
      * Use response.raise_for_status() to check for HTTP errors
      * Parse HTML with BeautifulSoup
      * Extract model information (selectors depend on actual page structure):
        - Model name from model name element
        - Release date from date element if available
        - Benchmark scores if available
        - Provider/organization if available
      * For each model found, append dict with:
        * model: str
        * release_date: str (or None)
        * benchmark_score: float (or None)
        * provider: str (or None)
        * source_url: str (base_url or specific model page)
        * retrieved_at: datetime.now().isoformat()
        * retrieved_by: "scrape_huggingface_models"
      * Add time.sleep(1) for rate limiting between requests
      * Handle pagination if present (loop through pages)
    - On exception: print error message, return empty pl.DataFrame()
    - Convert models_data list to pl.DataFrame
    - Return DataFrame

    Reference RESEARCH.md "Web Scraping for Model Release Dates" example
    Add comprehensive docstring explaining data source and limitations
    Add comments about rate limiting and respectful scraping practices
    Note that actual HTML selectors will need inspection and adjustment

    Define `scrape_provider_announcements() -> pl.DataFrame` function:
    - Similar structure to scrape_huggingface_models
    - Target provider blogs/news pages (OpenAI, Anthropic, Google, etc.)
    - Extract model announcements with dates
    - Add provenance columns (source_url, retrieved_at, retrieved_by)
    - Return DataFrame

    Note: According to CONTEXT.md, use automated collection where possible but accept best-effort coverage
  </action>
  <verify>
    `python -c "from src.enrich import scrape_huggingface_models, scrape_provider_announcements; print('Functions imported')"` confirms functions exist
  </verify>
  <done>
    Web scraping functions exist that can fetch external data from HuggingFace and provider sources with proper rate limiting, error handling, and provenance tracking
  </done>
</task>

<task type="auto">
  <name>Execute external data scraping</name>
  <files>scripts/06_enrich_external.py, data/external/huggingface_models.parquet</files>
  <action>
    Update scripts/06_enrich_external.py to execute web scraping:

    Import functions from src.enrich and src.utils

    Scraping pipeline:
    1. Create data/external/ directory if not exists
    2. Print "Starting external data collection..." to console
    3. Call scrape_huggingface_models():
       * Print progress messages (e.g., "Fetching HuggingFace leaderboard...")
       * Handle errors gracefully
       * Store result in huggingface_df variable
    4. If huggingface_df is not empty:
       * Save to data/external/huggingface_models.parquet
       * Print f"Retrieved {huggingface_df.shape[0]} models from HuggingFace"
    5. Optionally call scrape_provider_announcements():
       * Scrape from multiple provider sources
       * Save each to data/external/{provider}_announcements.parquet
       * Print progress for each source
    6. If all scraping fails (empty results):
       * Print warning: "No external data retrieved. Will proceed with base dataset only."
       * Print "Consider manual data entry for top 20 models."
    7. Combine all external data into single DataFrame if multiple sources
    8. Save combined external data to data/external/all_external_data.parquet

    Use verbose logging to show progress
    Implement rate limiting (1 second delay between requests)
    Handle HTTP errors, timeouts, and parsing errors gracefully
    Document any scraping issues in comments

    Reference CONTEXT.md: "best-effort coverage - document coverage rate and proceed regardless"
  </action>
  <verify>
    `python scripts/06_enrich_external.py` runs successfully (may produce warnings if scraping fails)
    `test -f data/external/huggingface_models.parquet` or `test -f data/external/all_external_data.parquet` confirms external data saved
    `ls data/external/*.parquet | wc -l` shows at least one external data file
  </verify>
  <done>
    External data scraping executes successfully, retrieving model metadata from HuggingFace and/or provider sources, with results saved to data/external/ directory
  </done>
</task>

</tasks>

<verification>
- [ ] src/enrich.py exists with web scraping functions
- [ ] scrape_huggingface_models function implements rate limiting with time.sleep
- [ ] scrape_provider_announcements function targets provider sources
- [ ] Both functions add provenance columns (source_url, retrieved_at, retrieved_by)
- [ ] scripts/06_enrich_external.py imports from src.enrich
- [ ] Running scripts/06_enrich_external.py completes without errors
- [ ] data/external/ directory contains scraped data parquet files
- [ ] Provenance metadata is tracked in all scraped data
</verification>

<success_criteria>
External data scraping utilities are implemented with proper error handling and rate limiting, scraping executes successfully fetching model metadata from external sources, and all scraped data includes provenance tracking for reproducibility.
</success_criteria>

<output>
After completion, create `.planning/phases/01-data-pipeline-&-quality-assessment**---load,-clean,-validate,-and-enrich-the-benchmark-dataset/01-05a-SUMMARY.md`
</output>
