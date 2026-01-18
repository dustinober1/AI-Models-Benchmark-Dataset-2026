"""
External data enrichment for AI models benchmark dataset.

This script scrapes and enriches the dataset with external data sources such as
model release dates, provider announcements, and market events from HuggingFace
and provider blogs/news sources.

Functions
---------
main()
    Execute external data scraping pipeline.

main_enrich()
    Execute enrichment pipeline (load, join external data, add derived columns).

Notes
-----
- Implements rate limiting (1 second delay) for respectful scraping
- Tracks provenance (source_url, retrieved_at, retrieved_by) for reproducibility
- Handles failures gracefully - continues with base dataset if scraping fails
- Saves scraped data to data/external/ directory
- Enriches base dataset with external data and derived analysis columns
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from src.enrich import (
    scrape_huggingface_models,
    scrape_provider_announcements,
    enrich_with_external_data,
    add_derived_columns,
    calculate_enrichment_coverage
)
from src.utils import setup_logging


def main() -> pl.DataFrame:
    """
    Execute external data scraping pipeline.

    Scrapes model metadata from HuggingFace Open LLM Leaderboard and provider
    announcement sources. Saves all scraped data with provenance tracking to
    data/external/ directory.

    Returns
    -------
    pl.DataFrame
        Combined external data from all sources.

    Examples
    --------
    >>> external_data = main()
    >>> print(f"Retrieved {external_data.height} external records")

    Notes
    -----
    Pipeline steps:
    1. Create data/external/ directory if not exists
    2. Scrape HuggingFace Open LLM Leaderboard
    3. Scrape provider announcements (OpenAI, Anthropic, Google, etc.)
    4. Save each source to separate parquet file
    5. Combine all sources into single DataFrame
    6. Save combined data to all_external_data.parquet

    Error handling:
    - Scraping failures are logged but don't stop pipeline
    - Empty DataFrames are saved if scraping fails
    - Warning printed if no external data retrieved
    """
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("=" * 60)
    logger.info("Starting external data collection")
    logger.info("=" * 60)

    # Create data/external/ directory if not exists
    external_dir = Path("data/external")
    external_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"External data directory: {external_dir.absolute()}")

    # Track all external data
    all_external_data = []

    # Task 1: Scrape HuggingFace Open LLM Leaderboard
    logger.info("\n" + "=" * 60)
    logger.info("Task 1: Fetching HuggingFace Open LLM Leaderboard...")
    logger.info("=" * 60)

    try:
        huggingface_df = scrape_huggingface_models()

        if huggingface_df.height > 0:
            # Save to data/external/
            huggingface_path = external_dir / "huggingface_models.parquet"
            huggingface_df.write_parquet(huggingface_path)
            logger.info(f"Saved {huggingface_df.height} models to {huggingface_path}")
            all_external_data.append(huggingface_df)
        else:
            logger.warning("HuggingFace scraping returned empty results")
    except Exception as e:
        logger.error(f"ERROR: Failed to scrape HuggingFace: {e}")

    # Task 2: Scrape provider announcements
    logger.info("\n" + "=" * 60)
    logger.info("Task 2: Fetching provider announcements...")
    logger.info("=" * 60)

    try:
        provider_df = scrape_provider_announcements()

        if provider_df.height > 0:
            # Save to data/external/
            provider_path = external_dir / "provider_announcements.parquet"
            provider_df.write_parquet(provider_path)
            logger.info(f"Saved {provider_df.height} announcements to {provider_path}")
            all_external_data.append(provider_df)

            # Print summary by provider
            if "provider" in provider_df.columns:
                provider_summary = provider_df.group_by("provider").count()
                logger.info(f"Announcements by provider:\n{provider_summary}")
        else:
            logger.warning("Provider announcement scraping returned empty results")
    except Exception as e:
        logger.error(f"ERROR: Failed to scrape provider announcements: {e}")

    # Task 3: Combine and save all external data
    logger.info("\n" + "=" * 60)
    logger.info("Task 3: Combining external data sources...")
    logger.info("=" * 60)

    if all_external_data:
        # Combine all DataFrames vertically
        combined_df = pl.concat(all_external_data, how="diagonal")

        # Save combined data
        combined_path = external_dir / "all_external_data.parquet"
        combined_df.write_parquet(combined_path)
        logger.info(f"Saved combined data ({combined_df.height} rows) to {combined_path}")
    else:
        # Create empty combined DataFrame with expected schema
        empty_schema = {
            "model": pl.Utf8,
            "release_date": pl.Utf8,
            "benchmark_score": pl.Float64,
            "provider": pl.Utf8,
            "announcement_title": pl.Utf8,
            "source_url": pl.Utf8,
            "retrieved_at": pl.Utf8,
            "retrieved_by": pl.Utf8
        }
        combined_df = pl.DataFrame(schema=empty_schema)

        # Save empty DataFrame
        combined_path = external_dir / "all_external_data.parquet"
        combined_df.write_parquet(combined_path)
        logger.warning(f"No external data retrieved from any source")
        logger.warning(f"Saved empty schema to {combined_path}")
        logger.warning("Will proceed with base dataset only")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("External data collection summary")
    logger.info("=" * 60)
    logger.info(f"Total external records retrieved: {combined_df.height}")
    logger.info(f"Data sources: {len(all_external_data)}")

    # List all files in data/external/
    external_files = list(external_dir.glob("*.parquet"))
    logger.info(f"\nFiles in {external_dir}:")
    for file in external_files:
        file_size = file.stat().st_size / 1024  # KB
        logger.info(f"  - {file.name} ({file_size:.1f} KB)")

    # Coverage analysis
    if combined_df.height > 0:
        logger.info(f"\nExternal data coverage:")
        if "model" in combined_df.columns:
            unique_models = combined_df.select(pl.col("model").drop_nulls()).height
            logger.info(f"  - Unique models: {unique_models}")
        if "release_date" in combined_df.columns:
            models_with_dates = combined_df.select(pl.col("release_date").drop_nulls()).height
            logger.info(f"  - Models with release dates: {models_with_dates}")
        if "provider" in combined_df.columns:
            unique_providers = combined_df.select(pl.col("provider").drop_nulls()).unique().height
            logger.info(f"  - Unique providers: {unique_providers}")

    logger.info("\n" + "=" * 60)
    logger.info("External data collection completed")
    logger.info("=" * 60)

    return combined_df


def main_enrich() -> pl.DataFrame:
    """
    Execute enrichment pipeline to create final analysis-ready dataset.

    Loads cleaned base dataset, joins with external data (if available),
    adds derived analysis columns, calculates coverage statistics, and
    saves the final enriched dataset for Phase 2.

    Returns
    -------
    pl.DataFrame
        Final enriched dataset with all columns ready for analysis.

    Examples
    --------
    >>> enriched_df = main_enrich()
    >>> print(f"Final dataset: {enriched_df.shape}")
    >>> print(enriched_df.columns)

    Notes
    -----
    Pipeline steps:
    1. Load cleaned dataset from data/interim/02_cleaned.parquet
    2. Load external data from data/external/ (if exists and non-empty)
    3. Standardize model names (handle case sensitivity)
    4. Enrich with external data via left join on "Model" column
    5. Add derived analysis columns (price_per_intelligence_point, etc.)
    6. Calculate enrichment coverage statistics
    7. Save final dataset to data/processed/ai_models_enriched.parquet

    Error handling:
    - Continues with base dataset if external data is empty/missing
    - Handles model name mismatches gracefully (nulls in enrichment columns)
    - Logs all enrichment steps for reproducibility

    References
    ----------
    DATA-08: External data enrichment with provenance tracking
    RESEARCH.md "Pitfall 5: External Data Enrichment Provenance Loss"
    """
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("=" * 60)
    logger.info("Starting enrichment pipeline")
    logger.info("=" * 60)

    # Step 1: Load cleaned dataset
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: Loading cleaned dataset...")
    logger.info("=" * 60)

    cleaned_path = Path("data/interim/02_cleaned.parquet")
    if not cleaned_path.exists():
        logger.error(f"ERROR: Cleaned dataset not found at {cleaned_path}")
        logger.error("Please run scripts/02_clean.py first")
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path}")

    base_df = pl.read_parquet(cleaned_path)
    logger.info(f"Loaded cleaned dataset: {base_df.shape} rows x columns")
    logger.info(f"Columns: {base_df.columns}")

    # Step 2: Load external data (if exists)
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Loading external data...")
    logger.info("=" * 60)

    external_path = Path("data/external/all_external_data.parquet")
    external_df = None

    if external_path.exists():
        external_df = pl.read_parquet(external_path)
        logger.info(f"Loaded external data: {external_df.shape} rows x columns")

        # Check if external data has any content
        if external_df.height == 0:
            logger.warning("External data is empty (0 rows)")
            logger.warning("Proceeding without external enrichment")
            external_df = None
        else:
            # Check if model column has any non-null values
            if "model" in external_df.columns:
                non_null_models = external_df.select(pl.col("model").drop_nulls()).height
                if non_null_models == 0:
                    logger.warning("External data has no model names (all nulls in 'model' column)")
                    logger.warning("This means model name extraction failed during scraping")
                    logger.warning("Proceeding without external enrichment")
                    external_df = None
                else:
                    logger.info(f"External data has {non_null_models} models with names")
            else:
                logger.warning("External data missing 'model' column")
                logger.warning("Proceeding without external enrichment")
                external_df = None
    else:
        logger.warning(f"External data not found at {external_path}")
        logger.warning("Proceeding without external enrichment")

    # Step 3: Enrich with external data (if available)
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Enriching dataset...")
    logger.info("=" * 60)

    if external_df is not None:
        # Standardize model names for better matching
        # Convert to lowercase for case-insensitive join
        base_df_with_key = base_df.with_columns([
            pl.col("Model").alias("model_original")
        ])

        external_df_with_key = external_df.with_columns([
            pl.col("model").str.strip().str.to_ascii().alias("model")
        ])

        # Perform left join
        try:
            enriched_df = enrich_with_external_data(
                base_df_with_key,
                external_df_with_key,
                join_key="Model"
            )
            logger.info(f"Enriched dataset: {enriched_df.shape}")
            logger.info(f"Coverage rate: {enriched_df.select(pl.col('coverage_rate').drop_nulls()).item():.2f}%")
        except ValueError as e:
            logger.error(f"ERROR: Enrichment failed: {e}")
            logger.warning("Proceeding with base dataset only")
            enriched_df = base_df.clone()
    else:
        logger.info("Proceeding without external enrichment (no valid external data)")
        enriched_df = base_df.clone()

    # Step 4: Add derived columns
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Adding derived analysis columns...")
    logger.info("=" * 60)

    enriched_df = add_derived_columns(enriched_df)
    logger.info(f"Added derived columns: {enriched_df.shape}")

    # List new derived columns
    derived_cols = [
        "price_per_intelligence_point",
        "speed_intelligence_ratio",
        "model_tier",
        "log_context_window",
        "price_per_1k_tokens"
    ]
    new_cols = [col for col in derived_cols if col in enriched_df.columns]
    logger.info(f"New columns: {new_cols}")

    # Step 5: Calculate enrichment coverage
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Calculating enrichment coverage...")
    logger.info("=" * 60)

    # Identify enrichment columns (external + derived)
    enrichment_cols = []
    if external_df is not None:
        external_cols = [col for col in external_df.columns if col != "Model"]
        enrichment_cols.extend(external_cols)
    enrichment_cols.extend(new_cols)

    # Add metadata columns
    metadata_cols = ["enriched_at", "enrichment_source", "coverage_rate"]
    enrichment_cols.extend([col for col in metadata_cols if col in enriched_df.columns])

    coverage_stats = calculate_enrichment_coverage(enriched_df, enrichment_cols)

    # Step 6: Save final enriched dataset
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Saving final enriched dataset...")
    logger.info("=" * 60)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "ai_models_enriched.parquet"
    enriched_df.write_parquet(output_path)
    logger.info(f"Saved final enriched dataset to {output_path}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Enrichment pipeline completed")
    logger.info("=" * 60)
    logger.info(f"Final dataset shape: {enriched_df.shape}")
    logger.info(f"Output file: {output_path.absolute()}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Count models by tier
    if "model_tier" in enriched_df.columns:
        tier_counts = enriched_df.group_by("model_tier").count().sort("count", descending=True)
        logger.info("\nModels by tier:")
        for row in tier_counts.iter_rows(named=True):
            logger.info(f"  - {row['model_tier']}: {row['count']} models")

    # Count columns by type
    logger.info("\nColumn summary:")
    logger.info(f"  - Total columns: {enriched_df.width}")
    logger.info(f"  - Original columns: {base_df.width}")
    logger.info(f"  - External columns: {len([c for c in enrichment_cols if c in external_df.columns]) if external_df is not None else 0}")
    logger.info(f"  - Derived columns: {len(new_cols)}")

    logger.info("\n" + "=" * 60)
    logger.info("Dataset ready for Phase 2: Statistical Analysis")
    logger.info("=" * 60)

    return enriched_df


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--enrich":
        # Run enrichment pipeline only
        enriched_data = main_enrich()

        # Print final summary
        print("\n" + "=" * 60)
        print("ENRICHMENT COMPLETE")
        print("=" * 60)
        print(f"Final dataset: {enriched_data.shape}")
        print(f"Saved to: data/processed/ai_models_enriched.parquet")
        print("\nReady for Phase 2: Statistical Analysis")
    else:
        # Run scraping pipeline (original behavior)
        external_data = main()

        # Print final summary
        print("\n" + "=" * 60)
        print("SCRAPING COMPLETE")
        print("=" * 60)
        print(f"Total external records: {external_data.height}")
        print(f"Columns: {external_data.columns}")

        if external_data.height > 0:
            print("\nSample data (first 5 rows):")
            print(external_data.head(5))
            print("\nNow run with --enrich flag to create final dataset:")
            print("  poetry run python scripts/06_enrich_external.py --enrich")
        else:
            print("\nNo external data retrieved.")
            print("Note: Web scraping may have failed due to:")
            print("  - Network connectivity issues")
            print("  - Site structure changes (selectors need updates)")
            print("  - Rate limiting or blocking by target sites")
            print("  - Missing dependencies (requests, beautifulsoup4)")
            print("\nConsider manual data entry for top 20 models.")
            print("\nProceeding with base dataset only.")
            print("Run with --enrich flag to create final dataset:")
            print("  poetry run python scripts/06_enrich_external.py --enrich")
