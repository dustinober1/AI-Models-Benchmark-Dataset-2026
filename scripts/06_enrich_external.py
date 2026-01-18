"""
External data enrichment for AI models benchmark dataset.

This script scrapes and enriches the dataset with external data sources such as
model release dates, provider announcements, and market events from HuggingFace
and provider blogs/news sources.

Functions
---------
main()
    Execute external data scraping pipeline.

Notes
-----
- Implements rate limiting (1 second delay) for respectful scraping
- Tracks provenance (source_url, retrieved_at, retrieved_by) for reproducibility
- Handles failures gracefully - continues with base dataset if scraping fails
- Saves scraped data to data/external/ directory
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from src.enrich import scrape_huggingface_models, scrape_provider_announcements
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


if __name__ == "__main__":
    # Execute external data scraping pipeline
    external_data = main()

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total external records: {external_data.height}")
    print(f"Columns: {external_data.columns}")

    if external_data.height > 0:
        print("\nSample data (first 5 rows):")
        print(external_data.head(5))
    else:
        print("\nNo external data retrieved.")
        print("Note: Web scraping may have failed due to:")
        print("  - Network connectivity issues")
        print("  - Site structure changes (selectors need updates)")
        print("  - Rate limiting or blocking by target sites")
        print("  - Missing dependencies (requests, beautifulsoup4)")
        print("\nConsider manual data entry for top 20 models.")
