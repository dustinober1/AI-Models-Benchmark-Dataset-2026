"""
External data enrichment for AI models benchmark dataset.

This script enriches the dataset with external data sources such as
model release dates, provider announcements, and market events.

Functions
---------
scrape_huggingface_models()
    Scrape model information from HuggingFace Open LLM Leaderboard.

enrich_with_external_data(df, external_df)
    Merge external data with main dataset.

run_enrichment_pipeline(input_path, output_path)
    Execute full enrichment pipeline.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import polars as pl
from src.utils import setup_logging, load_checkpoint, save_checkpoint


def scrape_huggingface_models() -> pl.DataFrame:
    """
    Scrape model release dates from HuggingFace Open LLM Leaderboard.

    Note: This is a template function. Actual implementation requires:
    - Handling pagination for large leaderboards
    - Respecting rate limits (add delays between requests)
    - Handling dynamic content (may need Selenium/Playwright)
    - Parsing JSON API if available (preferred over HTML scraping)

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - model: Model name
        - release_date: Model release date (if available)
        - source_url: URL where data was retrieved
        - retrieved_at: Timestamp of data retrieval
        - retrieved_by: Script identifier for provenance

    Examples
    --------
    >>> external_data = scrape_huggingface_models()
    >>> print(external_data.head())

    Notes
    -----
    - This is a placeholder implementation
    - Actual scraping logic depends on HuggingFace page structure
    - Consider using HuggingFace API if available
    - Always respect robots.txt and rate limits
    - Track provenance for reproducibility
    """
    # Placeholder implementation
    # In production, this would:
    # 1. Fetch HuggingFace Open LLM Leaderboard page
    # 2. Parse HTML or JSON response
    # 3. Extract model names and release dates
    # 4. Add provenance metadata

    logger = setup_logging()
    logger.warning("scrape_huggingface_models() is not implemented yet")
    logger.warning("This is a template for future external data enrichment")

    # Return empty DataFrame with expected schema
    schema = {
        "model": pl.Utf8,
        "release_date": pl.Utf8,
        "source_url": pl.Utf8,
        "retrieved_at": pl.Utf8,
        "retrieved_by": pl.Utf8
    }

    return pl.DataFrame(schema=schema)


def enrich_with_external_data(
    df: pl.DataFrame,
    external_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Merge external data with main dataset.

    Joins external enrichment data (e.g., release dates) with the
    main AI models benchmark dataset based on model name.

    Parameters
    ----------
    df : pl.DataFrame
        Main AI models benchmark dataset.
    external_df : pl.DataFrame
        External data to merge (must have 'model' column).

    Returns
    -------
    pl.DataFrame
        Enriched DataFrame with additional columns from external data.

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> external_data = scrape_huggingface_models()
    >>> enriched_df = enrich_with_external_data(df, external_data)
    >>> print(enriched_df.columns)

    Notes
    -----
    - Join is performed on 'Model' column (left join)
    - Unmatched models retain null values in enrichment columns
    - Provenance columns are preserved for traceability
    - Original dataset is not modified (new columns added)
    """
    logger = setup_logging()

    # Validate external data has required columns
    if "model" not in external_df.columns:
        raise ValueError("External data must have 'model' column for joining")

    # Perform left join to preserve all models in main dataset
    logger.info(f"Joining external data ({external_df.height} rows) with main dataset ({df.height} rows)")

    enriched_df = df.join(
        external_df,
        left_on="Model",
        right_on="model",
        how="left"
    )

    # Log join statistics
    matched_count = enriched_df.filter(
        pl.col("release_date").is_not_null()
    ).height if "release_date" in enriched_df.columns else 0

    logger.info(f"Matched {matched_count} models with external data ({matched_count/df.height*100:.1f}% coverage)")

    return enriched_df


def add_derived_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add derived metrics for analysis.

    Creates calculated columns that combine existing metrics for
    deeper analysis insights.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with base metrics.

    Returns
    -------
    pl.DataFrame
        DataFrame with additional derived columns:
        - price_per_intelligence_point: Price / Intelligence Index
        - intelligence_per_dollar: Intelligence Index / Price
        - speed_to_latency_ratio: Speed / Latency

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> enriched_df = add_derived_metrics(df)
    >>> print(enriched_df["price_per_intelligence_point"].describe())

    Notes
    -----
    - Derived metrics help identify value propositions
    - High intelligence_per_dollar = good value model
    - High speed_to_latency_ratio = responsive model
    - Division by zero results in null values
    """
    logger = setup_logging()
    logger.info("Adding derived metrics...")

    # Start with original DataFrame
    result = df

    # Price per intelligence point (lower = better value)
    if "price_usd" in df.columns and "Intelligence Index" in df.columns:
        result = result.with_columns(
            (pl.col("price_usd") / pl.col("Intelligence Index"))
            .alias("price_per_intelligence_point")
        )
        logger.info("Added: price_per_intelligence_point")

    # Intelligence per dollar (higher = better value)
    if "price_usd" in df.columns and "Intelligence Index" in df.columns:
        result = result.with_columns(
            (pl.col("Intelligence Index") / pl.col("price_usd"))
            .alias("intelligence_per_dollar")
        )
        logger.info("Added: intelligence_per_dollar")

    # Speed to latency ratio (higher = more responsive)
    if "Speed(median token/s)" in df.columns and "Latency (First Answer Chunk /s)" in df.columns:
        result = result.with_columns(
            (pl.col("Speed(median token/s)") / pl.col("Latency (First Answer Chunk /s)"))
            .alias("speed_to_latency_ratio")
        )
        logger.info("Added: speed_to_latency_ratio")

    return result


def run_enrichment_pipeline(
    input_path: str = "data/interim/02_cleaned.parquet",
    output_path: str = "data/processed/ai_models_enriched.parquet"
) -> pl.DataFrame:
    """
    Execute full enrichment pipeline.

    Enriches dataset with external data sources and derived metrics.

    Parameters
    ----------
    input_path : str, default="data/interim/02_cleaned.parquet"
        Path to cleaned data checkpoint.
    output_path : str, default="data/processed/ai_models_enriched.parquet"
        Path to save enriched dataset.

    Returns
    -------
    pl.DataFrame
        Enriched DataFrame ready for analysis.

    Examples
    --------
    >>> enriched_df = run_enrichment_pipeline()
    >>> print(enriched_df.columns)
    >>> print(f"Total columns: {enriched_df.width}")

    Notes
    -----
    - Pipeline steps:
      1. Load cleaned data
      2. Add derived metrics (price per intelligence, etc.)
      3. Enrich with external data (if available)
      4. Save enriched dataset to processed/
    - External data enrichment is optional (proceeds with nulls if not available)
    - All enrichment sources are tracked with provenance metadata
    """
    logger = setup_logging()
    logger.info("Starting enrichment pipeline")

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = load_checkpoint(input_path, logger)

    # Add derived metrics
    logger.info("Adding derived metrics...")
    df_enriched = add_derived_metrics(df)

    # Enrich with external data (if available)
    logger.info("Attempting to fetch external enrichment data...")
    try:
        external_data = scrape_huggingface_models()
        if external_data.height > 0:
            df_enriched = enrich_with_external_data(df_enriched, external_data)
            logger.info("External data enrichment completed")
        else:
            logger.info("No external data available, proceeding with derived metrics only")
    except Exception as e:
        logger.warning(f"External data enrichment failed: {e}")
        logger.info("Proceeding with derived metrics only")

    # Add enrichment timestamp
    df_enriched = df_enriched.with_columns(
        pl.lit(datetime.now().isoformat()).alias("enriched_at")
    )

    # Save enriched dataset
    logger.info(f"Saving enriched dataset to {output_path}")
    save_checkpoint(df_enriched, output_path, logger)

    # Print summary
    logger.info(f"Enriched dataset: {df_enriched.height} rows, {df_enriched.width} columns")
    logger.info("New columns added:")
    original_cols = set(df.columns)
    new_cols = set(df_enriched.columns) - original_cols
    for col in new_cols:
        logger.info(f"  - {col}")

    logger.info("Enrichment pipeline completed")

    return df_enriched


if __name__ == "__main__":
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("Starting external data enrichment process")

    # Run enrichment pipeline
    enriched_df = run_enrichment_pipeline()

    # Print summary
    logger.info("\n=== Enrichment Summary ===")
    print(f"Total rows: {enriched_df.height:,}")
    print(f"Total columns: {enriched_df.width}")
    print(f"\nNew columns:")
    original_cols = {"Model", "Context Window", "Creator", "Intelligence Index",
                     "Price (Blended USD/1M Tokens)", "Speed(median token/s)",
                     "Latency (First Answer Chunk /s)"}
    new_cols = [col for col in enriched_df.columns if col not in original_cols]
    for col in new_cols:
        print(f"  - {col}")

    # Show sample enriched data
    logger.info("\nSample enriched data (first 5 rows):")
    print(enriched_df.head(5))

    logger.info("External data enrichment completed successfully")
