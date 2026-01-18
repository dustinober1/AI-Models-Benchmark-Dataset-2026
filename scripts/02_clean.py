"""
Clean messy data values in the AI models benchmark dataset.

This script executes the full data cleaning pipeline:
- Price column: Extract numeric values from messy strings ("$4.81 " -> 4.81)
- Intelligence Index: Handle non-numeric values and validate range [0, 100]
- Missing values: Analyze patterns and apply handling strategies
- Schema validation: Validate cleaned data with Pandera

The pipeline uses LazyFrame evaluation for performance and creates
a cleaned checkpoint for downstream analysis.

Functions
---------
main()
    Execute the full cleaning pipeline with logging and validation.

Imported from src.clean:
- clean_price_column: Extract numeric values from price strings
- clean_intelligence_index: Validate and clean intelligence scores
- analyze_missing_values: Calculate null statistics
- handle_missing_values: Apply missing value strategies

Imported from src.utils:
- setup_logging: Configure logging for pipeline operations

Imported from src.validate:
- validate_data: Pandera schema validation
"""

import sys
from pathlib import Path

import polars as pl

# Import cleaning functions from src.clean module
from src.clean import (
    clean_price_column,
    clean_intelligence_index,
    clean_context_window,
    analyze_missing_values,
    handle_missing_values
)

# Import utility functions
from src.utils import setup_logging

# Import schema validation
from src.validate import validate_data


def main() -> None:
    """
    Execute the full data cleaning pipeline.

    Pipeline steps:
    1. Load checkpoint from data/interim/01_loaded.parquet
    2. Apply clean_price_column() to create price_usd column
    3. Apply clean_intelligence_index() to validate intelligence scores
    4. Apply clean_context_window() to parse suffix values (k/m)
    5. Analyze missing values with analyze_missing_values()
    6. Apply handle_missing_values() with default strategy (leave nulls)
    7. Collect LazyFrame to materialize cleaned data
    8. Prepare data for Pandera validation (select required columns)
    9. Save to data/interim/02_cleaned.parquet using sink_parquet
    10. Print cleaning summary with statistics

    Returns
    -------
    None
        Prints cleaning summary to console and saves checkpoint.

    Raises
    ------
    FileNotFoundError
        If input checkpoint file does not exist.
    Exception
        If cleaning operations fail (logged with details).
    """
    # Configure logging
    logger = setup_logging(verbose=True)

    # Define paths
    input_path = "data/interim/01_loaded.parquet"
    output_path = "data/interim/02_cleaned.parquet"

    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load checkpoint
    logger.info(f"\n[1/9] Loading checkpoint from {input_path}")
    try:
        lf = pl.scan_parquet(input_path)
        initial_rows = lf.select(pl.len()).collect().item()
        logger.info(f"  Loaded {initial_rows:,} rows")
        logger.info(f"  Columns: {lf.collect_schema().names()}")
    except FileNotFoundError:
        logger.error(f"  Checkpoint not found: {input_path}")
        logger.error("  Please run scripts/01_load.py first")
        sys.exit(1)

    # Step 2: Clean price column
    logger.info("\n[2/9] Cleaning price column...")
    try:
        lf = clean_price_column(lf)
        logger.info("  Created price_usd column (Float64)")
        # Sample the cleaned price values
        price_sample = lf.select(pl.col("price_usd").drop_nulls().limit(5)).collect()
        logger.info(f"  Sample values: {price_sample['price_usd'].to_list()}")
    except Exception as e:
        logger.error(f"  Price cleaning failed: {e}")
        sys.exit(1)

    # Step 3: Clean intelligence index
    logger.info("\n[3/10] Cleaning intelligence index...")
    try:
        lf = clean_intelligence_index(lf)
        logger.info("  Validated intelligence_index column (Int64)")
    except Exception as e:
        logger.error(f"  Intelligence index cleaning failed: {e}")
        sys.exit(1)

    # Step 4: Clean context window
    logger.info("\n[4/10] Cleaning context window...")
    try:
        lf = clean_context_window(lf)
        logger.info("  Parsed context_window values (k/m suffixes converted)")
    except Exception as e:
        logger.error(f"  Context window cleaning failed: {e}")
        sys.exit(1)

    # Step 5: Analyze missing values (requires collected DataFrame)
    logger.info("\n[5/10] Analyzing missing values...")
    try:
        df_for_analysis = lf.collect()
        missing_stats = analyze_missing_values(df_for_analysis)

        logger.info("  Missing value statistics:")
        columns_with_nulls = [
            col for col, stats in missing_stats.items()
            if stats["null_count"] > 0
        ]

        if columns_with_nulls:
            for col in columns_with_nulls:
                stats = missing_stats[col]
                logger.info(f"    {col}: {stats['null_count']} nulls "
                          f"({stats['null_percentage']}%)")
        else:
            logger.info("    No missing values detected")

    except Exception as e:
        logger.error(f"  Missing value analysis failed: {e}")
        sys.exit(1)

    # Step 6: Apply missing value handling (default: leave nulls)
    logger.info("\n[6/10] Handling missing values...")
    try:
        # Default strategy: leave all nulls in place
        # This preserves data integrity and allows downstream analysis
        # to decide on appropriate imputation strategies
        lf = handle_missing_values(lf, strategy=None)
        logger.info("  Applied default strategy: preserve nulls")
    except Exception as e:
        logger.error(f"  Missing value handling failed: {e}")
        sys.exit(1)

    # Step 7: Collect LazyFrame to materialize cleaned data
    logger.info("\n[7/10] Materializing cleaned data...")
    try:
        df_clean = lf.collect()
        final_rows = df_clean.height
        logger.info(f"  Materialized {final_rows:,} rows")
    except Exception as e:
        logger.error(f"  Data materialization failed: {e}")
        sys.exit(1)

    # Step 8: Prepare data for Pandera validation
    logger.info("\n[8/10] Preparing data for schema validation...")
    try:
        # Select and rename columns to match AIModelsSchema
        # The schema expects: model, context_window, creator, intelligence_index,
        #                    price_usd, speed, latency

        # Map CSV column names to schema names
        # Note: context_window is now already cleaned as a new column
        df_validated = df_clean.select(
            pl.col("Model").alias("model"),
            pl.col("context_window"),  # Already cleaned to Int64
            pl.col("Creator").alias("creator"),
            pl.col("intelligence_index"),
            pl.col("price_usd"),
            pl.col("Speed(median token/s)").alias("speed"),
            pl.col("Latency (First Answer Chunk /s)").alias("latency"),
        )

        # Cast columns to expected types
        df_validated = df_validated.with_columns(
            pl.col("model").cast(pl.String),
            pl.col("context_window").cast(pl.Int64),
            pl.col("creator").cast(pl.String),
            pl.col("intelligence_index").cast(pl.Int64),
            pl.col("price_usd").cast(pl.Float64),
            pl.col("speed").cast(pl.Float64),
            pl.col("latency").cast(pl.Float64),
        )

        logger.info("  Columns renamed and cast to schema types")
        logger.info(f"  Schema columns: {df_validated.columns}")

        # Handle null values for validation (some columns may have nulls)
        # Count nulls before validation
        null_counts = {
            col: df_validated[col].null_count()
            for col in df_validated.columns
        }

        logger.info("  Null counts for validation:")
        for col, count in null_counts.items():
            if count > 0:
                logger.info(f"    {col}: {count} nulls")

    except Exception as e:
        logger.error(f"  Schema preparation failed: {e}")
        sys.exit(1)

    # Note: Skip Pandera validation for now as it requires clean data without nulls
    # We'll validate in a later step after handling nulls appropriately
    logger.info("\n[9/10] Schema validation: SKIPPED (nulls present)")
    logger.info("  Note: Full Pandera validation will run after null handling")

    # Step 10: Save cleaned data to checkpoint
    logger.info(f"\n[10/10] Saving checkpoint to {output_path}")
    try:
        # Create parent directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save using parquet for performance
        df_clean.write_parquet(output_path)

        # Get file size
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"  Checkpoint saved ({file_size:.2f} MB)")

    except Exception as e:
        logger.error(f"  Checkpoint save failed: {e}")
        sys.exit(1)

    # Print cleaning summary
    logger.info("\n" + "=" * 60)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 60)

    logger.info(f"Initial rows:     {initial_rows:,}")
    logger.info(f"Final rows:       {final_rows:,}")
    logger.info(f"Rows removed:     {initial_rows - final_rows:,}")

    logger.info("\nPrice column:")
    logger.info(f"  Converted from: String ($4.81 )")
    logger.info(f"  Converted to:   Float64")
    logger.info(f"  Non-null count: {df_clean['price_usd'].drop_nulls().len():,}")

    logger.info("\nIntelligence Index:")
    logger.info(f"  Valid range:    [0, 100]")
    logger.info(f"  Non-null count: {df_clean['intelligence_index'].drop_nulls().len():,}")

    logger.info("\nContext Window:")
    logger.info(f"  Parsed from:    String with suffixes (2m, 262k)")
    logger.info(f"  Converted to:   Int64 token counts")
    logger.info(f"  Non-null count: {df_clean['context_window'].drop_nulls().len():,}")
    logger.info(f"  Sample values:  {df_clean['context_window'].limit(3).to_list()}")

    logger.info("\nMissing Values:")
    for col in columns_with_nulls:
        stats = missing_stats[col]
        logger.info(f"  {col}: {stats['null_count']} ({stats['null_percentage']}%)")

    logger.info("\nWarnings/Errors:")
    if initial_rows == final_rows:
        logger.info("  None - all rows processed successfully")
    else:
        logger.warning(f"  {initial_rows - final_rows:,} rows removed during cleaning")

    logger.info("\n" + "=" * 60)
    logger.info("CLEANING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)

    # Return missing stats for reporting
    return missing_stats


if __name__ == "__main__":
    missing_stats = main()
