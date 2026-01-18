"""
Load AI models benchmark dataset from CSV.

This script loads the raw CSV file using Polars with explicit schema
definition to ensure data type consistency and early error detection.

The schema is defined explicitly rather than inferred because:
1. CSV contains messy values (e.g., "$4.81 " with dollar sign and space)
2. Some values may have embedded newlines in quoted fields
3. Explicit schema catches data quality issues early

Functions
---------
load_data(path: str) -> pl.LazyFrame
    Load AI models performance data from CSV with schema validation.
"""

import polars as pl
from src.utils import setup_logging, save_checkpoint


def load_data(path: str) -> pl.LazyFrame:
    """
    Load AI models performance data from CSV with schema validation.

    Uses lazy evaluation (scan_csv) for efficient processing of large files.
    Schema is defined upfront to catch type mismatches early.

    Parameters
    ----------
    path : str
        Path to ai_models_performance.csv file.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with validated schema, ready for transformation.

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist at specified path.
    SchemaError
        If CSV columns don't match expected types or structure.

    Examples
    --------
    >>> lf = load_data("data/raw/ai_models_performance.csv")
    >>> print(lf.collect_schema())
    >>> df = lf.collect()  # Materialize when needed

    Notes
    -----
    Schema definition:
    - Model: Utf8 (string) - Model name/identifier
    - Context Window: Int64 - Maximum context size in tokens
    - Creator: Utf8 - Organization/lab that created the model
    - Intelligence Index: Int64 - Performance score (0-100)
    - Price (Blended USD/1M Tokens): Utf8 - Price string (will clean to Float64)
    - Speed(median token/s): Float64 - Median generation speed
    - Latency (First Answer Chunk /s): Float64 - Time to first token

    The price column is loaded as Utf8 because it contains messy formatting
    (dollar signs, spaces) that needs to be cleaned in the next step.
    """
    # Define explicit schema for type safety
    # Using Utf8 for price column because it contains "$4.81 " format
    schema = {
        "Model": pl.Utf8,
        "Context Window": pl.Int64,
        "Creator": pl.Utf8,
        "Intelligence Index": pl.Int64,
        "Price (Blended USD/1M Tokens)": pl.Utf8,  # Will clean to Float64 in 02_clean.py
        "Speed(median token/s)": pl.Float64,
        "Latency (First Answer Chunk /s)": pl.Float64,
    }

    # Use lazy evaluation for efficient processing
    lf = pl.scan_csv(path, schema_overrides=schema)

    return lf


if __name__ == "__main__":
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("Starting data loading process")

    # Load the data
    input_path = "data/raw/ai_models_performance.csv"
    logger.info(f"Loading data from {input_path}")

    lf = load_data(input_path)

    # Print schema information
    logger.info("Data schema:")
    print(lf.collect_schema())

    # Collect and show basic info
    df = lf.collect()
    logger.info(f"Loaded {df.height:,} rows and {df.width} columns")

    # Print first few rows
    logger.info("Sample data (first 5 rows):")
    print(df.head(5))

    # Save checkpoint
    checkpoint_path = "data/interim/01_loaded.parquet"
    save_checkpoint(df, checkpoint_path, logger)

    logger.info("Data loading completed successfully")
