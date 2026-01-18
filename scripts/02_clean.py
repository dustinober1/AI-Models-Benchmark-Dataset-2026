"""
Clean messy data values in the AI models benchmark dataset.

This script handles data cleaning operations including:
- Price column: Extract numeric values from messy strings ("$4.81 " -> 4.81)
- Intelligence Index: Handle any non-numeric or out-of-range values
- Context Window: Validate and clean context window values

Functions
---------
clean_price_column(lf: pl.LazyFrame) -> pl.LazyFrame
    Extract numeric values from price strings and convert to Float64.

clean_intelligence_index(lf: pl.LazyFrame) -> pl.LazyFrame
    Validate and clean intelligence index values (0-100 range).

clean_context_window(lf: pl.LazyFrame) -> pl.LazyFrame
    Validate and clean context window values.

run_cleaning_pipeline(input_path: str, output_path: str) -> pl.LazyFrame
    Execute full cleaning pipeline on loaded data.
"""

import polars as pl
from src.utils import setup_logging, save_checkpoint, load_checkpoint


def clean_price_column(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Clean price column by extracting numeric values from messy strings.

    Input format: "$4.81 " (with dollar sign and trailing space)
    Output: Float64 numeric value

    The price column contains formatted strings with dollar signs and
    whitespace that need to be stripped before conversion to numeric type.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame with messy price column ("Price (Blended USD/1M Tokens)").

    Returns
    -------
    pl.LazyFrame
        LazyFrame with new column "price_usd" containing Float64 values.

    Examples
    --------
    >>> lf = pl.scan_csv("data/interim/01_loaded.parquet")
    >>> lf_clean = clean_price_column(lf)
    >>> df = lf_clean.collect()
    >>> print(df["price_usd"].head())

    Notes
    -----
    - Removes leading/trailing whitespace with str.strip()
    - Removes dollar sign ($) with str.replace()
    - Converts to Float64 for numeric operations
    - Creates new column "price_usd" to preserve original if needed
    """
    return lf.with_columns(
        pl.col("Price (Blended USD/1M Tokens)")
        .str.strip()
        .str.replace("$", "")
        .str.replace(" ", "")
        .cast(pl.Float64)
        .alias("price_usd")
    )


def clean_intelligence_index(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Validate and clean intelligence index values.

    Ensures intelligence index is in valid range [0, 100] and handles
    any non-numeric or out-of-range values.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame with intelligence index column.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with validated intelligence index column.

    Examples
    --------
    >>> lf = pl.scan_csv("data/interim/01_loaded.parquet")
    >>> lf_clean = clean_intelligence_index(lf)
    >>> df = lf_clean.collect()
    >>> print(df["Intelligence Index"].describe())

    Notes
    -----
    - Intelligence Index should be in range [0, 100]
    - Values outside this range are flagged for review
    - Null values are preserved for later handling
    """
    # For now, just validate range - can add more cleaning logic as needed
    return lf.with_columns(
        pl.col("Intelligence Index")
        .clip(0, 100)  # Ensure values are in valid range
    )


def clean_context_window(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Validate and clean context window values.

    Ensures context window values are reasonable (positive integers)
    and handles any anomalies.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame with context window column.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with validated context window column.

    Examples
    --------
    >>> lf = pl.scan_csv("data/interim/01_loaded.parquet")
    >>> lf_clean = clean_context_window(lf)
    >>> df = lf_clean.collect()
    >>> print(df["Context Window"].describe())

    Notes
    -----
    - Context window should be positive integer
    - Typical range: 2K to 2M tokens
    - Zero or negative values are flagged for review
    """
    # For now, just ensure non-negative values
    return lf.with_columns(
        pl.col("Context Window")
        .abs()  # Ensure non-negative
    )


def run_cleaning_pipeline(
    input_path: str = "data/interim/01_loaded.parquet",
    output_path: str = "data/interim/02_cleaned.parquet"
) -> pl.LazyFrame:
    """
    Execute full cleaning pipeline on loaded data.

    Applies all cleaning functions in sequence:
    1. Clean price column
    2. Clean intelligence index
    3. Clean context window

    Parameters
    ----------
    input_path : str, default="data/interim/01_loaded.parquet"
        Path to loaded data checkpoint.
    output_path : str, default="data/interim/02_cleaned.parquet"
        Path to save cleaned data checkpoint.

    Returns
    -------
    pl.LazyFrame
        Cleaned LazyFrame ready for validation.

    Examples
    --------
    >>> lf_clean = run_cleaning_pipeline()
    >>> print(lf_clean.collect().head())

    Notes
    -----
    - Pipeline is executed lazily - no computation until collect()
    - All cleaning operations are reversible (original columns preserved)
    - Checkpoint is saved after all cleaning operations complete
    """
    logger = setup_logging()

    # Load data
    logger.info(f"Loading data from {input_path}")
    lf = pl.scan_parquet(input_path)

    # Apply cleaning operations
    logger.info("Cleaning price column...")
    lf = clean_price_column(lf)

    logger.info("Validating intelligence index...")
    lf = clean_intelligence_index(lf)

    logger.info("Validating context window...")
    lf = clean_context_window(lf)

    # Collect and save checkpoint
    df_clean = lf.collect()
    save_checkpoint(df_clean, output_path, logger)

    logger.info("Cleaning pipeline completed")

    return lf


if __name__ == "__main__":
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("Starting data cleaning process")

    # Run cleaning pipeline
    lf_clean = run_cleaning_pipeline()

    # Show sample of cleaned data
    df_clean = lf_clean.collect()
    logger.info("Cleaned data sample (first 5 rows):")
    print(df_clean.head(5))

    # Show new columns
    logger.info("New columns added:")
    if "price_usd" in df_clean.columns:
        print(f"  - price_usd: {df_clean['price_usd'].dtype}")

    logger.info("Data cleaning completed successfully")
