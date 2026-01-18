"""
Data loading utilities for AI models benchmark dataset.

This module provides functions for loading raw CSV data with explicit
schema definition and documenting the data structure for inspection.

The loading strategy uses Polars LazyFrame for efficient lazy evaluation:
- Schema is defined upfront for type safety
- Lazy evaluation defers computation until needed
- Explicit schema catches data quality issues early
"""

import polars as pl
from typing import Dict, Any, List
import logging

from src.utils import setup_logging


def load_data(path: str) -> pl.LazyFrame:
    """
    Load AI models performance data from CSV with explicit schema validation.

    Uses Polars LazyFrame (scan_csv) for lazy evaluation and efficient processing.
    Schema is defined upfront to catch type mismatches early in the pipeline.

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
    Schema definition uses lenient loading (all Utf8) to handle messy data:
    - Model: Utf8 (string) - Model name/identifier
    - Context Window: Utf8 - Contains "400k", "1m" suffixes (will clean to Int64)
    - Creator: Utf8 - Organization/lab that created the model
    - Intelligence Index: Utf8 - May have quoted values (will clean to Int64)
    - Price (Blended USD/1M Tokens): Utf8 - Contains "$4.81 " format (will clean to Float64)
    - Speed(median token/s): Utf8 - Will clean to Float64 in 02_clean.py
    - Latency (First Answer Chunk /s): Utf8 - Will clean to Float64 in 02_clean.py

    All columns are loaded as Utf8 because the raw data contains messy formatting
    that needs cleaning in the 02_clean.py script:
    - Context Window has "k" and "m" suffixes (e.g., "400k", "1m")
    - Price has dollar signs and trailing spaces (e.g., "$4.81 ")
    - Some values may be embedded in quoted multi-line fields

    The proper type conversion and validation happens in subsequent pipeline steps:
    1. Load (01_load.py) - Lenient loading as Utf8
    2. Clean (02_clean.py) - Remove formatting, convert to proper types
    3. Validate (02_clean.py) - Pandera schema validation with type/range checks
    """
    # Define lenient schema for loading (all Utf8 to handle messy data)
    # The dataset contains messy values that need cleaning:
    # - Context Window: "400k", "1m", "200k" (k/m suffixes)
    # - Price: "$4.81 " (dollar sign and trailing space)
    # - Intelligence Index: quoted multi-line values like "41\nE"
    # These will be cleaned in the 02_clean.py script
    schema = {
        "Model": pl.Utf8,
        "Context Window": pl.Utf8,  # Contains "400k", "1m" - will clean to Int64
        "Creator": pl.Utf8,
        "Intelligence Index": pl.Utf8,  # May have quoted values - will clean to Int64
        "Price (Blended USD/1M Tokens)": pl.Utf8,  # Contains "$4.81 " - will clean to Float64
        "Speed(median token/s)": pl.Utf8,  # Load as Utf8 first, will clean to Float64
        "Latency (First Answer Chunk /s)": pl.Utf8,  # Load as Utf8 first, will clean to Float64
    }

    # Use lazy evaluation for efficient processing
    # ignore_errors=True allows us to handle messy data during cleaning
    lf = pl.scan_csv(path, schema_overrides=schema, ignore_errors=True)

    return lf


def document_structure(lf: pl.LazyFrame, logger: logging.Logger = None) -> Dict[str, Any]:
    """
    Document the structure of a LazyFrame for inspection and analysis.

    Collects schema information, sample data, and row count to provide
    a comprehensive overview of the dataset structure. This is useful
    for understanding the data before processing.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame to document. Will be collected during analysis.
    logger : logging.Logger, optional
        Logger instance for console output. If None, uses default logger.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - column_names: List[str] - All column names in order
        - dtypes: Dict[str, str] - Mapping of column names to Polars types
        - sample_values: List[Dict] - First 5 rows as list of dicts
        - row_count: int - Total number of rows

    Examples
    --------
    >>> lf = load_data("data/raw/ai_models_performance.csv")
    >>> structure = document_structure(lf)
    >>> print(structure["column_names"])
    >>> print(structure["dtypes"])
    >>> print(structure["sample_values"])

    Notes
    -----
    This function prints documentation to the console when verbose logging
    is enabled. The output includes:
    - Column count and names
    - Data types for each column
    - First 5 rows of data
    - Total row count
    """
    if logger is None:
        logger = setup_logging(verbose=True)

    # Collect the LazyFrame to access schema and data
    df = lf.collect()

    # Get schema information
    schema = df.schema
    column_names = df.columns
    dtypes = {col: str(dtype) for col, dtype in schema.items()}

    # Get sample values (first 5 rows)
    sample_values = df.head(5).to_dicts()

    # Get row count
    row_count = df.height

    # Print documentation to console
    logger.info("=" * 70)
    logger.info("DATA STRUCTURE DOCUMENTATION")
    logger.info("=" * 70)
    logger.info(f"Total columns: {len(column_names)}")
    logger.info(f"Total rows: {row_count:,}")
    logger.info("")
    logger.info("Column schema:")
    for col_name in column_names:
        logger.info(f"  - {col_name}: {dtypes[col_name]}")
    logger.info("")
    logger.info("Sample data (first 5 rows):")
    for i, row in enumerate(sample_values, 1):
        logger.info(f"  Row {i}: {row}")
    logger.info("=" * 70)

    # Return dictionary for programmatic access
    structure = {
        "column_names": column_names,
        "dtypes": dtypes,
        "sample_values": sample_values,
        "row_count": row_count,
    }

    return structure
