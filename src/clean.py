"""
Data cleaning functions for AI models benchmark dataset.

This module provides reusable cleaning utilities that handle:
- Messy price strings (e.g., "$4.81 " -> 4.81)
- Intelligence index validation and range checking
- Missing value analysis and pattern documentation
- Configurable missing value handling strategies

These functions are designed to be imported and executed by the
cleaning pipeline script (scripts/02_clean.py).
"""

import re
from typing import Optional, Dict, Any
import logging

import polars as pl


def clean_price_column(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Extract numeric values from messy price strings and create Float64 column.

    The raw CSV column "Price (Blended USD/1M Tokens)" contains messy strings
    with dollar signs, spaces, and trailing whitespace. This function cleans
    those values and creates a new price_usd column with proper Float64 type.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame containing the raw "Price (Blended USD/1M Tokens)" column.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with new price_usd column (Float64) containing cleaned prices.

    Examples
    --------
    >>> import polars as pl
    >>> lf = pl.DataFrame({
    ...     "Price (Blended USD/1M Tokens)": ["$4.81 ", "$10.00", "$2.50 "]
    ... }).lazy()
    >>> cleaned = clean_price_column(lf).collect()
    >>> cleaned["price_usd"].to_list()
    [4.81, 10.0, 2.5]

    Notes
    -----
    Transformation sequence:
    1. str.strip() - Remove leading/trailing whitespace
    2. str.replace("$", "") - Remove dollar sign
    3. str.replace(" ", "") - Remove any internal spaces
    4. str.replace(",", "") - Remove commas from large values
    5. cast(Float64) - Convert to numeric type
    6. alias("price_usd") - Rename to clean column name

    Rows with conversion failures are flagged but not dropped immediately.
    """
    return lf.with_columns(
        pl.col("Price (Blended USD/1M Tokens)")
        .str.strip_chars()                # Remove whitespace: "$4.81 " -> "$4.81"
        .str.replace("$", "", literal=True)  # Remove dollar sign: "$4.81" -> "4.81"
        .str.replace(" ", "", literal=True)  # Remove any spaces
        .str.replace(",", "", literal=True)  # Remove commas from large values
        .cast(pl.Float64)                 # Convert to numeric: "4.81" -> 4.81
        .alias("price_usd")               # Rename to clean column name
    )


def clean_intelligence_index(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Validate and clean intelligence index scores, extracting numeric values.

    The Intelligence Index column may contain problematic values like:
    - Multi-line strings: "41\\nE" (from quoted CSV fields)
    - Missing value placeholders: "--"
    - Values outside valid range [0, 100]

    This function extracts numeric values, validates the range, and flags
    any problematic entries for review.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame containing the "Intelligence Index" column.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with validated intelligence_index column (Int64).

    Examples
    --------
    >>> import polars as pl
    >>> lf = pl.DataFrame({
    ...     "Intelligence Index": ["41", "49", "55"]
    ... }).lazy()
    >>> cleaned = clean_intelligence_index(lf).collect()
    >>> cleaned["intelligence_index"].to_list()
    [41, 49, 55]

    Notes
    -----
    Valid range for intelligence scores: 0 to 100 (inclusive).
    Values outside this range are flagged for review.

    For dirty string values like "41\\nE", the function uses regex
    extraction to pull out the leading numeric portion.
    """
    # Check if column is already numeric (Int64)
    # If so, we can pass through with validation only
    col = pl.col("Intelligence Index")

    # Try to extract numeric values if strings exist
    return lf.with_columns(
        col.str.extract(r"^(\d+)")       # Extract leading digits from strings
        .cast(pl.Int64)                   # Convert to integer
        .alias("intelligence_index")
    ).with_columns(
        # Flag values outside valid range [0, 100]
        pl.col("intelligence_index")
        .is_between(0, 100, closed="both")
        .alias("intelligence_index_valid")
    )


def clean_context_window(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Parse and clean context window values with suffixes (k, m).

    The Context Window column contains string values with suffixes:
    - "2m" = 2,000,000 tokens
    - "262k" = 262,000 tokens
    - "400k" = 400,000 tokens

    This function parses these suffixes and converts to integer token counts.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame containing the "Context Window" column.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with cleaned context_window column (Int64).

    Examples
    --------
    >>> import polars as pl
    >>> lf = pl.DataFrame({
    ...     "Context Window": ["2m", "262k", "400k"]
    ... }).lazy()
    >>> cleaned = clean_context_window(lf).collect()
    >>> cleaned["context_window"].to_list()
    [2000000, 262000, 400000]

    Notes
    -----
    Suffix multipliers:
    - "k" or "K" = multiply by 1,000
    - "m" or "M" = multiply by 1,000,000
    - No suffix = use value as-is
    """
    # Create context_window column by parsing the original
    # We need to handle cases like "2m", "262k", etc.
    return lf.with_columns(
        pl.col("Context Window")
        .str.strip_chars()
        .str.to_lowercase()
        .alias("context_window_str")
    ).with_columns(
        # Extract numeric part
        pl.col("context_window_str")
        .str.extract(r"([\d.]+)")
        .cast(pl.Float64)
        .alias("context_window_num")
    ).with_columns(
        # Extract suffix
        pl.col("context_window_str")
        .str.extract(r"([km]?)$")
        .alias("context_window_suffix")
    ).with_columns(
        # Calculate actual value based on suffix
        pl.when(pl.col("context_window_suffix") == "m")
        .then(pl.col("context_window_num") * 1_000_000)
        .when(pl.col("context_window_suffix") == "k")
        .then(pl.col("context_window_num") * 1_000)
        .otherwise(pl.col("context_window_num"))
        .cast(pl.Int64)
        .alias("context_window")
    ).drop(
        "context_window_str",
        "context_window_num",
        "context_window_suffix"
    )


def analyze_missing_values(df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze missing values across all columns and return summary statistics.

    This function calculates null counts and percentages for each column,
    identifies columns with any missing values, and provides detailed
    statistics for missing value pattern analysis.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to analyze for missing values.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping column names to missing value statistics:
        - null_count: Number of null values in column
        - null_percentage: Percentage of null values (0-100)
        - has_nulls: Boolean indicating if column has any nulls

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "a": [1, 2, None, 4],
    ...     "b": [1, None, None, 4]
    ... })
    >>> stats = analyze_missing_values(df)
    >>> stats["a"]["null_count"]
    1
    >>> stats["a"]["null_percentage"]
    25.0

    Notes
    -----
    Missing value patterns help determine appropriate handling strategies:
    - Random missing values: May be safe for imputation
    - Clustered missing values: May indicate data quality issues
    - Correlated missing values: May require domain knowledge to handle
    """
    total_rows = df.height
    missing_stats = {}

    for col_name in df.columns:
        null_count = df[col_name].null_count()
        null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0

        missing_stats[col_name] = {
            "null_count": null_count,
            "null_percentage": round(null_percentage, 2),
            "has_nulls": null_count > 0
        }

    return missing_stats


def handle_missing_values(
    lf: pl.LazyFrame,
    strategy: Optional[Dict[str, str]] = None
) -> pl.LazyFrame:
    """
    Apply missing value handling strategies according to configuration.

    This function applies various imputation and filtering strategies
    to handle null values in the dataset. Each column can have its own
    strategy, providing fine-grained control over missing value handling.

    Parameters
    ----------
    lf : pl.LazyFrame
        LazyFrame with potentially null values.
    strategy : Dict[str, str], optional
        Mapping of column names to handling strategies:
        - "drop": Remove rows with nulls in this column
        - "forward_fill": Fill with previous value (propagate forward)
        - "backward_fill": Fill with next value (propagate backward)
        - "mean": Fill with column mean (numeric columns only)
        - "median": Fill with column median (numeric columns only)
        - "zero": Fill with 0 (numeric columns only)
        - None: Leave nulls as-is (default)

        If None (default), all columns are left with nulls intact.

    Returns
    -------
    pl.LazyFrame
        LazyFrame with nulls handled according to strategy.

    Examples
    --------
    >>> import polars as pl
    >>> lf = pl.DataFrame({
    ...     "a": [1.0, None, 3.0],
    ...     "b": [None, 2.0, None]
    ... }).lazy()
    >>> strategy = {"a": "mean", "b": "zero"}
    >>> cleaned = handle_missing_values(lf, strategy).collect()
    >>> print(cleaned)

    Notes
    -----
    Default strategy (None): Leave all nulls in place
    This is recommended for enrichment columns where nulls represent
    truly missing data that should not be imputed.

    Strategy selection guidelines:
    - Use "drop" for critical columns where nulls invalidate the record
    - Use "mean"/"median" for numeric columns with random missing values
    - Use "forward_fill"/"backward_fill" for time-series data
    - Use None for optional/enrichment columns (preserve nulls)
    """
    if strategy is None:
        # Default: leave all nulls in place
        return lf

    result = lf

    for col_name, method in strategy.items():
        if col_name not in result.collect_schema():
            # Column doesn't exist, skip
            continue

        if method == "drop":
            # Remove rows with nulls in this column
            result = result.filter(pl.col(col_name).is_not_null())

        elif method == "forward_fill":
            # Fill with previous value
            result = result.with_columns(
                pl.col(col_name).fill_null(strategy="forward")
            )

        elif method == "backward_fill":
            # Fill with next value
            result = result.with_columns(
                pl.col(col_name).fill_null(strategy="backward")
            )

        elif method == "mean":
            # Fill with column mean (numeric only)
            # Collect once to get the mean value
            mean_value = result.select(pl.col(col_name).mean()).collect().item()
            result = result.with_columns(
                pl.col(col_name).fill_null(mean_value)
            )

        elif method == "median":
            # Fill with column median (numeric only)
            median_value = result.select(pl.col(col_name).median()).collect().item()
            result = result.with_columns(
                pl.col(col_name).fill_null(median_value)
            )

        elif method == "zero":
            # Fill with 0 (numeric only)
            result = result.with_columns(
                pl.col(col_name).fill_null(0)
            )

        # If method is None or unrecognized, leave nulls as-is

    return result
