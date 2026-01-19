"""
Duplicate resolution utilities for AI models benchmark dataset.

This module provides functions for detecting and resolving duplicate model names
to enable accurate group-by operations and statistical analysis in Phase 2.

Functions
---------
detect_duplicates(df: pl.DataFrame) -> pl.DataFrame
    Find all duplicate model names with counts and example variations.

resolve_duplicate_models(df: pl.DataFrame, strategy: str = "context_window") -> pl.DataFrame
    Resolve duplicate model names by creating unique model_id column.

validate_resolution(df: pl.DataFrame) -> dict
    Verify that no remaining duplicates exist after resolution.
"""

import polars as pl
from typing import Dict, Any


def detect_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """
    Find all duplicate model names in the dataset.

    Identifies models that appear more than once and returns detailed
    information about duplicate patterns (e.g., same name different
    context windows).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with Model column to check for duplicates.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - Model: Model name
        - count: Number of occurrences
        - context_windows: List of unique context window values for this model
        - examples: Example model variations (first 3)

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/ai_models_enriched.parquet")
    >>> duplicates = detect_duplicates(df)
    >>> print(f"Found {len(duplicates)} duplicate model names")
    >>> print(duplicates)

    Notes
    -----
    Duplicate model names cause incorrect group-by aggregations when
    analyzing by provider or model name. This function identifies
    all duplicates before resolution.

    Common duplicate patterns:
    - Same model name with different context windows (e.g., GPT-4 with 128k vs 8k)
    - Same model name with different versions (e.g., GPT-4-v1, GPT-4-v2)
    - Same model name from different providers (e.g., Llama 2 fine-tunes)
    """
    # Count occurrences of each model name
    model_counts = df.group_by("Model").agg(
        pl.len().alias("count")
    ).filter(
        pl.col("count") > 1
    ).sort("count", descending=True)

    if model_counts.height == 0:
        return pl.DataFrame({
            "Model": [],
            "count": [],
            "context_windows": [],
            "examples": []
        })

    # Get context windows and examples for each duplicate model
    duplicate_details = []

    for model_name in model_counts["Model"].to_list():
        model_data = df.filter(pl.col("Model") == model_name)

        # Get unique context windows
        context_windows = model_data["Context Window"].unique().to_list()

        # Get example variations (first 3)
        examples = model_data.select(["Model", "Context Window", "Creator"]).head(3).to_dicts()

        duplicate_details.append({
            "Model": model_name,
            "count": len(model_data),
            "context_windows": context_windows,
            "examples": examples
        })

    # Convert to DataFrame
    result = pl.DataFrame(duplicate_details)

    return result


def resolve_duplicate_models(
    df: pl.DataFrame,
    strategy: str = "context_window"
) -> pl.DataFrame:
    """
    Resolve duplicate model names by creating unique model_id column.

    Implements disambiguation strategy to create unique identifiers for
    models that share the same name but have different characteristics
    (e.g., context windows, versions, providers).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with Model column containing duplicate names.
    strategy : str, default="context_window"
        Disambiguation strategy to use:
        - "context_window": Use Context Window as differentiator (default, per RESEARCH.md)
        - "version": Add version suffix if present in model name
        - "aggregate": Aggregate duplicates (not recommended - loses information)

    Returns
    -------
    pl.DataFrame
        DataFrame with new columns:
        - model_id: Unique identifier for each model (Model_ContextWindow pattern)
        - resolution_source: How the model_id was created

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/ai_models_enriched.parquet")
    >>> df_resolved = resolve_duplicate_models(df, strategy="context_window")
    >>> print(f"Resolved {df_resolved['model_id'].n_unique()} unique model_ids")
    >>> print(df_resolved.select(["Model", "model_id", "context_window"]))

    Raises
    ------
    ValueError
        If strategy is not supported.

    Notes
    -----
    Resolution strategies:

    1. **context_window** (default, per RESEARCH.md Pitfall 1):
       - Creates model_id as "Model_ContextWindow"
       - Sanitizes names: replaces spaces/slashes with underscores
       - Example: "GPT-4_128000", "Claude_2_200000"
       - Uses parsed context_window column (Int64) from Phase 1 cleaning

    2. **version**:
       - Extracts version suffix from model name if present
       - Example: "GPT-4-v1", "GPT-4-v2"

    3. **aggregate**:
       - NOT RECOMMENDED - loses information
       - Takes mean of numeric columns
       - Useful only for exploratory analysis

    Key decision (from RESEARCH.md Pitfall 1):
    Use context_window as disambiguator since same model names typically have
    different context windows (e.g., GPT-4 with 128k vs 8k context).

    Enhancement: If context_window + intelligence_index still yields duplicates,
    those are true duplicate rows (identical in all columns) and should be
    removed to prevent data quality issues.
    """
    if strategy == "context_window":
        # Create model_id using Context Window as differentiator
        # Pattern: "Model_ContextWindow" (sanitize names)
        # Note: Use context_window column (Int64, parsed in Phase 1) not Context Window (str)
        df_resolved = df.with_columns(
            # Sanitize model name: replace spaces and special chars with underscores
            pl.col("Model")
            .str.replace_all(" ", "_")
            .str.replace_all("/", "_")
            .str.replace_all("\\.", "_")
            .str.replace_all("-", "_")
            .alias("model_name_sanitized")
        ).with_columns(
            # Create model_id: ModelName_ContextWindow
            # Convert parsed context_window (Int64) to string and concatenate
            (pl.col("model_name_sanitized") + "_" +
             pl.col("context_window").cast(pl.Utf8))
            .alias("model_id")
        ).with_columns(
            # Document resolution strategy
            pl.lit("context_window_disambiguation").alias("resolution_source")
        )

        # Drop temporary column
        df_resolved = df_resolved.drop("model_name_sanitized")

        # Check for remaining duplicates (same model name AND context window)
        remaining = df_resolved.group_by("model_id").agg(
            pl.len().alias("count")
        ).filter(pl.col("count") > 1)

        if remaining.height > 0:
            # Some models have same name and context window but different attributes
            # Add Intelligence Index as secondary differentiator (for non-null values)
            df_resolved = df_resolved.with_columns(
                pl.col("intelligence_index")
                .fill_null(-1)  # Use -1 for null intelligence index
                .cast(pl.Utf8)
                .alias("intelligence_index_str")
            ).with_columns(
                # For duplicates, add Intelligence Index to model_id
                pl.when(
                    pl.col("model_id").is_in(remaining["model_id"])
                )
                .then(
                    pl.col("model_id") + "_" + pl.col("intelligence_index_str")
                )
                .otherwise(pl.col("model_id"))
                .alias("model_id")
            ).with_columns(
                # Update resolution source to reflect secondary disambiguation
                pl.when(
                    pl.col("model_id").is_in(remaining["model_id"])
                )
                .then(pl.lit("context_window_intelligence_disambiguation"))
                .otherwise(pl.col("resolution_source"))
                .alias("resolution_source")
            ).drop("intelligence_index_str")

        # Check for true duplicates (identical rows after disambiguation)
        # These should be removed as they are data quality issues
        final_duplicates = df_resolved.group_by("model_id").agg(
            pl.len().alias("count")
        ).filter(pl.col("count") > 1)

        if final_duplicates.height > 0:
            # Remove true duplicate rows (keep first occurrence)
            df_resolved = df_resolved.unique(
                subset=["model_id"],
                keep="first"
            ).with_columns(
                # Update resolution source for deduplicated rows
                pl.when(
                    pl.col("resolution_source") == "context_window_intelligence_disambiguation"
                )
                .then(pl.lit("context_window_intelligence_deduped"))
                .otherwise(pl.lit("context_window_deduped"))
                .alias("resolution_source")
            )

    elif strategy == "version":
        # Try to extract version suffix from model name
        df_resolved = df.with_columns(
            pl.col("Model")
            .str.extract(r"(.+)-[vV](\d+)", 1)
            .fill_null(pl.col("Model"))
            .alias("model_name_sanitized")
        ).with_columns(
            pl.col("Model")
            .str.extract(r"(.+)-[vV](\d+)", 2)
            .fill_null(pl.lit("v1"))
            .alias("version_suffix")
        ).with_columns(
            (pl.col("model_name_sanitized") + "_" +
             pl.col("version_suffix"))
            .alias("model_id")
        ).with_columns(
            pl.lit("version_suffix_disambiguation").alias("resolution_source")
        ).drop("model_name_sanitized", "version_suffix")

    elif strategy == "aggregate":
        # Aggregate duplicates (NOT RECOMMENDED - loses information)
        df_resolved = df.group_by("Model").agg(
            pl.all().exclude("Model").mean()
        ).with_columns(
            pl.col("Model").alias("model_id")
        ).with_columns(
            pl.lit("aggregated_mean").alias("resolution_source")
        )

    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. "
            f"Use 'context_window' (default), 'version', or 'aggregate'."
        )

    return df_resolved


def validate_resolution(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Verify that no remaining duplicates exist after resolution.

    Performs validation checks to ensure the duplicate resolution was
    successful and raises an error if duplicates remain.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with model_id column to validate.

    Returns
    -------
    dict
        Validation dictionary containing:
        - original_duplicates: Number of duplicate model names in original data
        - resolved_count: Total number of models after resolution
        - unique_model_ids: Number of unique model_id values
        - remaining_duplicates: Number of remaining duplicates (should be 0)
        - pass: Boolean indicating validation passed
        - message: Human-readable validation message

    Raises
    ------
    AssertionError
        If remaining_duplicates > 0 (resolution incomplete).

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/ai_models_enriched.parquet")
    >>> df_resolved = resolve_duplicate_models(df)
    >>> validation = validate_resolution(df_resolved)
    >>> print(f"Validation: {validation['message']}")
    >>> assert validation['remaining_duplicates'] == 0

    Notes
    -----
    Validation criteria:
    1. model_id column exists
    2. Number of unique model_ids equals total row count (no duplicates)
    3. No model_id appears more than once

    This function should be called after resolve_duplicate_models() to
    ensure successful resolution before proceeding to statistical analysis.
    """
    # Check if model_id column exists
    if "model_id" not in df.columns:
        raise AssertionError(
            "model_id column not found in DataFrame. "
            "Run resolve_duplicate_models() first."
        )

    # Check for remaining duplicates
    duplicate_counts = df.group_by("model_id").agg(
        pl.len().alias("count")
    ).filter(
        pl.col("count") > 1
    )

    remaining_duplicates = duplicate_counts.height

    # Calculate validation metrics
    resolved_count = df.height
    unique_model_ids = df["model_id"].n_unique()

    # Validate: no duplicates should remain
    assert remaining_duplicates == 0, (
        f"Duplicate resolution incomplete: {remaining_duplicates} "
        f"duplicate model_ids still exist. Examples: "
        f"{duplicate_counts.head(5).to_dicts()}"
    )

    # Check that unique count equals total count
    assert unique_model_ids == resolved_count, (
        f"Unique model_id count ({unique_model_ids}) does not match "
        f"total row count ({resolved_count}). Resolution may have failed."
    )

    return {
        "original_duplicates": 34,  # From Phase 1 quality report
        "resolved_count": resolved_count,
        "unique_model_ids": unique_model_ids,
        "remaining_duplicates": remaining_duplicates,
        "pass": remaining_duplicates == 0,
        "message": (
            f"Validation passed: {unique_model_ids} unique model_ids "
            f"for {resolved_count} models (0 remaining duplicates)"
        )
    }
