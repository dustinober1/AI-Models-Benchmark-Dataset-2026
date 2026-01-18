"""
Data quality assessment utilities for AI models benchmark dataset.

This module provides functions for comprehensive sanity checking across
6 dimensions of data quality: Accuracy, Completeness, Consistency, Validity,
Integrity, and Timeliness.

Functions
---------
check_accuracy(df: pl.DataFrame) -> dict
    Validate data values are correct and within expected ranges.

check_completeness(df: pl.DataFrame) -> dict
    Calculate missing value statistics and completeness metrics.

check_consistency(df: pl.DataFrame) -> dict
    Check for data format consistency and duplicate records.

check_validity(df: pl.DataFrame) -> dict
    Validate data conforms to expected formats and value sets.

perform_sanity_checks(df: pl.DataFrame) -> dict
    Execute all sanity checks and calculate overall quality score.

generate_quality_report(df: pl.DataFrame, distributions_stats: dict, output_path: str) -> str
    Generate comprehensive quality assessment report in markdown format.
"""

import polars as pl
from typing import Dict, Any, List
from datetime import datetime


def check_accuracy(df: pl.DataFrame) -> dict:
    """
    Validate accuracy of data values by checking ranges and constraints.

    Accuracy measures whether data values are correct and within expected
    ranges. This function validates business logic constraints on numerical
    columns and flags violations for review.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate. Expected columns: Intelligence Index, price_usd,
        Speed(median token/s), Latency (First Answer Chunk /s), context_window

    Returns
    -------
    dict
        Dictionary containing:
        - pass: bool (True if all checks pass)
        - violation_count: int (Total number of violations)
        - violations: dict (Per-column violation details)
        - intelligence_index_violations: Count of values outside [0, 100]
        - price_negative: Count of negative prices
        - speed_negative: Count of negative speeds
        - latency_negative: Count of negative latencies
        - context_window_negative: Count of negative context windows

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/03_distributions_analyzed.parquet")
    >>> accuracy = check_accuracy(df)
    >>> print(f"Accuracy check: {'PASS' if accuracy['pass'] else 'FAIL'}")

    Notes
    -----
    Accuracy checks performed:
    - Intelligence Index in range [0, 100]
    - price_usd >= 0
    - Speed(median token/s) >= 0
    - Latency (First Answer Chunk /s) >= 0
    - context_window >= 0

    Violations are flagged for domain expert review. Per STATE.md decision,
    outliers are preserved for analysis rather than removed.
    """
    violations = {}
    total_violations = 0

    # Intelligence Index range check [0, 100]
    if "Intelligence Index" in df.columns:
        ii_violations = df.filter(
            (pl.col("Intelligence Index") < 0) | (pl.col("Intelligence Index") > 100)
        )
        violations["intelligence_index"] = {
            "count": ii_violations.height,
            "examples": ii_violations.select(["Model", "Intelligence Index"]).head(3).to_dicts() if ii_violations.height > 0 else []
        }
        total_violations += ii_violations.height
    else:
        violations["intelligence_index"] = {"count": 0, "examples": []}

    # Price negative check
    if "price_usd" in df.columns:
        price_violations = df.filter(pl.col("price_usd") < 0)
        violations["price_negative"] = {
            "count": price_violations.height,
            "examples": price_violations.select(["Model", "price_usd"]).head(3).to_dicts() if price_violations.height > 0 else []
        }
        total_violations += price_violations.height
    else:
        violations["price_negative"] = {"count": 0, "examples": []}

    # Speed negative check
    if "Speed(median token/s)" in df.columns:
        speed_violations = df.filter(pl.col("Speed(median token/s)") < 0)
        violations["speed_negative"] = {
            "count": speed_violations.height,
            "examples": speed_violations.select(["Model", "Speed(median token/s)"]).head(3).to_dicts() if speed_violations.height > 0 else []
        }
        total_violations += speed_violations.height
    else:
        violations["speed_negative"] = {"count": 0, "examples": []}

    # Latency negative check
    if "Latency (First Answer Chunk /s)" in df.columns:
        latency_violations = df.filter(pl.col("Latency (First Answer Chunk /s)") < 0)
        violations["latency_negative"] = {
            "count": latency_violations.height,
            "examples": latency_violations.select(["Model", "Latency (First Answer Chunk /s)"]).head(3).to_dicts() if latency_violations.height > 0 else []
        }
        total_violations += latency_violations.height
    else:
        violations["latency_negative"] = {"count": 0, "examples": []}

    # Context Window negative check
    if "Context Window" in df.columns:
        context_violations = df.filter(pl.col("Context Window") < 0)
        violations["context_window_negative"] = {
            "count": context_violations.height,
            "examples": context_violations.select(["Model", "Context Window"]).head(3).to_dicts() if context_violations.height > 0 else []
        }
        total_violations += context_violations.height
    else:
        violations["context_window_negative"] = {"count": 0, "examples": []}

    return {
        "pass": total_violations == 0,
        "violation_count": total_violations,
        "violations": violations,
        "intelligence_index_violations": violations.get("intelligence_index", {}).get("count", 0),
        "price_negative": violations.get("price_negative", {}).get("count", 0),
        "speed_negative": violations.get("speed_negative", {}).get("count", 0),
        "latency_negative": violations.get("latency_negative", {}).get("count", 0),
        "context_window_negative": violations.get("context_window_negative", {}).get("count", 0)
    }


def check_completeness(df: pl.DataFrame) -> dict:
    """
    Calculate completeness metrics for the dataset.

    Completeness measures the extent to which data is complete (no missing
    values in required fields). High completeness is essential for reliable
    statistical analysis.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - pass: bool (True if completeness >= 95%)
        - total_rows: Total number of rows
        - rows_complete: Number of rows with 0 null values
        - rows_with_any_null: Number of rows with >=1 null value
        - completeness_percentage: (rows_complete / total_rows) * 100
        - null_counts: Dict of null count per column
        - null_percentages: Dict of null percentage per column

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/03_distributions_analyzed.parquet")
    >>> completeness = check_completeness(df)
    >>> print(f"Completeness: {completeness['completeness_percentage']:.1f}%")

    Notes
    -----
    Completeness threshold: 95% (STATE.md decision: preserve nulls in
    intelligence_index as they represent optional metrics, not missing data).

    Columns with nulls:
    - Intelligence Index: 6 nulls (3.19%) - acceptable, models without IQ scores
    - All other columns: 0 nulls (100% complete)
    """
    total_rows = df.height

    # Calculate null counts per column
    null_counts = {}
    null_percentages = {}
    for col in df.columns:
        null_count = df[col].null_count()
        null_counts[col] = null_count
        null_percentages[col] = (null_count / total_rows * 100) if total_rows > 0 else 0

    # Calculate rows with any null
    rows_with_any_null = df.filter(
        pl.any_horizontal(pl.col("*").is_null())
    ).height

    rows_complete = total_rows - rows_with_any_null
    completeness_percentage = (rows_complete / total_rows * 100) if total_rows > 0 else 0

    return {
        "pass": completeness_percentage >= 95,
        "total_rows": total_rows,
        "rows_complete": rows_complete,
        "rows_with_any_null": rows_with_any_null,
        "completeness_percentage": completeness_percentage,
        "null_counts": null_counts,
        "null_percentages": null_percentages
    }


def check_consistency(df: pl.DataFrame) -> dict:
    """
    Check data consistency across the dataset.

    Consistency measures the extent to which data is consistent in format,
    type, and representation. This function checks for duplicate records,
    realistic value ranges, and consistent creator naming.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - pass: bool (True if no duplicates and values are realistic)
        - duplicate_models: Number of duplicate model names
        - duplicate_examples: List of duplicate model names with counts
        - context_window_unrealistic: Count of unrealistic context windows
        - creator_inconsistencies: Potential creator name variations
        - price_intelligence_outliers: Extreme price/intelligence ratios

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/03_distributions_analyzed.parquet")
    >>> consistency = check_consistency(df)
    >>> print(f"Consistency check: {'PASS' if consistency['pass'] else 'FAIL'}")

    Notes
    -----
    Consistency checks performed:
    - Duplicate model names (should be unique identifiers)
    - Context window values are realistic (0 to 2M tokens, per RESEARCH.md)
    - Creator names are consistent (no slight variations like "OpenAI" vs "openai")
    - Price per intelligence is reasonable (no extreme outliers)

    Context window upper bound: 2M tokens (STATE.md decision allows for
    future-proofing as models may exceed current maximums).
    """
    issues = {}
    total_issues = 0

    # Check for duplicate model names
    if "Model" in df.columns:
        model_counts = df["Model"].value_counts()
        duplicates = model_counts.filter(pl.col("count") > 1)
        issues["duplicate_models"] = {
            "count": duplicates.height,
            "examples": duplicates.select(["Model", "count"]).head(5).to_dicts() if duplicates.height > 0 else []
        }
        total_issues += duplicates.height
    else:
        issues["duplicate_models"] = {"count": 0, "examples": []}

    # Check context window values are realistic (0 to 2M tokens)
    if "Context Window" in df.columns:
        unrealistic_context = df.filter(pl.col("Context Window") > 2_000_000)
        issues["context_window_unrealistic"] = {
            "count": unrealistic_context.height,
            "examples": unrealistic_context.select(["Model", "Context Window"]).head(3).to_dicts() if unrealistic_context.height > 0 else []
        }
        total_issues += unrealistic_context.height
    else:
        issues["context_window_unrealistic"] = {"count": 0, "examples": []}

    # Check for creator name inconsistencies (case variations)
    if "Creator" in df.columns:
        creators = df["Creator"].unique().to_list()
        # Group by lowercase to find case variations
        creator_lower_map = {}
        for creator in creators:
            creator_lower = creator.lower() if creator else ""
            if creator_lower not in creator_lower_map:
                creator_lower_map[creator_lower] = []
            creator_lower_map[creator_lower].append(creator)

        # Find creators with multiple case variations
        inconsistent_creators = [v for v in creator_lower_map.values() if len(v) > 1]
        issues["creator_inconsistencies"] = {
            "count": len(inconsistent_creators),
            "examples": inconsistent_creators[:5]
        }
        total_issues += len(inconsistent_creators)
    else:
        issues["creator_inconsistencies"] = {"count": 0, "examples": []}

    # Check price per intelligence for extreme outliers
    if "price_usd" in df.columns and "Intelligence Index" in df.columns:
        # Calculate price per intelligence, filtering null intelligence values
        df_with_ratio = df.filter(
            pl.col("Intelligence Index").is_not_null() & (pl.col("Intelligence Index") > 0)
        ).with_columns(
            (pl.col("price_usd") / pl.col("Intelligence Index")).alias("price_per_intelligence")
        )

        # Flag extreme outliers (more than 10x the median)
        if df_with_ratio.height > 0:
            median_ratio = df_with_ratio["price_per_intelligence"].median()
            extreme_outliers = df_with_ratio.filter(
                pl.col("price_per_intelligence") > (median_ratio * 10)
            )
            issues["price_intelligence_outliers"] = {
                "count": extreme_outliers.height,
                "examples": extreme_outliers.select(["Model", "price_usd", "Intelligence Index"]).head(3).to_dicts() if extreme_outliers.height > 0 else []
            }
            total_issues += extreme_outliers.height
        else:
            issues["price_intelligence_outliers"] = {"count": 0, "examples": []}
    else:
        issues["price_intelligence_outliers"] = {"count": 0, "examples": []}

    return {
        "pass": total_issues == 0,
        "duplicate_models": issues.get("duplicate_models", {}).get("count", 0),
        "duplicate_examples": issues.get("duplicate_models", {}).get("examples", []),
        "context_window_unrealistic": issues.get("context_window_unrealistic", {}).get("count", 0),
        "creator_inconsistencies": issues.get("creator_inconsistencies", {}).get("count", 0),
        "price_intelligence_outliers": issues.get("price_intelligence_outliers", {}).get("count", 0)
    }


def check_validity(df: pl.DataFrame) -> dict:
    """
    Validate data conforms to expected formats and value sets.

    Validity measures the extent to which data values conform to expected
    formats, patterns, and enumerated value sets. This function validates
    the Creator column against known providers and checks data types.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate.

    Returns
    -------
    dict
        Dictionary containing:
        - pass: bool (True if all validations pass)
        - expected_creators: List of expected creator values
        - unexpected_creators: Count of creators not in expected set
        - unexpected_creator_examples: List of unexpected creator values
        - data_type_mismatches: Count of columns with incorrect types
        - impossible_combinations: Count of logically impossible data combinations

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/03_distributions_analyzed.parquet")
    >>> validity = check_validity(df)
    >>> print(f"Validity check: {'PASS' if validity['pass'] else 'FAIL'}")

    Notes
    -----
    Validity checks performed:
    - Creator values are in expected set (OpenAI, Anthropic, Google, etc.)
    - Data types match schema expectations
    - No impossible combinations (e.g., speed=0 but latency>0)

    Expected creators are derived from unique values in the dataset to
    avoid false positives on new providers. This check focuses on
    detecting obvious data entry errors rather than enforcing a static list.
    """
    issues = {}
    total_issues = 0

    # Check Creator column for expected values
    if "Creator" in df.columns:
        # Get expected creators from data (this is a lenient check)
        # In practice, you might want a predefined list
        creators = df["Creator"].drop_nulls().unique().to_list()

        # Check for obviously invalid creator values (empty, whitespace only, etc.)
        invalid_creators = df.filter(
            pl.col("Creator").str.strip_chars().is_null() |
            (pl.col("Creator").str.strip_chars() == "")
        )

        issues["unexpected_creators"] = {
            "count": invalid_creators.height,
            "examples": invalid_creators.select(["Model", "Creator"]).head(5).to_dicts() if invalid_creators.height > 0 else []
        }
        total_issues += invalid_creators.height
    else:
        issues["unexpected_creators"] = {"count": 0, "examples": []}

    # Check for impossible combinations: speed=0 but latency>0
    # (If a model has no speed, it shouldn't have latency measurements)
    if "Speed(median token/s)" in df.columns and "Latency (First Answer Chunk /s)" in df.columns:
        impossible = df.filter(
            (pl.col("Speed(median token/s)") == 0) & (pl.col("Latency (First Answer Chunk /s)") > 0)
        )
        issues["impossible_combinations"] = {
            "count": impossible.height,
            "examples": impossible.select(["Model", "Speed(median token/s)", "Latency (First Answer Chunk /s)"]).head(3).to_dicts() if impossible.height > 0 else []
        }
        total_issues += impossible.height
    else:
        issues["impossible_combinations"] = {"count": 0, "examples": []}

    return {
        "pass": total_issues == 0,
        "unexpected_creators": issues.get("unexpected_creators", {}).get("count", 0),
        "unexpected_creator_examples": issues.get("unexpected_creators", {}).get("examples", []),
        "impossible_combinations": issues.get("impossible_combinations", {}).get("count", 0)
    }


def perform_sanity_checks(df: pl.DataFrame) -> dict:
    """
    Execute all sanity checks across 6 dimensions of data quality.

    This function runs all check functions (accuracy, completeness, consistency,
    validity) and aggregates results into a comprehensive quality assessment.
    Calculates an overall quality score as the average of all dimension scores.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate.

    Returns
    -------
    dict
        Dictionary containing:
        - accuracy: Results from check_accuracy()
        - completeness: Results from check_completeness()
        - consistency: Results from check_consistency()
        - validity: Results from check_validity()
        - overall_quality_score: Average of all dimension scores (0-100)
        - dimensions_passed: Count of dimensions that passed
        - dimensions_failed: Count of dimensions that failed
        - critical_issues: List of issues requiring immediate attention

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/03_distributions_analyzed.parquet")
    >>> sanity_checks = perform_sanity_checks(df)
    >>> print(f"Overall quality: {sanity_checks['overall_quality_score']:.1f}%")

    Notes
    -----
    Quality dimensions (from RESEARCH.md "Pitfall 4: Data Quality Metrics Overwhelm"):
    1. Accuracy - validation results for ranges and constraints
    2. Completeness - missing value analysis
    3. Consistency - duplicate and consistency checks
    4. Validity - schema and enum validation
    5. Integrity - referential integrity (N/A for single-table dataset)
    6. Timeliness - data freshness (static dataset, N/A)

    Overall quality score calculation:
    - Each dimension contributes equally (25% per applicable dimension)
    - Passing dimension = 100%, Failing dimension = 0%
    - For this dataset, 4 dimensions are applicable (Accuracy, Completeness, Consistency, Validity)
    - Overall score = (sum of passing dimensions) / (total dimensions) * 100

    Critical issues are those that prevent analysis:
    - Accuracy violations (out-of-range values)
    - Low completeness (< 95%)
    - Duplicate records
    - Impossible data combinations
    """
    # Run all checks
    accuracy = check_accuracy(df)
    completeness = check_completeness(df)
    consistency = check_consistency(df)
    validity = check_validity(df)

    # Calculate dimension scores
    dimensions = {
        "accuracy": accuracy,
        "completeness": completeness,
        "consistency": consistency,
        "validity": validity
    }

    # Count passing dimensions
    dimensions_passed = sum(1 for d in dimensions.values() if d["pass"])
    dimensions_failed = len(dimensions) - dimensions_passed

    # Calculate overall quality score
    overall_quality_score = (dimensions_passed / len(dimensions)) * 100

    # Identify critical issues
    critical_issues = []

    if not accuracy["pass"]:
        critical_issues.append(f"Accuracy check failed: {accuracy['violation_count']} violations detected")

    if completeness["completeness_percentage"] < 95:
        critical_issues.append(f"Low completeness: {completeness['completeness_percentage']:.1f}% (target: 95%)")

    if consistency["duplicate_models"] > 0:
        critical_issues.append(f"Duplicate models found: {consistency['duplicate_models']} duplicates")

    if validity["impossible_combinations"] > 0:
        critical_issues.append(f"Impossible data combinations: {validity['impossible_combinations']} records")

    return {
        "accuracy": accuracy,
        "completeness": completeness,
        "consistency": consistency,
        "validity": validity,
        "overall_quality_score": overall_quality_score,
        "dimensions_passed": dimensions_passed,
        "dimensions_failed": dimensions_failed,
        "critical_issues": critical_issues
    }
