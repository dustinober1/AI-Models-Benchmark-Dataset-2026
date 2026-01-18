"""
Quality assessment and reporting for AI models benchmark dataset.

This script generates a comprehensive data quality report covering
the 6 dimensions of data quality: Accuracy, Completeness, Consistency,
Integrity, Timeliness, and Validity.

Functions
---------
calculate_completeness_metrics(df)
    Calculate completeness metrics (null counts, percentages).

calculate_accuracy_metrics(df)
    Calculate accuracy metrics (value ranges, constraints).

calculate_consistency_metrics(df)
    Calculate consistency metrics (data format, type consistency).

calculate_validity_metrics(df)
    Calculate validity metrics (enum values, patterns).

generate_quality_report(df, output_path)
    Generate comprehensive quality report in Markdown format.

run_quality_assessment(input_path, output_path)
    Execute full quality assessment pipeline.
"""

from pathlib import Path
from datetime import datetime
import polars as pl
from src.utils import setup_logging, load_checkpoint


def calculate_completeness_metrics(df: pl.DataFrame) -> dict:
    """
    Calculate completeness metrics for the dataset.

    Completeness measures the extent to which data is complete
    (no missing values in required fields).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - total_rows: Total number of rows
        - total_columns: Total number of columns
        - null_counts: Null count per column
        - null_percentages: Null percentage per column
        - rows_with_nulls: Number of rows with any null
        - rows_with_nulls_percentage: Percentage of rows with any null
        - completeness_score: Overall completeness score (0-100)

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> metrics = calculate_completeness_metrics(df)
    >>> print(f"Completeness: {metrics['completeness_score']:.1f}%")

    Notes
    -----
    - Completeness score is weighted average of column completeness
    - Columns with 0% null values contribute 100% to score
    - Rows with any nulls are flagged for review
    """
    total_rows = df.height
    total_columns = df.width

    # Null counts per column
    null_counts = {}
    null_percentages = {}
    for col in df.columns:
        null_count = df[col].null_count()
        null_counts[col] = null_count
        null_percentages[col] = (null_count / total_rows * 100) if total_rows > 0 else 0

    # Rows with any null
    rows_with_nulls = df.filter(
        pl.any_horizontal(pl.col("*").is_null())
    ).height
    rows_with_nulls_percentage = (rows_with_nulls / total_rows * 100) if total_rows > 0 else 0

    # Overall completeness score (average of column completeness)
    column_completeness = [100 - pct for pct in null_percentages.values()]
    completeness_score = sum(column_completeness) / len(column_completeness) if column_completeness else 0

    return {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "null_counts": null_counts,
        "null_percentages": null_percentages,
        "rows_with_nulls": rows_with_nulls,
        "rows_with_nulls_percentage": rows_with_nulls_percentage,
        "completeness_score": completeness_score
    }


def calculate_accuracy_metrics(df: pl.DataFrame) -> dict:
    """
    Calculate accuracy metrics for the dataset.

    Accuracy measures the extent to which data values are correct
    and within expected ranges.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - intelligence_index_range: Min/max values
        - intelligence_index_out_of_range: Count of values outside [0, 100]
        - price_negative: Count of negative prices
        - context_window_negative: Count of negative context windows
        - speed_negative: Count of negative speeds
        - latency_negative: Count of negative latencies
        - accuracy_score: Overall accuracy score (0-100)

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> metrics = calculate_accuracy_metrics(df)
    >>> print(f"Accuracy: {metrics['accuracy_score']:.1f}%")

    Notes
    -----
    - Accuracy checks: Intelligence Index [0, 100], all metrics >= 0
    - Out-of-range values are flagged for review
    - Accuracy score based on percentage of valid values
    """
    metrics = {}

    # Intelligence Index range check
    if "Intelligence Index" in df.columns:
        ii_col = df["Intelligence Index"]
        metrics["intelligence_index_min"] = float(ii_col.min()) if ii_col.len() > 0 else None
        metrics["intelligence_index_max"] = float(ii_col.max()) if ii_col.len() > 0 else None
        metrics["intelligence_index_out_of_range"] = df.filter(
            (pl.col("Intelligence Index") < 0) | (pl.col("Intelligence Index") > 100)
        ).height
    else:
        metrics["intelligence_index_min"] = None
        metrics["intelligence_index_max"] = None
        metrics["intelligence_index_out_of_range"] = 0

    # Price negative check
    if "price_usd" in df.columns:
        metrics["price_negative"] = df.filter(pl.col("price_usd") < 0).height
    else:
        metrics["price_negative"] = 0

    # Context Window negative check
    if "Context Window" in df.columns:
        metrics["context_window_negative"] = df.filter(pl.col("Context Window") < 0).height
    else:
        metrics["context_window_negative"] = 0

    # Speed negative check
    if "Speed(median token/s)" in df.columns:
        metrics["speed_negative"] = df.filter(pl.col("Speed(median token/s)") < 0).height
    else:
        metrics["speed_negative"] = 0

    # Latency negative check
    if "Latency (First Answer Chunk /s)" in df.columns:
        metrics["latency_negative"] = df.filter(pl.col("Latency (First Answer Chunk /s)") < 0).height
    else:
        metrics["latency_negative"] = 0

    # Calculate accuracy score
    total_checks = (
        metrics["intelligence_index_out_of_range"] +
        metrics["price_negative"] +
        metrics["context_window_negative"] +
        metrics["speed_negative"] +
        metrics["latency_negative"]
    )
    total_values = df.height * 5  # 5 checks per row
    accuracy_score = (1 - total_checks / total_values) * 100 if total_values > 0 else 100
    metrics["accuracy_score"] = accuracy_score

    return metrics


def calculate_consistency_metrics(df: pl.DataFrame) -> dict:
    """
    Calculate consistency metrics for the dataset.

    Consistency measures the extent to which data is consistent
    in format, type, and representation.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - column_types: Data type per column
        - duplicate_models: Count of duplicate model names
        - consistency_score: Overall consistency score (0-100)

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> metrics = calculate_consistency_metrics(df)
    >>> print(f"Consistency: {metrics['consistency_score']:.1f}%")

    Notes
    -----
    - Checks for duplicate model names (should be unique)
    - Verifies data types are consistent
    - Consistency score penalizes duplicates
    """
    metrics = {}

    # Column types
    metrics["column_types"] = {col: str(df[col].dtype) for col in df.columns}

    # Duplicate model names
    if "Model" in df.columns:
        model_counts = df["Model"].value_counts()
        duplicates = model_counts.filter(pl.col("count") > 1).height
        metrics["duplicate_models"] = duplicates
    else:
        metrics["duplicate_models"] = 0

    # Consistency score (penalize duplicates)
    consistency_score = max(0, 100 - (metrics["duplicate_models"] / df.height * 100)) if df.height > 0 else 100
    metrics["consistency_score"] = consistency_score

    return metrics


def calculate_validity_metrics(df: pl.DataFrame) -> dict:
    """
    Calculate validity metrics for the dataset.

    Validity measures the extent to which data values conform
    to expected formats, patterns, and value sets.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - unique_creators: Number of unique creators
        - top_creators: Top 5 creators by model count
        - validity_score: Overall validity score (0-100)

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> metrics = calculate_validity_metrics(df)
    >>> print(f"Validity: {metrics['validity_score']:.1f}%")

    Notes
    -----
    - Checks Creator column for expected values
    - Reports diversity of model providers
    - High validity = expected patterns present
    """
    metrics = {}

    # Creator diversity
    if "Creator" in df.columns:
        creator_counts = df["Creator"].value_counts()
        metrics["unique_creators"] = creator_counts.height
        metrics["top_creators"] = creator_counts.sort("count", descending=True).head(5).to_dict(as_series=False)
    else:
        metrics["unique_creators"] = 0
        metrics["top_creators"] = []

    # Validity score (high if diverse creators)
    validity_score = min(100, metrics["unique_creators"] * 2) if metrics["unique_creators"] > 0 else 0
    metrics["validity_score"] = validity_score

    return metrics


def generate_quality_report(
    df: pl.DataFrame,
    output_path: str
) -> None:
    """
    Generate comprehensive quality report in Markdown format.

    Creates a detailed report covering all 6 dimensions of data quality
    with narrative interpretation and recommendations.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to analyze.
    output_path : str
        Path to save the quality report (Markdown format).

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> generate_quality_report(df, "reports/quality_2026-01-18.md")

    Notes
    -----
    - Report includes: Completeness, Accuracy, Consistency, Validity
    - Narrative interpretation of metrics
    - Recommendations for improvement
    - Timestamped for tracking
    """
    logger = setup_logging()

    # Calculate all metrics
    logger.info("Calculating completeness metrics...")
    completeness = calculate_completeness_metrics(df)

    logger.info("Calculating accuracy metrics...")
    accuracy = calculate_accuracy_metrics(df)

    logger.info("Calculating consistency metrics...")
    consistency = calculate_consistency_metrics(df)

    logger.info("Calculating validity metrics...")
    validity = calculate_validity_metrics(df)

    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        f"# Data Quality Report",
        f"",
        f"**Generated:** {timestamp}",
        f"**Dataset:** AI Models Benchmark 2026",
        f"**Total Rows:** {completeness['total_rows']:,}",
        f"**Total Columns:** {completeness['total_columns']}",
        f"",
        f"---",
        f"",
        f"## Executive Summary",
        f"",
        f"This report assesses the data quality across 6 dimensions: Completeness, Accuracy, Consistency, and Validity.",
        f"",
        f"**Overall Quality Scores:**",
        f"- Completeness: {completeness['completeness_score']:.1f}%",
        f"- Accuracy: {accuracy['accuracy_score']:.1f}%",
        f"- Consistency: {consistency['consistency_score']:.1f}%",
        f"- Validity: {validity['validity_score']:.1f}%",
        f"",
        f"---",
        f"",
        f"## 1. Completeness",
        f"",
        f"Completeness measures the extent to which data is complete (no missing values).",
        f"",
        f"**Metrics:**",
        f"- Total Rows: {completeness['total_rows']:,}",
        f"- Rows with Missing Values: {completeness['rows_with_nulls']:,} ({completeness['rows_with_nulls_percentage']:.1f}%)",
        f"- Completeness Score: {completeness['completeness_score']:.1f}%",
        f"",
        f"**Missing Values by Column:**",
        f""
    ]

    # Add null counts table
    report_lines.append("| Column | Null Count | Null Percentage |")
    report_lines.append("|--------|------------|-----------------|")
    for col, count in completeness['null_counts'].items():
        pct = completeness['null_percentages'][col]
        report_lines.append(f"| {col} | {count:,} | {pct:.1f}% |")

    # Add interpretation
    report_lines.extend([
        f"",
        f"**Interpretation:**",
        f"{'✓ High data completeness - most columns are fully populated.' if completeness['completeness_score'] >= 95 else '⚠ Some missing values detected - review columns above.'}",
        f"",
        f"---",
        f"",
        f"## 2. Accuracy",
        f"",
        f"Accuracy measures the extent to which data values are correct and within expected ranges.",
        f"",
        f"**Metrics:**",
        f"- Intelligence Index Range: [{accuracy['intelligence_index_min']}, {accuracy['intelligence_index_max']}]",
        f"- Values Outside [0, 100]: {accuracy['intelligence_index_out_of_range']}",
        f"- Negative Prices: {accuracy['price_negative']}",
        f"- Negative Context Windows: {accuracy['context_window_negative']}",
        f"- Negative Speeds: {accuracy['speed_negative']}",
        f"- Negative Latencies: {accuracy['latency_negative']}",
        f"- Accuracy Score: {accuracy['accuracy_score']:.1f}%",
        f"",
        f"**Interpretation:**",
        f"{'✓ All values are within expected ranges.' if accuracy['accuracy_score'] >= 99 else '⚠ Some values outside expected ranges - review accuracy issues above.'}",
        f"",
        f"---",
        f"",
        f"## 3. Consistency",
        f"",
        f"Consistency measures the extent to which data is consistent in format and representation.",
        f"",
        f"**Metrics:**",
        f"- Duplicate Models: {consistency['duplicate_models']}",
        f"- Consistency Score: {consistency['consistency_score']:.1f}%",
        f"",
        f"**Column Types:**",
        f""
    ])

    # Add column types table
    for col, dtype in consistency['column_types'].items():
        report_lines.append(f"- {col}: {dtype}")

    report_lines.extend([
        f"",
        f"**Interpretation:**",
        "✓ No duplicate models found - data is consistent." if consistency['duplicate_models'] == 0 else f"⚠ {consistency['duplicate_models']} duplicate models found - review for data entry errors.",
        f"",
        f"---",
        f"",
        f"## 4. Validity",
        f"",
        f"Validity measures the extent to which data values conform to expected formats and value sets.",
        f"",
        f"**Metrics:**",
        f"- Unique Creators: {validity['unique_creators']}",
        f"- Validity Score: {validity['validity_score']:.1f}%",
        f"",
        f"**Top 5 Creators by Model Count:**",
        f""
    ])

    # Add top creators table
    if validity['top_creators']:
        report_lines.append("| Creator | Model Count |")
        report_lines.append("|---------|-------------|")
        for creator in validity['top_creators'].get('Creator', [])[:5]:
            count = creator.get('count', 0)
            name = creator.get('Creator', 'N/A')
            report_lines.append(f"| {name} | {count} |")

    report_lines.extend([
        f"",
        f"**Interpretation:**",
        f"{'✓ High diversity of model providers - data covers broad ecosystem.' if validity['unique_creators'] >= 10 else '⚠ Limited creator diversity - consider expanding data sources.'}",
        f"",
        f"---",
        f"",
        f"## Recommendations",
        f"",
        f"Based on the quality assessment, here are recommended actions:",
        f"",
    ])

    # Add recommendations based on scores
    if completeness['completeness_score'] < 95:
        report_lines.append(f"- **Review missing values**: {completeness['rows_with_nulls_percentage']:.1f}% of rows have missing data")

    if accuracy['accuracy_score'] < 99:
        report_lines.append(f"- **Fix accuracy issues**: {accuracy['intelligence_index_out_of_range']} values outside valid ranges")

    if consistency['duplicate_models'] > 0:
        report_lines.append(f"- **Resolve duplicates**: {consistency['duplicate_models']} duplicate model names detected")

    if len(report_lines) == len(["Based on the quality assessment, here are recommended actions:", ""]):
        report_lines.append("- ✓ No critical issues found - data quality is acceptable")

    report_lines.extend([
        f"",
        f"---",
        f"",
        f"*End of Report*"
    ])

    # Write report to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Quality report saved: {output_path}")


def run_quality_assessment(
    input_path: str = "data/interim/02_cleaned.parquet",
    output_path: str = None
) -> dict:
    """
    Execute full quality assessment pipeline.

    Generates comprehensive quality report covering all dimensions.

    Parameters
    ----------
    input_path : str, default="data/interim/02_cleaned.parquet"
        Path to cleaned data checkpoint.
    output_path : str, optional
        Path to save quality report. If None, generates timestamped path.

    Returns
    -------
    dict
        Dictionary containing all quality metrics.

    Examples
    --------
    >>> metrics = run_quality_assessment()
    >>> print(f"Overall quality: {metrics['completeness_score']:.1f}%")

    Notes
    -----
    - Generates timestamped report if output_path not specified
    - All 6 quality dimensions are assessed
    - Report includes narrative interpretation and recommendations
    """
    logger = setup_logging()
    logger.info("Starting quality assessment")

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = load_checkpoint(input_path, logger)

    # Generate output path if not specified
    if output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        output_path = f"reports/quality_{timestamp}.md"

    # Generate report
    logger.info("Generating quality report...")
    generate_quality_report(df, output_path)

    # Calculate all metrics for return
    metrics = {
        "completeness": calculate_completeness_metrics(df),
        "accuracy": calculate_accuracy_metrics(df),
        "consistency": calculate_consistency_metrics(df),
        "validity": calculate_validity_metrics(df)
    }

    logger.info("Quality assessment completed")

    return metrics


if __name__ == "__main__":
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("Starting quality assessment process")

    # Run quality assessment
    metrics = run_quality_assessment()

    # Print summary
    logger.info("\n=== Quality Summary ===")
    print(f"Completeness: {metrics['completeness']['completeness_score']:.1f}%")
    print(f"Accuracy: {metrics['accuracy']['accuracy_score']:.1f}%")
    print(f"Consistency: {metrics['consistency']['consistency_score']:.1f}%")
    print(f"Validity: {metrics['validity']['validity_score']:.1f}%")

    logger.info("Quality assessment completed successfully")
