"""
Distribution analysis for AI models benchmark dataset.

This script performs statistical distribution analysis on numerical columns
to understand data characteristics, identify patterns, and detect anomalies.

Functions
---------
run_distribution_analysis(input_path: str, output_dir: str, output_checkpoint: str)
    Execute distribution analysis on all numerical columns including:
    - Calculate comprehensive statistics for each numerical variable
    - Detect outliers using Isolation Forest algorithm
    - Generate distribution visualizations (histogram, box plot, Q-Q plot)
    - Create markdown report with statistics and interpretations
"""

from pathlib import Path
from datetime import datetime
import polars as pl
from src.utils import setup_logging, load_checkpoint
from src.analyze import (
    analyze_distribution,
    detect_outliers_isolation_forest,
    plot_distribution,
)


def run_distribution_analysis(
    input_path: str = "data/interim/02_cleaned.parquet",
    output_dir: str = "reports/figures",
    output_checkpoint: str = "data/interim/03_distributions_analyzed.parquet",
    output_report: str = "reports/distributions.md"
) -> dict:
    """
    Execute distribution analysis on all numerical columns.

    Analyzes distribution statistics, detects outliers, and generates plots for:
    - context_window
    - intelligence_index
    - price_usd
    - Speed(median token/s)
    - Latency (First Answer Chunk /s)

    Parameters
    ----------
    input_path : str, default="data/interim/02_cleaned.parquet"
        Path to cleaned data checkpoint.
    output_dir : str, default="reports/figures"
        Directory to save distribution plots.
    output_checkpoint : str, default="data/interim/03_distributions_analyzed.parquet"
        Path to save checkpoint with outlier flags and statistics.
    output_report : str, default="reports/distributions.md"
        Path to save markdown analysis report.

    Returns
    -------
    dict
        Dictionary of distribution statistics for each column.

    Examples
    --------
    >>> stats = run_distribution_analysis()
    >>> print(stats["price_usd"]["mean"])

    Notes
    -----
    - Creates output directory if it doesn't exist
    - Generates one plot per numerical column
    - Detects outliers using Isolation Forest with 5% contamination
    - Returns statistics for all analyzed columns
    - Saves checkpoint with outlier flags to data/interim/03_distributions_analyzed.parquet
    - Creates comprehensive markdown report with statistics and visualizations
    """
    logger = setup_logging()

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = load_checkpoint(input_path, logger)
    logger.info(f"Loaded {len(df)} records")

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Numerical columns to analyze
    numerical_columns = [
        "context_window",
        "intelligence_index",
        "price_usd",
        "Speed(median token/s)",
        "Latency (First Answer Chunk /s)"
    ]

    # Analyze each column
    all_stats = {}
    for column in numerical_columns:
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found, skipping")
            continue

        logger.info(f"Analyzing distribution of {column}...")

        # Calculate statistics
        stats_result = analyze_distribution(df[column])
        all_stats[column] = stats_result

        # Log summary
        logger.info(f"  Count: {stats_result.get('count', 'N/A')}")
        logger.info(f"  Mean: {stats_result.get('mean', 'N/A'):.2f}")
        logger.info(f"  Std: {stats_result.get('std', 'N/A'):.2f}")
        logger.info(f"  Median: {stats_result.get('median', 'N/A'):.2f}")
        logger.info(f"  Range: [{stats_result.get('min', 'N/A'):.2f}, {stats_result.get('max', 'N/A'):.2f}]")
        logger.info(f"  Skewness: {stats_result.get('skewness', 'N/A'):.2f}")
        logger.info(f"  Kurtosis: {stats_result.get('kurtosis', 'N/A'):.2f}")

        # Generate plot
        plot_output = output_path / f"{column.replace(' ', '_').replace('/', '_')}_distribution.png"
        try:
            plot_distribution(df, column, str(plot_output))
            logger.info(f"  Plot saved: {plot_output}")
        except Exception as e:
            logger.error(f"  Failed to generate plot: {e}")

    logger.info(f"Distribution analysis completed for {len(all_stats)} columns")

    # Detect outliers using Isolation Forest
    logger.info("Detecting outliers using Isolation Forest...")
    outlier_columns = [col for col in numerical_columns if col in df.columns]
    df_with_outliers = detect_outliers_isolation_forest(df, outlier_columns, contamination=0.05)

    outlier_count = df_with_outliers.filter(pl.col("is_outlier")).height
    outlier_pct = (outlier_count / len(df_with_outliers)) * 100
    logger.info(f"  Total outliers detected: {outlier_count} ({outlier_pct:.2f}%)")
    logger.info(f"  Inliers: {len(df_with_outliers) - outlier_count} ({100 - outlier_pct:.2f}%)")

    # Save checkpoint with outlier flags
    logger.info(f"Saving checkpoint to {output_checkpoint}")
    df_with_outliers.write_parquet(output_checkpoint)
    logger.info("Checkpoint saved successfully")

    # Generate markdown report
    logger.info(f"Generating distribution analysis report at {output_report}")
    _generate_markdown_report(all_stats, df_with_outliers, outlier_columns, output_report)
    logger.info("Report generated successfully")

    return all_stats


def _generate_markdown_report(
    stats: dict,
    df: pl.DataFrame,
    columns: list[str],
    output_path: str
) -> None:
    """
    Generate markdown report with distribution statistics and analysis.

    Parameters
    ----------
    stats : dict
        Dictionary of distribution statistics for each column.
    df : pl.DataFrame
        DataFrame with outlier flags.
    columns : list[str]
        List of numerical columns analyzed.
    output_path : str
        Path to save markdown report.
    """
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        # Header
        f.write("# Distribution Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n")
        f.write(f"**Total Records:** {len(df)}\n\n")
        f.write("---\n\n")

        # Statistics table
        f.write("## Summary Statistics\n\n")
        f.write("| Column | Count | Mean | Std | Median | Min | Max | Skewness | Kurtosis |\n")
        f.write("|--------|-------|------|-----|--------|-----|-----|----------|----------|\n")

        for col, col_stats in stats.items():
            if "error" in col_stats:
                continue

            f.write(
                f"| {col} | "
                f"{col_stats.get('count', 0):,} | "
                f"{col_stats.get('mean', 0):.2f} | "
                f"{col_stats.get('std', 0):.2f} | "
                f"{col_stats.get('median', 0):.2f} | "
                f"{col_stats.get('min', 0):.2f} | "
                f"{col_stats.get('max', 0):.2f} | "
                f"{col_stats.get('skewness', 0):.2f} | "
                f"{col_stats.get('kurtosis', 0):.2f} |\n"
            )

        f.write("\n")

        # Distribution interpretation
        f.write("## Distribution Interpretation\n\n")

        for col, col_stats in stats.items():
            if "error" in col_stats:
                continue

            skew = col_stats.get('skewness', 0)
            kurt = col_stats.get('kurtosis', 3)

            f.write(f"### {col}\n\n")
            f.write(f"- **Skewness:** {skew:.2f} - ")
            if abs(skew) < 0.5:
                f.write("Approximately symmetric\n")
            elif skew > 0.5:
                f.write("Right-skewed (tail extends toward higher values)\n")
            else:
                f.write("Left-skewed (tail extends toward lower values)\n")

            f.write(f"- **Kurtosis:** {kurt:.2f} - ")
            if kurt > 3.5:
                f.write("Heavy-tailed (more outliers than normal distribution)\n")
            elif kurt < 2.5:
                f.write("Light-tailed (fewer outliers than normal distribution)\n")
            else:
                f.write("Normal-like tail behavior\n")

            # Normality test
            norm_test = col_stats.get("normality_test", {})
            if "p_value" in norm_test:
                p_val = norm_test["p_value"]
                f.write(f"- **Normality test:** p-value = {p_val:.4f} - ")
                if p_val < 0.05:
                    f.write("**Not normally distributed** (reject null hypothesis)\n")
                else:
                    f.write("Normally distributed (cannot reject null hypothesis)\n")

            f.write("\n")

        # Outlier analysis
        f.write("## Outlier Analysis\n\n")
        outlier_count = df.filter(pl.col("is_outlier")).height
        outlier_pct = (outlier_count / len(df)) * 100

        f.write(f"- **Method:** Isolation Forest (contamination=0.05)\n")
        f.write(f"- **Total outliers detected:** {outlier_count} ({outlier_pct:.2f}%)\n")
        f.write(f"- **Total inliers:** {len(df) - outlier_count} ({100 - outlier_pct:.2f}%)\n\n")

        if outlier_count > 0:
            outliers_df = df.filter(pl.col("is_outlier"))
            f.write("### Outlier Details\n\n")
            f.write("Models flagged as outliers:\n\n")
            f.write("| Model | Creator | Price | Intelligence | Speed | Latency | Outlier Score |\n")
            f.write("|-------|---------|-------|--------------|-------|---------|---------------|\n")

            for row in outliers_df.iter_rows(named=True):
                # Safely extract and format numeric values
                price = row.get('price_usd', 0)
                if price is not None and price != '':
                    try:
                        price_str = f"${float(price):.2f}"
                    except (ValueError, TypeError):
                        price_str = str(price)
                else:
                    price_str = "N/A"

                intelligence = row.get('intelligence_index')
                intelligence_str = str(intelligence) if intelligence is not None else "N/A"

                speed = row.get('Speed(median token/s)', 0)
                if speed is not None and speed != '':
                    try:
                        speed_str = f"{float(speed):.1f}"
                    except (ValueError, TypeError):
                        speed_str = str(speed)
                else:
                    speed_str = "N/A"

                latency = row.get('Latency (First Answer Chunk /s)', 0)
                if latency is not None and latency != '':
                    try:
                        latency_str = f"{float(latency):.1f}"
                    except (ValueError, TypeError):
                        latency_str = str(latency)
                else:
                    latency_str = "N/A"

                outlier_score = row.get('outlier_score', 0)
                if outlier_score is not None:
                    try:
                        score_str = f"{float(outlier_score):.3f}"
                    except (ValueError, TypeError):
                        score_str = str(outlier_score)
                else:
                    score_str = "N/A"

                f.write(
                    f"| {row.get('Model', 'N/A')} | "
                    f"{row.get('Creator', 'N/A')} | "
                    f"{price_str} | "
                    f"{intelligence_str} | "
                    f"{speed_str} | "
                    f"{latency_str} | "
                    f"{score_str} |\n"
                )

        f.write("\n")

        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("Distribution plots have been generated for all numerical columns:\n\n")

        for col in columns:
            safe_name = col.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            f.write(f"- **{col}:** `reports/figures/{safe_name}_distribution.png`\n")

        f.write("\n")
        f.write("Each plot includes:\n")
        f.write("- Histogram with KDE curve (distribution shape)\n")
        f.write("- Box plot (quartiles and outliers)\n")
        f.write("- Q-Q plot (normality assessment)\n")

    logger = setup_logging()
    logger.info(f"Markdown report saved to {output_path}")


if __name__ == "__main__":
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("Starting distribution analysis")

    # Run analysis
    stats = run_distribution_analysis()

    # Print summary statistics
    logger.info("\n=== Distribution Summary ===")
    for column, stat in stats.items():
        print(f"\n{column}:")
        print(f"  Count: {stat.get('count', 'N/A'):,}")
        print(f"  Mean: {stat.get('mean', 'N/A'):.2f}")
        print(f"  Std: {stat.get('std', 'N/A'):.2f}")
        print(f"  Median: {stat.get('median', 'N/A'):.2f}")
        print(f"  Range: [{stat.get('min', 'N/A'):.2f}, {stat.get('max', 'N/A'):.2f}]")
        print(f"  Skewness: {stat.get('skewness', 'N/A'):.2f}")
        print(f"  Kurtosis: {stat.get('kurtosis', 'N/A'):.2f}")

    logger.info("\nDistribution analysis completed successfully")
    logger.info(f"Check report: reports/distributions.md")
    logger.info(f"Check figures: reports/figures/")
    logger.info(f"Checkpoint saved: data/interim/03_distributions_analyzed.parquet")
