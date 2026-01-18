"""
Distribution analysis for AI models benchmark dataset.

This script performs statistical distribution analysis on numerical columns
to understand data characteristics, identify patterns, and detect anomalies.

Functions
---------
analyze_distribution(series: pl.Series) -> dict
    Calculate comprehensive distribution statistics using scipy.stats.

plot_distribution(df: pl.DataFrame, column: str, output_path: str)
    Create comprehensive distribution plot with histogram, box plot, and Q-Q plot.

run_distribution_analysis(input_path: str, output_dir: str)
    Execute distribution analysis on all numerical columns.
"""

from pathlib import Path
from scipy import stats
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import setup_logging, load_checkpoint


def analyze_distribution(series: pl.Series) -> dict:
    """
    Calculate comprehensive distribution statistics for a numerical column.

    Computes descriptive statistics and tests for normality using
    scipy.stats functions. Handles missing values appropriately.

    Parameters
    ----------
    series : pl.Series
        Polars Series with numerical data to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - count: Number of non-null values
        - mean: Arithmetic mean
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - median: Median (50th percentile)
        - q25: 25th percentile (first quartile)
        - q75: 75th percentile (third quartile)
        - skewness: Measure of distribution asymmetry
        - kurtosis: Measure of distribution tailedness
        - normality_test: Result of scipy.stats.normaltest

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> stats = analyze_distribution(df["price_usd"])
    >>> print(f"Mean: {stats['mean']:.2f}, Skewness: {stats['skewness']:.2f}")

    Notes
    -----
    - Skewness > 0: Right-skewed (tail on right)
    - Skewness < 0: Left-skewed (tail on left)
    - Skewness = 0: Symmetric distribution
    - Kurtosis > 3: Heavy-tailed (more outliers than normal)
    - Kurtosis < 3: Light-tailed (fewer outliers than normal)
    - Normality test p-value < 0.05: Not normally distributed
    """
    # Drop null values for statistical calculations
    data = series.drop_nulls().to_numpy()

    if len(data) == 0:
        return {
            "count": 0,
            "error": "No valid data points"
        }

    # Descriptive statistics
    result = {
        "count": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data, fisher=False)),  # Excess kurtosis
    }

    # Normality test (requires at least 8 data points)
    if len(data) >= 8:
        normality_result = stats.normaltest(data)
        result["normality_test"] = {
            "statistic": float(normality_result.statistic),
            "p_value": float(normality_result.pvalue)
        }
    else:
        result["normality_test"] = {
            "error": "Insufficient data for normality test (need >= 8 points)"
        }

    return result


def plot_distribution(
    df: pl.DataFrame,
    column: str,
    output_path: str
) -> None:
    """
    Create comprehensive distribution plot for a numerical column.

    Generates a figure with three subplots:
    1. Histogram with KDE curve
    2. Box plot for outlier detection
    3. Q-Q plot for normality assessment

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the column to plot.
    column : str
        Name of the column to plot.
    output_path : str
        Path to save the generated figure (PNG format).

    Raises
    ------
    ValueError
        If column is not found in DataFrame or is not numerical.

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> plot_distribution(df, "price_usd", "reports/figures/price_distribution.png")

    Notes
    -----
    - Figure size: 15x4 inches (three subplots side-by-side)
    - DPI: 300 for high-quality publication figures
    - File format: PNG with tight bounding box
    - Color palette: seaborn default style
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    # Extract data and drop nulls
    data = df[column].drop_nulls().to_numpy()

    if len(data) == 0:
        raise ValueError(f"Column '{column}' has no valid data to plot")

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Histogram with KDE
    sns.histplot(data, kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution of {column}")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Frequency")

    # 2. Box plot
    sns.boxplot(y=data, ax=axes[1])
    axes[1].set_title(f"Box Plot of {column}")
    axes[1].set_ylabel(column)

    # 3. Q-Q plot for normality
    stats.probplot(data, dist="norm", plot=axes[2])
    axes[2].set_title("Q-Q Plot (Normal Distribution)")
    axes[2].set_xlabel("Theoretical Quantiles")
    axes[2].set_ylabel("Sample Quantiles")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_distribution_analysis(
    input_path: str = "data/interim/02_cleaned.parquet",
    output_dir: str = "reports/figures"
) -> dict:
    """
    Execute distribution analysis on all numerical columns.

    Analyzes distribution statistics and generates plots for:
    - price_usd
    - Intelligence Index
    - Speed(median token/s)
    - Latency (First Answer Chunk /s)
    - Context Window

    Parameters
    ----------
    input_path : str, default="data/interim/02_cleaned.parquet"
        Path to cleaned data checkpoint.
    output_dir : str, default="reports/figures"
        Directory to save distribution plots.

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
    - Returns statistics for all analyzed columns
    """
    logger = setup_logging()

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = load_checkpoint(input_path, logger)

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Numerical columns to analyze
    numerical_columns = [
        "price_usd",
        "Intelligence Index",
        "Speed(median token/s)",
        "Latency (First Answer Chunk /s)",
        "Context Window"
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
        logger.info(f"  Mean: {stats_result.get('mean', 'N/A'):.2f}")
        logger.info(f"  Std: {stats_result.get('std', 'N/A'):.2f}")
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

    return all_stats


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

    logger.info("Distribution analysis completed successfully")
