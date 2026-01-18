"""
Statistical analysis utilities for AI models benchmark dataset.

This module provides functions for distribution analysis, outlier detection,
and visualization of numerical data using scipy.stats, sklearn, and Polars.

Functions
---------
analyze_distribution(series: pl.Series) -> dict
    Calculate comprehensive distribution statistics using scipy.stats.

detect_outliers_isolation_forest(df: pl.DataFrame, columns: list[str], contamination: float = 0.05) -> pl.DataFrame
    Detect multivariate outliers using sklearn Isolation Forest algorithm.

plot_distribution(df: pl.DataFrame, column: str, output_path: str) -> None
    Create comprehensive distribution plot with histogram, box plot, and Q-Q plot.

plot_all_distributions(df: pl.DataFrame, columns: list[str], output_dir: str) -> None
    Generate distribution plots for all specified numerical columns.
"""

from scipy import stats
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


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
        - skewness: Measure of distribution asymmetry (positive=right-skewed, negative=left-skewed)
        - kurtosis: Measure of distribution tailedness (Fisher=False, so >3=heavy-tailed)
        - normality_test: Result of scipy.stats.normaltest (tests if distribution is Gaussian)

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> stats = analyze_distribution(df["price_usd"])
    >>> print(f"Mean: {stats['mean']:.2f}, Skewness: {stats['skewness']:.2f}")

    Notes
    -----
    - Skewness > 0: Right-skewed (tail on right, more high values)
    - Skewness < 0: Left-skewed (tail on left, more low values)
    - Skewness = 0: Symmetric distribution
    - Kurtosis > 3: Heavy-tailed (more outliers than normal distribution)
    - Kurtosis < 3: Light-tailed (fewer outliers than normal distribution)
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
        "kurtosis": float(stats.kurtosis(data, fisher=False)),  # Excess kurtosis (Fisher=False)
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


def detect_outliers_isolation_forest(
    df: pl.DataFrame,
    columns: list[str],
    contamination: float = 0.05
) -> pl.DataFrame:
    """
    Detect multivariate outliers using Isolation Forest algorithm.

    Isolation Forest is an unsupervised learning algorithm that identifies
    anomalies by isolating observations in random feature subsets. It's
    particularly effective for high-dimensional datasets and is robust to
    the "masking effect" where outliers hide each other in traditional methods
    like IQR or z-score.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with numerical features.
    columns : list[str]
        List of numerical column names to use for outlier detection.
    contamination : float, default=0.05
        Expected proportion of outliers in the dataset (0.05 = 5%).
        This parameter should be adjusted based on domain knowledge and
        data exploration. Lower values detect fewer, more extreme outliers.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with two new columns:
        - is_outlier: boolean (True for outliers, False for inliers)
        - outlier_score: float (lower = more anomalous, negative scores are outliers)

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> df_with_outliers = detect_outliers_isolation_forest(
    ...     df,
    ...     columns=["price_usd", "Speed(median token/s)", "intelligence_index"],
    ...     contamination=0.05
    ... )
    >>> print(df_with_outliers.filter(pl.col("is_outlier")))

    Notes
    -----
    Algorithm advantages over IQR/z-score:
    - Handles multivariate outliers (considers relationships between features)
    - Robust to masking effect (outliers can't hide each other)
    - Works well with high-dimensional data
    - No assumption of normal distribution
    - Computationally efficient for large datasets

    Contamination parameter guidance:
    - 0.01 (1%): Detect only extreme outliers
    - 0.05 (5%): Default, balanced approach for general use
    - 0.10 (10%): More sensitive, detects more potential anomalies
    - Adjust based on domain knowledge and data quality

    Reference
    ----------
    Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation forest.
    In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.
    """
    # Extract numerical features
    X = df.select(columns).to_numpy()

    # Initialize Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,  # Reproducibility
        n_jobs=-1  # Use all CPU cores
    )

    # Fit model and predict outliers
    iso_forest.fit(X)
    outlier_labels = iso_forest.predict(X)  # -1 for outliers, 1 for inliers
    outlier_scores = iso_forest.score_samples(X)  # Lower = more anomalous

    # Add columns to DataFrame
    result = df.with_columns(
        pl.Series("is_outlier", outlier_labels == -1),
        pl.Series("outlier_score", outlier_scores)
    )

    return result


def plot_distribution(
    df: pl.DataFrame,
    column: str,
    output_path: str
) -> None:
    """
    Create comprehensive distribution plot for a numerical column.

    Generates a figure with three subplots for comprehensive distribution analysis:
    1. Histogram with KDE curve - shows overall distribution shape
    2. Box plot - identifies outliers and quartiles
    3. Q-Q plot - assesses normality by comparing to theoretical normal distribution

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
    - Color palette: seaborn default style with context="talk"
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


def plot_all_distributions(
    df: pl.DataFrame,
    columns: list[str],
    output_dir: str
) -> None:
    """
    Generate distribution plots for all specified numerical columns.

    Creates comprehensive distribution plots (histogram+KDE, box plot, Q-Q plot)
    for each column and saves them as individual PNG files.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the columns to plot.
    columns : list[str]
        List of column names to generate plots for.
    output_dir : str
        Directory to save the generated plots.

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> plot_all_distributions(
    ...     df,
    ...     ["price_usd", "intelligence_index", "Speed(median token/s)"],
    ...     "reports/figures/"
    ... )

    Notes
    -----
    - Creates output directory if it doesn't exist
    - Saves each plot as {column_name}_distribution.png
    - Column names are sanitized (spaces and special characters replaced with underscores)
    - Skips columns that are not found in DataFrame with a warning
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found, skipping")
            continue

        # Sanitize column name for filename
        safe_filename = column.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        plot_output = output_path / f"{safe_filename}_distribution.png"

        try:
            plot_distribution(df, column, str(plot_output))
            print(f"Plot saved: {plot_output}")
        except Exception as e:
            print(f"Failed to generate plot for '{column}': {e}")
