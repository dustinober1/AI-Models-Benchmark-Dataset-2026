"""
Outlier detection for AI models benchmark dataset.

This script uses Isolation Forest algorithm to detect multivariate outliers
in the dataset. Outliers are models with unusual combinations of features
(e.g., very high price but low intelligence, extreme latency, etc.).

Functions
---------
detect_outliers_isolation_forest(df, columns, contamination)
    Detect outliers using Isolation Forest algorithm.

quarantine_outliers(df, output_path, columns)
    Detect and quarantine outliers to separate file.

run_outlier_detection(input_path, output_path, columns)
    Execute full outlier detection pipeline.
"""

from pathlib import Path
from sklearn.ensemble import IsolationForest
import polars as pl
from src.utils import setup_logging, load_checkpoint, save_checkpoint, get_quarantine_path


def detect_outliers_isolation_forest(
    df: pl.DataFrame,
    columns: list[str],
    contamination: float = 0.05
) -> pl.DataFrame:
    """
    Detect outliers using Isolation Forest algorithm.

    Isolation Forest is an unsupervised learning algorithm that isolates
    anomalies by randomly selecting features and split values. Anomalies
    are identified as observations with short average path lengths.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with features for outlier detection.
    columns : list of str
        Numerical columns to use for outlier detection.
    contamination : float, default=0.05
        Expected proportion of outliers in the dataset (0.0 to 0.5).
        Higher values = more outliers detected.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with two additional columns:
        - is_outlier: Boolean flag (True if outlier)
        - outlier_score: Anomaly score (lower = more anomalous)

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> df_with_outliers = detect_outliers_isolation_forest(
    ...     df,
    ...     columns=["price_usd", "speed", "latency", "intelligence_index"],
    ...     contamination=0.05
    ... )
    >>> outliers = df_with_outliers.filter(pl.col("is_outlier"))
    >>> print(f"Detected {outliers.height} outliers")

    Notes
    -----
    - Isolation Forest is robust to the "masking effect" where outliers
      hide each other in univariate methods like IQR or z-score.
    - Algorithm works by isolating observations through random splits
    - Outliers have shorter average path lengths (easier to isolate)
    - contamination parameter should be set based on domain knowledge
    - Random state is fixed at 42 for reproducibility
    - All CPU cores are used (n_jobs=-1) for performance
    """
    # Validate columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Extract numerical features
    X = df.select(columns).to_numpy()

    # Handle missing values (IsolationForest doesn't support NaN)
    # For now, we'll drop rows with any NaN in selected columns
    nan_mask = df.select(columns).null_count().row(0)
    if any(nan_mask):
        raise ValueError(
            f"Missing values detected in columns: "
            f"{[col for col, count in zip(columns, nan_mask) if count > 0]}. "
            "Please handle missing values before outlier detection."
        )

    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1  # Use all cores
    )
    iso_forest.fit(X)

    # Predict outliers (-1) vs inliers (1)
    outlier_labels = iso_forest.predict(X)
    outlier_scores = iso_forest.score_samples(X)

    # Add to DataFrame
    result = df.with_columns(
        [
            pl.Series("is_outlier", outlier_labels == -1),
            pl.Series("outlier_score", outlier_scores)
        ]
    )

    return result


def quarantine_outliers(
    df: pl.DataFrame,
    output_path: str,
    columns: list[str],
    contamination: float = 0.05
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Detect and quarantine outliers to separate file.

    Splits dataset into clean and outlier DataFrames based on
    Isolation Forest outlier detection.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame to analyze.
    output_path : str
        Base path for quarantined outliers file.
    columns : list of str
        Numerical columns to use for outlier detection.
    contamination : float, default=0.05
        Expected proportion of outliers.

    Returns
    -------
    clean_df : pl.DataFrame
        DataFrame with outliers removed.
    outliers_df : pl.DataFrame
        DataFrame containing only outlier records.

    Examples
    --------
    >>> df = pl.read_parquet("data/interim/02_cleaned.parquet")
    >>> clean, outliers = quarantine_outliers(
    ...     df,
    ...     "data/quarantine/outliers.csv",
    ...     columns=["price_usd", "intelligence_index", "speed"],
    ...     contamination=0.05
    ... )
    >>> print(f"Clean: {clean.height}, Outliers: {outliers.height}")

    Notes
    -----
    - Quarantine file includes timestamp to prevent overwrites
    - Both DataFrames preserve outlier_score column for transparency
    - Output path is generated with timestamp via get_quarantine_path()
    """
    logger = setup_logging()

    # Detect outliers
    logger.info(f"Detecting outliers with contamination={contamination}")
    df_with_outliers = detect_outliers_isolation_forest(df, columns, contamination)

    # Split into clean and outliers
    outliers_df = df_with_outliers.filter(pl.col("is_outlier"))
    clean_df = df_with_outliers.filter(pl.col("is_outlier").not_())

    logger.info(f"Detected {outliers_df.height} outliers out of {df.height} records")

    # Save quarantined outliers
    quarantine_path = get_quarantine_path(output_path, reason="isolation_forest")
    save_checkpoint(outliers_df, quarantine_path, logger)

    # Save clean dataset
    clean_path = output_path.replace(".csv", "_clean.parquet").replace("_clean", "")
    # Extract base path and add _clean suffix
    clean_path = Path(output_path).parent / f"{Path(output_path).stem}_clean{Path(output_path).suffix}"
    clean_path = str(clean_path).replace("_clean_clean", "_clean")
    save_checkpoint(clean_df, clean_path, logger)

    return clean_df, outliers_df


def run_outlier_detection(
    input_path: str = "data/interim/02_cleaned.parquet",
    output_path: str = "data/quarantine/outliers.csv",
    columns: list[str] = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Execute full outlier detection pipeline.

    Detects outliers using Isolation Forest on specified columns and
    quarantines them to a separate file for review.

    Parameters
    ----------
    input_path : str, default="data/interim/02_cleaned.parquet"
        Path to cleaned data checkpoint.
    output_path : str, default="data/quarantine/outliers.csv"
        Base path for quarantined outliers file.
    columns : list of str, optional
        Numerical columns for outlier detection.
        If None, uses default columns: price_usd, intelligence_index, speed, latency.

    Returns
    -------
    clean_df : pl.DataFrame
        DataFrame with outliers removed.
    outliers_df : pl.DataFrame
        DataFrame containing only outlier records.

    Examples
    --------
    >>> clean, outliers = run_outlier_detection()
    >>> print(f"Clean dataset: {clean.height} rows")
    >>> print(f"Outliers: {outliers.height} rows")

    Notes
    -----
    - Default columns: price_usd, Intelligence Index, Speed, Latency, Context Window
    - Contamination is set to 0.05 (expect 5% outliers)
    - Both clean and outlier datasets are saved as checkpoints
    - Outlier scores are preserved in both datasets for analysis
    """
    logger = setup_logging()
    logger.info("Starting outlier detection pipeline")

    # Load data
    logger.info(f"Loading data from {input_path}")
    df = load_checkpoint(input_path, logger)

    # Default columns for outlier detection
    if columns is None:
        columns = [
            "price_usd",
            "Intelligence Index",
            "Speed(median token/s)",
            "Latency (First Answer Chunk /s)",
            "Context Window"
        ]

    # Validate columns exist
    available_columns = [col for col in columns if col in df.columns]
    if len(available_columns) < len(columns):
        missing = set(columns) - set(available_columns)
        logger.warning(f"Some columns not found, using: {available_columns}")
        columns = available_columns

    logger.info(f"Using columns for outlier detection: {columns}")

    # Detect and quarantine outliers
    clean_df, outliers_df = quarantine_outliers(
        df,
        output_path,
        columns,
        contamination=0.05
    )

    # Print outlier summary
    logger.info("\n=== Outlier Summary ===")
    outlier_summary = outliers_df.describe()
    print(outlier_summary)

    logger.info("Outlier detection completed")

    return clean_df, outliers_df


if __name__ == "__main__":
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("Starting outlier detection process")

    # Run outlier detection
    clean_df, outliers_df = run_outlier_detection()

    # Print summary
    logger.info(f"\nClean dataset: {clean_df.height} rows")
    logger.info(f"Outliers detected: {outliers_df.height} rows")

    # Show sample outliers
    if outliers_df.height > 0:
        logger.info("\nSample outliers (first 5):")
        print(outliers_df.head(5))

    logger.info("Outlier detection completed successfully")
