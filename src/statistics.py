"""
Statistical analysis utilities for AI models benchmark dataset.

This module provides functions for correlation analysis, multiple testing correction,
and quartile-based grouping using scipy.stats and Polars.

Functions
---------
compute_spearman_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]
    Compute Spearman rank correlation coefficient and p-value.

compute_correlation_matrix(df: pl.DataFrame, columns: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]
    Compute Spearman correlation matrix for all column pairs.

apply_fdr_correction(p_values: np.ndarray, method: str = 'bh') -> np.ndarray
    Apply False Discovery Rate (FDR) correction to p-values.

interpret_correlation(corr: float, p_value: float, p_adjusted: float, alpha: float = 0.05) -> dict
    Classify correlation strength, direction, and significance.

group_by_quartile(df: pl.DataFrame, value_col: str, group_col: str = None) -> pl.DataFrame
    Create quartile bins for a numerical column and optionally group by another column.
"""

from scipy import stats
import numpy as np
import polars as pl


def compute_spearman_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Compute Spearman rank correlation coefficient and p-value.

    Spearman correlation is a non-parametric measure of rank correlation.
    It assesses how well the relationship between two variables can be
    described using a monotonic function. Unlike Pearson correlation,
    Spearman does not assume normality and is robust to outliers.

    Parameters
    ----------
    x : np.ndarray
        First variable's data.
    y : np.ndarray
        Second variable's data.

    Returns
    -------
    tuple[float, float]
        (correlation_coefficient, p_value)
        - correlation_coefficient: Range [-1, 1], where -1 is perfect negative
          monotonic relationship, 0 is no monotonic relationship, 1 is perfect
          positive monotonic relationship.
        - p_value: Two-tailed p-value for testing non-correlation.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> corr, p = compute_spearman_correlation(x, y)
    >>> print(f"Correlation: {corr:.3f}, p-value: {p:.4f}")

    Notes
    -----
    - Spearman correlation is appropriate for non-normal distributions.
    - Uses scipy.stats.spearmanr with nan_policy='omit' for pairwise null handling.
    - Rank-based method: Converts data to ranks, then computes Pearson correlation on ranks.

    References
    ----------
    Spearman, C. (1904). The proof and measurement of association between two things.
    American Journal of Psychology, 15(1), 72-101.
    """
    # Remove NaN pairs (pairwise deletion)
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    # Compute Spearman correlation
    corr, p_value = stats.spearmanr(x_clean, y_clean, nan_policy='omit')

    return float(corr), float(p_value)


def compute_correlation_matrix(df: pl.DataFrame, columns: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Compute Spearman correlation matrix for all numerical column pairs.

    Computes pairwise Spearman correlations for all combinations of columns.
    Returns both correlation coefficients and p-values as separate DataFrames.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with numerical columns.
    columns : list[str]
        List of column names to include in correlation matrix.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        (correlation_df, p_value_df)
        - correlation_df: DataFrame with correlation coefficients, columns as row
          headers and column headers.
        - p_value_df: DataFrame with raw p-values matching correlation_df structure.

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/ai_models_deduped.parquet")
    >>> columns = ["Intelligence Index", "price_usd", "Speed(median token/s)"]
    >>> corr_df, p_df = compute_correlation_matrix(df, columns)
    >>> print(corr_df)

    Notes
    -----
    - Matrix is symmetric (correlation of A-B equals B-A).
    - Diagonal values are 1.0 (perfect self-correlation).
    - Uses pairwise deletion for missing values (drops nulls for each pair separately).
    - Builds matrices as numpy arrays first, then converts to Polars DataFrames.

    See Also
    --------
    compute_spearman_correlation : Computes correlation for a single pair.
    apply_fdr_correction : Adjusts p-values for multiple testing.
    """
    n = len(columns)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    # Compute correlation for all pairs (i <= j for symmetry)
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            # Extract data, handle nulls
            x = df[col1].drop_nulls().cast(pl.Float64).to_numpy()
            y = df[col2].drop_nulls().cast(pl.Float64).to_numpy()

            # Compute Spearman correlation
            corr, p_val = compute_spearman_correlation(x, y)

            corr_matrix[i, j] = corr
            p_matrix[i, j] = p_val

    # Convert to Polars DataFrames
    corr_df = pl.DataFrame(corr_matrix, schema=columns)
    corr_df = corr_df.insert_column(0, pl.Series("column", columns))

    p_df = pl.DataFrame(p_matrix, schema=columns)
    p_df = p_df.insert_column(0, pl.Series("column", columns))

    return corr_df, p_df


def apply_fdr_correction(p_values: np.ndarray, method: str = 'bh') -> np.ndarray:
    """
    Apply False Discovery Rate (FDR) correction to p-values.

    FDR correction controls the expected proportion of false discoveries
    (Type I errors) among all rejected hypotheses. More powerful than
    Bonferroni correction, especially for large numbers of tests.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values from multiple statistical tests.
    method : str, default='bh'
        FDR correction method:
        - 'bh': Benjamini-Hochberg (for independent or positively correlated tests)
        - 'by': Benjamini-Yekutieli (more conservative, for any dependency structure)

    Returns
    -------
    np.ndarray
        Array of adjusted p-values (same length as input).

    Examples
    --------
    >>> p_values = np.array([0.001, 0.03, 0.04, 0.12, 0.57])
    >>> adjusted = apply_fdr_correction(p_values, method='bh')
    >>> print(f"Raw: {p_values}")
    >>> print(f"Adjusted: {adjusted}")

    Notes
    -----
    - FDR controls false discovery rate, not family-wise error rate.
    - More powerful than Bonferroni (less likely to miss true effects).
    - Benjamini-Hochberg assumes tests are independent or positively correlated.
    - Benjamini-Yekutieli is valid for any dependency structure (more conservative).

    Interpretation
    -------------
    - If adjusted_p < 0.05: Result is significant after FDR correction.
    - Report both raw and adjusted p-values for transparency.

    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
    A practical and powerful approach to multiple testing. Journal of the
    Royal Statistical Society Series B, 57(1), 289-300.

    See Also
    --------
    scipy.stats.false_discovery_control : Underlying FDR correction function.
    """
    # Use scipy.stats.false_discovery_control (available in SciPy 1.15+)
    adjusted = stats.false_discovery_control(p_values, method=method)
    return adjusted


def interpret_correlation(
    corr: float,
    p_value: float,
    p_adjusted: float,
    alpha: float = 0.05
) -> dict:
    """
    Classify correlation strength, direction, and statistical significance.

    Provides human-readable interpretation of correlation coefficients and
    statistical test results following standard classification thresholds.

    Parameters
    ----------
    corr : float
        Spearman correlation coefficient (range: -1 to 1).
    p_value : float
        Raw p-value from correlation test.
    p_adjusted : float
        FDR-adjusted p-value.
    alpha : float, default=0.05
        Significance level threshold.

    Returns
    -------
    dict
        Dictionary containing:
        - strength: Correlation strength category (very weak, weak, moderate, strong, very strong)
        - direction: Positive or negative
        - significant: Boolean (True if p_adjusted < alpha)
        - interpretation: Human-readable summary string

    Examples
    --------
    >>> result = interpret_correlation(0.65, 0.002, 0.008)
    >>> print(result['interpretation'])
    'Strong positive correlation, statistically significant'

    Notes
    -----
    Strength thresholds (absolute correlation coefficient):
    - 0.0-0.19: Very weak
    - 0.2-0.39: Weak
    - 0.4-0.59: Moderate
    - 0.6-0.79: Strong
    - 0.8-1.0: Very strong

    Direction:
    - Positive (corr > 0): Variables increase together
    - Negative (corr < 0): One variable increases as the other decreases

    Significance:
    - Uses FDR-adjusted p-value, not raw p-value
    - alpha=0.05 means 5% false discovery rate

    References
    ----------
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).
    Routledge. (Standard effect size interpretation)
    """
    # Determine strength
    abs_corr = abs(corr)
    if abs_corr < 0.2:
        strength = "very weak"
    elif abs_corr < 0.4:
        strength = "weak"
    elif abs_corr < 0.6:
        strength = "moderate"
    elif abs_corr < 0.8:
        strength = "strong"
    else:
        strength = "very strong"

    # Determine direction
    direction = "positive" if corr > 0 else "negative"

    # Determine significance
    significant = p_adjusted < alpha

    # Build interpretation string
    sig_text = "statistically significant" if significant else "not statistically significant"
    interpretation = f"{strength.capitalize()} {direction} correlation, {sig_text}"

    return {
        "strength": strength,
        "direction": direction,
        "significant": significant,
        "interpretation": interpretation,
        "correlation": corr,
        "p_value_raw": p_value,
        "p_value_adjusted": p_adjusted
    }


def group_by_quartile(df: pl.DataFrame, value_col: str, group_col: str = None) -> pl.DataFrame:
    """
    Create quartile bins for a numerical column and optionally group by another column.

    Divides data into four equal-sized groups (quartiles) based on the specified
    numerical column. Useful for creating intelligence tiers, price segments, etc.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    value_col : str
        Column name to create quartiles from (e.g., "Intelligence Index").
    group_col : str, optional
        If provided, also group by this column and compute aggregations.

    Returns
    -------
    pl.DataFrame
        DataFrame with new "intelligence_quartile" column added.
        If group_col provided, returns grouped aggregations.

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     'intelligence': [10, 20, 30, 40, 50, 60, 70, 80],
    ...     'context_window': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    ... })
    >>> grouped = group_by_quartile(df, 'intelligence')
    >>> print(grouped['intelligence_quartile'].unique())

    Notes
    -----
    - Quartiles: Q1 (0-25%), Q2 (25-50%), Q3 (50-75%), Q4 (75-100%).
    - Uses pl.col(value_col).qcut() for equal-sized bins.
    - Q1 = low tier, Q2 = mid-low, Q3 = mid-high, Q4 = high tier.
    - For STAT-05: Groups models by intelligence quartile to analyze context window distribution.

    Quartile Labels
    ---------------
    - Q1: Lowest 25% (e.g., low intelligence tier)
    - Q2: 25-50% (mid-low tier)
    - Q3: 50-75% (mid-high tier)
    - Q4: Highest 25% (high intelligence tier)

    See Also
    --------
    pl.col.qcut : Polars quantile cut function for equal-sized bins.
    """
    # Create quartile bins using qcut
    # qcut creates equal-sized bins based on quantiles
    df_result = df.with_columns(
        pl.col(value_col)
        .qcut([0.25, 0.5, 0.75], labels=["Q1", "Q2", "Q3", "Q4"])
        .alias(f"{value_col}_quartile")
    )

    # If group_col provided, compute aggregations
    if group_col:
        result = df_result.group_by(f"{value_col}_quartile").agg([
            pl.col(group_col).count().alias("count"),
            pl.col(group_col).mean().alias(f"mean_{group_col}"),
            pl.col(group_col).median().alias(f"median_{group_col}"),
            pl.col(group_col).std().alias(f"std_{group_col}"),
            pl.col(group_col).min().alias(f"min_{group_col}"),
            pl.col(group_col).max().alias(f"max_{group_col}"),
        ]).sort(f"{value_col}_quartile")

        return result

    return df_result
