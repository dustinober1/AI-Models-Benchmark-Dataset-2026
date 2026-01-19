"""
Bootstrap confidence interval and non-parametric statistical testing utilities.

This module provides functions for bootstrap resampling to compute confidence
intervals for means, medians, correlations, and group differences. Also includes
non-parametric statistical tests (Mann-Whitney U, Kruskal-Wallis) appropriate
for non-normal distributions identified in Phase 1.

Functions
---------
bootstrap_mean_ci(data, confidence_level, n_resamples, random_state) -> dict
    Compute bootstrap confidence interval for mean with BCa method.

bootstrap_median_ci(data, confidence_level, n_resamples, random_state) -> dict
    Compute bootstrap confidence interval for median with BCa method.

bootstrap_correlation_ci(x, y, confidence_level, n_resamples, random_state) -> dict
    Compute bootstrap confidence interval for Spearman correlation.

bootstrap_group_difference_ci(group1, group2, confidence_level, n_resamples, random_state) -> dict
    Compute bootstrap confidence interval for difference in group means.

mann_whitney_u_test(group1, group2, alternative) -> dict
    Perform Mann-Whitney U test (non-parametric alternative to t-test).

kruskal_wallis_test(*groups) -> dict
    Perform Kruskal-Wallis test (non-parametric alternative to ANOVA).
"""

from scipy.stats import bootstrap, mannwhitneyu, kruskal
import numpy as np


def bootstrap_mean_ci(
    data: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    random_state: int = 42
) -> dict:
    """
    Compute bootstrap confidence interval for mean using BCa method.

    Uses scipy.stats.bootstrap with bias-corrected and accelerated (BCa) method,
    which provides more accurate confidence intervals than the percentile method
    for skewed distributions.

    Parameters
    ----------
    data : np.ndarray
        1D array of numerical data.
    confidence_level : float, default=0.95
        Confidence level for interval (e.g., 0.95 for 95% CI).
    n_resamples : int, default=9999
        Number of bootstrap resamples. Higher values provide more accurate CIs
        but take longer to compute.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - mean: Sample mean
        - ci_low: Lower bound of confidence interval
        - ci_high: Upper bound of confidence interval
        - standard_error: Standard error of the mean
        - n_resamples: Number of bootstrap resamples performed
        - method: Bootstrap method used ('BCa')

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
    >>> result = bootstrap_mean_ci(data, confidence_level=0.95)
    >>> print(f"Mean: {result['mean']:.2f}")
    >>> print(f"95% CI: [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")

    Notes
    -----
    - BCa method adjusts for bias and skewness in the bootstrap distribution
    - Requires sufficient data (n >= 10 recommended for stable BCa estimation)
    - Falls back to percentile method if BCa fails (rare)
    - Standard error computed from bootstrap distribution, not formula

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1994). An introduction to the bootstrap.
    CRC press. (Chapter 14 for BCa method)
    """
    # Remove NaN values
    data_clean = data[~np.isnan(data)]

    if len(data_clean) < 3:
        raise ValueError(f"Need at least 3 data points for bootstrap, got {len(data_clean)}")

    # Reshape data as 2D tuple (required by scipy.stats.bootstrap)
    data_tuple = (data_clean,)

    # Define statistic function (mean)
    def statistic(data):
        return np.mean(data)

    # Compute bootstrap CI with BCa method
    try:
        result = bootstrap(
            data_tuple,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='BCa',  # Bias-corrected and accelerated
            random_state=random_state
        )

        return {
            'mean': float(np.mean(data_clean)),
            'ci_low': float(result.confidence_interval.low),
            'ci_high': float(result.confidence_interval.high),
            'standard_error': float(result.standard_error),
            'n_resamples': n_resamples,
            'method': 'BCa'
        }
    except Exception as e:
        # Fall back to percentile method if BCa fails
        # (typically due to insufficient data for bias estimation)
        result = bootstrap(
            data_tuple,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='percentile',
            random_state=random_state
        )

        return {
            'mean': float(np.mean(data_clean)),
            'ci_low': float(result.confidence_interval.low),
            'ci_high': float(result.confidence_interval.high),
            'standard_error': float(result.standard_error),
            'n_resamples': n_resamples,
            'method': 'percentile (BCa failed)'
        }


def bootstrap_median_ci(
    data: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    random_state: int = 42
) -> dict:
    """
    Compute bootstrap confidence interval for median using BCa method.

    Bootstrap CI for median is particularly useful for skewed distributions
    where median is a better measure of central tendency than mean.

    Parameters
    ----------
    data : np.ndarray
        1D array of numerical data.
    confidence_level : float, default=0.95
        Confidence level for interval.
    n_resamples : int, default=9999
        Number of bootstrap resamples.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with same structure as bootstrap_mean_ci:
        - median: Sample median
        - ci_low: Lower bound of confidence interval
        - ci_high: Upper bound of confidence interval
        - standard_error: Standard error of the median
        - n_resamples: Number of bootstrap resamples performed
        - method: Bootstrap method used

    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10)
    >>> result = bootstrap_median_ci(data, confidence_level=0.95)
    >>> print(f"Median: {result['median']:.2f}")
    >>> print(f"95% CI: [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")

    Notes
    -----
    - Median is more robust to outliers than mean
    - Use for highly skewed distributions (skewness > 2)
    - CI for median is typically wider than for mean (less efficient estimator)
    """
    # Remove NaN values
    data_clean = data[~np.isnan(data)]

    if len(data_clean) < 3:
        raise ValueError(f"Need at least 3 data points for bootstrap, got {len(data_clean)}")

    # Reshape data as 2D tuple
    data_tuple = (data_clean,)

    # Define statistic function (median)
    def statistic(data):
        return np.median(data)

    # Compute bootstrap CI with BCa method
    try:
        result = bootstrap(
            data_tuple,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='BCa',
            random_state=random_state
        )

        return {
            'median': float(np.median(data_clean)),
            'ci_low': float(result.confidence_interval.low),
            'ci_high': float(result.confidence_interval.high),
            'standard_error': float(result.standard_error),
            'n_resamples': n_resamples,
            'method': 'BCa'
        }
    except Exception as e:
        # Fall back to percentile method if BCa fails
        result = bootstrap(
            data_tuple,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='percentile',
            random_state=random_state
        )

        return {
            'median': float(np.median(data_clean)),
            'ci_low': float(result.confidence_interval.low),
            'ci_high': float(result.confidence_interval.high),
            'standard_error': float(result.standard_error),
            'n_resamples': n_resamples,
            'method': 'percentile (BCa failed)'
        }


def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    random_state: int = 42
) -> dict:
    """
    Compute bootstrap confidence interval for Spearman correlation.

    Uses pair-wise resampling (resample indices, apply to both x and y)
    to preserve the paired structure of the data.

    Parameters
    ----------
    x : np.ndarray
        First variable's data (1D array).
    y : np.ndarray
        Second variable's data (1D array, same length as x).
    confidence_level : float, default=0.95
        Confidence level for interval.
    n_resamples : int, default=9999
        Number of bootstrap resamples.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - correlation: Sample Spearman correlation coefficient
        - ci_low: Lower bound of confidence interval
        - ci_high: Upper bound of confidence interval
        - standard_error: Standard error of the correlation
        - n_resamples: Number of bootstrap resamples performed

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    >>> result = bootstrap_correlation_ci(x, y, confidence_level=0.95)
    >>> print(f"Correlation: {result['correlation']:.3f}")
    >>> print(f"95% CI: [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")

    Notes
    -----
    - Uses Spearman correlation (rank-based, non-parametric)
    - Pair-wise resampling preserves data pairing structure
    - CI for correlation can be asymmetric around the point estimate
    - Particularly useful for assessing uncertainty in correlation estimates

    References
    ----------
    - Bootstrap CI for correlation: Efron & Tibshirani (1993), Chapter 5
    """
    # Remove NaN pairs (pairwise deletion)
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        raise ValueError(f"Need at least 3 complete pairs for bootstrap, got {len(x_clean)}")

    if len(x_clean) != len(y_clean):
        raise ValueError(f"x and y must have same length after NaN removal: {len(x_clean)} vs {len(y_clean)}")

    # Combine data as tuple of arrays
    data_tuple = (x_clean, y_clean)

    # Define statistic function (Spearman correlation)
    def statistic(data):
        x_sample, y_sample = data
        # Use scipy.stats.spearmanr for Spearman correlation
        from scipy.stats import spearmanr
        corr, _ = spearmanr(x_sample, y_sample, nan_policy='omit')
        return corr

    # Compute bootstrap CI
    try:
        result = bootstrap(
            data_tuple,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='BCa',
            random_state=random_state
        )

        # Compute sample correlation
        from scipy.stats import spearmanr
        corr_sample, _ = spearmanr(x_clean, y_clean, nan_policy='omit')

        return {
            'correlation': float(corr_sample),
            'ci_low': float(result.confidence_interval.low),
            'ci_high': float(result.confidence_interval.high),
            'standard_error': float(result.standard_error),
            'n_resamples': n_resamples
        }
    except Exception as e:
        # Fall back to percentile method if BCa fails
        result = bootstrap(
            data_tuple,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='percentile',
            random_state=random_state
        )

        from scipy.stats import spearmanr
        corr_sample, _ = spearmanr(x_clean, y_clean, nan_policy='omit')

        return {
            'correlation': float(corr_sample),
            'ci_low': float(result.confidence_interval.low),
            'ci_high': float(result.confidence_interval.high),
            'standard_error': float(result.standard_error),
            'n_resamples': n_resamples
        }


def bootstrap_group_difference_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    confidence_level: float = 0.95,
    n_resamples: int = 9999,
    random_state: int = 42
) -> dict:
    """
    Compute bootstrap confidence interval for difference in group means.

    Tests whether two groups have significantly different means by
    bootstrapping the difference in means and computing a confidence interval.
    If CI excludes 0, the difference is statistically significant.

    Parameters
    ----------
    group1 : np.ndarray
        First group's data (1D array).
    group2 : np.ndarray
        Second group's data (1D array).
    confidence_level : float, default=0.95
        Confidence level for interval.
    n_resamples : int, default=9999
        Number of bootstrap resamples.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - mean_difference: Difference in means (group1 - group2)
        - ci_low: Lower bound of confidence interval
        - ci_high: Upper bound of confidence interval
        - group1_mean: Mean of group1
        - group2_mean: Mean of group2
        - significant: True if CI excludes 0 (significant difference)

    Examples
    --------
    >>> group1 = np.array([1, 2, 3, 4, 5])
    >>> group2 = np.array([4, 5, 6, 7, 8])
    >>> result = bootstrap_group_difference_ci(group1, group2)
    >>> print(f"Mean difference: {result['mean_difference']:.2f}")
    >>> print(f"95% CI: [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")
    >>> print(f"Significant: {result['significant']}")

    Notes
    -----
    - If CI excludes 0, groups are significantly different at alpha level
    - Non-parametric alternative to two-sample t-test
    - Does not assume normality or equal variances
    - Particularly useful for skewed distributions or small sample sizes
    """
    # Remove NaN values from each group
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]

    if len(group1_clean) < 3:
        raise ValueError(f"Group1 needs at least 3 data points, got {len(group1_clean)}")
    if len(group2_clean) < 3:
        raise ValueError(f"Group2 needs at least 3 data points, got {len(group2_clean)}")

    # Combine data as tuple of groups
    data_tuple = (group1_clean, group2_clean)

    # Define statistic function (difference in means)
    def statistic(data):
        g1, g2 = data
        return np.mean(g1) - np.mean(g2)

    # Compute bootstrap CI
    try:
        result = bootstrap(
            data_tuple,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='BCa',
            random_state=random_state
        )

        mean_diff = float(np.mean(group1_clean) - np.mean(group2_clean))

        return {
            'mean_difference': mean_diff,
            'ci_low': float(result.confidence_interval.low),
            'ci_high': float(result.confidence_interval.high),
            'group1_mean': float(np.mean(group1_clean)),
            'group2_mean': float(np.mean(group2_clean)),
            'significant': (result.confidence_interval.low > 0) or (result.confidence_interval.high < 0)
        }
    except Exception as e:
        # Fall back to percentile method if BCa fails
        result = bootstrap(
            data_tuple,
            statistic,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            method='percentile',
            random_state=random_state
        )

        mean_diff = float(np.mean(group1_clean) - np.mean(group2_clean))

        return {
            'mean_difference': mean_diff,
            'ci_low': float(result.confidence_interval.low),
            'ci_high': float(result.confidence_interval.high),
            'group1_mean': float(np.mean(group1_clean)),
            'group2_mean': float(np.mean(group2_clean)),
            'significant': (result.confidence_interval.low > 0) or (result.confidence_interval.high < 0)
        }


def mann_whitney_u_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = 'two-sided'
) -> dict:
    """
    Perform Mann-Whitney U test (non-parametric alternative to two-sample t-test).

    Tests whether two independent samples were drawn from the same distribution.
    More robust than t-test for non-normal distributions and outliers.

    Parameters
    ----------
    group1 : np.ndarray
        First group's data (1D array).
    group2 : np.ndarray
        Second group's data (1D array).
    alternative : str, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': distributions are not equal
        - 'less': distribution of group1 is shifted left of group2
        - 'greater': distribution of group1 is shifted right of group2

    Returns
    -------
    dict
        Dictionary containing:
        - statistic: Mann-Whitney U statistic
        - p_value: Two-tailed p-value
        - significant: True if p_value < 0.05
        - effect_size: Rank-biserial correlation (effect size)

    Examples
    --------
    >>> group1 = np.array([1, 2, 3, 4, 5])
    >>> group2 = np.array([4, 5, 6, 7, 8])
    >>> result = mann_whitney_u_test(group1, group2)
    >>> print(f"U statistic: {result['statistic']:.2f}")
    >>> print(f"p-value: {result['p_value']:.4f}")
    >>> print(f"Significant: {result['significant']}")

    Notes
    -----
    - Non-parametric alternative to independent t-test
    - Tests if one distribution is stochastically dominant
    - Robust to outliers and non-normal distributions
    - Assumes observations are independent within and between groups
    - Does not assume equal variances

    Effect size interpretation (rank-biserial correlation):
    - 0.1: Small effect
    - 0.3: Medium effect
    - 0.5: Large effect

    References
    ----------
    Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two
    random variables is stochastically larger than the other. The annals
    of mathematical statistics, 50-60.
    """
    # Remove NaN values
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]

    # Perform Mann-Whitney U test
    statistic, p_value = mannwhitneyu(
        group1_clean,
        group2_clean,
        alternative=alternative,
        nan_policy='omit'
    )

    # Compute effect size (rank-biserial correlation)
    # Formula: 1 - (2U / (n1 * n2))
    n1 = len(group1_clean)
    n2 = len(group2_clean)
    effect_size = 1 - (2 * statistic / (n1 * n2))

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'effect_size': float(effect_size)
    }


def kruskal_wallis_test(*groups: np.ndarray) -> dict:
    """
    Perform Kruskal-Wallis test (non-parametric alternative to one-way ANOVA).

    Tests whether three or more independent samples were drawn from the
    same distribution. Extension of Mann-Whitney U test to multiple groups.

    Parameters
    ----------
    *groups : np.ndarray
        Variable number of groups (each a 1D array).
        Must provide at least 2 groups, typically 3+.

    Returns
    -------
    dict
        Dictionary containing:
        - statistic: Kruskal-Wallis H statistic
        - p_value: P-value for the test
        - significant: True if p_value < 0.05
        - n_groups: Number of groups tested

    Examples
    --------
    >>> group1 = np.array([1, 2, 3, 4, 5])
    >>> group2 = np.array([4, 5, 6, 7, 8])
    >>> group3 = np.array([7, 8, 9, 10, 11])
    >>> result = kruskal_wallis_test(group1, group2, group3)
    >>> print(f"H statistic: {result['statistic']:.2f}")
    >>> print(f"p-value: {result['p_value']:.4f}")
    >>> print(f"Significant: {result['significant']}")

    Notes
    -----
    - Non-parametric alternative to one-way ANOVA
    - Tests if all groups have the same median
    - Does not assume normality
    - Assumes observations are independent within and between groups
    - Assumes groups have similar shape distributions (not necessarily same location)

    Post-hoc tests:
    - If significant, perform pairwise Mann-Whitney U tests
    - Apply FDR correction to pairwise p-values (multiple testing)

    References
    ----------
    Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion
    variance analysis. Journal of the American statistical association,
    47(260), 583-621.
    """
    # Remove NaN values from each group
    groups_clean = [g[~np.isnan(g)] for g in groups]

    # Check minimum number of groups
    if len(groups_clean) < 2:
        raise ValueError(f"Need at least 2 groups for Kruskal-Wallis, got {len(groups_clean)}")

    # Perform Kruskal-Wallis test
    statistic, p_value = kruskal(*groups_clean, nan_policy='omit')

    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n_groups': len(groups_clean)
    }
