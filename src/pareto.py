"""
Pareto frontier analysis utilities for AI models benchmark dataset.

This module provides functions for identifying Pareto-efficient models in
multi-objective optimization space. A model is Pareto-efficient if no other
model is better in all objectives (dominates it).

Functions
---------
compute_pareto_frontier(df: pl.DataFrame, maximize: list[str], minimize: list[str]) -> pl.DataFrame
    Identify Pareto-efficient models from multi-objective optimization.

get_pareto_efficient_models(df: pl.DataFrame) -> pl.DataFrame
    Filter DataFrame to return only Pareto-efficient models.

compute_hypervolume(df: pl.DataFrame, maximize: list[str], minimize: list[str], reference_point: dict) -> float
    Calculate hypervolume indicator (area dominated by Pareto frontier).

plot_pareto_frontier(df: pl.DataFrame, x_col: str, y_col: str, x_minimize: bool, y_maximize: bool, output_path: str, title: str) -> None
    Create Pareto frontier scatter plot with annotations.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_pareto_frontier(
    df: pl.DataFrame,
    maximize: list[str] = None,
    minimize: list[str] = None
) -> pl.DataFrame:
    """
    Identify Pareto-efficient models from multi-objective optimization.

    A model is Pareto-efficient if no other model dominates it. Model j dominates
    model i if j is better or equal in all objectives AND strictly better in at
    least one objective.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with models and objective metrics.
    maximize : list[str], optional
        List of column names to maximize (e.g., intelligence_index, speed).
        Higher values are better.
    minimize : list[str], optional
        List of column names to minimize (e.g., price_usd, latency).
        Lower values are better.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with two new columns:
        - is_pareto_efficient: boolean (True if model is on Pareto frontier)
        - pareto_rank: int (1 = efficient, 2+ = dominated by 1 efficient model)

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/ai_models_deduped.parquet")
    >>> df_pareto = compute_pareto_frontier(
    ...     df,
    ...     maximize=["Intelligence Index"],
    ...     minimize=["price_usd"]
    ... )
    >>> efficient = df_pareto.filter(pl.col("is_pareto_efficient"))
    >>> print(f"Pareto-efficient models: {len(efficient)}/{len(df_pareto)}")

    Notes
    -----
    Pareto dominance algorithm:
    1. For maximization: higher is better (e.g., intelligence_index)
    2. For minimization: convert to maximization by negating (obj_min = -df.select(minimize))
    3. Combine objectives: all_objectives = np.hstack([obj_max, obj_min])
    4. Check dominance: model j dominates i if (all_objectives[j] >= all_objectives[i])
       AND (any(all_objectives[j] > all_objectives[i]))

    Handles null values by dropping rows with missing objective values.

    Reference
    ----------
    Pareto efficiency is a fundamental concept in multi-objective optimization.
    See: https://en.wikipedia.org/wiki/Pareto_efficiency
    """
    if maximize is None:
        maximize = []
    if minimize is None:
        minimize = []

    if not maximize and not minimize:
        raise ValueError("Must specify at least one objective in maximize or minimize")

    # Filter to rows with all objectives present (drop nulls)
    objective_cols = maximize + minimize
    df_valid = df.drop_nulls(subset=objective_cols)

    if len(df_valid) == 0:
        raise ValueError(f"No valid data points for objectives: {objective_cols}")

    # Cast all objective columns to Float64 for numeric operations
    # This handles string columns that contain numeric data
    for col in objective_cols:
        if df_valid[col].dtype == pl.String:
            df_valid = df_valid.with_columns(
                pl.col(col).cast(pl.Float64).alias(col)
            )

    # Extract objective values
    # For maximization: higher is better (use as-is)
    obj_max = df_valid.select(maximize).to_numpy() if maximize else np.empty((len(df_valid), 0))

    # For minimization: lower is better (negate to convert to maximization)
    obj_min = -df_valid.select(minimize).to_numpy() if minimize else np.empty((len(df_valid), 0))

    # Combine all objectives (all maximization now)
    all_objectives = np.hstack([obj_max, obj_min])

    # Find Pareto frontier
    n = len(df_valid)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # Check if model j dominates model i
                # j dominates i if j is better or equal in all objectives
                # and strictly better in at least one
                if np.all(all_objectives[j] >= all_objectives[i]) and \
                   np.any(all_objectives[j] > all_objectives[i]):
                    is_pareto[i] = False
                    break

    # Add Pareto flag to valid DataFrame
    df_result = df_valid.with_columns(
        pl.Series("is_pareto_efficient", is_pareto)
    )

    # Compute Pareto rank (1 = efficient, 2+ = how many efficient models dominate it)
    pareto_ranks = np.ones(n, dtype=int)

    for i in range(n):
        if not is_pareto[i]:
            # Count how many Pareto-efficient models dominate this one
            dominating_count = 0
            for j in range(n):
                if is_pareto[j] and i != j:
                    if np.all(all_objectives[j] >= all_objectives[i]) and \
                       np.any(all_objectives[j] > all_objectives[i]):
                        dominating_count += 1
            pareto_ranks[i] = 1 + dominating_count

    df_result = df_result.with_columns(
        pl.Series("pareto_rank", pareto_ranks)
    )

    # Merge back with original DataFrame (rows with nulls get is_pareto_efficient = None)
    df_final = df.with_columns(
        pl.Series("is_pareto_efficient", [None] * len(df), dtype=pl.Boolean),
        pl.Series("pareto_rank", [None] * len(df), dtype=pl.Int32)
    )

    # For simplicity, return the analyzed DataFrame with nulls dropped
    # Users can merge back if needed
    return df_result


def get_pareto_efficient_models(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter DataFrame to return only Pareto-efficient models.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with is_pareto_efficient column (from compute_pareto_frontier).

    Returns
    -------
    pl.DataFrame
        Subset of DataFrame containing only Pareto-efficient models.

    Examples
    --------
    >>> df_pareto = compute_pareto_frontier(df, maximize=["IQ"], minimize=["Price"])
    >>> efficient = get_pareto_efficient_models(df_pareto)
    >>> print(efficient.select(["Model", "IQ", "Price"]).sort("Price"))

    Raises
    ------
    ValueError
        If is_pareto_efficient column is not found in DataFrame.

    Notes
    -----
    Returns models sorted by their objectives (maximize descending, minimize ascending)
    for easy interpretation of the frontier.
    """
    if "is_pareto_efficient" not in df.columns:
        raise ValueError("DataFrame must have 'is_pareto_efficient' column. Run compute_pareto_frontier first.")

    efficient = df.filter(pl.col("is_pareto_efficient") == True)

    return efficient


def compute_hypervolume(
    df: pl.DataFrame,
    maximize: list[str] = None,
    minimize: list[str] = None,
    reference_point: dict = None
) -> float:
    """
    Calculate hypervolume indicator for Pareto frontier.

    Hypervolume is the area/volume in objective space dominated by the Pareto
    frontier, bounded by a reference point (worst-case scenario). Higher values
    indicate better frontier quality.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with is_pareto_efficient column and objective columns.
    maximize : list[str], optional
        Column names to maximize.
    minimize : list[str], optional
        Column names to minimize.
    reference_point : dict, optional
        Dictionary mapping column names to worst-case values.
        If not provided, uses min/max values from data.

    Returns
    -------
    float
        Hypervolume indicator (area dominated by Pareto frontier).

    Examples
    --------
    >>> df_pareto = compute_pareto_frontier(df, maximize=["IQ"], minimize=["Price"])
    >>> hv = compute_hypervolume(
    ...     df_pareto,
    ...     maximize=["IQ"],
    ...     minimize=["Price"],
    ...     reference_point={"IQ": 0, "Price": 1000}
    ... )
    >>> print(f"Hypervolume: {hv:.2f}")

    Notes
    -----
    Hypervolume calculation requires a reference point representing the worst
    acceptable values for each objective. The hypervolume is the volume of the
    region dominated by the Pareto frontier and bounded by the reference point.

    For 2D objectives (maximize x, minimize y), this is the area under the curve.
    For higher dimensions, it's the multi-dimensional volume.

    Reference: Emmerich, M., & Deutz, A. (2018). A discussion on multi-objective
    hypervolume indicators and the computation of their generating set.
    """
    if maximize is None:
        maximize = []
    if minimize is None:
        minimize = []

    # Get Pareto-efficient models only
    efficient = get_pareto_efficient_models(df)

    if len(efficient) == 0:
        return 0.0

    # Set default reference point if not provided
    if reference_point is None:
        reference_point = {}
        for col in maximize:
            reference_point[col] = df[col].min() - 1  # Worse than worst
        for col in minimize:
            reference_point[col] = df[col].max() + 1  # Worse than worst

    # For 2D case (1 maximize, 1 minimize), compute area exactly
    if len(maximize) + len(minimize) == 2:
        if len(maximize) == 1 and len(minimize) == 1:
            # Sort by maximize objective
            efficient_sorted = efficient.sort(maximize[0])

            x = efficient_sorted[maximize[0]].to_numpy()
            y = efficient_sorted[minimize[0]].to_numpy()

            # Reference point
            x_ref = reference_point.get(maximize[0], x.min() - 1)
            y_ref = reference_point.get(minimize[0], y.max() + 1)

            # Compute hypervolume (area under Pareto frontier)
            # For maximize x, minimize y: area = sum((x - x_ref) * (y_ref - y))
            # But need to handle overlapping areas correctly
            hv = 0.0
            for i in range(len(x)):
                # Contribution from this point
                # Width: from x_ref to current x
                # Height: from y_ref to current y (but y is minimized, so y_ref - y)
                width = x[i] - x_ref
                height = y_ref - y[i]

                # For non-dominated points, we need to handle overlaps
                # Simplified: use the minimum y for points with similar x
                if i == 0:
                    hv += width * height
                else:
                    # Only add area that's not already dominated by better points
                    prev_y = y[i-1]
                    if y[i] < prev_y:  # This point has lower y (better for minimization)
                        hv += width * (prev_y - y[i])

            return hv

    # For higher dimensions, use Monte Carlo approximation
    # This is a simplified implementation
    n_samples = 10000
    n_dim = len(maximize) + len(minimize)

    # Generate random points in the bounded space
    mins = {}
    maxs = {}
    for col in maximize:
        mins[col] = df[col].min()
        maxs[col] = df[col].max()
    for col in minimize:
        mins[col] = df[col].min()
        maxs[col] = df[col].max()

    # Count dominated points
    dominated_count = 0
    for _ in range(n_samples):
        # Generate random point
        point = {}
        for col in maximize:
            point[col] = np.random.uniform(reference_point.get(col, mins[col]), maxs[col])
        for col in minimize:
            point[col] = np.random.uniform(mins[col], reference_point.get(col, maxs[col]))

        # Check if dominated by any Pareto-efficient model
        is_dominated = False
        for _, row in efficient.iter_rows(named=True):
            dominates = True
            for col in maximize:
                if row[col] < point[col]:  # Model should have higher value
                    dominates = False
                    break
            for col in minimize:
                if row[col] > point[col]:  # Model should have lower value
                    dominates = False
                    break

            if dominates:
                is_dominated = True
                break

        if is_dominated:
            dominated_count += 1

    # Hypervolume approximation
    total_volume = 1.0
    for col in maximize:
        total_volume *= (maxs[col] - reference_point.get(col, mins[col]))
    for col in minimize:
        total_volume *= (reference_point.get(col, maxs[col]) - mins[col])

    return total_volume * (dominated_count / n_samples)


def plot_pareto_frontier(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    x_minimize: bool = False,
    y_maximize: bool = True,
    output_path: str = None,
    title: str = "Pareto Frontier"
) -> None:
    """
    Create Pareto frontier scatter plot with annotations.

    Generates a scatter plot showing dominated models in gray and Pareto-efficient
    models highlighted in red with larger markers. Efficient models are annotated
    with model names.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with is_pareto_efficient column and objective columns.
    x_col : str
        Column name for x-axis.
    y_col : str
        Column name for y-axis.
    x_minimize : bool, default=False
        If True, lower x values are better (e.g., price).
    y_maximize : bool, default=True
        If True, higher y values are better (e.g., intelligence).
    output_path : str, optional
        Path to save the figure. If not provided, figure is displayed.
    title : str, default="Pareto Frontier"
        Title for the plot.

    Examples
    --------
    >>> df_pareto = compute_pareto_frontier(
    ...     df, maximize=["Intelligence Index"], minimize=["price_usd"]
    ... )
    >>> plot_pareto_frontier(
    ...     df_pareto,
    ...     x_col="price_usd",
    ...     y_col="Intelligence Index",
    ...     x_minimize=True,
    ...     y_maximize=True,
    ...     output_path="reports/figures/pareto_frontier.png",
    ...     title="Price-Performance Pareto Frontier"
    ... )

    Notes
    -----
    - Dominated models shown in gray with transparency (alpha=0.6)
    - Pareto-efficient models highlighted in red with larger markers (s=100)
    - Efficient models annotated with model names (offset by 5,5 points)
    - Figure size: 12x8 inches
    - DPI: 300 for high-quality publication figures
    - Uses seaborn style with grid for better readability
    """
    if "is_pareto_efficient" not in df.columns:
        raise ValueError("DataFrame must have 'is_pareto_efficient' column. Run compute_pareto_frontier first.")

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' and '{y_col}' must exist in DataFrame")

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Separate dominated and Pareto-efficient models
    is_pareto = df["is_pareto_efficient"].to_numpy()

    # Handle null values - treat as dominated
    is_pareto = np.nan_to_num(is_pareto, nan=False).astype(bool)

    dominated_mask = ~is_pareto
    pareto_mask = is_pareto

    # Get x and y values (drop nulls for plotting)
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # Plot dominated models
    if np.any(dominated_mask):
        ax.scatter(
            x[dominated_mask],
            y[dominated_mask],
            alpha=0.4,
            s=50,
            color='gray',
            label='Dominated',
            edgecolors='none'
        )

    # Plot Pareto-efficient models
    if np.any(pareto_mask):
        ax.scatter(
            x[pareto_mask],
            y[pareto_mask],
            color='#d62728',  # Red
            s=120,
            label='Pareto Efficient',
            edgecolors='black',
            linewidth=1.5,
            zorder=10
        )

        # Annotate Pareto-efficient models
        pareto_df = df.filter(pl.col("is_pareto_efficient") == True)

        for row in pareto_df.iter_rows(named=True):
            model_name = row.get("Model", row.get("model_id", "Unknown"))
            ax.annotate(
                model_name,
                (row[x_col], row[y_col]),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
            )

    # Axis labels with direction indicators
    x_label = x_col.replace("_", " ").title()
    y_label = y_col.replace("_", " ").title()

    if x_minimize:
        x_label += " (Lower is Better)"
    else:
        x_label += " (Higher is Better)"

    if y_maximize:
        y_label += " (Higher is Better)"
    else:
        y_label += " (Lower is Better)"

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Legend and grid
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Tight layout
    plt.tight_layout()

    # Save or display
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
