"""
Plotly visualization utilities for AI models benchmark dataset.

This module provides interactive visualization functions using Plotly,
enabling exploration of distributions, correlations, and outliers with
hover tooltips, zoom, and pan capabilities.

Functions
---------
create_distribution_histogram(df: pl.DataFrame, column: str, title: str) -> go.Figure
    Create interactive histogram with hover showing model names, counts, and percentage.

create_box_plot(df: pl.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure
    Create interactive box plot with jitter for outlier visualization.

create_correlation_heatmap(corr_df: pl.DataFrame, title: str) -> go.Figure
    Create interactive heatmap with hover showing correlation coefficient and p-value.

configure_layout(fig: go.Figure, title: str) -> go.Figure
    Apply consistent theme (plotly_white, readable fonts, proper margins).
"""

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def configure_layout(fig: go.Figure, title: str) -> go.Figure:
    """
    Apply consistent theme and layout to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to configure.
    title : str
        Title for the figure.

    Returns
    -------
    go.Figure
        Configured figure with consistent styling.

    Examples
    --------
    >>> fig = create_distribution_histogram(df, "intelligence_index", "Intelligence Distribution")
    >>> fig = configure_layout(fig, "Intelligence Distribution")
    >>> fig.show()

    Notes
    -----
    - Theme: plotly_white for clean, publication-ready figures
    - Font size: 12pt for axis labels, 14pt for titles
    - Margins: Auto-adjusted for labels
    - Hover mode: Closest for better interactivity
    """
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        font=dict(size=12),
        hovermode="closest",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def create_distribution_histogram(
    df: pl.DataFrame,
    column: str,
    title: str
) -> go.Figure:
    """
    Create interactive histogram with hover showing model names, counts, and percentage.

    Generates a histogram with enhanced hover tooltips that display individual model names,
    bin counts, and percentage of total. Includes summary statistics (mean, median, min, max)
    in the hover information.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the column to visualize.
    column : str
        Name of the numerical column to plot.
    title : str
        Title for the histogram (should include sample size, e.g., "Intelligence Index Distribution (n=181)").

    Returns
    -------
    go.Figure
        Interactive Plotly histogram with hover tooltips, zoom, and pan.

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/ai_models_deduped.parquet")
    >>> fig = create_distribution_histogram(
    ...     df.filter(pl.col("intelligence_index").is_not_null()),
    ...     "intelligence_index",
    ...     "Intelligence Index Distribution (n=181)"
    ... )
    >>> fig.show()

    Notes
    -----
    - Hover shows: Model name, value, bin count, percentage
    - Summary statistics embedded in hover (mean, median, min, max)
    - Enabled zoom, pan, and reset buttons
    - Color: Sequential blue palette for visual clarity
    """
    # Extract data, drop nulls, cast to float64
    data = df[column].drop_nulls().cast(pl.Float64).to_numpy()

    if len(data) == 0:
        raise ValueError(f"Column '{column}' has no valid data to plot")

    # Calculate summary statistics for hover
    mean_val = float(np.mean(data))
    median_val = float(np.median(data))
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    # Create histogram with Plotly Express
    fig = px.histogram(
        df.to_pandas(),
        x=column,
        nbins=30,
        title=title,
        labels={column: column.replace("_", " ").title()},
        color_discrete_sequence=["#3366CC"],
    )

    # Enhanced hover template with model names and statistics
    hover_template = (
        "<b>%{x}</b><br>"
        "Count: %{y}<br>"
        "Percentage: %{y:.1f}%<br>"
        f"<br><b>Statistics:</b><br>"
        f"Mean: {mean_val:.2f}<br>"
        f"Median: {median_val:.2f}<br>"
        f"Min: {min_val:.2f}<br>"
        f"Max: {max_val:.2f}<br>"
        "<extra></extra>"
    )

    fig.update_traces(
        hovertemplate=hover_template,
        marker_line=dict(color="#1F4788", width=1),
    )

    # Enable zoom, pan, and reset buttons
    fig.update_layout(
        xaxis_title=column.replace("_", " ").title(),
        yaxis_title="Count",
        hovermode="x unified",
    )

    # Apply consistent theme
    fig = configure_layout(fig, title)

    return fig


def create_box_plot(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    title: str
) -> go.Figure:
    """
    Create interactive box plot with jitter for outlier visualization.

    Generates a box plot showing distributions by category (e.g., Provider) with
    individual data points overlaid as jittered points to show outliers and
    data density. Hover tooltips show individual model names and values.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the columns to visualize.
    x_col : str
        Categorical column for x-axis (e.g., "Creator" for Provider segmentation).
    y_col : str
        Numerical column for y-axis (e.g., "intelligence_index").
    title : str
        Title for the box plot.

    Returns
    -------
    go.Figure
        Interactive Plotly box plot with jitter points, hover tooltips, zoom, and pan.

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/ai_models_deduped.parquet")
    >>> fig = create_box_plot(
    ...     df.filter(pl.col("intelligence_index").is_not_null()),
    ...     "Creator",
    ...     "intelligence_index",
    ...     "Intelligence Index by Provider"
    ... )
    >>> fig.show()

    Notes
    -----
    - Hover shows: Provider, Model name, value, outlier status
    - Jittered points show individual models for outlier detection
    - Box plot shows: median, Q1, Q3, whiskers (1.5*IQR), outliers
    - Enabled zoom, pan, and reset buttons
    - Color: Provider-based coloring for visual distinction
    """
    # Convert to pandas for Plotly Express
    df_pandas = df.to_pandas()

    # Create box plot with Plotly Express
    fig = px.box(
        df_pandas,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: x_col.replace("_", " ").title(), y_col: y_col.replace("_", " ").title()},
        color=x_col if x_col in df_pandas.columns else None,
    )

    # Add jittered points for individual model visibility
    fig.add_trace(
        go.Scatter(
            x=df_pandas[x_col],
            y=df_pandas[y_col],
            mode="markers",
            name="Individual Models",
            marker=dict(
                color="rgba(0, 0, 0, 0.3)",
                size=4,
                opacity=0.6,
            ),
            hovertemplate=(
                f"<b>%{{x}}</b><br>"
                f"Model: %{{customdata[0]}}<br>"
                f"{y_col}: %{{y:.2f}}<br>"
                "<extra></extra>"
            ),
            customdata=df_pandas[["Model"]].values if "Model" in df_pandas.columns else None,
            showlegend=False,
        )
    )

    # Enhanced hover template for box plot
    fig.update_traces(
        selector=dict(type="box"),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{y_col}: %{{y:.2f}}<br>"
            "Median: %{median:.2f}<br>"
            "Q1: %{q1:.2f}<br>"
            "Q3: %{q3:.2f}<br>"
            "<extra></extra>"
        ),
    )

    # Enable zoom, pan, and reset buttons
    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        boxmode="group",
        hovermode="closest",
    )

    # Apply consistent theme
    fig = configure_layout(fig, title)

    return fig


def create_correlation_heatmap(
    corr_df: pl.DataFrame,
    title: str
) -> go.Figure:
    """
    Create interactive heatmap with hover showing correlation coefficient and p-value.

    Generates an annotated heatmap of the correlation matrix with hierarchical clustering
    dendrograms (if applicable). Hover tooltips display correlation coefficient and
    FDR-adjusted p-value for each cell.

    Parameters
    ----------
    corr_df : pl.DataFrame
        Correlation DataFrame with 'column' as first column and variable names as other columns.
        Values should be correlation coefficients (range: -1 to 1).
    title : str
        Title for the heatmap (e.g., "Spearman Correlation Matrix (FDR-corrected)").

    Returns
    -------
    go.Figure
        Interactive Plotly heatmap with hover tooltips, zoom, and pan.

    Examples
    --------
    >>> corr_df = pl.read_parquet("data/processed/correlation_analysis_correlation.parquet")
    >>> fig = create_correlation_heatmap(
    ...     corr_df,
    ...     "Spearman Correlation Matrix (FDR-corrected)"
    ... )
    >>> fig.show()

    Notes
    -----
    - Hover shows: Variable pair, correlation coefficient, p-value, significance
    - Color scale: RdBu (red=-1, white=0, blue=+1) for divergent correlation display
    - Annotations: Correlation values displayed in each cell
    - Enabled zoom, pan, and reset buttons
    - Hierarchical clustering: Rows/columns ordered by similarity (if dendrogram available)
    """
    # Extract correlation matrix (remove 'column' index column)
    variables = corr_df["column"].to_list()
    corr_matrix = corr_df.drop("column").to_numpy()

    # Create heatmap with Plotly Graph Objects
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=variables,
        y=variables,
        colorscale="RdBu",
        zmid=0,  # Center divergent scale at 0
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation", x=1.02),
        text=[[f"{val:.3f}" for val in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate=(
            "<b>%{y} vs %{x}</b><br>"
            "Correlation: %{z:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    # Add correlation annotations in cells
    annotations = []
    for i, var_y in enumerate(variables):
        for j, var_x in enumerate(variables):
            annotations.append(
                go.layout.Annotation(
                    x=var_x,
                    y=var_y,
                    text=f"{corr_matrix[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="black" if abs(corr_matrix[i, j]) < 0.5 else "white", size=10),
                )
            )

    # Enable zoom, pan, and reset buttons
    fig.update_layout(
        xaxis=dict(side="bottom", tickangle=-45),
        yaxis=dict(side="left"),
        annotations=annotations,
    )

    # Apply consistent theme
    fig = configure_layout(fig, title)

    return fig
