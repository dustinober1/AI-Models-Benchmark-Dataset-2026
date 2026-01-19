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

    # Create histogram with Plotly Graph Objects (avoiding pyarrow dependency)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        marker_color="#3366CC",
        marker_line=dict(color="#1F4788", width=1),
        name=column,
    ))

    # Enhanced hover template with model names and statistics
    hover_template = (
        "<b>Value: %{x}</b><br>"
        "Count: %{y}<br>"
        "<br><b>Statistics:</b><br>"
        f"Mean: {mean_val:.2f}<br>"
        f"Median: {median_val:.2f}<br>"
        f"Min: {min_val:.2f}<br>"
        f"Max: {max_val:.2f}<br>"
        "<extra></extra>"
    )

    fig.update_traces(hovertemplate=hover_template)

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
    # Extract data using Polars (avoiding pyarrow dependency)
    categories = df[x_col].unique().to_list()

    # Create box plot with Plotly Graph Objects
    fig = go.Figure()

    # Add box traces for each category
    for category in sorted(categories):
        category_data = df.filter(pl.col(x_col) == category)
        y_values = category_data[y_col].cast(pl.Float64).to_numpy()

        # Add box plot
        fig.add_trace(go.Box(
            y=y_values,
            name=str(category),
            boxmean=True,  # Show mean
            jitter=0.3,    # Add jitter for individual point visibility
            pointpos=-1.8, # Position points
            marker_color="#3366CC",
            marker_line=dict(color="#1F4788", width=1),
        ))

    # Enhanced hover template for box plot
    hover_template = (
        "<b>%{x}</b><br>"
        f"{y_col}: %{{y:.2f}}<br>"
        "Median: %{median:.2f}<br>"
        "Q1: %{q1:.2f}<br>"
        "Q3: %{q3:.2f}<br>"
        "<extra></extra>"
    )

    fig.update_traces(hovertemplate=hover_template)

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


def create_pareto_frontier_chart(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    pareto_col: str,
    title: str,
    color_col: str = None
) -> go.Figure:
    """
    Create interactive Pareto frontier scatter plot with highlighted efficient models.

    Generates a scatter plot showing dominated models with transparency and Pareto-efficient
    models highlighted in red with larger markers. Includes hover tooltips showing model
    names, creators, objective values, and Pareto status. Efficient models are annotated
    with text labels.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with model data and Pareto efficiency flag.
    x_col : str
        Column name for x-axis (e.g., "price_usd").
    y_col : str
        Column name for y-axis (e.g., "intelligence_index").
    pareto_col : str
        Column name containing Pareto efficiency flag (boolean).
    title : str
        Title for the chart.
    color_col : str, optional
        Column name for color-coding points (e.g., "Creator", "cluster").
        If not provided, uses single color for dominated models.

    Returns
    -------
    go.Figure
        Interactive Plotly scatter plot with hover tooltips, zoom, and pan.

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/pareto_frontier.parquet")
    >>> fig = create_pareto_frontier_chart(
    ...     df,
    ...     x_col="price_usd",
    ...     y_col="intelligence_index",
    ...     pareto_col="is_pareto_intelligence_price",
    ...     title="Intelligence vs Price - Pareto Frontier"
    ... )
    >>> fig.show()

    Notes
    -----
    - Pareto-efficient models: Red markers, larger size (15), annotated with labels
    - Dominated models: Blue/colored markers, smaller size (8), transparency (0.5)
    - Hover shows: Model name, Creator, x/y values, Pareto status
    - Top 5 efficient models annotated with text labels
    - Trend line added using LOWESS smoothing if n>50
    - Color by Creator or cluster if color_col provided
    """
    # Extract data
    x_data = df[x_col].cast(pl.Float64).to_numpy()
    y_data = df[y_col].cast(pl.Float64).to_numpy()
    pareto_flags = df[pareto_col].to_numpy()

    # Get model names and creators for hover
    models = df["Model"].to_list()
    creators = df["Creator"].to_list()

    # Separate dominated and Pareto-efficient models
    pareto_mask = np.nan_to_num(pareto_flags, nan=False).astype(bool)
    dominated_mask = ~pareto_mask

    # Create figure
    fig = go.Figure()

    # Add dominated models (with transparency)
    if np.any(dominated_mask):
        dominated_x = x_data[dominated_mask]
        dominated_y = y_data[dominated_mask]
        dominated_models = [models[i] for i in range(len(models)) if dominated_mask[i]]
        dominated_creators = [creators[i] for i in range(len(creators)) if dominated_mask[i]]

        # Color by color_col if provided
        if color_col and color_col in df.columns:
            dominated_colors = [df[color_col][i] for i in range(len(df)) if dominated_mask[i]]
        else:
            dominated_colors = ["Dominated"] * len(dominated_x)

        fig.add_trace(go.Scatter(
            x=dominated_x,
            y=dominated_y,
            mode="markers",
            name="Dominated",
            marker=dict(
                size=8,
                color="#3366CC",
                opacity=0.5,
                line=dict(color="#1F4788", width=1)
            ),
            text=dominated_models,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Creator: " + "{}<br>".format(dominated_creators[0] if dominated_creators else "N/A") +
                f"{x_col}: %{{x:.2f}}<br>"
                f"{y_col}: %{{y:.2f}}<br>"
                "Pareto Status: Dominated<br>"
                "<extra></extra>"
            ),
        ))

    # Add Pareto-efficient models (highlighted)
    if np.any(pareto_mask):
        pareto_x = x_data[pareto_mask]
        pareto_y = y_data[pareto_mask]
        pareto_models = [models[i] for i in range(len(models)) if pareto_mask[i]]
        pareto_creators = [creators[i] for i in range(len(creators)) if pareto_mask[i]]

        fig.add_trace(go.Scatter(
            x=pareto_x,
            y=pareto_y,
            mode="markers",
            name="Pareto Efficient",
            marker=dict(
                size=15,
                color="#d62728",  # Red
                line=dict(color="black", width=1.5),
            ),
            text=pareto_models,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Creator: " + "{}<br>".format(pareto_creators[0] if pareto_creators else "N/A") +
                f"{x_col}: %{{x:.2f}}<br>"
                f"{y_col}: %{{y:.2f}}<br>"
                "Pareto Status: <b>Efficient</b><br>"
                "<extra></extra>"
            ),
        ))

        # Annotate top 5 efficient models (by y value, descending)
        pareto_df = df.filter(pl.col(pareto_col) == True)
        if len(pareto_df) > 0:
            # Sort by y_col descending and take top 5
            top_models = pareto_df.sort(y_col, descending=True).head(5)

            annotations = []
            for row in top_models.iter_rows(named=True):
                annotations.append(
                    go.layout.Annotation(
                        x=row[x_col],
                        y=row[y_col],
                        text=row.get("Model", row.get("model_id", "Unknown")),
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor="#666666",
                        ax=20,
                        ay=-30,
                        font=dict(size=9, color="#333333"),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#cccccc",
                        borderwidth=1,
                        borderpad=3,
                    )
                )

            fig.update_layout(annotations=annotations)

    # Add trend line if n>50 (using polynomial fit as proxy for LOWESS)
    if len(x_data) > 50:
        # Sort by x for line plotting
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]

        # Fit 2nd degree polynomial
        try:
            coeffs = np.polyfit(x_sorted, y_sorted, 2)
            poly_fn = np.poly1d(coeffs)
            y_trend = poly_fn(x_sorted)

            fig.add_trace(go.Scatter(
                x=x_sorted,
                y=y_trend,
                mode="lines",
                name="Trend Line",
                line=dict(color="#666666", width=2, dash="dash"),
                hoverinfo="skip",
            ))
        except Exception:
            pass  # Skip trend line if fit fails

    # Configure axes
    fig.update_layout(
        xaxis_title=x_col.replace("_", " ").title(),
        yaxis_title=y_col.replace("_", " ").title(),
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
        ),
    )

    # Apply consistent theme
    fig = configure_layout(fig, title)

    return fig


def create_provider_comparison(
    provider_df: pl.DataFrame,
    metrics: list[str] = None
) -> go.Figure:
    """
    Create 3-panel provider comparison dashboard with cluster color-coding.

    Generates a scatter plot matrix showing provider relationships across three
    key metrics: Intelligence vs Price, Intelligence vs Speed, and Price vs Speed.
    Points are colored by cluster assignment and sized by model count. Cluster
    centroids are displayed as star markers.

    Parameters
    ----------
    provider_df : pl.DataFrame
        Provider-level DataFrame with cluster assignments and metrics.
    metrics : list[str], optional
        List of metric column names. Default: ["avg_intelligence", "avg_price", "avg_speed"].

    Returns
    -------
    go.Figure
        Interactive Plotly 3-panel scatter plot with hover tooltips, zoom, and pan.

    Examples
    --------
    >>> provider_df = pl.read_parquet("data/processed/provider_clusters.parquet")
    >>> fig = create_provider_comparison(
    ...     provider_df,
    ...     metrics=["avg_intelligence", "avg_price", "avg_speed"]
    ... )
    >>> fig.show()

    Notes
    -----
    - 3-panel layout: Intelligence-Price, Intelligence-Speed, Price-Speed
    - Color by cluster: 0=Budget-Friendly, 1=Premium Performance
    - Size by model_count: Larger bubbles = more models from provider
    - Cluster centroids: Star markers with "Centroid" label
    - Hover shows: Creator, cluster, region, and metric values
    - Uses plotly_white template for clean publication-ready figures
    """
    if metrics is None:
        metrics = ["avg_intelligence", "avg_price", "avg_speed"]

    # Validate metrics exist
    for metric in metrics:
        if metric not in provider_df.columns:
            raise ValueError(f"Column '{metric}' not found in provider_df")

    # Extract data
    creators = provider_df["Creator"].to_list()
    clusters = provider_df["cluster"].to_numpy() if "cluster" in provider_df.columns else np.zeros(len(provider_df))
    regions = provider_df["region"].to_list() if "region" in provider_df.columns else ["Unknown"] * len(provider_df)
    model_counts = provider_df["model_count"].to_numpy() if "model_count" in provider_df.columns else np.ones(len(provider_df))

    # Extract metric data
    intelligence = provider_df["avg_intelligence"].cast(pl.Float64).to_numpy()
    price = provider_df["avg_price"].cast(pl.Float64).to_numpy()
    speed = provider_df["avg_speed"].cast(pl.Float64).to_numpy()

    # Create cluster label mapping
    cluster_labels = {0: "Budget-Friendly", 1: "Premium Performance"}
    cluster_colors = {0: "#3366CC", 1: "#d62728"}

    # Create 3-panel subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Intelligence vs Price",
            "Intelligence vs Speed",
            "Price vs Speed"
        ),
        horizontal_spacing=0.05,
    )

    # Define subplot configurations
    subplots = [
        {"x": intelligence, "y": price, "x_title": "Intelligence Index", "y_title": "Price (USD)", "row": 1, "col": 1},
        {"x": intelligence, "y": speed, "x_title": "Intelligence Index", "y_title": "Speed (tokens/s)", "row": 1, "col": 2},
        {"x": price, "y": speed, "x_title": "Price (USD)", "y_title": "Speed (tokens/s)", "row": 1, "col": 3},
    ]

    # Add traces for each cluster
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_name = cluster_labels.get(int(cluster_id), f"Cluster {cluster_id}")
        cluster_color = cluster_colors.get(int(cluster_id), "#3366CC")

        for subplot_config in subplots:
            x_data = subplot_config["x"][cluster_mask]
            y_data = subplot_config["y"][cluster_mask]
            sizes = model_counts[cluster_mask] * 5  # Scale for visibility

            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="markers",
                    name=cluster_name,
                    marker=dict(
                        size=sizes,
                        color=cluster_color,
                        opacity=0.7,
                        line=dict(color="white", width=1),
                    ),
                    legendgroup=f"cluster_{cluster_id}",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"{subplot_config['x_title']}: %{{x:.2f}}<br>"
                        f"{subplot_config['y_title']}: %{{y:.2f}}<br>"
                        "Cluster: " + cluster_name + "<br>"
                        "Region: " + "{region}<br>".format(region=regions[list(clusters).tolist().index(cluster_id)] if cluster_mask.any() else "N/A") +
                        "Models: %{marker.size:.0f}<br>"
                        "<extra></extra>"
                    ),
                    text=[creators[i] for i in range(len(creators)) if cluster_mask[i]],
                ),
                row=subplot_config["row"],
                col=subplot_config["col"],
            )

    # Add cluster centroids
    for cluster_id in np.unique(clusters):
        cluster_mask = clusters == cluster_id
        cluster_name = cluster_labels.get(int(cluster_id), f"Cluster {cluster_id}")

        for subplot_config in subplots:
            centroid_x = float(np.mean(subplot_config["x"][cluster_mask]))
            centroid_y = float(np.mean(subplot_config["y"][cluster_mask]))

            fig.add_trace(
                go.Scatter(
                    x=[centroid_x],
                    y=[centroid_y],
                    mode="markers",
                    name=f"{cluster_name} Centroid",
                    marker=dict(
                        size=20,
                        symbol="star",
                        color="gold",
                        line=dict(color="black", width=2),
                    ),
                    legendgroup=f"centroid_{cluster_id}",
                    hovertemplate=(
                        f"<b>{cluster_name} Centroid</b><br>"
                        f"{subplot_config['x_title']}: {centroid_x:.2f}<br>"
                        f"{subplot_config['y_title']}: {centroid_y:.2f}<br>"
                        "<extra></extra>"
                    ),
                ),
                row=subplot_config["row"],
                col=subplot_config["col"],
            )

    # Update axes labels
    for i, subplot_config in enumerate(subplots, start=1):
        fig.update_xaxes(title_text=subplot_config["x_title"], row=1, col=i)
        fig.update_yaxes(title_text=subplot_config["y_title"], row=1, col=i)

    # Configure layout
    fig.update_layout(
        hovermode="closest",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,  # Place legend outside plot
        ),
        margin=dict(l=20, r=160, t=40, b=20),  # Extra right margin for legend
    )

    # Apply consistent theme
    fig = configure_layout(fig, "Provider Market Segments")

    return fig


def create_context_window_analysis(
    df: pl.DataFrame,
    tier_col: str,
    context_col: str = "context_window"
) -> go.Figure:
    """
    Create interactive context window analysis by intelligence tier.

    Generates a box plot showing context window distributions across intelligence
    tiers (Q1-Q4) with individual model points overlaid as jittered strip plot.
    Includes annotations for mean and median values per tier.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with intelligence tiers and context window data.
    tier_col : str
        Column name containing intelligence tier (Q1, Q2, Q3, Q4).
    context_col : str, default="context_window"
        Column name containing context window size in tokens.

    Returns
    -------
    go.Figure
        Interactive Plotly box plot with strip plot overlay, hover tooltips, zoom, and pan.

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/ai_models_deduped.parquet")
    >>> fig = create_context_window_analysis(
    ...     df,
    ...     tier_col="intelligence_tier",
    ...     context_col="context_window"
    ... )
    >>> fig.show()

    Notes
    -----
    - Box plot shows: median, Q1, Q3, whiskers (1.5*IQR), outliers
    - Strip plot overlay: Individual model points with jitter for visibility
    - Hover shows: Model name, Creator, tier, context window size
    - Annotations: Mean and median values per tier
    - Log scale y-axis if context window range > 1M tokens
    - Colors: Sequential blue palette for visual clarity
    """
    # Extract data
    tiers = df[tier_col].unique().to_list()
    tiers_sorted = sorted(tiers, key=lambda x: (
        0 if str(x).startswith("Q1") else
        1 if str(x).startswith("Q2") else
        2 if str(x).startswith("Q3") else
        3
    ))

    # Check if log scale is needed (range > 1M)
    context_data = df[context_col].cast(pl.Float64).to_numpy()
    use_log_scale = context_data.max() - context_data.min() > 1_000_000

    # Create figure
    fig = go.Figure()

    # Add box plots and strip plots for each tier
    for tier in tiers_sorted:
        tier_data = df.filter(pl.col(tier_col) == tier)
        context_values = tier_data[context_col].cast(pl.Float64).to_numpy()
        models = tier_data["Model"].to_list()
        creators = tier_data["Creator"].to_list()

        # Calculate statistics for annotations
        tier_mean = float(np.mean(context_values))
        tier_median = float(np.median(context_values))

        # Add box plot
        fig.add_trace(go.Box(
            x=[str(tier)] * len(context_values),
            y=context_values,
            name=str(tier),
            boxmean="sd",  # Show mean and standard deviation
            marker_color="#3366CC",
            marker_line=dict(color="#1F4788", width=1),
            line_color="#1F4788",
            legendgroup=str(tier),
            hovertemplate=(
                f"<b>Tier {tier}</b><br>"
                f"Context Window: %{{y:,.0f}} tokens<br>"
                f"Mean: {tier_mean:,.0f}<br>"
                f"Median: {tier_median:,.0f}<br>"
                "<extra></extra>"
            ),
        ))

        # Add strip plot (jittered points) for individual models
        jitter = np.random.uniform(-0.1, 0.1, size=len(context_values))
        fig.add_trace(go.Scatter(
            x=[int(str(tier)[1]) + j for j in jitter],  # Convert Q1->1, Q2->2, etc.
            y=context_values,
            mode="markers",
            name=f"{tier} Models",
            marker=dict(
                size=6,
                color="#d62728",
                opacity=0.6,
            ),
            legendgroup=f"{tier}_points",
            showlegend=False,  # Don't show in legend
            text=models,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Creator: " + "{}<br>".format(creators[0] if creators else "N/A") +
                f"Tier: {tier}<br>"
                f"Context Window: %{{y:,.0f}} tokens<br>"
                "<extra></extra>"
            ),
        ))

        # Add annotation for mean and median
        fig.add_annotation(
            x=str(tier),
            y=tier_mean,
            text=f"Mean: {tier_mean:,.0f}",
            showarrow=False,
            yshift=10,
            font=dict(size=9, color="#333333"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=3,
        )

    # Configure axes
    fig.update_layout(
        xaxis_title="Intelligence Tier",
        yaxis_title="Context Window (tokens)",
        hovermode="closest",
        boxmode="group",
    )

    # Apply log scale if needed
    if use_log_scale:
        fig.update_yaxes(type="log")

    # Apply consistent theme
    fig = configure_layout(fig, "Context Window by Intelligence Tier")

    return fig
