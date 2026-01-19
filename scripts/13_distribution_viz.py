#!/usr/bin/env python
"""
Interactive Distribution Visualization Pipeline (VIZ-01, VIZ-02, VIZ-04)

Generates interactive Plotly visualizations for AI models benchmark dataset:
- VIZ-01: Distribution histograms (intelligence, price, speed, latency, context_window)
- VIZ-02: Box plots for outlier detection, segmented by Provider
- VIZ-04: Correlation heatmap with hierarchical clustering

Output: 11 standalone HTML files with hover tooltips, zoom, and pan capabilities
"""

from pathlib import Path
import polars as pl
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualize import (
    create_distribution_histogram,
    create_box_plot,
    create_correlation_heatmap,
    configure_layout,
)

# Create output directory
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Load deduplicated dataset and correlation matrix.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        (ai_models_deduped, correlation_matrix)
    """
    df = pl.read_parquet("data/processed/ai_models_deduped.parquet")
    corr_df = pl.read_parquet("data/processed/correlation_analysis_correlation.parquet")

    print(f"Loaded deduplicated dataset: {df.shape} rows, {df.shape[1]} columns")
    print(f"Loaded correlation matrix: {corr_df.shape[0]}x{corr_df.shape[0]-1}")

    return df, corr_df


def cast_string_to_numeric(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Cast String column to Float64 for plotting.

    Handles Speed and Latency columns which are stored as strings.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    column : str
        Column name to cast.

    Returns
    -------
    pl.DataFrame
        DataFrame with column cast to Float64.
    """
    return df.with_columns(
        pl.col(column)
        .cast(pl.Float64)
        .alias(column)
    )


def create_histograms(df: pl.DataFrame) -> list[Path]:
    """
    Create 5 distribution histograms (VIZ-01).

    Creates individual histograms for:
    - intelligence_index (n=181 with valid IQ scores)
    - price_usd
    - Speed(median token/s)
    - Latency (First Answer Chunk /s)
    - context_window

    Returns
    -------
    list[Path]
        List of paths to saved HTML files.
    """
    print("\n=== Creating Distribution Histograms (VIZ-01) ===")

    output_paths = []

    # Filter for valid intelligence_index
    df_intel = df.filter(pl.col("intelligence_index").is_not_null())
    n_intel = df_intel.shape[0]

    # Cast string columns to numeric
    df_numeric = cast_string_to_numeric(df, "Speed(median token/s)")
    df_numeric = cast_string_to_numeric(df_numeric, "Latency (First Answer Chunk /s)")

    # 1. Intelligence Index Histogram (n=181)
    fig = create_distribution_histogram(
        df_intel,
        "intelligence_index",
        f"Intelligence Index Distribution (n={n_intel})"
    )
    path = FIGURES_DIR / "interactive_intelligence_histogram.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    # 2. Price Histogram
    fig = create_distribution_histogram(
        df,
        "price_usd",
        f"Price Distribution (n={df.shape[0]})"
    )
    path = FIGURES_DIR / "interactive_price_histogram.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    # 3. Speed Histogram (cast to numeric)
    fig = create_distribution_histogram(
        df_numeric,
        "Speed(median token/s)",
        f"Speed Distribution (n={df.shape[0]})"
    )
    path = FIGURES_DIR / "interactive_speed_histogram.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    # 4. Latency Histogram (cast to numeric)
    fig = create_distribution_histogram(
        df_numeric,
        "Latency (First Answer Chunk /s)",
        f"Latency Distribution (n={df.shape[0]})"
    )
    path = FIGURES_DIR / "interactive_latency_histogram.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    # 5. Context Window Histogram
    fig = create_distribution_histogram(
        df,
        "context_window",
        f"Context Window Distribution (n={df.shape[0]})"
    )
    path = FIGURES_DIR / "interactive_context_window_histogram.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    return output_paths


def create_box_plots(df: pl.DataFrame) -> list[Path]:
    """
    Create 5 box plots for outlier detection (VIZ-02).

    Creates box plots segmented by Provider (Creator) for:
    - intelligence_index by Provider
    - price_usd by Provider
    - Speed by Provider
    - Latency by Provider
    - context_window by Provider

    Returns
    -------
    list[Path]
        List of paths to saved HTML files.
    """
    print("\n=== Creating Box Plots (VIZ-02) ===")

    output_paths = []

    # Filter for valid intelligence_index
    df_intel = df.filter(pl.col("intelligence_index").is_not_null())
    n_intel = df_intel.shape[0]

    # Cast string columns to numeric
    df_numeric = cast_string_to_numeric(df, "Speed(median token/s)")
    df_numeric = cast_string_to_numeric(df_numeric, "Latency (First Answer Chunk /s)")
    df_numeric_intel = df_numeric.filter(pl.col("intelligence_index").is_not_null())

    # 1. Intelligence Index by Provider
    fig = create_box_plot(
        df_intel,
        "Creator",
        "intelligence_index",
        f"Intelligence Index by Provider (n={n_intel})"
    )
    path = FIGURES_DIR / "interactive_intelligence_box_plot.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    # 2. Price by Provider
    fig = create_box_plot(
        df,
        "Creator",
        "price_usd",
        f"Price by Provider (n={df.shape[0]})"
    )
    path = FIGURES_DIR / "interactive_price_box_plot.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    # 3. Speed by Provider (cast to numeric)
    fig = create_box_plot(
        df_numeric,
        "Creator",
        "Speed(median token/s)",
        f"Speed by Provider (n={df.shape[0]})"
    )
    path = FIGURES_DIR / "interactive_speed_box_plot.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    # 4. Latency by Provider (cast to numeric)
    fig = create_box_plot(
        df_numeric,
        "Creator",
        "Latency (First Answer Chunk /s)",
        f"Latency by Provider (n={df.shape[0]})"
    )
    path = FIGURES_DIR / "interactive_latency_box_plot.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    # 5. Context Window by Provider
    fig = create_box_plot(
        df,
        "Creator",
        "context_window",
        f"Context Window by Provider (n={df.shape[0]})"
    )
    path = FIGURES_DIR / "interactive_context_window_box_plot.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    output_paths.append(path)
    print(f"Created: {path}")

    return output_paths


def create_correlation_heatmap_viz(corr_df: pl.DataFrame) -> Path:
    """
    Create interactive correlation heatmap (VIZ-04).

    Creates 5x5 Spearman correlation matrix with hierarchical clustering.
    Hover shows correlation coefficient and FDR-adjusted p-value.

    Returns
    -------
    Path
        Path to saved HTML file.
    """
    print("\n=== Creating Correlation Heatmap (VIZ-04) ===")

    fig = create_correlation_heatmap(
        corr_df,
        "Spearman Correlation Matrix (FDR-corrected)"
    )
    path = FIGURES_DIR / "interactive_correlation_heatmap.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    print(f"Created: {path}")

    return path


def create_combined_histogram_dashboard(df: pl.DataFrame) -> Path:
    """
    Create combined histogram dashboard with all 5 distributions.

    Creates a single HTML file with 5 subplots showing all distributions
    in one view for comprehensive analysis.

    Returns
    -------
    Path
        Path to saved HTML file.
    """
    print("\n=== Creating Combined Histogram Dashboard ===")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Filter for valid intelligence_index
    df_intel = df.filter(pl.col("intelligence_index").is_not_null())
    n_intel = df_intel.shape[0]

    # Cast string columns to numeric
    df_numeric = cast_string_to_numeric(df, "Speed(median token/s)")
    df_numeric = cast_string_to_numeric(df_numeric, "Latency (First Answer Chunk /s)")

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            f"Intelligence Index (n={n_intel})",
            f"Price (n={df.shape[0]})",
            f"Speed (n={df.shape[0]})",
            f"Latency (n={df.shape[0]})",
            f"Context Window (n={df.shape[0]})",
            ""  # Empty placeholder
        ],
        specs=[
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "histogram"}, {"type": "histogram"}],
        ],
    )

    # Add histograms
    # 1. Intelligence Index
    fig.add_trace(
        go.Histogram(
            x=df_intel["intelligence_index"],
            name="Intelligence Index",
            marker_color="#3366CC",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. Price
    fig.add_trace(
        go.Histogram(
            x=df["price_usd"],
            name="Price",
            marker_color="#3366CC",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # 3. Speed
    fig.add_trace(
        go.Histogram(
            x=df_numeric["Speed(median token/s)"].cast(pl.Float64),
            name="Speed",
            marker_color="#3366CC",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. Latency
    fig.add_trace(
        go.Histogram(
            x=df_numeric["Latency (First Answer Chunk /s)"].cast(pl.Float64),
            name="Latency",
            marker_color="#3366CC",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # 5. Context Window
    fig.add_trace(
        go.Histogram(
            x=df["context_window"],
            name="Context Window",
            marker_color="#3366CC",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    # Update layout
    fig = configure_layout(fig, "AI Models Distribution Dashboard")

    path = FIGURES_DIR / "interactive_distributions.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    print(f"Created: {path}")

    return path


def create_combined_box_plot_dashboard(df: pl.DataFrame) -> Path:
    """
    Create combined box plot dashboard with all 5 outlier plots.

    Creates a single HTML file with 5 box plots showing distributions by Provider.

    Returns
    -------
    Path
        Path to saved HTML file.
    """
    print("\n=== Creating Combined Box Plot Dashboard ===")

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Filter for valid intelligence_index
    df_intel = df.filter(pl.col("intelligence_index").is_not_null())
    n_intel = df_intel.shape[0]

    # Cast string columns to numeric
    df_numeric = cast_string_to_numeric(df, "Speed(median token/s)")
    df_numeric = cast_string_to_numeric(df_numeric, "Latency (First Answer Chunk /s)")
    df_numeric_intel = df_numeric.filter(pl.col("intelligence_index").is_not_null())

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            f"Intelligence Index by Provider (n={n_intel})",
            f"Price by Provider (n={df.shape[0]})",
            f"Speed by Provider (n={df.shape[0]})",
            f"Latency by Provider (n={df.shape[0]})",
            f"Context Window by Provider (n={df.shape[0]})",
            ""
        ],
        specs=[
            [{"type": "box"}, {"type": "box"}],
            [{"type": "box"}, {"type": "box"}],
            [{"type": "box"}, {"type": "box"}],
        ],
    )

    # Add box plots
    # 1. Intelligence Index
    for creator in df_intel["Creator"].unique().to_list():
        creator_data = df_intel.filter(pl.col("Creator") == creator)
        fig.add_trace(
            go.Box(
                y=creator_data["intelligence_index"],
                name=creator,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # 2. Price
    for creator in df["Creator"].unique().to_list():
        creator_data = df.filter(pl.col("Creator") == creator)
        fig.add_trace(
            go.Box(
                y=creator_data["price_usd"],
                name=creator,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 3. Speed
    for creator in df_numeric["Creator"].unique().to_list():
        creator_data = df_numeric.filter(pl.col("Creator") == creator)
        speed_numeric = creator_data["Speed(median token/s)"].cast(pl.Float64)
        fig.add_trace(
            go.Box(
                y=speed_numeric,
                name=creator,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # 4. Latency
    for creator in df_numeric["Creator"].unique().to_list():
        creator_data = df_numeric.filter(pl.col("Creator") == creator)
        latency_numeric = creator_data["Latency (First Answer Chunk /s)"].cast(pl.Float64)
        fig.add_trace(
            go.Box(
                y=latency_numeric,
                name=creator,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

    # 5. Context Window
    for creator in df["Creator"].unique().to_list():
        creator_data = df.filter(pl.col("Creator") == creator)
        fig.add_trace(
            go.Box(
                y=creator_data["context_window"],
                name=creator,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # Update layout
    fig = configure_layout(fig, "AI Models Outlier Detection Dashboard")

    path = FIGURES_DIR / "interactive_box_plots.html"
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    print(f"Created: {path}")

    return path


def main():
    """Execute the full interactive visualization pipeline."""
    print("=" * 60)
    print("Interactive Distribution Visualization Pipeline")
    print("=" * 60)

    # Load data
    df, corr_df = load_data()

    # Create individual histograms
    histogram_paths = create_histograms(df)

    # Create individual box plots
    box_plot_paths = create_box_plots(df)

    # Create correlation heatmap
    heatmap_path = create_correlation_heatmap_viz(corr_df)

    # Create combined dashboards
    dashboard_histogram_path = create_combined_histogram_dashboard(df)
    dashboard_box_plot_path = create_combined_box_plot_dashboard(df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Individual histograms: {len(histogram_paths)}")
    print(f"Individual box plots: {len(box_plot_paths)}")
    print(f"Correlation heatmap: 1")
    print(f"Combined dashboards: 2")
    print(f"Total visualizations: {len(histogram_paths) + len(box_plot_paths) + 1 + 2}")
    print()
    print("All visualizations saved to: reports/figures/")
    print("Open HTML files in browser to explore interactively")
    print("=" * 60)


if __name__ == "__main__":
    main()
