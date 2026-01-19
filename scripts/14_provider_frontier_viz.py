#!/usr/bin/env python3
"""
Provider and Frontier Visualization Pipeline

This script creates interactive visualizations for Pareto frontier analysis,
provider comparisons, and context window analysis by intelligence tier.

Outputs
-------
- reports/figures/interactive_pareto_intelligence_price.html
- reports/figures/interactive_pareto_speed_intelligence.html
- reports/figures/interactive_provider_dashboard.html
- reports/figures/interactive_context_window_analysis.html
- reports/figures/interactive_provider_frontier.html (combined dashboard)

Usage
-----
poetry run python scripts/14_provider_frontier_viz.py
"""

import polars as pl
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualize import (
    create_pareto_frontier_chart,
    create_provider_comparison,
    create_context_window_analysis,
)


def main():
    """Execute provider and frontier visualization pipeline."""

    print("=" * 60)
    print("Provider and Frontier Visualization Pipeline")
    print("=" * 60)

    # Create output directory
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/4] Loading data...")

    # Load Pareto frontier data
    pareto_path = Path("data/processed/pareto_frontier.parquet")
    if not pareto_path.exists():
        raise FileNotFoundError(f"Pareto frontier data not found: {pareto_path}")
    df_pareto = pl.read_parquet(pareto_path)
    print(f"  - Loaded {len(df_pareto)} models from {pareto_path}")

    # Load provider cluster data
    provider_path = Path("data/processed/provider_clusters.parquet")
    if not provider_path.exists():
        raise FileNotFoundError(f"Provider cluster data not found: {provider_path}")
    df_provider = pl.read_parquet(provider_path)
    print(f"  - Loaded {len(df_provider)} providers from {provider_path}")

    # Load deduped model data for intelligence quartiles
    models_path = Path("data/processed/ai_models_deduped.parquet")
    if not models_path.exists():
        raise FileNotFoundError(f"Models data not found: {models_path}")
    df_models = pl.read_parquet(models_path)
    print(f"  - Loaded {len(df_models)} models from {models_path}")

    # Create intelligence quartiles (for context window analysis)
    print("\n[2/4] Computing intelligence quartiles...")
    df_valid = df_models.filter(pl.col("intelligence_index").is_not_null())

    # Compute quartile boundaries
    q1_bound = df_valid["intelligence_index"].quantile(0.25)
    q2_bound = df_valid["intelligence_index"].quantile(0.50)
    q3_bound = df_valid["intelligence_index"].quantile(0.75)

    # Assign quartiles
    df_models = df_models.with_columns(
        pl.when(pl.col("intelligence_index") <= q1_bound)
        .then(pl.lit("Q1"))
        .when(pl.col("intelligence_index") <= q2_bound)
        .then(pl.lit("Q2"))
        .when(pl.col("intelligence_index") <= q3_bound)
        .then(pl.lit("Q3"))
        .otherwise(pl.lit("Q4"))
        .alias("intelligence_tier")
    )

    # Handle null intelligence_index (assign to "Unknown")
    df_models = df_models.with_columns(
        pl.when(pl.col("intelligence_index").is_null())
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("intelligence_tier"))
        .alias("intelligence_tier")
    )

    print(f"  - Q1: ≤ {q1_bound:.2f}")
    print(f"  - Q2: {q1_bound:.2f} - {q2_bound:.2f}")
    print(f"  - Q3: {q2_bound:.2f} - {q3_bound:.2f}")
    print(f"  - Q4: > {q3_bound:.2f}")

    # Create Pareto frontier charts (VIZ-05)
    print("\n[3/4] Creating Pareto frontier charts...")

    # Filter to models with valid intelligence for Pareto analysis
    df_pareto_valid = df_pareto.filter(pl.col("intelligence_index").is_not_null())
    print(f"  - Using {len(df_pareto_valid)} models with valid intelligence scores")

    # Intelligence vs Price Pareto frontier
    print("  - Creating Intelligence vs Price Pareto frontier...")
    fig_pareto_price = create_pareto_frontier_chart(
        df_pareto_valid,
        x_col="price_usd",
        y_col="intelligence_index",
        pareto_col="is_pareto_intelligence_price",
        title="Intelligence vs Price - Pareto Frontier (n={})".format(len(df_pareto_valid)),
        color_col="Creator",
    )

    pareto_price_path = output_dir / "interactive_pareto_intelligence_price.html"
    fig_pareto_price.write_html(
        str(pareto_price_path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"    Saved: {pareto_price_path}")

    # Speed vs Intelligence Pareto frontier
    print("  - Creating Speed vs Intelligence Pareto frontier...")
    # Convert Speed column to numeric if it's a string
    df_pareto_speed = df_pareto_valid.with_columns(
        pl.col("Speed(median token/s)")
        .cast(pl.Float64)
        .alias("speed_numeric")
    )

    fig_pareto_speed = create_pareto_frontier_chart(
        df_pareto_speed,
        x_col="speed_numeric",
        y_col="intelligence_index",
        pareto_col="is_pareto_speed_intelligence",
        title="Speed vs Intelligence - Pareto Frontier (n={})".format(len(df_pareto_speed)),
        color_col="Creator",
    )

    pareto_speed_path = output_dir / "interactive_pareto_speed_intelligence.html"
    fig_pareto_speed.write_html(
        str(pareto_speed_path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"    Saved: {pareto_speed_path}")

    # Create provider comparison dashboard (VIZ-06)
    print("\n  - Creating provider comparison dashboard...")
    fig_provider = create_provider_comparison(
        df_provider,
        metrics=["avg_intelligence", "avg_price", "avg_speed"],
    )

    provider_dashboard_path = output_dir / "interactive_provider_dashboard.html"
    fig_provider.write_html(
        str(provider_dashboard_path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"    Saved: {provider_dashboard_path}")

    # Create context window analysis (VIZ-08)
    print("  - Creating context window analysis by intelligence tier...")
    fig_context = create_context_window_analysis(
        df_models,
        tier_col="intelligence_tier",
        context_col="context_window",
    )

    context_analysis_path = output_dir / "interactive_context_window_analysis.html"
    fig_context.write_html(
        str(context_analysis_path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"    Saved: {context_analysis_path}")

    # Create combined dashboard
    print("\n[4/4] Creating combined dashboard...")

    from plotly.subplots import make_subplots

    # Create 2x2 combined dashboard
    fig_combined = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Intelligence vs Price - Pareto Frontier",
            "Speed vs Intelligence - Pareto Frontier",
            "Provider Market Segments",
            "Context Window by Intelligence Tier",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "box"}],
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Add traces from individual figures (simplified approach - recreate key elements)
    # Note: This is a simplified combined view. Full implementation would extract
    # traces from individual figures and re-add them to the combined layout.

    # For now, save individual files and create a simple HTML wrapper
    combined_html_path = output_dir / "interactive_provider_frontier.html"

    # Create HTML wrapper with iframe embeds
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Provider and Frontier Analysis Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .dashboard {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            max-width: 1600px;
            margin: 0 auto;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart-container h2 {{
            font-size: 14px;
            color: #666;
            margin: 0 0 10px 0;
            text-align: center;
        }}
        iframe {{
            width: 100%;
            height: 500px;
            border: none;
            border-radius: 4px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
    </style>
</head>
<body>
    <h1>Provider and Frontier Analysis Dashboard</h1>
    <div class="dashboard">
        <div class="chart-container">
            <h2>Intelligence vs Price - Pareto Frontier</h2>
            <iframe src="interactive_pareto_intelligence_price.html"></iframe>
        </div>
        <div class="chart-container">
            <h2>Speed vs Intelligence - Pareto Frontier</h2>
            <iframe src="interactive_pareto_speed_intelligence.html"></iframe>
        </div>
        <div class="chart-container full-width">
            <h2>Provider Market Segments (3-Panel Comparison)</h2>
            <iframe src="interactive_provider_dashboard.html" style="height: 400px;"></iframe>
        </div>
        <div class="chart-container full-width">
            <h2>Context Window by Intelligence Tier</h2>
            <iframe src="interactive_context_window_analysis.html" style="height: 400px;"></iframe>
        </div>
    </div>
</body>
</html>
"""

    with open(combined_html_path, "w") as f:
        f.write(html_content)

    print(f"    Saved: {combined_html_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nGenerated {len(list(output_dir.glob("interactive_*.html")))} interactive visualizations:")
    print(f"  - {pareto_price_path.name}")
    print(f"  - {pareto_speed_path.name}")
    print(f"  - {provider_dashboard_path.name}")
    print(f"  - {context_analysis_path.name}")
    print(f"  - {combined_html_path.name} (combined dashboard)")
    print("\nOpen files in a web browser to explore interactive features.")
    print("=" * 60)


if __name__ == "__main__":
    main()
