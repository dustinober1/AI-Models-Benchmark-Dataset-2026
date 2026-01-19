#!/usr/bin/env python3
"""
Linked Brushing Visualization Pipeline (VIZ-07, VIZ-10)

This script creates advanced interactive visualizations including speed-intelligence
tradeoff analysis with use case zones and linked brushing dashboard for cross-filtering.

Outputs
-------
- reports/figures/interactive_tradeoff_analysis.html
- reports/figures/interactive_linked_dashboard.html
- reports/figures/all_visualizations.html

Usage
-----
poetry run python scripts/15_linked_brushing_viz.py
"""

import polars as pl
from pathlib import Path
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualize import (
    create_speed_intelligence_tradeoff,
    create_linked_brushing_dashboard,
)


def main():
    """Execute linked brushing visualization pipeline."""

    print("=" * 60)
    print("Linked Brushing Visualization Pipeline")
    print("=" * 60)

    # Create output directory
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[1/3] Loading data...")

    # Load deduplicated models
    models_path = Path("data/processed/ai_models_deduped.parquet")
    if not models_path.exists():
        raise FileNotFoundError(f"Models data not found: {models_path}")
    df_models = pl.read_parquet(models_path)
    print(f"  - Loaded {len(df_models)} models from {models_path}")

    # Load provider clusters
    provider_path = Path("data/processed/provider_clusters.parquet")
    if not provider_path.exists():
        raise FileNotFoundError(f"Provider cluster data not found: {provider_path}")
    df_provider = pl.read_parquet(provider_path)
    print(f"  - Loaded {len(df_provider)} providers from {provider_path}")

    # Merge cluster assignments
    df_models = df_models.join(
        df_provider.select(["Creator", "cluster"]),
        on="Creator",
        how="left"
    )
    print(f"  - Merged cluster assignments (total: {len(df_models)} models)")

    # Filter to models with valid intelligence
    print("\n[2/3] Creating visualizations...")
    df_valid = df_models.filter(pl.col("intelligence_index").is_not_null())
    n_valid = len(df_valid)
    print(f"  - Using {n_valid} models with valid intelligence scores")

    # Create speed-intelligence tradeoff chart (VIZ-07)
    print("  - Creating Speed-Intelligence Tradeoff Analysis (VIZ-07)...")
    fig_tradeoff = create_speed_intelligence_tradeoff(
        df_valid,
        title=f"Speed vs Intelligence Tradeoff Analysis (n={n_valid})"
    )

    tradeoff_path = output_dir / "interactive_tradeoff_analysis.html"
    fig_tradeoff.write_html(
        str(tradeoff_path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"    Saved: {tradeoff_path}")

    # Create linked brushing dashboard (VIZ-10)
    print("  - Creating Linked Brushing Dashboard (VIZ-10)...")
    fig_linked = create_linked_brushing_dashboard(df_valid)

    linked_path = output_dir / "interactive_linked_dashboard.html"
    fig_linked.write_html(
        str(linked_path),
        include_plotlyjs="cdn",
        full_html=True,
    )
    print(f"    Saved: {linked_path}")

    # Create master index file
    print("\n[3/3] Creating master index file...")

    # List all interactive visualizations
    distributions = [
        ("interactive_intelligence_histogram.html", "Intelligence Index Distribution", "Distribution of intelligence scores across all models (n=181)"),
        ("interactive_price_histogram.html", "Price Distribution", "Distribution of pricing across all models (n=187)"),
        ("interactive_speed_histogram.html", "Speed Distribution", "Distribution of inference speed in tokens/s (n=187)"),
        ("interactive_latency_histogram.html", "Latency Distribution", "Distribution of first answer latency (n=187)"),
        ("interactive_context_window_histogram.html", "Context Window Distribution", "Distribution of context window sizes (n=187)"),
        ("interactive_intelligence_box_plot.html", "Intelligence by Provider", "Intelligence scores segmented by provider (n=181)"),
        ("interactive_price_box_plot.html", "Price by Provider", "Pricing segmented by provider (n=187)"),
        ("interactive_speed_box_plot.html", "Speed by Provider", "Inference speed segmented by provider (n=187)"),
        ("interactive_latency_box_plot.html", "Latency by Provider", "First answer latency segmented by provider (n=187)"),
        ("interactive_context_window_box_plot.html", "Context Window by Provider", "Context window sizes segmented by provider (n=187)"),
        ("interactive_correlation_heatmap.html", "Correlation Heatmap", "5x5 Spearman correlation matrix with FDR correction"),
    ]

    provider_frontier = [
        ("interactive_pareto_intelligence_price.html", "Intelligence vs Price - Pareto Frontier", "Pareto-efficient models in price-performance tradeoff"),
        ("interactive_pareto_speed_intelligence.html", "Speed vs Intelligence - Pareto Frontier", "Pareto-efficient models in speed-intelligence tradeoff"),
        ("interactive_provider_dashboard.html", "Provider Market Segments", "3-panel comparison: Intelligence-Price, Intelligence-Speed, Price-Speed"),
        ("interactive_context_window_analysis.html", "Context Window by Intelligence Tier", "Context window distribution across Q1-Q4 intelligence tiers"),
    ]

    advanced = [
        ("interactive_tradeoff_analysis.html", "Speed-Intelligence Tradeoff Analysis", "Use case zones: Real-time, High-IQ, Balanced, Budget"),
        ("interactive_linked_dashboard.html", "Linked Brushing Dashboard", "4-panel cross-filtering: Intelligence, Price histograms and scatter plots"),
    ]

    # Create HTML index
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Models Benchmark - Interactive Visualizations</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.8;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            transition: all 0.3s ease;
            text-decoration: none;
            color: inherit;
            display: block;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }}
        .card h3 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.2em;
        }}
        .card p {{
            color: #666;
            font-size: 0.95em;
            line-height: 1.5;
        }}
        .badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            margin-bottom: 10px;
        }}
        footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e9ecef;
        }}
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
            header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>AI Models Benchmark Dataset</h1>
            <p>Interactive Visualization Dashboard (2026)</p>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">187</div>
                    <div class="stat-label">Models</div>
                </div>
                <div class="stat">
                    <div class="stat-value">15</div>
                    <div class="stat-label">Visualizations</div>
                </div>
                <div class="stat">
                    <div class="stat-value">37</div>
                    <div class="stat-label">Providers</div>
                </div>
            </div>
        </header>

        <div class="content">

            <div class="section">
                <h2>Distribution Analysis (6 charts)</h2>
                <div class="grid">
"""

    # Add distribution cards
    for filename, title, description in distributions:
        html_content += f"""
                    <a href="{filename}" target="_blank" class="card">
                        <span class="badge">VIZ-01, VIZ-02</span>
                        <h3>{title}</h3>
                        <p>{description}</p>
                    </a>
"""

    html_content += """
                </div>
            </div>

            <div class="section">
                <h2>Correlation Analysis (1 chart)</h2>
                <div class="grid">
"""

    # Add correlation card
    html_content += f"""
                    <a href="{distributions[-1][0]}" target="_blank" class="card">
                        <span class="badge">VIZ-04</span>
                        <h3>{distributions[-1][1]}</h3>
                        <p>{distributions[-1][2]}</p>
                    </a>
"""

    html_content += """
                </div>
            </div>

            <div class="section">
                <h2>Provider & Frontier Analysis (4 charts)</h2>
                <div class="grid">
"""

    # Add provider/frontier cards
    for filename, title, description in provider_frontier:
        html_content += f"""
                    <a href="{filename}" target="_blank" class="card">
                        <span class="badge">VIZ-05, VIZ-06, VIZ-08</span>
                        <h3>{title}</h3>
                        <p>{description}</p>
                    </a>
"""

    html_content += """
                </div>
            </div>

            <div class="section">
                <h2>Advanced Analysis (2 charts)</h2>
                <div class="grid">
"""

    # Add advanced cards
    for filename, title, description in advanced:
        html_content += f"""
                    <a href="{filename}" target="_blank" class="card">
                        <span class="badge">VIZ-07, VIZ-10</span>
                        <h3>{title}</h3>
                        <p>{description}</p>
                    </a>
"""

    html_content += f"""
                </div>
            </div>

            <div class="section">
                <h2>Combined Dashboards (2 charts)</h2>
                <div class="grid">
                    <a href="interactive_distributions.html" target="_blank" class="card">
                        <span class="badge">Combined</span>
                        <h3>All Distribution Histograms</h3>
                        <p>5-panel dashboard showing all distributions in one view</p>
                    </a>
                    <a href="interactive_box_plots.html" target="_blank" class="card">
                        <span class="badge">Combined</span>
                        <h3>All Box Plots by Provider</h3>
                        <p>5-panel dashboard showing all provider-segmented outliers</p>
                    </a>
                    <a href="interactive_provider_frontier.html" target="_blank" class="card">
                        <span class="badge">Combined</span>
                        <h3>Provider and Frontier Analysis</h3>
                        <p>4-panel dashboard with Pareto frontiers and provider market segments</p>
                    </a>
                </div>
            </div>

        </div>

        <footer>
            <p><strong>AI Models Benchmark Dataset 2026</strong></p>
            <p>Generated from Phase 3: Interactive Visualizations</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                All visualizations are standalone HTML files with interactive features (hover, zoom, pan)
            </p>
        </footer>
    </div>
</body>
</html>
"""

    index_path = output_dir / "all_visualizations.html"
    with open(index_path, "w") as f:
        f.write(html_content)
    print(f"    Saved: {index_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nGenerated {len(list(output_dir.glob('interactive_*.html')))} interactive visualizations:")
    print(f"  - {tradeoff_path.name} (VIZ-07: Speed-Intelligence Tradeoff)")
    print(f"  - {linked_path.name} (VIZ-10: Linked Brushing Dashboard)")
    print(f"  - {index_path.name} (Master Index)")
    print(f"\nTotal: 15 interactive visualizations ready for Phase 4 notebook")
    print("\nOpen files in a web browser to explore interactive features.")
    print("=" * 60)


if __name__ == "__main__":
    main()
