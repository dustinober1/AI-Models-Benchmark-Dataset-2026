#!/usr/bin/env python3
"""
Pareto Frontier Analysis Pipeline - Phase 2 Plan 03

Identifies Pareto-efficient models that dominate in multi-objective space.
Analyzes price-performance and speed-intelligence tradeoffs.

Usage:
    PYTHONPATH=. python3 scripts/09_pareto_frontier.py
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
from src.pareto import (
    compute_pareto_frontier,
    get_pareto_efficient_models,
    plot_pareto_frontier
)


def generate_pareto_report(
    df: pl.DataFrame,
    efficient1: pl.DataFrame,
    efficient2: pl.DataFrame,
    efficient3: pl.DataFrame,
    output_path: str
) -> str:
    """
    Generate comprehensive Pareto analysis report in markdown format.

    Creates detailed report with:
    - Summary of each Pareto frontier analysis
    - List of Pareto-efficient models for each objective
    - Market leaders identification
    - Value propositions (best bang for buck)
    - Frontier interpretation and recommendations

    Parameters
    ----------
    df : pl.DataFrame
        Full dataset with Pareto flags.
    efficient1 : pl.DataFrame
        Intelligence vs Price Pareto-efficient models.
    efficient2 : pl.DataFrame
        Speed vs Intelligence Pareto-efficient models.
    efficient3 : pl.DataFrame
        Multi-objective Pareto-efficient models.
    output_path : str
        Path to save the report (Markdown format).

    Returns
    -------
    str
        Path to the generated report file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate statistics
    total_models = len(df)
    total_models_with_iq = df.filter(pl.col("intelligence_index").is_not_null()).height
    total_with_iq = total_models_with_iq  # Alias for compatibility

    # Build report
    report_lines = [
        "# Pareto Frontier Analysis Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Input:** ai_models_deduped.parquet",
        f"**Models Analyzed:** {total_models_with_iq}/{total_models} (with valid Intelligence Index)",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "Pareto frontier analysis identifies models that offer optimal tradeoffs between competing objectives. A model is **Pareto-efficient** if no other model dominates it (is better in all objectives). These models represent the \"efficient frontier\" - the best choices for different use cases.",
        "",
        "This analysis examines three objective spaces:",
        "",
        "1. **Intelligence vs Price** - Value proposition: Best intelligence per dollar",
        "2. **Speed vs Intelligence** - Performance leadership: Fast models with high intelligence",
        "3. **Multi-Objective** - Overall excellence: Balancing intelligence, speed, price, and latency",
        "",
        "---",
        "",
        "## Key Findings",
        ""
    ]

    # Analysis 1: Intelligence vs Price
    report_lines.extend([
        "### 1. Price-Performance Frontier (Intelligence vs Price)",
        "",
        f"**Pareto-Efficient Models:** {len(efficient1)}/{total_models_with_iq} ({len(efficient1)/total_models_with_iq*100:.1f}%)",
        "",
        "These models offer the best intelligence per dollar. No other model provides higher intelligence at a lower price.",
        "",
        "**Top Value Leaders:**",
        "",
        "| Model | Intelligence Index | Price ($/1M tokens) | Creator |",
        "|-------|-------------------|---------------------|---------|"
    ])

    for row in efficient1.sort("intelligence_index", descending=True).head(10).to_dicts():
        model = row.get("Model", row.get("model_id", "Unknown"))[:40]
        iq = row.get("intelligence_index", 0)
        price = row.get("price_usd", 0)
        creator = row.get("Creator", "Unknown")[:20]
        # Format price safely
        try:
            price_val = float(price) if price is not None else 0
            price_str = f"${price_val:.2f}"
        except (ValueError, TypeError):
            price_str = str(price)
        report_lines.append(f"| {model} | {iq:.0f} | {price_str} | {creator} |")

    report_lines.extend([
        "",
        "**Value Proposition:**",
        "- Budget-friendly options with competitive intelligence",
        "- Premium models with intelligence justifying higher cost",
        "- Best \"bang for buck\" models at each price point",
        ""
    ])

    # Analysis 2: Speed vs Intelligence
    report_lines.extend([
        "---",
        "",
        "### 2. Speed-Intelligence Frontier",
        "",
        f"**Pareto-Efficient Models:** {len(efficient2)}/{total_models_with_iq} ({len(efficient2)/total_models_with_iq*100:.1f}%)",
        "",
        "These models dominate in both speed and intelligence. No other model is faster AND smarter.",
        "",
        "**Performance Leaders:**",
        "",
        "| Model | Intelligence Index | Speed (tokens/s) | Creator |",
        "|-------|-------------------|-----------------|---------|"
    ])

    for row in efficient2.sort("intelligence_index", descending=True).head(10).to_dicts():
        model = row.get("Model", row.get("model_id", "Unknown"))[:40]
        iq = row.get("intelligence_index", 0)
        speed = row.get("Speed(median token/s)", 0)
        creator = row.get("Creator", "Unknown")[:20]
        # Format speed safely
        try:
            speed_val = float(speed) if speed is not None else 0
            speed_str = f"{speed_val:.1f}"
        except (ValueError, TypeError):
            speed_str = str(speed)
        report_lines.append(f"| {model} | {iq:.0f} | {speed_str} | {creator} |")

    report_lines.extend([
        "",
        "**Performance Insights:**",
        "- High-intelligence models with competitive speed",
        "- Real-time capable models with good intelligence",
        "- Throughput leaders for high-volume applications",
        ""
    ])

    # Analysis 3: Multi-Objective
    report_lines.extend([
        "---",
        "",
        "### 3. Multi-Objective Frontier (Overall Excellence)",
        "",
        f"**Pareto-Efficient Models:** {len(efficient3)}/{total_models_with_iq} ({len(efficient3)/total_models_with_iq*100:.1f}%)",
        "",
        "These models balance all four objectives: intelligence, speed, price, and latency. No other model is better in all dimensions.",
        "",
        "**Overall Optimal Models:**",
        "",
        "| Model | Intelligence | Speed | Price | Latency | Creator |",
        "|-------|-------------|-------|-------|---------|---------|"
    ])

    for row in efficient3.sort("intelligence_index", descending=True).head(10).to_dicts():
        model = row.get("Model", row.get("model_id", "Unknown"))[:35]
        iq = row.get("intelligence_index", 0)
        speed = row.get("Speed(median token/s)", 0)
        price = row.get("price_usd", 0)
        latency = row.get("Latency (First Answer Chunk /s)", 0)
        creator = row.get("Creator", "Unknown")[:15]
        # Format numeric values safely
        try:
            speed_val = float(speed) if speed is not None else 0
            speed_str = f"{speed_val:.1f}"
        except (ValueError, TypeError):
            speed_str = str(speed)
        try:
            price_val = float(price) if price is not None else 0
            price_str = f"${price_val:.2f}"
        except (ValueError, TypeError):
            price_str = str(price)
        try:
            latency_val = float(latency) if latency is not None else 0
            latency_str = f"{latency_val:.1f}"
        except (ValueError, TypeError):
            latency_str = str(latency)
        report_lines.append(f"| {model} | {iq:.0f} | {speed_str} | {price_str} | {latency_str} | {creator} |")

    report_lines.extend([
        "",
        "**Market Leaders:**",
        "- These models represent the state-of-the-art across multiple dimensions",
        "- No single model dominates all objectives - tradeoffs exist",
        "- Different leaders for different use cases (budget, speed, intelligence)",
        ""
    ])

    # Market Insights
    report_lines.extend([
        "---",
        "",
        "## Market Insights",
        "",
        "### Provider Dominance",
        "",
        "Which providers dominate the Pareto frontiers:",
        ""
    ])

    # Provider analysis for each frontier
    for i, (efficient, name) in enumerate([
        (efficient1, "Price-Performance"),
        (efficient2, "Speed-Intelligence"),
        (efficient3, "Multi-Objective")
    ], 1):
        provider_counts = efficient.group_by("Creator").agg(
            pl.len().alias("count")
        ).sort("count", descending=True)

        report_lines.extend([
            f"**{name} Frontier:**",
            ""
        ])

        for row in provider_counts.head(5).to_dicts():
            creator = row["Creator"][:30]
            count = row["count"]
            pct = count / len(efficient) * 100
            report_lines.append(f"- {creator}: {count} models ({pct:.1f} of frontier)")

        report_lines.append("")

    # Recommendations
    report_lines.extend([
        "---",
        "",
        "## Model Selection Recommendations",
        "",
        "### For Different Use Cases",
        "",
        "**1. Budget-Conscious Applications**",
        "- Choose from: Intelligence vs Price Pareto frontier",
        "- Prioritize: Low price per 1M tokens",
        "- Best for: High-volume applications, cost-sensitive projects",
        "",
        "**2. Performance-Critical Applications**",
        "- Choose from: Speed vs Intelligence Pareto frontier",
        "- Prioritize: High token throughput",
        "- Best for: Real-time applications, high-volume processing",
        "",
        "**3. Balanced Requirements**",
        "- Choose from: Multi-objective Pareto frontier",
        "- Prioritize: Overall excellence across all dimensions",
        "- Best for: General-purpose applications, uncertain requirements",
        "",
        "**4. Intelligence-Critical Applications**",
        "- Choose: Highest intelligence model within budget",
        "- Tradeoff: Accept higher price or lower speed",
        "- Best for: Complex reasoning, code generation, analysis",
        ""
    ])

    # Frontier Interpretation
    report_lines.extend([
        "---",
        "",
        "## Frontier Interpretation",
        "",
        "### What Does Pareto-Efficient Mean?",
        "",
        "A model is **Pareto-efficient** if:",
        "- No other model is better in **all** objectives being considered",
        "- Any improvement in one objective would require sacrifice in another",
        "- These models form the \"efficient frontier\" of the solution space",
        "",
        "### What About Non-Efficient Models?",
        "",
        "Models NOT on the Pareto frontier are **dominated** - there exists at least one other model that is better in all objectives. These models are generally not recommended unless:",
        "- You have specific constraints not captured in the analysis",
        "- You require features not measured (e.g., specific capabilities, ecosystem)",
        "- The model has other advantages (e.g., ease of use, documentation)",
        "",
        "### Frontier Density",
        "",
        f"- **Price-Performance:** {len(efficient1)} efficient models ({len(efficient1)/total_models_with_iq*100:.1f}% of analyzed models)",
        f"- **Speed-Intelligence:** {len(efficient2)} efficient models ({len(efficient2)/total_models_with_iq*100:.1f}% of analyzed models)",
        f"- **Multi-Objective:** {len(efficient3)} efficient models ({len(efficient3)/total_models_with_iq*100:.1f}% of analyzed models)",
        "",
        "A smaller frontier indicates clearer leaders. A larger frontier indicates more tradeoff options.",
        ""
    ])

    # Limitations
    report_lines.extend([
        "---",
        "",
        "## Limitations and Considerations",
        "",
        "**Analysis Scope:**",
        "- Only models with valid Intelligence Index scores ({total_models_with_iq} of {total_models} models)",
        "- Objectives limited to: Intelligence, Speed, Price, Latency",
        "- Does not consider: Capabilities, ecosystem, documentation, ease of use",
        "",
        "**Data Quality:**",
        "- Scores represent benchmarks at time of data collection",
        "- Performance may vary by task and implementation",
        "- Prices may change over time",
        "",
        "**Recommendation:**",
        "Use Pareto analysis as a starting point, but consider qualitative factors (features, support, ecosystem) for final model selection.",
        ""
    ])

    # Technical Details
    report_lines.extend([
        "---",
        "",
        "## Technical Details",
        "",
        "### Pareto Dominance Algorithm",
        "",
        "For two models A and B with objectives [o1, o2, ..., on]:",
        "",
        "**A dominates B if:**",
        "- A is better or equal to B in **all** objectives",
        "- A is strictly better than B in **at least one** objective",
        "",
        "**Implementation:**",
        "- Maximization objectives: Higher is better (e.g., intelligence)",
        "- Minimization objectives: Lower is better (e.g., price, latency)",
        "- Negate minimization objectives to convert to maximization",
        "- Check dominance using vectorized numpy operations",
        "",
        "### Output Files",
        "",
        "- `data/processed/pareto_frontier.parquet` - Dataset with Pareto flags",
        "- `reports/figures/pareto_frontier_intelligence_price.png` - Price-performance visualization",
        "- `reports/figures/pareto_frontier_speed_intelligence.png` - Speed-intelligence visualization",
        "",
        "---",
        "",
        "## Metadata",
        "",
        f"**Generation Timestamp:** {timestamp}",
        f"**Pipeline Version:** Phase 2 - Statistical Analysis & Domain Insights",
        f"**Plan:** 02-03 (Pareto Frontier Analysis)",
        "",
        "**Dependencies:**",
        "- polars >= 1.0.0",
        "- numpy >= 1.24.0",
        "- matplotlib >= 3.10.0",
        "- Input: `data/processed/ai_models_deduped.parquet`",
        "- Output: `data/processed/pareto_frontier.parquet`",
        "",
        "*End of Report*"
    ])

    # Write report to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    return str(output_path)


def main():
    """Main execution function for Pareto frontier analysis pipeline."""
    # Define paths
    input_path = "data/processed/ai_models_deduped.parquet"
    output_path = "data/processed/pareto_frontier.parquet"
    report_path = "reports/pareto_analysis_2026-01-18.md"

    print("=" * 70)
    print("PARETO FRONTIER ANALYSIS PIPELINE")
    print("=" * 70)
    print()

    # Load deduplicated dataset
    print(f"Loading: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"Loaded {df.height} models, {df.width} columns")
    print()

    # Filter to models with valid intelligence scores
    df_valid = df.filter(pl.col("intelligence_index").is_not_null())
    print(f"Models with valid Intelligence Index: {df_valid.height} ({df_valid.height/df.height*100:.1f}%)")
    print(f"Models with null Intelligence Index: {df.height - df_valid.height} (excluded from analysis)")
    print()

    # Analysis 1: Intelligence vs Price (maximize intelligence, minimize price)
    print("=" * 70)
    print("PARETO FRONTIER 1: INTELLIGENCE VS PRICE")
    print("=" * 70)
    print("Objectives: Maximize Intelligence Index, Minimize Price")
    print()

    df_pareto1 = compute_pareto_frontier(
        df_valid,
        maximize=["intelligence_index"],
        minimize=["price_usd"]
    )
    efficient1 = df_pareto1.filter(pl.col("is_pareto_efficient"))
    print(f"Pareto-efficient models: {len(efficient1)}/{len(df_pareto1)} ({len(efficient1)/len(df_pareto1)*100:.1f}%)")
    print()

    # Show top 5 value leaders
    print("Top 5 Value Leaders (Intelligence per Dollar):")
    value_leaders = efficient1.sort("intelligence_index", descending=True).head(5)
    for row in value_leaders.to_dicts():
        model = row.get("Model", row.get("model_id", "Unknown"))[:40]
        iq = row.get("intelligence_index", 0)
        price = row.get("price_usd", 0)
        # Handle price safely (might be string or float)
        try:
            price_val = float(price) if price is not None else 0
            price_str = f"${price_val:.2f}"
        except (ValueError, TypeError):
            price_str = f"${price}"
        print(f"  - {model}: IQ={iq:.0f}, Price={price_str}")
    print()

    # Plot Intelligence vs Price frontier
    plot_dir = Path("reports/figures")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_output = plot_dir / "pareto_frontier_intelligence_price.png"
    print(f"Generating plot: {plot_output}")
    plot_pareto_frontier(
        df_pareto1,
        x_col="price_usd",
        y_col="intelligence_index",
        x_minimize=True,
        y_maximize=True,
        output_path=str(plot_output),
        title="Price-Performance Pareto Frontier (Intelligence vs Price)"
    )
    print(f"Plot saved: {plot_output}")
    print()

    # Analysis 2: Speed vs Intelligence (maximize both)
    print("=" * 70)
    print("PARETO FRONTIER 2: SPEED VS INTELLIGENCE")
    print("=" * 70)
    print("Objectives: Maximize Speed, Maximize Intelligence Index")
    print()

    df_pareto2 = compute_pareto_frontier(
        df_valid,
        maximize=["Speed(median token/s)", "intelligence_index"],
        minimize=[]
    )
    efficient2 = df_pareto2.filter(pl.col("is_pareto_efficient"))
    print(f"Pareto-efficient models: {len(efficient2)}/{len(df_pareto2)} ({len(efficient2)/len(df_pareto2)*100:.1f}%)")
    print()

    # Show top 5 performance leaders
    print("Top 5 Performance Leaders (Speed + Intelligence):")
    perf_leaders = efficient2.sort("intelligence_index", descending=True).head(5)
    for row in perf_leaders.to_dicts():
        model = row.get("Model", row.get("model_id", "Unknown"))[:40]
        iq = row.get("intelligence_index", 0)
        speed = row.get("Speed(median token/s)", 0)
        # Handle speed safely (might be string or float)
        try:
            speed_val = float(speed) if speed is not None else 0
            speed_str = f"{speed_val:.1f} tok/s"
        except (ValueError, TypeError):
            speed_str = f"{speed} tok/s"
        print(f"  - {model}: IQ={iq:.0f}, Speed={speed_str}")
    print()

    # Plot Speed vs Intelligence frontier
    plot_output = plot_dir / "pareto_frontier_speed_intelligence.png"
    print(f"Generating plot: {plot_output}")
    plot_pareto_frontier(
        df_pareto2,
        x_col="Speed(median token/s)",
        y_col="intelligence_index",
        x_minimize=False,
        y_maximize=True,
        output_path=str(plot_output),
        title="Speed-Intelligence Pareto Frontier"
    )
    print(f"Plot saved: {plot_output}")
    print()

    # Analysis 3: Multi-objective (Intelligence, Speed, Price, Latency)
    print("=" * 70)
    print("PARETO FRONTIER 3: MULTI-OBJECTIVE")
    print("=" * 70)
    print("Objectives: Maximize Intelligence & Speed, Minimize Price & Latency")
    print()

    df_pareto3 = compute_pareto_frontier(
        df_valid,
        maximize=["intelligence_index", "Speed(median token/s)"],
        minimize=["price_usd", "Latency (First Answer Chunk /s)"]
    )
    efficient3 = df_pareto3.filter(pl.col("is_pareto_efficient"))
    print(f"Pareto-efficient models: {len(efficient3)}/{len(df_pareto3)} ({len(efficient3)/len(df_pareto3)*100:.1f}%)")
    print()

    # Show top 5 overall optimal models
    print("Top 5 Overall Optimal Models (All Objectives):")
    overall_leaders = efficient3.sort("intelligence_index", descending=True).head(5)
    for row in overall_leaders.to_dicts():
        model = row.get("Model", row.get("model_id", "Unknown"))[:35]
        iq = row.get("intelligence_index", 0)
        speed = row.get("Speed(median token/s)", 0)
        price = row.get("price_usd", 0)
        latency = row.get("Latency (First Answer Chunk /s)", 0)
        # Handle numeric values safely
        try:
            speed_val = float(speed) if speed is not None else 0
            speed_str = f"{speed_val:.1f}"
        except (ValueError, TypeError):
            speed_str = str(speed)
        try:
            price_val = float(price) if price is not None else 0
            price_str = f"${price_val:.2f}"
        except (ValueError, TypeError):
            price_str = f"${price}"
        try:
            latency_val = float(latency) if latency is not None else 0
            latency_str = f"{latency_val:.1f}"
        except (ValueError, TypeError):
            latency_str = str(latency)
        print(f"  - {model}")
        print(f"    IQ={iq:.0f}, Speed={speed_str} tok/s, Price={price_str}, Latency={latency_str}/s")
    print()

    # Merge Pareto flags into main dataset
    # Join on model_id to ensure correct alignment
    print("Merging Pareto flags into main dataset...")

    # Create DataFrames with only model_id and Pareto flags for joining
    flags1 = df_pareto1.select(["model_id", "is_pareto_efficient"]).rename(
        {"is_pareto_efficient": "is_pareto_intelligence_price"}
    )
    flags2 = df_pareto2.select(["model_id", "is_pareto_efficient"]).rename(
        {"is_pareto_efficient": "is_pareto_speed_intelligence"}
    )
    flags3 = df_pareto3.select(["model_id", "is_pareto_efficient"]).rename(
        {"is_pareto_efficient": "is_pareto_multi_objective"}
    )

    # Join flags back to main dataset
    df_final = df.join(flags1, on="model_id", how="left")
    df_final = df_final.join(flags2, on="model_id", how="left")
    df_final = df_final.join(flags3, on="model_id", how="left")

    print(f"Added Pareto flag columns to dataset")
    print()

    # Save results
    print("=" * 70)
    print(f"Saving: {output_path}")
    df_final.write_parquet(output_path)
    print(f"Saved {df_final.height} models to {output_path}")
    print()

    # Generate Pareto analysis report
    print("Generating Pareto analysis report...")
    report_path = generate_pareto_report(
        df_final,
        efficient1,
        efficient2,
        efficient3,
        report_path
    )
    print(f"Report: {report_path}")
    print()

    # Summary statistics
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total models: {df.height}")
    print(f"Models analyzed (valid IQ): {df_valid.height}")
    print()
    print(f"Frontier 1 (Intelligence vs Price): {len(efficient1)} efficient models")
    print(f"Frontier 2 (Speed vs Intelligence): {len(efficient2)} efficient models")
    print(f"Frontier 3 (Multi-objective): {len(efficient3)} efficient models")
    print()
    print("=" * 70)
    print("PARETO FRONTIER ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print()
    print("Frontier visualizations:")
    print(f"  - {plot_dir / 'pareto_frontier_intelligence_price.png'}")
    print(f"  - {plot_dir / 'pareto_frontier_speed_intelligence.png'}")


if __name__ == "__main__":
    main()
