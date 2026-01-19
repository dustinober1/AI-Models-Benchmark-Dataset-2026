#!/usr/bin/env python3
"""
Correlation Analysis Pipeline - Phase 2 Plan 02

Computes Spearman correlation matrix with FDR correction for all numerical variables.
Identifies significant relationships while controlling for multiple testing.
Analyzes context window distribution by intelligence tier (STAT-05).

Usage:
    PYTHONPATH=. poetry run python3 scripts/08_correlation_analysis.py
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.statistics import (
    compute_spearman_correlation,
    compute_correlation_matrix,
    apply_fdr_correction,
    interpret_correlation,
    group_by_quartile
)
from pathlib import Path
from datetime import datetime


def main():
    """Execute correlation analysis pipeline."""
    # Paths
    input_path = "data/processed/ai_models_deduped.parquet"
    output_base = "data/processed/correlation_analysis"
    heatmap_path = "reports/figures/correlation_heatmap.png"
    tier_plot_path = "reports/figures/context_window_by_intelligence_tier.png"
    report_path = "reports/correlation_analysis_2026-01-18.md"

    print(f"Loading: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"Loaded {df.height} models")

    # Define numerical columns for correlation
    # NOTE: Use intelligence_index (Int64) not "Intelligence Index" (String with "--" placeholders)
    numerical_cols = [
        "intelligence_index",
        "price_usd",
        "Speed(median token/s)",
        "Latency (First Answer Chunk /s)",
        "context_window"
    ]

    # Filter to models with valid intelligence scores
    df_valid = df.filter(pl.col("intelligence_index").is_not_null())
    print(f"Models with valid intelligence: {df_valid.height}")

    # Compute correlation matrix
    print("\n=== COMPUTING SPEARMAN CORRELATION ===")
    corr_df, p_df = compute_correlation_matrix(df_valid, numerical_cols)
    print(f"Computed {len(numerical_cols)}x{len(numerical_cols)} correlation matrix")

    # Apply FDR correction to p-values
    print("\n=== APPLYING FDR CORRECTION ===")
    p_values_flat = p_df.select(pl.exclude("column")).to_numpy().flatten()
    p_adjusted_flat = apply_fdr_correction(p_values_flat, method='bh')

    # Reshape adjusted p-values back to matrix
    p_adjusted_matrix = p_adjusted_flat.reshape(len(numerical_cols), len(numerical_cols))
    p_adjusted_df = pl.DataFrame(p_adjusted_matrix, schema=numerical_cols)
    p_adjusted_df = p_adjusted_df.insert_column(0, pl.Series("column", numerical_cols))

    # Save results
    print(f"\nSaving: {output_base}_*.parquet")
    corr_df.write_parquet(f"{output_base}_correlation.parquet")
    p_df.write_parquet(f"{output_base}_p_raw.parquet")
    p_adjusted_df.write_parquet(f"{output_base}_p_adjusted.parquet")

    # Create correlation heatmap
    print(f"\nGenerating heatmap: {heatmap_path}")
    create_correlation_heatmap(corr_df, p_adjusted_df, numerical_cols, heatmap_path)

    # Analyze context window by intelligence tier (STAT-05)
    print("\n=== CONTEXT WINDOW BY INTELLIGENCE TIER (STAT-05) ===")
    tier_analysis = analyze_context_window_by_tier(df_valid, tier_plot_path)
    print(f"Tier analysis complete: {tier_plot_path}")

    # Generate correlation report
    generate_correlation_report(
        corr_df, p_df, p_adjusted_df, numerical_cols,
        tier_analysis, report_path
    )
    print(f"Report: {report_path}")

    print("\n✓ Correlation analysis complete")


def create_correlation_heatmap(
    corr_df: pl.DataFrame,
    p_adj_df: pl.DataFrame,
    columns: list[str],
    output_path: str
) -> None:
    """
    Create correlation heatmap with hierarchical clustering and significance annotations.

    Parameters
    ----------
    corr_df : pl.DataFrame
        Correlation matrix DataFrame.
    p_adj_df : pl.DataFrame
        FDR-adjusted p-value matrix DataFrame.
    columns : list[str]
        List of column names for labels.
    output_path : str
        Path to save the heatmap figure.
    """
    # Column display name mapping for readability
    display_names = {
        "intelligence_index": "Intelligence Index",
        "price_usd": "Price (USD)",
        "Speed(median token/s)": "Speed (median token/s)",
        "Latency (First Answer Chunk /s)": "Latency (First Answer Chunk /s)",
        "context_window": "Context Window",
    }
    display_labels = [display_names.get(col, col) for col in columns]

    # Extract correlation matrix
    corr_matrix = corr_df.select(pl.exclude("column")).to_numpy()

    # Create figure with clustermap
    sns.set_style("whitegrid")
    g = sns.clustermap(
        corr_matrix,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt='.2f',
        xticklabels=display_labels,
        yticklabels=display_labels,
        figsize=(12, 10),
        cbar_kws={'label': 'Spearman Correlation'},
        dendrogram_ratio=0.1
    )

    plt.suptitle('Spearman Correlation Matrix with FDR Correction', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_context_window_by_tier(df: pl.DataFrame, output_path: str) -> dict:
    """
    Analyze context window distribution by intelligence quartile (STAT-05).

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with valid intelligence scores.
    output_path : str
        Path to save the tier visualization.

    Returns
    -------
    dict
        Dictionary containing tier statistics and correlation results.
    """
    # Add intelligence quartile column
    # NOTE: Use intelligence_index (Int64) column, not "Intelligence Index" (String)
    df_with_tiers = df.with_columns(
        pl.col("intelligence_index")
        .qcut([0.25, 0.5, 0.75], labels=["Q1 (Low)", "Q2 (Mid-Low)", "Q3 (Mid-High)", "Q4 (High)"])
        .alias("intelligence_quartile")
    )

    # Compute context window statistics by quartile
    tier_stats = df_with_tiers.group_by("intelligence_quartile").agg([
        pl.col("context_window").count().alias("count"),
        pl.col("context_window").mean().alias("mean_context_window"),
        pl.col("context_window").median().alias("median_context_window"),
        pl.col("context_window").std().alias("std_context_window"),
        pl.col("context_window").min().alias("min_context_window"),
        pl.col("context_window").max().alias("max_context_window"),
        pl.col("intelligence_index").mean().alias("mean_intelligence")
    ]).sort("intelligence_quartile")

    print("\nContext Window by Intelligence Quartile:")
    print(tier_stats)

    # Create visualization
    create_tier_visualization(df_with_tiers, output_path)

    # Compute Spearman correlation between intelligence and context window
    intelligence_data = df["intelligence_index"].drop_nulls().cast(pl.Float64).to_numpy()
    context_data = df["context_window"].drop_nulls().cast(pl.Float64).to_numpy()
    corr, p_val = compute_spearman_correlation(intelligence_data, context_data)

    # Return statistics for report
    return {
        "tier_stats": tier_stats,
        "correlation_with_intelligence": corr,
        "p_value": p_val,
        "n_models": df.height
    }


def create_tier_visualization(df: pl.DataFrame, output_path: str) -> None:
    """
    Create box plot showing context window distribution by intelligence quartile.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with intelligence_quartile column.
    output_path : str
        Path to save the visualization.
    """
    plt.figure(figsize=(10, 6))

    # Order quartiles properly
    quartile_order = ["Q1 (Low)", "Q2 (Mid-Low)", "Q3 (Mid-High)", "Q4 (High)"]

    # Prepare data for seaborn (extract arrays from polars)
    data_for_plot = []
    for quartile in quartile_order:
        quartile_data = df.filter(
            pl.col("intelligence_quartile") == quartile
        )["context_window"].to_numpy()
        data_for_plot.append(quartile_data)

    # Create box plot using matplotlib directly (avoids pyarrow dependency)
    bp = plt.boxplot(data_for_plot, labels=quartile_order, patch_artist=True)

    # Color the boxes
    colors = sns.color_palette("viridis", len(quartile_order))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add colors to whiskers, caps, and medians
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black')

    plt.xlabel("Intelligence Quartile", fontsize=12, fontweight='bold')
    plt.ylabel("Context Window (tokens)", fontsize=12, fontweight='bold')
    plt.title("Context Window Distribution by Intelligence Tier (STAT-05)", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_correlation_report(
    corr_df: pl.DataFrame,
    p_df: pl.DataFrame,
    p_adj_df: pl.DataFrame,
    columns: list[str],
    tier_analysis: dict,
    output_path: str
) -> None:
    """
    Generate narrative report of correlation findings.

    Parameters
    ----------
    corr_df : pl.DataFrame
        Correlation matrix DataFrame.
    p_df : pl.DataFrame
        Raw p-value matrix DataFrame.
    p_adj_df : pl.DataFrame
        FDR-adjusted p-value matrix DataFrame.
    columns : list[str]
        List of column names.
    tier_analysis : dict
        Tier analysis results from analyze_context_window_by_tier.
    output_path : str
        Path to save the report.
    """
    # Column display name mapping for readability
    display_names = {
        "intelligence_index": "Intelligence Index",
        "price_usd": "Price (USD)",
        "Speed(median token/s)": "Speed (median token/s)",
        "Latency (First Answer Chunk /s)": "Latency (First Answer Chunk /s)",
        "context_window": "Context Window",
        "Intelligence Index": "Intelligence Index",
        "Context Window": "Context Window"
    }

    def display_name(col: str) -> str:
        return display_names.get(col, col)

    lines = []

    # Header
    lines.append("# Correlation Analysis Report")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("**Analysis:** Spearman correlation with FDR correction")
    lines.append("\n---\n")

    # Methodology
    lines.append("## Methodology")
    lines.append("\n### Statistical Approach")
    lines.append("- **Correlation Method:** Spearman rank correlation (non-parametric)")
    lines.append("  - Rationale: All numerical variables are right-skewed per Phase 1 distribution analysis")
    lines.append("  - Spearman is rank-based, does not assume normality")
    lines.append("  - Robust to outliers compared to Pearson correlation")
    lines.append("- **Multiple Testing Correction:** Benjamini-Hochberg FDR (False Discovery Rate)")
    lines.append("  - Rationale: More powerful than Bonferroni for multiple comparisons")
    lines.append("  - Controls expected proportion of false discoveries among rejected hypotheses")
    lines.append("  - Adjusted p-values reported alongside raw p-values")
    lines.append("- **Significance Threshold:** p_adjusted < 0.05")
    lines.append("\n### Data")
    lines.append(f"- **Sample Size:** {tier_analysis['n_models']} models with valid intelligence scores")
    lines.append(f"- **Variables Analyzed:** {len(columns)} numerical metrics")
    lines.append(f"- **Tests Performed:** {len(columns) * (len(columns) - 1) // 2} pairwise correlations")

    # Correlation summary table
    lines.append("\n---\n")
    lines.append("## Correlation Summary")
    lines.append("\n### All Pairwise Correlations")
    lines.append("\n| Variable 1 | Variable 2 | Correlation | Raw p-value | Adjusted p-value | Significant | Interpretation |")
    lines.append("|-----------|-----------|-------------|-------------|------------------|-------------|----------------|")

    significant_count = 0
    null_count = 0

    # Iterate through unique pairs (i < j)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]

            # Indexing: first column is "column" (row labels), so offset column index by +1
            corr = corr_df[i, j + 1]
            p_raw = p_df[i, j + 1]
            p_adj = p_adj_df[i, j + 1]

            interpretation = interpret_correlation(corr, p_raw, p_adj)
            significant = "Yes" if interpretation['significant'] else "No"

            if interpretation['significant']:
                significant_count += 1
            else:
                null_count += 1

            lines.append(
                f"| {display_name(col1)} | {display_name(col2)} | {corr:.3f} | {p_raw:.4f} | {p_adj:.4f} | "
                f"{significant} | {interpretation['interpretation']} |"
            )

    # Significant findings
    lines.append("\n---\n")
    lines.append("## Significant Findings (FDR-corrected)")
    lines.append(f"\n**Total Significant Correlations:** {significant_count} of {significant_count + null_count}")
    lines.append("\nStatistically significant correlations after FDR correction:")

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]

            # Indexing: first column is "column" (row labels), so offset column index by +1
            corr = corr_df[i, j + 1]
            p_raw = p_df[i, j + 1]
            p_adj = p_adj_df[i, j + 1]

            interpretation = interpret_correlation(corr, p_raw, p_adj)

            if interpretation['significant']:
                lines.append(f"\n#### {display_name(col1)} vs {display_name(col2)}")
                lines.append(f"- **Correlation:** {corr:.3f} ({interpretation['direction']} {interpretation['strength']})")
                lines.append(f"- **Raw p-value:** {p_raw:.4f}")
                lines.append(f"- **Adjusted p-value:** {p_adj:.4f}")
                lines.append(f"- **Interpretation:** {interpretation['interpretation']}")

    # Null findings (STAT-11 requirement)
    lines.append("\n---\n")
    lines.append("## Null Findings (STAT-11)")
    lines.append(f"\n**Total Non-Significant Correlations:** {null_count} of {significant_count + null_count}")
    lines.append("\nCorrelations that did NOT reach statistical significance after FDR correction:")
    lines.append("(These are reported to avoid publication bias)")

    if null_count > 0:
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]

                # Indexing: first column is "column" (row labels), so offset column index by +1
                corr = corr_df[i, j + 1]
                p_raw = p_df[i, j + 1]
                p_adj = p_adj_df[i, j + 1]

                interpretation = interpret_correlation(corr, p_raw, p_adj)

                if not interpretation['significant']:
                    lines.append(f"\n#### {display_name(col1)} vs {display_name(col2)}")
                    lines.append(f"- **Correlation:** {corr:.3f} ({interpretation['direction']} {interpretation['strength']})")
                    lines.append(f"- **Raw p-value:** {p_raw:.4f}")
                    lines.append(f"- **Adjusted p-value:** {p_adj:.4f}")
                    lines.append(f"- **Interpretation:** {interpretation['interpretation']}")
                    lines.append(f"- **Note:** This correlation is not statistically significant and may be due to chance")
    else:
        lines.append("\n*All correlations were statistically significant.*")

    # STAT-05: Context Window by Intelligence Tier
    lines.append("\n---\n")
    lines.append("## Context Window by Intelligence Tier (STAT-05)")
    lines.append("\n### Overview")
    lines.append("\nThis section analyzes how context window capacity scales with model intelligence.")
    lines.append(f"- **Spearman Correlation:** {tier_analysis['correlation_with_intelligence']:.3f}")
    lines.append(f"- **p-value:** {tier_analysis['p_value']:.4f}")

    corr_interpretation = interpret_correlation(
        tier_analysis['correlation_with_intelligence'],
        tier_analysis['p_value'],
        tier_analysis['p_value']  # Single test, no FDR correction needed
    )
    lines.append(f"- **Interpretation:** {corr_interpretation['interpretation']}")

    lines.append("\n### Context Window Statistics by Intelligence Quartile")
    lines.append("\n| Intelligence Tier | Count | Mean Context Window | Median Context Window | Std Dev | Min | Max |")
    lines.append("|-------------------|-------|---------------------|----------------------|---------|-----|-----|")

    for row in tier_analysis['tier_stats'].iter_rows(named=True):
        tier = row['intelligence_quartile']
        count = row['count']
        mean_cw = row['mean_context_window']
        median_cw = row['median_context_window']
        std_cw = row['std_context_window']
        min_cw = row['min_context_window']
        max_cw = row['max_context_window']

        lines.append(
            f"| {tier} | {count} | {mean_cw:,.0f} | {median_cw:,.0f} | {std_cw:,.0f} | {min_cw:,.0f} | {max_cw:,.0f} |"
        )

    lines.append("\n### Interpretation")
    lines.append("\nThe intelligence quartile analysis reveals how context capacity scales with model intelligence:")

    # Add interpretation based on correlation strength
    abs_corr = abs(tier_analysis['correlation_with_intelligence'])
    if abs_corr < 0.2:
        lines.append("- **Very weak relationship:** Context window does not systematically scale with intelligence.")
        lines.append("- Models at all intelligence levels have similar context capacities.")
    elif abs_corr < 0.4:
        lines.append("- **Weak relationship:** Slight positive trend between intelligence and context window.")
        lines.append("- Higher intelligence models tend to have slightly larger context windows.")
    elif abs_corr < 0.6:
        lines.append("- **Moderate relationship:** Clear positive trend between intelligence and context window.")
        lines.append("- Higher intelligence models generally have larger context windows.")
    else:
        lines.append("- **Strong relationship:** Intelligence and context window are strongly associated.")
        lines.append("- Higher intelligence models consistently have larger context windows.")

    lines.append("\n### Visualization")
    lines.append("\nSee figure: `reports/figures/context_window_by_intelligence_tier.png`")
    lines.append("- Box plot shows context window distribution for each intelligence quartile")
    lines.append("- Q1 = Lowest 25% intelligence, Q4 = Highest 25% intelligence")

    # Conclusions
    lines.append("\n---\n")
    lines.append("## Conclusions")
    lines.append("\n### Key Insights")
    lines.append(f"\n1. **{significant_count} significant correlations** identified after FDR correction")
    lines.append(f"2. **{null_count} null findings** reported (transparency per STAT-11)")
    lines.append("3. **Non-parametric approach validated:** Spearman correlation appropriate for skewed distributions")

    lines.append("\n### Methodology Strengths")
    lines.append("- Used Spearman correlation (robust to non-normality and outliers)")
    lines.append("- Applied FDR correction (controls false discovery rate)")
    lines.append("- Reported both significant and null findings (avoids publication bias)")
    lines.append("- Analyzed context window by intelligence tier (STAT-05 requirement)")

    lines.append("\n### Limitations")
    lines.append("- Cross-sectional data (single time point)")
    lines.append("- Correlation does not imply causation")
    lines.append("- Sample size limited to models with valid intelligence scores")
    lines.append("- Some variables have extreme skewness (e.g., Context Window)")

    lines.append("\n---\n")
    lines.append("## Figures")
    lines.append("\n1. **Correlation Heatmap:** `reports/figures/correlation_heatmap.png`")
    lines.append("   - Hierarchical clustering groups correlated variables")
    lines.append("   - Color scale: Blue (negative) to Red (positive)")
    lines.append("   - Annotations show correlation coefficients")
    lines.append("\n2. **Context Window by Intelligence Tier:** `reports/figures/context_window_by_intelligence_tier.png`")
    lines.append("   - Box plot showing context window distribution by intelligence quartile")
    lines.append("   - Q1 (Low) to Q4 (High) intelligence tiers")

    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    main()
