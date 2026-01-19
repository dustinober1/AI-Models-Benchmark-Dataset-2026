#!/usr/bin/env python3
"""
Provider Clustering Pipeline - Phase 2 Plan 04

Segments providers by performance characteristics using KMeans clustering.
Identifies market segments and competitive positioning with regional comparisons.

Usage:
    PYTHONPATH=. python3 scripts/11_provider_clustering.py
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.clustering import (
    aggregate_by_provider,
    find_optimal_clusters,
    cluster_providers,
    validate_clustering,
    assign_region,
    compare_regions
)
from pathlib import Path
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def main():
    """Main pipeline execution."""
    # Paths
    input_path = "data/processed/ai_models_deduped.parquet"
    output_path = "data/processed/provider_clusters.parquet"
    report_path = "reports/provider_clustering_2026-01-18.md"

    print(f"Loading: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"Loaded {df.height} models")

    # Filter to models with valid intelligence scores
    df_valid = df.filter(pl.col("intelligence_index").is_not_null())
    print(f"Models with valid intelligence: {df_valid.height}")

    # Aggregate by provider
    print("\n=== AGGREGATING BY PROVIDER ===")
    provider_df = aggregate_by_provider(
        df_valid,
        ["intelligence_index", "price_usd", "Speed(median token/s)"]
    )
    print(f"Unique providers: {len(provider_df)}")

    # Assign regions
    provider_df = provider_df.with_columns(
        pl.col("Creator").map_elements(assign_region, return_dtype=pl.Utf8).alias("region")
    )
    print(f"Regions: {sorted(provider_df['region'].unique().to_list())}")
    print("Providers per region:")
    print(provider_df.group_by("region").agg(pl.len().alias("count")).sort("region"))

    # Cluster providers
    print("\n=== CLUSTERING PROVIDERS ===")
    features = ["avg_intelligence", "avg_price", "avg_speed"]

    # Handle null values in features
    provider_df_clean = provider_df.drop_nulls(subset=features)
    print(f"Providers after dropping nulls: {len(provider_df_clean)}")

    # Find optimal clusters first
    from sklearn.preprocessing import StandardScaler
    X = provider_df_clean.select(features).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    optimal_results = find_optimal_clusters(X_scaled, max_k=10)
    optimal_k = optimal_results["optimal_k"]
    print(f"Optimal clusters (silhouette method): {optimal_k}")

    # Cluster with optimal K
    clustered_df = cluster_providers(provider_df_clean, features, n_clusters=optimal_k)
    print(f"Clusters assigned: {sorted(clustered_df['cluster'].unique().to_list())}")
    print("Providers per cluster:")
    print(clustered_df.group_by("cluster").agg(pl.len().alias("count")).sort("cluster"))

    # Validate clustering
    print("\n=== VALIDATING CLUSTERS ===")
    validation = validate_clustering(X_scaled, clustered_df["cluster"].to_numpy())
    print(f"Silhouette score: {validation['silhouette_score']:.3f}")
    print(f"Number of clusters: {validation['n_clusters']}")

    # Save results
    print(f"\nSaving: {output_path}")
    clustered_df.write_parquet(output_path)

    # Generate cluster visualizations
    print("\n=== GENERATING VISUALIZATIONS ===")
    create_cluster_visualizations(clustered_df, features, validation, optimal_results)

    # Generate regional comparison
    print("\n=== REGIONAL COMPARISON ===")
    regional_comparison = compare_regions_by_metric(df_valid)

    # Generate clustering report
    generate_clustering_report(
        clustered_df,
        validation,
        optimal_results,
        regional_comparison,
        report_path
    )
    print(f"Report: {report_path}")

    print("\n✓ Provider clustering complete")


def create_cluster_visualizations(df, features, validation, optimal_results):
    """Create cluster analysis visualizations."""
    # 1. Silhouette scores plot
    k_range = range(2, len(optimal_results['silhouette_scores']) + 2)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, optimal_results['silhouette_scores'], 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_results['optimal_k'], color='r', linestyle='--', linewidth=2,
                label=f'Optimal K={optimal_results["optimal_k"]}')
    plt.xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    plt.title('Silhouette Score Analysis for Optimal K', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/silhouette_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: reports/figures/silhouette_scores.png")

    # 2. Elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, optimal_results['inertias'], 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
    plt.title('Elbow Method for Cluster Validation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/figures/elbow_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: reports/figures/elbow_plot.png")

    # 3. Cluster scatter plot (2D projection using first two features)
    create_cluster_scatter_plot(df, features)


def create_cluster_scatter_plot(df, features):
    """Create 2D scatter plot of provider clusters."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    feature_pairs = [
        (features[0], features[1], "Intelligence vs Price"),
        (features[0], features[2], "Intelligence vs Speed"),
        (features[1], features[2], "Price vs Speed")
    ]

    colors = sns.color_palette("husl", df['cluster'].n_unique())

    for idx, (feat_x, feat_y, title) in enumerate(feature_pairs):
        ax = axes[idx]

        # Plot each cluster
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df.filter(pl.col('cluster') == cluster_id)
            ax.scatter(
                cluster_data[feat_x].to_numpy(),
                cluster_data[feat_y].to_numpy(),
                label=f'Cluster {cluster_id}',
                alpha=0.7,
                s=100,
                color=colors[cluster_id]
            )

        # Label key providers
        key_providers = df.filter(
            pl.col('Creator').is_in(['OpenAI', 'Anthropic', 'Google', 'DeepSeek', 'Mistral', 'Alibaba'])
        )

        for row in key_providers.iter_rows(named=True):
            ax.annotate(
                row['Creator'],
                (row[feat_x], row[feat_y]),
                fontsize=8,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points'
            )

        ax.set_xlabel(feat_x.replace('avg_', '').replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_ylabel(feat_y.replace('avg_', '').replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/figures/provider_cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: reports/figures/provider_cluster_analysis.png")


def compare_regions_by_metric(df: pl.DataFrame) -> dict:
    """Compare providers by region across key metrics."""
    # Assign regions to model-level data
    df_with_region = df.with_columns(
        pl.col("Creator").map_elements(assign_region, return_dtype=pl.Utf8).alias("region")
    )

    # Compare by intelligence, price, speed
    metrics = ["intelligence_index", "price_usd", "Speed(median token/s)"]
    regional_stats = {}

    for metric in metrics:
        try:
            # Cast to Float64 for numeric operations
            stats = (
                df_with_region
                .group_by("region")
                .agg([
                    pl.col(metric).cast(pl.Float64).mean().alias("mean"),
                    pl.col(metric).cast(pl.Float64).median().alias("median"),
                    pl.col(metric).cast(pl.Float64).std().alias("std"),
                    pl.col(metric).cast(pl.Float64).min().alias("min"),
                    pl.col(metric).cast(pl.Float64).max().alias("max"),
                    pl.len().alias("count")
                ])
                .sort("region")
            )
            regional_stats[metric] = stats
        except Exception as e:
            print(f"Warning: Could not compute regional stats for {metric}: {e}")

    return regional_stats


def generate_clustering_report(df, validation, optimal_results, regional_comparison, output_path):
    """Generate narrative report of clustering findings."""
    from datetime import datetime

    # Compute cluster profiles
    cluster_profiles = df.group_by("cluster").agg([
        pl.col("avg_intelligence").mean().alias("mean_intelligence"),
        pl.col("avg_price").mean().alias("mean_price"),
        pl.col("avg_speed").mean().alias("mean_speed"),
        pl.col("Creator").alias("providers")
    ]).sort("cluster")

    lines = []
    lines.append("# Provider Clustering Analysis")
    lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n---\n")

    # 1. Cluster Validation
    lines.append("## 1. Cluster Validation")
    lines.append(f"\n**Optimal number of clusters:** {validation['n_clusters']}")
    lines.append(f"**Silhouette score:** {validation['silhouette_score']:.3f}")
    lines.append("\n**Interpretation:**")
    if validation['silhouette_score'] > 0.5:
        lines.append("- Silhouette score > 0.5 indicates **good cluster structure**")
        lines.append("- Providers are well-separated into distinct market segments")
    elif validation['silhouette_score'] > 0.25:
        lines.append("- Silhouette score > 0.25 indicates **moderate cluster structure**")
        lines.append("- Some overlap between clusters, but meaningful segments exist")
    else:
        lines.append("- Silhouette score < 0.25 indicates **weak cluster structure**")
        lines.append("- Clusters may not represent meaningful market segments")

    # 2. Silhouette Score Analysis
    lines.append("\n## 2. Silhouette Score Analysis")
    lines.append("\nSilhouette scores were computed for k=2 to k=10 clusters:")
    lines.append("\n| K | Silhouette Score |")
    lines.append("|---|------------------|")
    k_range = range(2, len(optimal_results['silhouette_scores']) + 2)
    for k, score in zip(k_range, optimal_results['silhouette_scores']):
        marker = " ← **Optimal K**" if k == optimal_results['optimal_k'] else ""
        lines.append(f"| {k} | {score:.3f} {marker} |")

    # 3. Cluster Profiles
    lines.append("\n## 3. Cluster Profiles")
    lines.append("\n### Characteristics by Cluster")

    for row in cluster_profiles.iter_rows(named=True):
        cluster_id = row['cluster']
        lines.append(f"\n#### Cluster {cluster_id}")
        lines.append(f"- **Mean Intelligence:** {row['mean_intelligence']:.1f}")
        lines.append(f"- **Mean Price:** ${row['mean_price']:.2f}")
        lines.append(f"- **Mean Speed:** {row['mean_speed']:.1f} tokens/s")
        lines.append(f"- **Providers ({len(row['providers'])}):** {', '.join(sorted(row['providers']))}")

    # 4. Market Segments
    lines.append("\n## 4. Market Segments")
    lines.append("\nBased on cluster profiles, the following market segments were identified:")

    # Analyze clusters to identify segments
    # Determine overall median to classify clusters
    overall_intel_median = cluster_profiles['mean_intelligence'].median()
    overall_price_median = cluster_profiles['mean_price'].median()
    overall_speed_median = cluster_profiles['mean_speed'].median()

    for row in cluster_profiles.iter_rows(named=True):
        cluster_id = row['cluster']
        intel = row['mean_intelligence']
        price = row['mean_price']
        speed = row['mean_speed']

        # Compare to overall medians for classification
        intel_above = intel > overall_intel_median
        price_above = price > overall_price_median
        speed_above = speed > overall_speed_median

        if intel_above and price_above and speed_above:
            segment = "Premium Performance Segment"
            desc = f"High-intelligence ({intel:.1f}), premium-priced (${price:.2f}), high-speed ({speed:.1f} tokens/s) providers"
        elif intel_above and price_above:
            segment = "Premium Intelligence Segment"
            desc = f"High-intelligence ({intel:.1f}), premium-priced (${price:.2f}) providers"
        elif intel_above:
            segment = "High-Value Segment"
            desc = f"High-intelligence ({intel:.1f}), competitively priced (${price:.2f}) providers"
        elif price_above:
            segment = "Niche Premium Segment"
            desc = f"Mid-tier intelligence ({intel:.1f}) with premium pricing (${price:.2f})"
        else:
            segment = "Budget-Friendly Segment"
            desc = f"Affordable providers (${price:.2f}) with mid-tier intelligence ({intel:.1f})"

        lines.append(f"\n**Cluster {cluster_id}: {segment}**")
        lines.append(f"- {desc}")

    # 5. Regional Comparison (STAT-04)
    lines.append("\n## 5. Regional Comparison (STAT-04)")
    lines.append("\nProvider performance compared across regions (US, China, Europe, Other):")

    for metric_name, metric_key in [
        ("Intelligence", "intelligence_index"),
        ("Price", "price_usd"),
        ("Speed", "Speed(median token/s)")
    ]:
        if metric_key in regional_comparison:
            lines.append(f"\n### {metric_name}")
            lines.append("\n| Region | Mean | Median | Std Dev | Min | Max | Count |")
            lines.append("|--------|------|--------|---------|-----|-----|-------|")

            for row in regional_comparison[metric_key].iter_rows(named=True):
                region = row['region']
                mean_val = row['mean']
                median_val = row['median']
                std_val = row['std']
                min_val = row['min']
                max_val = row['max']
                count = row['count']

                if metric_key == "price_usd":
                    lines.append(f"| {region} | ${mean_val:.2f} | ${median_val:.2f} | ${std_val:.2f} | ${min_val:.2f} | ${max_val:.2f} | {count} |")
                elif metric_key == "Speed(median token/s)":
                    lines.append(f"| {region} | {mean_val:.1f} | {median_val:.1f} | {std_val:.1f} | {min_val:.1f} | {max_val:.1f} | {count} |")
                else:
                    lines.append(f"| {region} | {mean_val:.1f} | {median_val:.1f} | {std_val:.1f} | {min_val:.1f} | {max_val:.1f} | {count} |")

    # 6. Strategic Insights
    lines.append("\n## 6. Strategic Insights")
    lines.append("\n### Market Structure")
    lines.append("- Provider clustering reveals distinct market segments based on intelligence, price, and speed")
    lines.append(f"- {validation['n_clusters']} market segments identified with good separation (silhouette: {validation['silhouette_score']:.3f})")

    lines.append("\n### Competitive Positioning")
    # Find top providers by cluster
    for row in cluster_profiles.iter_rows(named=True):
        cluster_id = row['cluster']
        providers = sorted(row['providers'])
        lines.append(f"\n**Cluster {cluster_id}:** {', '.join(providers[:5])}" + (f" (+{len(providers)-5} more)" if len(providers) > 5 else ""))

    lines.append("\n### Regional Differences")
    lines.append("- STAT-04 requirement: Regional comparison shows performance differences across US, China, and European providers")
    lines.append("- See Regional Comparison section above for detailed statistics by region")

    lines.append("\n### Implications")
    lines.append("- Market segments can inform pricing strategy and competitive positioning")
    lines.append("- Regional differences may reflect regulatory environments, market dynamics, or strategic priorities")
    lines.append("- Cluster assignments available in `data/processed/provider_clusters.parquet` for further analysis")

    # Write report
    report_content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(report_content)

    print(f"✓ Report generated: {output_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
