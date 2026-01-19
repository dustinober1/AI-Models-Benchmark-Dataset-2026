#!/usr/bin/env python3
"""
Statistical Testing Pipeline - Phase 2 Plan 05 (Part 1)

Performs group comparisons with non-parametric tests and bootstrap CIs.
Quantifies uncertainty for all statistical estimates.

Usage:
    PYTHONPATH=. python3 scripts/10_statistical_tests.py
"""

import polars as pl
import numpy as np
from scipy import stats
from src.bootstrap import (
    bootstrap_mean_ci,
    bootstrap_median_ci,
    bootstrap_group_difference_ci,
    mann_whitney_u_test,
    kruskal_wallis_test
)
from src.clustering import assign_region
from src.statistics import apply_fdr_correction
from pathlib import Path
from datetime import datetime


def main():
    # Load deduplicated dataset
    input_path = "data/processed/ai_models_deduped.parquet"
    report_path = "reports/statistical_tests_2026-01-18.md"

    print(f"Loading: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"Loaded {df.height} models")

    # Filter to models with valid intelligence scores
    df_valid = df.filter(pl.col("intelligence_index").is_not_null())
    print(f"Models with valid intelligence: {df_valid.height}")

    # Assign regions
    df_with_region = df_valid.with_columns(
        pl.col("Creator").map_elements(assign_region, return_dtype=pl.Utf8).alias("region")
    )
    print(f"Regions assigned: {df_with_region['region'].unique().to_list()}")

    # Test 1: Regional comparison (US vs China vs Europe) - STAT-04
    print("\n=== TEST 1: REGIONAL COMPARISON (KRUSKAL-WALLIS) ===")
    regional_results = test_regional_differences(df_with_region)

    # Test 2: Pareto-efficient vs dominated models - Mann-Whitney U
    print("\n=== TEST 2: PARETO EFFICIENT VS DOMINATED ===")
    if "is_pareto_multi_objective" in df.columns:
        pareto_results = test_pareto_differences(df_with_region)
    else:
        print("Skipping: Pareto flags not found")
        pareto_results = None

    # Test 3: Bootstrap CIs for key metrics - STAT-07, STAT-09
    print("\n=== TEST 3: BOOTSTRAP CONFIDENCE INTERVALS ===")
    bootstrap_results = compute_bootstrap_cis(df_valid)

    # Generate report with significant AND null findings - STAT-11
    print("\n=== GENERATING REPORT ===")
    generate_statistical_report(
        df_with_region,
        regional_results,
        pareto_results,
        bootstrap_results,
        report_path
    )
    print(f"Report: {report_path}")

    print("\n✓ Statistical testing complete")


def test_regional_differences(df: pl.DataFrame) -> dict:
    """Compare US vs China vs Europe providers using Kruskal-Wallis test."""
    metrics = ["intelligence_index", "price_usd", "Speed(median token/s)"]
    results = {}
    pairwise_results = []

    for metric in metrics:
        try:
            # Extract groups by region (cast to Float64)
            groups = []
            group_names = []
            for region in ["US", "China", "Europe"]:
                region_df = df.filter(pl.col("region") == region)
                region_data = region_df[metric].drop_nulls()
                if len(region_data) > 0:
                    # Cast to Float64 for string columns
                    if metric == "Speed(median token/s)":
                        groups.append(region_data.cast(pl.Utf8).cast(pl.Float64).to_numpy())
                    else:
                        groups.append(region_data.cast(pl.Float64).to_numpy())
                    group_names.append(region)

            if len(groups) >= 2:
                # Kruskal-Wallis test
                kw_result = kruskal_wallis_test(*groups)
                kw_result["groups"] = group_names
                kw_result["n_per_group"] = [len(g) for g in groups]
                results[metric] = kw_result

                # Pairwise Mann-Whitney U tests (if significant)
                if kw_result["significant"]:
                    print(f"{metric}: Kruskal-Wallis p={kw_result['p_value']:.4f} (significant)")
                    # Perform pairwise comparisons with FDR correction
                    for i in range(len(group_names)):
                        for j in range(i + 1, len(group_names)):
                            mw_result = mann_whitney_u_test(groups[i], groups[j])
                            mw_result["group1"] = group_names[i]
                            mw_result["group2"] = group_names[j]
                            pairwise_results.append(mw_result)
                else:
                    print(f"{metric}: Kruskal-Wallis p={kw_result['p_value']:.4f} (NOT significant)")

        except Exception as e:
            print(f"Error testing {metric}: {e}")
            results[metric] = {"error": str(e)}

    # Apply FDR correction to pairwise tests
    if pairwise_results:
        p_values = np.array([r["p_value"] for r in pairwise_results])
        p_adjusted = apply_fdr_correction(p_values, method='bh')

        for i, result in enumerate(pairwise_results):
            result["p_adjusted"] = float(p_adjusted[i])
            result["significant_adj"] = p_adjusted[i] < 0.05

    return {
        "kruskal_wallis": results,
        "pairwise": pairwise_results
    }


def test_pareto_differences(df: pl.DataFrame) -> dict:
    """Compare Pareto-efficient vs dominated models."""
    results = {}
    metrics = ["intelligence_index", "price_usd", "Speed(median token/s)"]

    for metric in metrics:
        try:
            # Extract groups
            pareto_df = df.filter(pl.col("is_pareto_multi_objective") == True)
            dominated_df = df.filter(pl.col("is_pareto_multi_objective") == False)
            pareto_data = pareto_df[metric].drop_nulls()
            dominated_data = dominated_df[metric].drop_nulls()

            if len(pareto_data) > 0 and len(dominated_data) > 0:
                # Cast to Float64 for string columns
                if metric == "Speed(median token/s)":
                    group1 = pareto_data.cast(pl.Utf8).cast(pl.Float64).to_numpy()
                    group2 = dominated_data.cast(pl.Utf8).cast(pl.Float64).to_numpy()
                else:
                    group1 = pareto_data.cast(pl.Float64).to_numpy()
                    group2 = dominated_data.cast(pl.Float64).to_numpy()

                # Mann-Whitney U test
                mw_result = mann_whitney_u_test(group1, group2)
                mw_result["n_pareto"] = len(group1)
                mw_result["n_dominated"] = len(group2)

                # Bootstrap CI for difference in means
                bootstrap_result = bootstrap_group_difference_ci(group1, group2)
                mw_result["bootstrap_ci"] = bootstrap_result

                results[metric] = mw_result

                if mw_result["significant"]:
                    print(f"{metric}: Mann-Whitney p={mw_result['p_value']:.4f} (significant)")
                else:
                    print(f"{metric}: Mann-Whitney p={mw_result['p_value']:.4f} (NOT significant)")

        except Exception as e:
            print(f"Error testing Pareto difference for {metric}: {e}")
            results[metric] = {"error": str(e)}

    return results


def compute_bootstrap_cis(df: pl.DataFrame) -> dict:
    """Compute bootstrap confidence intervals for key metrics."""
    metrics = {
        "Intelligence Index": df["intelligence_index"].drop_nulls().cast(pl.Float64).to_numpy(),
        "price_usd": df["price_usd"].drop_nulls().to_numpy(),
        "Speed(median token/s)": df["Speed(median token/s)"].drop_nulls().cast(pl.Utf8).cast(pl.Float64).to_numpy()
    }

    results = {}
    for metric_name, data in metrics.items():
        if len(data) > 0:
            # Bootstrap mean CI
            mean_result = bootstrap_mean_ci(data)
            # Bootstrap median CI
            median_result = bootstrap_median_ci(data)

            results[metric_name] = {
                "mean": mean_result,
                "median": median_result,
                "n": len(data)
            }
            print(f"{metric_name}: Mean 95% CI [{mean_result['ci_low']:.2f}, {mean_result['ci_high']:.2f}]")

    return results


def generate_statistical_report(df, regional, pareto, bootstrap, output_path):
    """Generate narrative report with significant and null findings."""
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Get date
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Build report sections
    sections = []

    # Title
    sections.append(f"# Statistical Tests Report")
    sections.append(f"**Generated:** {date_str}\n")

    # Section 1: Methodology (NARR-07 requirement)
    sections.append("## Methodology")
    sections.append("### Statistical Methods")
    sections.append("This analysis uses **non-parametric statistical tests** appropriate for the")
    sections.append("highly right-skewed distributions identified in Phase 1 (skewness 2.34-9.63).")
    sections.append("")
    sections.append("**Non-parametric tests:**")
    sections.append("- **Mann-Whitney U test:** Two-group comparison (alternative to independent t-test)")
    sections.append("- **Kruskal-Wallis test:** Three+ group comparison (alternative to one-way ANOVA)")
    sections.append("- **Bootstrap confidence intervals:** BCa method with 9,999 resamples")
    sections.append("")
    sections.append("**Why non-parametric?**")
    sections.append("- No assumption of normality")
    sections.append("- Robust to outliers")
    sections.append("- Appropriate for skewed distributions")
    sections.append("- Rank-based methods (Spearman, Mann-Whitney U, Kruskal-Wallis)")
    sections.append("")
    sections.append("**Multiple testing correction:**")
    sections.append("- Benjamini-Hochberg FDR correction applied to pairwise comparisons")
    sections.append("- Controls false discovery rate while maintaining power")
    sections.append("- Significance threshold: α = 0.05")
    sections.append("")
    sections.append("**Bootstrap methodology:**")
    sections.append(f"- BCa (bias-corrected and accelerated) method")
    sections.append(f"- n_resamples = 9,999 for accuracy")
    sections.append(f"- Confidence level = 95%")
    sections.append(f"- Fallback to percentile method if BCa fails")
    sections.append("")

    # Section 2: Regional Comparison Results
    sections.append("## Regional Comparison Results")
    sections.append("### Kruskal-Wallis Test (US vs China vs Europe)")
    sections.append("")

    kw_results = regional["kruskal_wallis"]
    for metric, result in kw_results.items():
        if "error" not in result:
            metric_display = metric.replace("_", " ").title()
            sections.append(f"#### {metric_display}")
            sections.append(f"- **H statistic:** {result['statistic']:.2f}")
            sections.append(f"- **p-value:** {result['p_value']:.4f}")
            sections.append(f"- **Significant:** {'Yes' if result['significant'] else 'No'}")
            sections.append(f"- **Groups:** {', '.join(result['groups'])}")
            sections.append(f"- **Sample sizes:** {result['n_per_group']}")
            sections.append("")

    # Section 3: Pairwise Comparisons
    if regional["pairwise"]:
        sections.append("### Pairwise Mann-Whitney U Tests (with FDR correction)")
        sections.append("")

        # Display name mapping
        display_names = {
            "intelligence_index": "Intelligence Index",
            "price_usd": "Price (USD)",
            "Speed(median token/s)": "Speed (median token/s)"
        }

        for result in regional["pairwise"]:
            group1 = result["group1"]
            group2 = result["group2"]
            sections.append(f"**{group1} vs {group2}:**")
            sections.append(f"- **Mann-Whitney U:** {result['statistic']:.2f}")
            sections.append(f"- **Raw p-value:** {result['p_value']:.4f}")
            sections.append(f"- **FDR-adjusted p-value:** {result['p_adjusted']:.4f}")
            sections.append(f"- **Significant (after FDR):** {'Yes' if result['significant_adj'] else 'No'}")
            sections.append(f"- **Effect size (r):** {result['effect_size']:.3f}")
            sections.append("")

    # Section 4: Pareto-Efficient vs Dominated
    if pareto:
        sections.append("## Pareto-Efficient vs Dominated Models")
        sections.append("### Mann-Whitney U Test Results")
        sections.append("")

        for metric, result in pareto.items():
            if "error" not in result:
                metric_display = metric.replace("_", " ").replace("(", "").replace(")", "").title()
                sections.append(f"#### {metric_display}")
                sections.append(f"- **Mann-Whitney U:** {result['statistic']:.2f}")
                sections.append(f"- **p-value:** {result['p_value']:.4f}")
                sections.append(f"- **Significant:** {'Yes' if result['significant'] else 'No'}")
                sections.append(f"- **Effect size (r):** {result['effect_size']:.3f}")
                sections.append(f"- **Sample sizes:** Pareto (n={result['n_pareto']}), Dominated (n={result['n_dominated']})")

                if "bootstrap_ci" in result:
                    ci = result["bootstrap_ci"]
                    sections.append(f"- **Mean difference:** {ci['mean_difference']:.2f}")
                    sections.append(f"- **95% CI:** [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
                    sections.append(f"- **Pareto mean:** {ci['group1_mean']:.2f}")
                    sections.append(f"- **Dominated mean:** {ci['group2_mean']:.2f}")

                sections.append("")

    # Section 5: Bootstrap Confidence Intervals (STAT-09)
    sections.append("## Bootstrap Confidence Intervals")
    sections.append("### Uncertainty Quantification for Key Metrics")
    sections.append("")
    sections.append("All confidence intervals computed using BCa method with 9,999 resamples.")
    sections.append("")

    for metric_name, results_dict in bootstrap.items():
        sections.append(f"#### {metric_name}")
        mean_res = results_dict["mean"]
        median_res = results_dict["median"]

        sections.append(f"**Mean:**")
        sections.append(f"- Point estimate: {mean_res['mean']:.2f}")
        sections.append(f"- 95% CI: [{mean_res['ci_low']:.2f}, {mean_res['ci_high']:.2f}]")
        sections.append(f"- Standard error: {mean_res['standard_error']:.4f}")
        sections.append(f"- Sample size: n={results_dict['n']}")
        sections.append("")

        sections.append(f"**Median:**")
        sections.append(f"- Point estimate: {median_res['median']:.2f}")
        sections.append(f"- 95% CI: [{median_res['ci_low']:.2f}, {median_res['ci_high']:.2f}]")
        sections.append(f"- Standard error: {median_res['standard_error']:.4f}")
        sections.append("")

    # Section 6: Significant Findings
    sections.append("## Significant Findings")
    sections.append("### Tests with p < 0.05 (after FDR correction)")
    sections.append("")

    significant_count = 0

    # Regional significant results
    for metric, result in kw_results.items():
        if "error" not in result and result["significant"]:
            significant_count += 1
            metric_display = metric.replace("_", " ").title()
            sections.append(f"**{metric_display} (Regional):**")
            sections.append(f"- Kruskal-Wallis test: H={result['statistic']:.2f}, p={result['p_value']:.4f}")
            sections.append(f"- {', '.join(result['groups'])} show significant differences")
            sections.append("")

    # Pairwise significant results
    for result in regional["pairwise"]:
        if result["significant_adj"]:
            significant_count += 1
            group1 = result["group1"]
            group2 = result["group2"]
            sections.append(f"**{group1} vs {group2}:**")
            sections.append(f"- Mann-Whitney U: {result['statistic']:.2f}, p_adj={result['p_adjusted']:.4f}")
            sections.append(f"- Effect size: r={result['effect_size']:.3f}")
            sections.append("")

    # Pareto significant results
    if pareto:
        for metric, result in pareto.items():
            if "error" not in result and result["significant"]:
                significant_count += 1
                metric_display = metric.replace("_", " ").title()
                sections.append(f"**{metric_display} (Pareto vs Dominated):**")
                sections.append(f"- Mann-Whitney U: {result['statistic']:.2f}, p={result['p_value']:.4f}")
                sections.append(f"- Effect size: r={result['effect_size']:.3f}")
                sections.append("")

    if significant_count == 0:
        sections.append("*No significant findings detected.*\n")

    # Section 7: Null Findings (STAT-11 - CRITICAL)
    sections.append("## Null Findings")
    sections.append("### Tests with p >= 0.05 (no significant difference)")
    sections.append("")
    sections.append("*This section reports all non-significant findings to avoid publication bias.*")
    sections.append("")

    null_count = 0

    # Regional null results
    for metric, result in kw_results.items():
        if "error" not in result and not result["significant"]:
            null_count += 1
            metric_display = metric.replace("_", " ").title()
            sections.append(f"**{metric_display} (Regional):**")
            sections.append(f"- Kruskal-Wallis test: H={result['statistic']:.2f}, p={result['p_value']:.4f}")
            sections.append(f"- No significant differences between {', '.join(result['groups'])}")
            sections.append("")

    # Pairwise null results
    for result in regional["pairwise"]:
        if not result["significant_adj"]:
            null_count += 1
            group1 = result["group1"]
            group2 = result["group2"]
            sections.append(f"**{group1} vs {group2}:**")
            sections.append(f"- Mann-Whitney U: {result['statistic']:.2f}, p_adj={result['p_adjusted']:.4f}")
            sections.append(f"- No significant difference detected")
            sections.append("")

    # Pareto null results
    if pareto:
        for metric, result in pareto.items():
            if "error" not in result and not result["significant"]:
                null_count += 1
                metric_display = metric.replace("_", " ").title()
                sections.append(f"**{metric_display} (Pareto vs Dominated):**")
                sections.append(f"- Mann-Whitney U: {result['statistic']:.2f}, p={result['p_value']:.4f}")
                sections.append(f"- No significant difference between groups")
                sections.append("")

    if null_count == 0:
        sections.append("*No null findings to report.*\n")

    # Section 8: Limitations and Recommendations
    sections.append("## Limitations and Recommendations")
    sections.append("")
    sections.append("### Limitations")
    sections.append("- Sample sizes vary by region (some regions have few providers)")
    sections.append("- Cross-sectional data (2026 snapshot, not longitudinal)")
    sections.append("- String columns (Speed, Latency) required casting to Float64")
    sections.append("- Some providers classified as 'Other' due to unknown region")
    sections.append("- Non-parametric tests have lower power than parametric alternatives")
    sections.append("")
    sections.append("### Recommendations")
    sections.append("- Use bootstrap CIs for uncertainty quantification in all analyses")
    sections.append("- Apply FDR correction when performing multiple tests")
    sections.append("- Report both significant and null findings to avoid publication bias")
    sections.append("- Consider log transformation for highly skewed variables")
    sections.append("- Increase sample size for regional comparisons if possible")
    sections.append("")

    # Write report
    report_content = "\n".join(sections)

    with open(output_path, 'w') as f:
        f.write(report_content)

    print(f"\nReport summary:")
    print(f"- Significant findings: {significant_count}")
    print(f"- Null findings: {null_count}")


if __name__ == "__main__":
    main()
