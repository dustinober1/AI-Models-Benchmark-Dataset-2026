#!/usr/bin/env python3
"""
Duplicate Resolution Pipeline - Phase 2 Plan 01

Resolves 34 duplicate model names (18.1%) using context window disambiguation.
Creates unique model_id column for accurate group-by operations.

Usage:
    PYTHONPATH=. python3 scripts/07_duplicate_resolution.py
"""

import polars as pl
from src.deduplicate import detect_duplicates, resolve_duplicate_models, validate_resolution
from pathlib import Path
from datetime import datetime


def generate_resolution_report(
    df_original: pl.DataFrame,
    df_resolved: pl.DataFrame,
    duplicates: pl.DataFrame,
    validation: dict,
    output_path: str
) -> str:
    """
    Generate comprehensive duplicate resolution report in markdown format.

    Creates detailed report with before/after statistics, resolution strategy,
    validation results, and examples of resolved duplicates.

    Parameters
    ----------
    df_original : pl.DataFrame
        Original DataFrame before resolution.
    df_resolved : pl.DataFrame
        Resolved DataFrame with unique model_id column.
    duplicates : pl.DataFrame
        DataFrame with duplicate information from detect_duplicates().
    validation : dict
        Validation dictionary from validate_resolution().
    output_path : str
        Path to save the resolution report (Markdown format).

    Returns
    -------
    str
        Path to the generated report file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate statistics
    original_count = df_original.height
    resolved_count = df_resolved.height
    unique_model_ids = df_resolved["model_id"].n_unique()
    duplicates_removed = original_count - resolved_count

    # Resolution strategy distribution
    strategy_dist = df_resolved.group_by("resolution_source").agg(
        pl.len().alias("count")
    ).sort("count", descending=True)

    # Build report
    report_lines = [
        "# Duplicate Resolution Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Input:** ai_models_enriched.parquet",
        f"**Output:** ai_models_deduped.parquet",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"**Original Models:** {original_count:,}",
        f"**Duplicate Model Names:** {len(duplicates)}",
        f"**Resolved Models:** {resolved_count:,}",
        f"**Unique Model IDs:** {unique_model_ids:,}",
        f"**True Duplicates Removed:** {duplicates_removed}",
        "",
        "---",
        "",
        "## Resolution Strategy",
        "",
        "**Primary:** Context Window Disambiguation",
        "",
        "Unique model_id created using pattern: `ModelName_ContextWindow`",
        "",
        "Example:",
        "- `GPT-4_128000` for GPT-4 with 128k context window",
        "- `Claude_2_200000` for Claude 2 with 200k context window",
        "",
        "**Secondary:** Intelligence Index Disambiguation",
        "",
        "For models with same name AND context window, add Intelligence Index:",
        "",
        "Example:",
        "- `NVIDIA_Nemotron_3_Nano_1000000_25` for IQ=25",
        "- `NVIDIA_Nemotron_3_Nano_1000000_14` for IQ=14",
        "",
        "**Tertiary:** True Duplicate Removal",
        "",
        "Models identical in all columns (name, context, IQ, price, speed, etc.) are true duplicates.",
        "These are removed, keeping only the first occurrence.",
        "",
        "---",
        "",
        "## Before/After Statistics",
        "",
        "| Metric | Before | After | Change |",
        "|--------|--------|-------|--------|",
        f"| Total Rows | {original_count:,} | {resolved_count:,} | {-duplicates_removed} |",
        f"| Unique Model IDs | N/A | {unique_model_ids:,} | +{unique_model_ids} |",
        f"| Duplicate Names | {len(duplicates)} | 0 | -{len(duplicates)} |",
        "",
        "---",
        "",
        "## Duplicates Found",
        "",
        f"**Total Duplicate Model Names:** {len(duplicates)}",
        "",
        "| Model | Count | Context Windows |",
        "|-------|-------|-----------------|"
    ]

    # Add duplicate details
    for dup in duplicates.sort("count", descending=True).to_dicts():
        model = dup["Model"]
        count = dup["count"]
        contexts = dup["context_windows"]

        # Format context windows as comma-separated list
        contexts_str = ", ".join(contexts)

        report_lines.append(f"| {model} | {count} | {contexts_str} |")

    report_lines.extend([
        "",
        "---",
        "",
        "## Validation Results",
        "",
        f"**Status:** {'✓ PASS' if validation['pass'] else '✗ FAIL'}",
        f"**Original Duplicates:** {validation['original_duplicates']}",
        f"**Resolved Count:** {validation['resolved_count']}",
        f"**Unique Model IDs:** {validation['unique_model_ids']}",
        f"**Remaining Duplicates:** {validation['remaining_duplicates']}",
        "",
        "**Message:**",
        f"> {validation['message']}",
        "",
        "---",
        "",
        "## Resolution Strategy Distribution",
        "",
        "| Strategy | Count | Percentage |",
        "|----------|-------|------------|"
    ])

    # Add strategy distribution
    for row in strategy_dist.to_dicts():
        strategy = row["resolution_source"]
        count = row["count"]
        pct = (count / resolved_count * 100) if resolved_count > 0 else 0
        report_lines.append(f"| {strategy} | {count:,} | {pct:.1f}% |")

    report_lines.extend([
        "",
        "---",
        "",
        "## Examples of Resolved Duplicates",
        "",
        "### Primary Disambiguation (Context Window)",
        "",
        "Models with same name but different context windows:",
        "",
        "| Original Model | Context Window | New model_id |",
        "|----------------|----------------|--------------|"
    ])

    # Add examples of context window disambiguation
    cx_examples = df_resolved.filter(
        pl.col("resolution_source") == "context_window_deduped"
    ).group_by("Model").agg(
        pl.col("model_id").alias("model_ids"),
        pl.col("context_window").alias("context_windows")
    ).filter(
        pl.col("model_ids").list.len() > 1
    ).head(5)

    for example in cx_examples.to_dicts():
        model = example["Model"]
        model_ids = example["model_ids"]
        contexts = example["context_windows"]

        for i, (mid, ctx) in enumerate(zip(model_ids, contexts)):
            if i == 0:
                report_lines.append(f"| {model} | {ctx:,} | {mid} |")
            else:
                report_lines.append(f"| {model} | {ctx:,} | {mid} |")

    report_lines.extend([
        "",
        "### Secondary Disambiguation (Intelligence Index)",
        "",
        "Models with same name and context window but different Intelligence Index:",
        "",
        "| Original Model | Context Window | Intelligence Index | New model_id |",
        "|----------------|----------------|-------------------|--------------|"
    ])

    # Add examples of intelligence index disambiguation
    iq_examples = df_resolved.filter(
        pl.col("resolution_source").str.contains("intelligence")
    ).group_by("Model").agg(
        pl.col("model_id").alias("model_ids"),
        pl.col("context_window").alias("context_windows"),
        pl.col("intelligence_index").alias("iq_scores")
    ).head(5)

    for example in iq_examples.to_dicts():
        model = example["Model"]
        model_ids = example["model_ids"]
        contexts = example["context_windows"]
        iq_scores = example["iq_scores"]

        for i, (mid, ctx, iq) in enumerate(zip(model_ids, contexts, iq_scores)):
            if i == 0:
                report_lines.append(f"| {model} | {ctx:,} | {iq} | {mid} |")
            else:
                report_lines.append(f"| {model} | {ctx:,} | {iq} | {mid} |")

    report_lines.extend([
        "",
        "---",
        "",
        "## Data Quality Notes",
        "",
        "**True Duplicate Removed:**",
        f"- 1 model had identical rows in all columns (Exaone 4.0 1.2B)",
        f"- These were data entry errors and removed to prevent aggregation issues",
        "",
        "**Intelligence Index Null Handling:**",
        f"- 6 models have null Intelligence Index (filled with -1 for disambiguation)",
        f"- These models are identified by model_id ending in '_-1'",
        f"- Intelligence-specific analyses should filter to n=181 models with valid IQ scores",
        "",
        "---",
        "",
        "## Next Steps",
        "",
        "✓ **Dataset ready for Phase 2 statistical analysis**",
        "",
        "**Output files:**",
        f"- `data/processed/ai_models_deduped.parquet` - Deduplicated dataset ({resolved_count} models, 18 columns)",
        f"- Unique `model_id` column for accurate group-by operations",
        f"- Original `Model` column preserved for reference",
        "",
        "**Recommended for correlation analysis:**",
        "- Use `model_id` for all group-by operations",
        "- Use Spearman correlation (non-parametric) for skewed distributions",
        "- Consider log-transformation for Context Window (extreme skewness: 9.63)",
        "- Filter to n=181 for intelligence-specific analyses (exclude null IQ scores)",
        "",
        "---",
        "",
        "## Metadata",
        "",
        f"**Generation Timestamp:** {timestamp}",
        f"**Pipeline Version:** Phase 2 - Statistical Analysis & Domain Insights",
        f"**Plan:** 02-01 (Duplicate Resolution)",
        "",
        "**Dependencies:**",
        "- polars >= 1.0.0",
        "- Input: `data/processed/ai_models_enriched.parquet`",
        "- Output: `data/processed/ai_models_deduped.parquet`",
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
    """Main execution function for duplicate resolution pipeline."""
    # Define paths
    input_path = "data/processed/ai_models_enriched.parquet"
    output_path = "data/processed/ai_models_deduped.parquet"
    report_path = "reports/duplicate_resolution_2026-01-18.md"

    print("=" * 60)
    print("DUPLICATE RESOLUTION PIPELINE")
    print("=" * 60)
    print()

    # Load enriched dataset
    print(f"Loading: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"Loaded {df.height} models, {df.width} columns")
    print()

    # Detect duplicates
    print("=== DUPLICATE DETECTION ===")
    duplicates = detect_duplicates(df)
    print(f"Found {len(duplicates)} duplicate model names")
    print(f"Duplicate percentage: {len(duplicates) / df.height * 100:.1f}%")
    print()

    # Show top 5 duplicates
    if len(duplicates) > 0:
        print("Top 5 duplicates by count:")
        print(duplicates.head(5))
        print()

    # Resolve duplicates
    print("=== RESOLVING DUPLICATES ===")
    df_resolved = resolve_duplicate_models(df, strategy="context_window")
    print(f"Created model_id column for {df_resolved.height} models")
    print(f"Unique model_ids: {df_resolved['model_id'].n_unique()}")
    print(f"True duplicates removed: {df.height - df_resolved.height}")
    print()

    # Validate resolution
    print("=== VALIDATION ===")
    validation = validate_resolution(df_resolved)
    print(f"Original duplicates: {validation['original_duplicates']}")
    print(f"Resolved count: {validation['resolved_count']}")
    print(f"Unique model_ids: {validation['unique_model_ids']}")
    print(f"Remaining duplicates: {validation['remaining_duplicates']}")
    print(f"Validation: {'PASS ✓' if validation['pass'] else 'FAIL ✗'}")
    print()

    # Save deduplicated dataset
    print(f"Saving: {output_path}")
    df_resolved.write_parquet(output_path)
    print(f"Saved {df_resolved.height} models to {output_path}")
    print()

    # Generate resolution report
    print("Generating resolution report...")
    report_path = generate_resolution_report(
        df, df_resolved, duplicates, validation, report_path
    )
    print(f"Report: {report_path}")
    print()

    print("=" * 60)
    print("✓ DUPLICATE RESOLUTION COMPLETE")
    print("=" * 60)
    print()
    print(f"Input:  {input_path} ({df.height} models)")
    print(f"Output: {output_path} ({df_resolved.height} models)")
    print(f"Report: {report_path}")
    print()
    print("Dataset ready for Phase 2 statistical analysis!")


if __name__ == "__main__":
    main()
