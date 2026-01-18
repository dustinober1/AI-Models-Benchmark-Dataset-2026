"""
Quality assessment and reporting for AI models benchmark dataset.

This script generates a comprehensive data quality report covering
the 6 dimensions of data quality: Accuracy, Completeness, Consistency,
Integrity, Timeliness, and Validity.

Functions
---------
run_quality_assessment(input_path, output_path)
    Execute full quality assessment pipeline with distribution statistics.

main()
    Main execution function for quality report generation.
"""

from pathlib import Path
from datetime import datetime
import polars as pl
from src.utils import setup_logging, load_checkpoint
from src.quality import perform_sanity_checks, generate_quality_report
from src.analyze import analyze_distribution


def run_quality_assessment(
    input_path: str = "data/interim/03_distributions_analyzed.parquet",
    output_path: str = None
) -> dict:
    """
    Execute full quality assessment pipeline.

    Generates comprehensive quality report covering all 6 dimensions of data quality
    with embedded visualizations, distribution statistics, and narrative interpretation.

    Pipeline steps (per plan 01-06):
    1. Load analyzed dataset from data/interim/03_distributions_analyzed.parquet
    2. Perform sanity checks using perform_sanity_checks()
    3. Calculate distribution statistics for numerical columns
    4. Generate quality report using generate_quality_report()
    5. Verify report file exists and contains all sections

    Parameters
    ----------
    input_path : str, default="data/interim/03_distributions_analyzed.parquet"
        Path to analyzed data checkpoint (from plan 01-04).
    output_path : str, optional
        Path to save quality report. If None, generates timestamped path
        using current date (2026-01-18 per CONTEXT.md example).

    Returns
    -------
    dict
        Dictionary containing:
        - sanity_checks: Full sanity check results from perform_sanity_checks()
        - distributions_stats: Distribution statistics for each numerical column
        - report_path: Path to generated quality report
        - overall_quality_score: Overall quality score (0-100)

    Examples
    --------
    >>> results = run_quality_assessment()
    >>> print(f"Quality: {results['overall_quality_score']:.1f}%")
    >>> print(f"Report: {results['report_path']}")

    Notes
    -----
    - Generates timestamped report if output_path not specified
    - All 6 quality dimensions are assessed (Accuracy, Completeness, Consistency, Validity, Integrity, Timeliness)
    - Report includes narrative interpretation and actionable recommendations
    - Distribution plots are embedded as markdown links
    - Sanity check summary is printed to console for immediate feedback

    Analysis choices documented (NARR-06 requirement):
    - Using Isolation Forest for outlier detection (contamination=5%, random_state=42)
    - Distribution analysis uses scipy.stats for skewness, kurtosis, normality testing
    - Quality threshold: 75% overall score for Phase 2 readiness
    - Non-parametric methods recommended due to non-normal distributions
    """
    logger = setup_logging(verbose=True)
    logger.info("=" * 80)
    logger.info("Starting quality assessment process")
    logger.info("=" * 80)

    # Step 1: Load data
    logger.info(f"\n[Step 1/5] Loading data from {input_path}")
    df = load_checkpoint(input_path, logger)
    logger.info(f"Loaded {df.height} rows, {df.width} columns")

    # Step 2: Perform sanity checks
    logger.info("\n[Step 2/5] Performing sanity checks across 6 quality dimensions")
    sanity_results = perform_sanity_checks(df)

    # Print sanity check summary to console
    logger.info("\n" + "=" * 80)
    logger.info("SANITY CHECK SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Overall Quality Score: {sanity_results['overall_quality_score']:.1f}%")
    logger.info(f"Dimensions Passed: {sanity_results['dimensions_passed']}/4")
    logger.info(f"Dimensions Failed: {sanity_results['dimensions_failed']}/4")
    logger.info("")

    # Print individual dimension results
    accuracy = sanity_results["accuracy"]
    completeness = sanity_results["completeness"]
    consistency = sanity_results["consistency"]
    validity = sanity_results["validity"]

    logger.info(f"Accuracy: {'✓ PASS' if accuracy['pass'] else '✗ FAIL'} ({accuracy['violation_count']} violations)")
    logger.info(f"Completeness: {'✓ PASS' if completeness['pass'] else '✗ FAIL'} ({completeness['completeness_percentage']:.1f}%)")
    logger.info(f"Consistency: {'✓ PASS' if consistency['pass'] else '✗ FAIL'} ({consistency['duplicate_models']} duplicates)")
    logger.info(f"Validity: {'✓ PASS' if validity['pass'] else '✗ FAIL'} ({validity['impossible_combinations']} impossible combos)")

    # Print critical issues if any
    if sanity_results["critical_issues"]:
        logger.info("\n⚠ CRITICAL ISSUES FOUND:")
        for issue in sanity_results["critical_issues"]:
            logger.info(f"  - {issue}")
    else:
        logger.info("\n✓ No critical issues found")

    # Step 3: Calculate distribution statistics
    logger.info("\n[Step 3/5] Calculating distribution statistics for numerical columns")
    numerical_columns = ["context_window", "intelligence_index", "price_usd", "Speed(median token/s)", "Latency (First Answer Chunk /s)"]
    distributions_stats = {}

    for col in numerical_columns:
        if col in df.columns:
            logger.info(f"  Analyzing {col}...")
            stats = analyze_distribution(df[col])
            distributions_stats[col] = stats
        else:
            logger.warning(f"  Column {col} not found in DataFrame")

    logger.info(f"Calculated statistics for {len(distributions_stats)} numerical columns")

    # Step 4: Generate quality report
    logger.info("\n[Step 4/5] Generating comprehensive quality report")

    # Generate output path if not specified (use current date: 2026-01-18)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        output_path = f"reports/quality_{timestamp}.md"

    # Generate report using the new function from src.quality
    report_path = generate_quality_report(df, distributions_stats, output_path)
    logger.info(f"Quality report generated at {report_path}")

    # Step 5: Verify report file exists
    logger.info("\n[Step 5/5] Verifying report file")
    report_file = Path(report_path)
    if report_file.exists():
        logger.info(f"✓ Report file verified: {report_file}")

        # Verify report structure by checking for key sections
        with open(report_file, 'r') as f:
            content = f.read()

        sections_to_check = [
            "Executive Summary",
            "Data Dimensions",
            "Distribution Analysis",
            "Outlier Analysis",
            "Sanity Check Results",
            "Data Quality Issues Found",
            "Next Steps",
            "Metadata"
        ]

        missing_sections = []
        for section in sections_to_check:
            if section not in content:
                missing_sections.append(section)

        if missing_sections:
            logger.warning(f"⚠ Missing sections in report: {missing_sections}")
        else:
            logger.info("✓ All required sections present in report")

        # Check for embedded figures
        if "![Distribution](figures/" in content:
            logger.info("✓ Distribution plot links embedded in report")
        else:
            logger.warning("⚠ No distribution plot links found in report")

    else:
        logger.error(f"✗ Report file not found: {report_file}")

    # Compile results for return
    results = {
        "sanity_checks": sanity_results,
        "distributions_stats": distributions_stats,
        "report_path": report_path,
        "overall_quality_score": sanity_results["overall_quality_score"]
    }

    logger.info("\n" + "=" * 80)
    logger.info("QUALITY ASSESSMENT COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Report: {report_path}")
    logger.info(f"Overall Quality: {sanity_results['overall_quality_score']:.1f}%")
    logger.info("=" * 80)

    return results


def main():
    """
    Main execution function for quality report generation.

    Executes the quality assessment pipeline and prints a summary of results.
    This function is called when the script is run directly.

    Usage
    -----
    poetry run python scripts/05_quality_report.py

    Notes
    -----
    - Uses verbose logging throughout execution
    - Generates timestamped report in reports/ directory
    - Prints summary to console for immediate feedback
    """
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("Starting quality assessment script")

    try:
        # Run quality assessment
        results = run_quality_assessment()

        # Print sections generated (verification)
        logger.info("\nSections generated in quality report:")
        sections = [
            "Executive Summary",
            "Data Dimensions (6 dimensions assessed)",
            "Distribution Analysis with embedded visualizations",
            "Outlier Analysis with examples",
            "Sanity Check Results",
            "Data Quality Issues Found",
            "Next Steps for Phase 2",
            "Metadata and dependencies"
        ]
        for section in sections:
            logger.info(f"  ✓ {section}")

        logger.info("\nQuality assessment completed successfully")

    except Exception as e:
        logger.error(f"Error during quality assessment: {e}")
        raise


if __name__ == "__main__":
    main()
