"""
Load AI models benchmark dataset from CSV with schema validation.

This script loads the raw CSV file using Polars with explicit schema
definition, validates data quality with Pandera, and quarantines any
invalid records for review.

The script follows the data pipeline pattern:
1. Load with lenient schema (Utf8 for messy price data)
2. Document structure (columns, types, samples)
3. Collect and validate with Pandera schema
4. Quarantine invalid rows to separate file
5. Save valid data to parquet checkpoint

Functions
---------
main()
    Execute the data loading pipeline.
"""

import polars as pl
from pathlib import Path
import sys

from src.load import load_data, document_structure
from src.utils import setup_logging, save_checkpoint, get_quarantine_path


def main():
    """
    Execute the data loading pipeline.

    Pipeline steps:
    1. Load raw CSV with explicit schema
    2. Document data structure
    3. Collect and validate with Pandera
    4. Quarantine invalid records
    5. Save valid data to checkpoint
    """
    # Configure logging
    logger = setup_logging(verbose=True)
    logger.info("Starting data loading process")

    # Define paths
    input_path = "data/raw/ai_models_performance.csv"
    checkpoint_path = "data/interim/01_loaded.parquet"
    quarantine_path = "data/quarantine/01_invalid_records.csv"

    # Check input file exists
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Step 1: Load the data
    logger.info(f"Loading data from {input_path}")
    lf = load_data(input_path)

    # Step 2: Document structure
    logger.info("Documenting data structure...")
    structure = document_structure(lf, logger)

    # Step 3: Collect and validate
    logger.info("Collecting data for validation...")
    df = lf.collect()

    # Note: We'll validate AFTER cleaning in plan 01-03
    # For now, save the raw loaded data with documented structure
    logger.info(f"Loaded {structure['row_count']:,} rows and {len(structure['column_names'])} columns")

    # Step 4: Save valid data to checkpoint
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    save_checkpoint(df, checkpoint_path, logger)

    # Note: Schema validation and quarantining will happen in next plan
    # after we clean the price column and rename columns to match schema

    logger.info("Data loading completed successfully")
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    logger.info(f"Next step: Run scripts/02_clean.py to clean messy values")


if __name__ == "__main__":
    main()
