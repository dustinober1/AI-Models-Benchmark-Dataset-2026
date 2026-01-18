"""
Schema validation for AI models benchmark dataset using Pandera.

This module defines strict validation rules for the dataset to ensure
data quality and catch issues early in the pipeline.

The schema enforces:
- Correct data types for all columns
- Valid value ranges (e.g., Intelligence Index 0-100)
- Reasonable constraints (e.g., prices >= 0)
- Custom dataframe-level validation checks
"""

import pandera.polars as pa
import polars as pl
from pandera import Column
from pandera.typing import Series


class AIModelsSchema(pa.DataFrameModel):
    """
    Schema validation for AI models benchmark dataset.

    This schema enforces type safety and business logic constraints
    on the AI models dataset to catch data quality issues early.

    Attributes
    ----------
    model : str
        Model name/identifier (e.g., "GPT-5.2 (xhigh)", "Claude Opus 4.5")
    context_window : int
        Context window size in tokens. Valid range: 0 to 2,000,000.
        The upper bound allows for future-proofing as models may exceed
        current maximums (currently ~1M tokens).
    creator : str
        Model creator/organization (e.g., "OpenAI", "Anthropic", "Google")
    intelligence_index : int
        IQ/performance score (0-100 scale). This is a normalized score
        where higher values indicate better performance.
    price_usd : float
        Price per 1M tokens in USD. Must be non-negative.
        Note: The raw CSV column is named "Price (Blended USD/1M Tokens)"
        and will be renamed during the cleaning phase.
    speed : float
        Median tokens per second generated. Must be non-negative.
        Higher values indicate faster generation.
    latency : float
        First chunk latency in seconds. Must be non-negative.
        Lower values indicate faster time-to-first-token.

    Notes
    -----
    Schema validation runs AFTER data loading and cleaning:
    1. Load with lenient schema (all Utf8 for messy data)
    2. Clean data (remove $ signs, handle missing values)
    3. Validate with strict Pandera schema
    4. Quarantine records that fail validation
    """

    # Column definitions with type constraints and validation rules
    model: str = pa.Field(description="Model name/identifier")
    context_window: int = pa.Field(
        ge=0,
        le=2_000_000,
        description="Context window size in tokens (0 to 2M)"
    )
    creator: str = pa.Field(description="Model creator/organization")
    intelligence_index: int = pa.Field(
        ge=0,
        le=100,
        description="IQ score (0-100 scale)"
    )
    price_usd: float = pa.Field(
        ge=0,
        description="Price per 1M tokens in USD"
    )
    speed: float = pa.Field(
        ge=0,
        description="Median tokens per second"
    )
    latency: float = pa.Field(
        ge=0,
        description="First chunk latency in seconds"
    )

    @pa.dataframe_check
    def check_context_window_range(cls, df: pa.PolarsData) -> pl.LazyFrame:
        """
        Validate that context window values are realistic.

        This custom check ensures context window sizes are within
        reasonable bounds for current LLM technology (0 to 2M tokens).
        The upper bound is set conservatively to allow for future models
        that may exceed current maximums.

        Parameters
        ----------
        df : pa.PolarsData
            The dataframe to validate.

        Returns
        -------
        pl.LazyFrame
            LazyFrame with boolean result for each row.

        Notes
        -----
        Current state-of-the-art context windows (2026):
        - Gemini 3: 1M tokens
        - Claude 3.5 Sonnet: 200K tokens
        - GPT-5: 400K tokens

        The 2M upper bound allows for 5x growth in context window sizes.
        """
        return df.lazyframe.select(
            pl.col("context_window").le(2_000_000)
        )


def validate_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Validate DataFrame against AIModelsSchema.

    Converts the schema class to a validation object and runs
    comprehensive validation checks on the provided DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to validate. Must have columns matching AIModelsSchema:
        model, context_window, creator, intelligence_index, price_usd,
        speed, latency.

    Returns
    -------
    pl.DataFrame
        Validated DataFrame. Returns the same DataFrame if validation passes.

    Raises
    ------
    pa.errors.SchemaError
        If validation fails. The error message includes detailed information
        about which columns/rows failed validation and why.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({
    ...     "model": ["GPT-5.2", "Claude Opus 4.5"],
    ...     "context_window": [400_000, 200_000],
    ...     "creator": ["OpenAI", "Anthropic"],
    ...     "intelligence_index": [51, 49],
    ...     "price_usd": [4.81, 10.00],
    ...     "speed": [100.0, 79.0],
    ...     "latency": [44.29, 1.7]
    ... })
    >>> validated_df = validate_data(df)  # Returns df if valid

    Notes
    -----
    This function should be called AFTER data cleaning:
    1. Load data from CSV (lenient schema)
    2. Clean messy values (remove $, handle nulls)
    3. Rename columns to match schema (price_usd, etc.)
    4. Call validate_data() to enforce schema rules
    5. Quarantine any records that fail validation
    """
    # Convert schema class to validation object
    schema = AIModelsSchema.to_schema()

    # Run validation
    # This will raise SchemaError with detailed message if validation fails
    validated = schema.validate(df)

    return validated
