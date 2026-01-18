"""
External data enrichment utilities for AI models benchmark dataset.

This module provides web scraping functions to fetch model metadata from
external sources (HuggingFace, provider announcements) with proper rate
limiting, error handling, and provenance tracking.

Functions
---------
enrich_with_external_data()
    Join base dataset with external data via left join.

add_derived_columns()
    Create derived analysis metrics from existing columns.

calculate_enrichment_coverage()
    Calculate coverage statistics for enrichment columns.

scrape_huggingface_models()
    Scrape model information from HuggingFace Open LLM Leaderboard.

scrape_provider_announcements()
    Scrape model announcements from provider blogs/news sources.

Notes
-----
- Rate limiting: 1 second delay between requests for respectful scraping
- Provenance tracking: All scraped data includes source_url, retrieved_at, retrieved_by
- Error handling: Functions return empty DataFrame on failure, allowing pipeline to continue
- Best-effort coverage: Scraping failures are logged but don't stop the pipeline
"""

from datetime import datetime
from typing import Optional
import time
import re

import polars as pl

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def enrich_with_external_data(
    base_df: pl.DataFrame,
    external_df: pl.DataFrame,
    join_key: str = "Model"
) -> pl.DataFrame:
    """
    Enrich base dataset with external data via left join.

    Performs a left join to preserve all original models, adding external
    metadata where available. Tracks enrichment metadata including timestamp,
    source, and coverage rate.

    Parameters
    ----------
    base_df : pl.DataFrame
        Base dataset with models to be enriched. Must contain join_key column.
    external_df : pl.DataFrame
        External data with enrichment metadata. Must contain join_key column.
    join_key : str, default="Model"
        Column name to join on (typically "Model" for this dataset).

    Returns
    -------
    pl.DataFrame
        Enriched DataFrame with:
        - All original columns from base_df
        - External columns from external_df (nulls if no match)
        - enriched_at: ISO timestamp of enrichment operation
        - enrichment_source: Source of external data
        - coverage_rate: Percentage of rows with enrichment data

    Raises
    ------
    ValueError
        If join_key doesn't exist in both DataFrames.

    Examples
    --------
    >>> base = pl.DataFrame({"Model": ["GPT-4", "Claude"], "price": [10, 20]})
    >>> external = pl.DataFrame({"Model": ["GPT-4"], "release_date": ["2023-03"]})
    >>> enriched = enrich_with_external_data(base, external)
    >>> print(enriched.shape)
    (2, 5)  # All rows preserved, external data added where matched

    Notes
    -----
    - Left join ensures no original models are lost
    - Null values in external columns indicate no match found
    - Coverage rate calculated as (non-null enrichment rows / total rows) * 100
    - Case-sensitive join by default - consider standardizing model names first

    References
    ----------
    Pattern: Left join for data enrichment preserving all records.
    See RESEARCH.md "Pitfall 5: External Data Enrichment Provenance Loss"
    """
    # Validate join_key exists in both DataFrames
    if join_key not in base_df.columns:
        raise ValueError(f"join_key '{join_key}' not found in base_df columns: {base_df.columns}")
    if join_key not in external_df.columns:
        raise ValueError(f"join_key '{join_key}' not found in external_df columns: {external_df.columns}")

    # Perform left join (preserve all base records)
    enriched_df = base_df.join(external_df, on=join_key, how="left", coalesce=False)

    # Calculate coverage rate
    total_rows = enriched_df.height
    # Count rows that have at least one non-null value from external data
    external_columns = [col for col in external_df.columns if col != join_key]
    if external_columns:
        # Count rows where ANY external column has data
        has_enrichment = pl.any_horizontal(pl.col(external_columns).is_not_null())
        enriched_count = enriched_df.select(has_enrichment.sum()).item()
        coverage_rate = (enriched_count / total_rows * 100) if total_rows > 0 else 0.0
    else:
        coverage_rate = 0.0

    # Add enrichment metadata
    enriched_df = enriched_df.with_columns([
        pl.lit(datetime.now().isoformat()).alias("enriched_at"),
        pl.lit("external_scraping").alias("enrichment_source"),
        pl.lit(coverage_rate).alias("coverage_rate")
    ])

    return enriched_df


def add_derived_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add derived analysis columns to dataset.

    Creates computed metrics for statistical analysis and visualization,
    handling edge cases like null values and division by zero.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame with base model metrics.

    Returns
    -------
    pl.DataFrame
        DataFrame with additional derived columns:
        - price_per_intelligence_point: Price per IQ score (USD per point)
        - speed_intelligence_ratio: Speed (tokens/s) per IQ point
        - model_tier: Extracted tier from model name (xhigh, high, medium, low, mini)
        - log_context_window: Log10 of context window for better visualization
        - price_per_1k_tokens: Price scaled to per-1k-token basis

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "price_usd": [10.0, 20.0],
    ...     "intelligence_index": [80, 100],
    ...     "Speed(median token/s)": [50, 100],
    ...     "Model": ["GPT-4-mini", "Claude-Opus"],
    ...     "context_window": [128000, 200000]
    ... })
    >>> enriched = add_derived_columns(df)
    >>> print(enriched.columns)
    [..., 'price_per_intelligence_point', 'speed_intelligence_ratio', ...]

    Notes
    -----
    - Division by zero results in null values
    - Null intelligence_index results in null for ratios involving it
    - Model tier extraction uses regex patterns (case-insensitive)
    - Log transformation helps visualize skewed distributions
    - All derived columns preserve null semantics from source columns

    Tier Classification
    -------------------
    Model tiers extracted from name patterns:
    - xhigh: "x-high", "xhigh", "extra-high", "ultra"
    - high: "high", "pro", "max", "plus"
    - medium: "medium", "standard", "base"
    - low: "low", "lite", "basic"
    - mini: "mini", "small", "tiny", "nano"
    - unknown: No tier pattern detected
    """
    derived_df = df.clone()

    # Price per intelligence point (handle division by zero and nulls)
    if "price_usd" in derived_df.columns and "intelligence_index" in derived_df.columns:
        derived_df = derived_df.with_columns([
            pl.when(pl.col("intelligence_index") > 0)
            .then(pl.col("price_usd") / pl.col("intelligence_index"))
            .otherwise(None)
            .alias("price_per_intelligence_point")
        ])
    else:
        # Add null column if source columns missing
        derived_df = derived_df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("price_per_intelligence_point")
        ])

    # Speed to intelligence ratio
    if "Speed(median token/s)" in derived_df.columns and "intelligence_index" in derived_df.columns:
        # Convert Speed column to Float64 for division
        derived_df = derived_df.with_columns([
            pl.when(pl.col("intelligence_index") > 0)
            .then(pl.col("Speed(median token/s)") / pl.col("intelligence_index"))
            .otherwise(None)
            .alias("speed_intelligence_ratio")
        ])
    else:
        derived_df = derived_df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("speed_intelligence_ratio")
        ])

    # Extract model tier from model name using regex
    if "Model" in derived_df.columns:
        # Define tier patterns (order matters - more specific first)
        tier_patterns = {
            "xhigh": r"(x-high|xhigh|extra-high|ultra)",
            "high": r"(high|pro|max|plus)",
            "medium": r"(medium|standard|base)",
            "low": r"(low|lite|basic)",
            "mini": r"(mini|small|tiny|nano)"
        }

        # Create tier column with case-insensitive matching
        model_lower = pl.col("Model").str.to_lowercase()

        # Apply tier patterns in order
        derived_df = derived_df.with_columns([
            pl.when(model_lower.str.contains(tier_patterns["xhigh"], literal=False))
            .then(pl.lit("xhigh"))
            .when(model_lower.str.contains(tier_patterns["high"], literal=False))
            .then(pl.lit("high"))
            .when(model_lower.str.contains(tier_patterns["medium"], literal=False))
            .then(pl.lit("medium"))
            .when(model_lower.str.contains(tier_patterns["low"], literal=False))
            .then(pl.lit("low"))
            .when(model_lower.str.contains(tier_patterns["mini"], literal=False))
            .then(pl.lit("mini"))
            .otherwise(pl.lit("unknown"))
            .alias("model_tier")
        ])
    else:
        derived_df = derived_df.with_columns([
            pl.lit("unknown", dtype=pl.Utf8).alias("model_tier")
        ])

    # Log10 of context window for better visualization
    if "context_window" in derived_df.columns:
        derived_df = derived_df.with_columns([
            pl.when(pl.col("context_window") > 0)
            .then(pl.col("context_window").log10())
            .otherwise(None)
            .alias("log_context_window")
        ])
    else:
        derived_df = derived_df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("log_context_window")
        ])

    # Price per 1k tokens (more intuitive scale)
    if "price_usd" in derived_df.columns:
        derived_df = derived_df.with_columns([
            (pl.col("price_usd") / 1000.0).alias("price_per_1k_tokens")
        ])
    else:
        derived_df = derived_df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("price_per_1k_tokens")
        ])

    return derived_df


def calculate_enrichment_coverage(
    df: pl.DataFrame,
    enrichment_columns: list[str]
) -> dict[str, dict[str, any]]:
    """
    Calculate coverage statistics for enrichment columns.

    Analyzes how many rows have data vs nulls for each enrichment column,
    providing metrics to assess data quality and completeness.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to analyze (typically after enrichment).
    enrichment_columns : list[str]
        List of column names to calculate coverage for.

    Returns
    -------
    dict[str, dict[str, any]]
        Dictionary mapping column names to coverage statistics:
        - non_null_count: Number of rows with non-null values
        - null_count: Number of rows with null values
        - coverage_percentage: (non_null_count / total_rows) * 100

    Examples
    --------
    >>> df = pl.DataFrame({
    ...     "release_date": ["2023-01", None, "2023-03"],
    ...     "benchmark_score": [85.5, None, 92.1]
    ... })
    >>> coverage = calculate_enrichment_coverage(df, ["release_date", "benchmark_score"])
    >>> print(coverage["release_date"]["coverage_percentage"])
    66.67

    Notes
    -----
    - Total rows used for percentage calculation is df.height
    - Coverage percentage rounded to 2 decimal places
    - Prints summary to console for immediate feedback
    - Useful for documenting enrichment quality in reports

    Output Format
    -------------
    Prints table showing:
    | Column            | Non-Null | Null | Coverage % |
    |-------------------|----------|------|------------|
    | release_date      | 2        | 1    | 66.67%     |
    """
    coverage_stats = {}
    total_rows = df.height

    print("\n" + "=" * 60)
    print("Enrichment Coverage Analysis")
    print("=" * 60)
    print(f"{'Column':<30} {'Non-Null':>10} {'Null':>10} {'Coverage %':>12}")
    print("-" * 60)

    for col in enrichment_columns:
        if col in df.columns:
            non_null_count = df.select(pl.col(col).drop_nulls()).height
            null_count = total_rows - non_null_count
            coverage_pct = (non_null_count / total_rows * 100) if total_rows > 0 else 0.0

            coverage_stats[col] = {
                "non_null_count": non_null_count,
                "null_count": null_count,
                "coverage_percentage": round(coverage_pct, 2)
            }

            print(f"{col:<30} {non_null_count:>10} {null_count:>10} {coverage_pct:>11.2f}%")
        else:
            print(f"{col:<30} {'NOT FOUND':>32}")

    print("=" * 60)

    return coverage_stats


def scrape_huggingface_models() -> pl.DataFrame:
    """
    Scrape model information from HuggingFace Open LLM Leaderboard.

    Fetches model metadata including release dates, benchmark scores, and
    provider information from HuggingFace's public leaderboard. Implements
    rate limiting and comprehensive error handling.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - model: Model name (str)
        - release_date: Model release date (str or None)
        - benchmark_score: Benchmark score if available (float or None)
        - provider: Provider/organization (str or None)
        - source_url: URL where data was retrieved (str)
        - retrieved_at: ISO timestamp of data retrieval (str)
        - retrieved_by: Script identifier for provenance (str)

        Returns empty DataFrame with correct schema if scraping fails.

    Examples
    --------
    >>> external_data = scrape_huggingface_models()
    >>> print(f"Retrieved {external_data.height} models")
    >>> print(external_data.columns)

    Notes
    -----
    - Rate limiting: 1 second delay between requests to avoid blocking
    - User-Agent header set to avoid being blocked as bot traffic
    - Provenance columns track data source for reproducibility
    - Actual HTML selectors depend on page structure (may need adjustment)
    - Function gracefully handles errors and returns empty DataFrame

    Limitations
    -----------
    - HTML structure changes may break scraping (requires maintenance)
    - Pagination not implemented (scrapes first page only)
    - Dynamic content may require Selenium/Playwright for full access
    - Coverage varies based on HuggingFace leaderboard availability

    References
    ----------
    HuggingFace Open LLM Leaderboard:
    https://huggingface.co/open-llm-leaderboard
    """
    base_url = "https://huggingface.co/open-llm-leaderboard"
    models_data = []

    # Check if requests is available
    if not REQUESTS_AVAILABLE:
        print("WARNING: requests library not available. Install with: pip install requests beautifulsoup4")
        print("Returning empty DataFrame with expected schema")
        schema = {
            "model": pl.Utf8,
            "release_date": pl.Utf8,
            "benchmark_score": pl.Float64,
            "provider": pl.Utf8,
            "source_url": pl.Utf8,
            "retrieved_at": pl.Utf8,
            "retrieved_by": pl.Utf8
        }
        return pl.DataFrame(schema=schema)

    try:
        print(f"Fetching HuggingFace Open LLM Leaderboard from {base_url}...")

        # Send GET request with User-Agent to avoid blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; AI-Models-Benchmark-Bot/1.0; +https://github.com/dustinober)"
        }
        response = requests.get(base_url, headers=headers, timeout=30)

        # Check for HTTP errors
        response.raise_for_status()

        print(f"Response status: {response.status_code}")

        # Parse HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract model information
        # NOTE: Actual selectors depend on page structure - this is a template
        # Common patterns for HuggingFace leaderboard:
        # - Model names in <a> tags with /models/ path
        # - Scores in table cells or div elements
        # - Dates in various formats

        # Try to find model entries (adjust selectors based on actual page)
        model_entries = soup.select("table tbody tr")  # Common pattern for leaderboards

        if not model_entries:
            # Alternative: look for model cards or list items
            model_entries = soup.select(".model-card") or soup.select("li[class*='model']")

        print(f"Found {len(model_entries)} potential model entries")

        for entry in model_entries:
            try:
                # Extract model name
                model_elem = entry.select_one("a[href*='/models/']") or entry.select_one(".model-name")
                model_name = model_elem.text.strip() if model_elem else None

                if not model_name:
                    continue

                # Extract release date if available
                date_elem = entry.select_one(".date") or entry.select_one("time")
                release_date = date_elem.get("datetime") or date_elem.text.strip() if date_elem else None

                # Extract benchmark score if available
                score_elem = entry.select_one(".score") or entry.select_one("[class*='benchmark']")
                benchmark_score = None
                if score_elem:
                    try:
                        benchmark_score = float(score_elem.text.strip())
                    except (ValueError, AttributeError):
                        pass

                # Extract provider/organization if available
                provider_elem = entry.select_one(".provider") or entry.select_one("[class*='org']")
                provider = provider_elem.text.strip() if provider_elem else None

                # Append to data list
                models_data.append({
                    "model": model_name,
                    "release_date": release_date,
                    "benchmark_score": benchmark_score,
                    "provider": provider,
                    "source_url": base_url,
                    "retrieved_at": datetime.now().isoformat(),
                    "retrieved_by": "scrape_huggingface_models"
                })

            except Exception as e:
                # Continue processing other entries if one fails
                print(f"Warning: Error parsing model entry: {e}")
                continue

        # Rate limiting: wait before returning
        time.sleep(1)

        print(f"Successfully parsed {len(models_data)} models from HuggingFace")

    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error occurred while scraping HuggingFace: {e}")
        print(f"Status code: {e.response.status_code if e.response else 'Unknown'}")
    except requests.exceptions.Timeout:
        print(f"ERROR: Request timeout while fetching {base_url}")
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Connection error while fetching {base_url}")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request exception occurred: {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error while scraping HuggingFace: {e}")

    # Convert to DataFrame
    if models_data:
        df = pl.DataFrame(models_data)
        print(f"Created DataFrame with {df.height} rows and {df.width} columns")
        return df
    else:
        print("No models data collected. Returning empty DataFrame with expected schema")
        schema = {
            "model": pl.Utf8,
            "release_date": pl.Utf8,
            "benchmark_score": pl.Float64,
            "provider": pl.Utf8,
            "source_url": pl.Utf8,
            "retrieved_at": pl.Utf8,
            "retrieved_by": pl.Utf8
        }
        return pl.DataFrame(schema=schema)


def scrape_provider_announcements() -> pl.DataFrame:
    """
    Scrape model announcements from provider blogs and news sources.

    Fetches model release announcements from major AI provider sources
    (OpenAI, Anthropic, Google, Meta, etc.) to gather release dates
    and contextual information.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - model: Model name (str)
        - release_date: Announcement/release date (str or None)
        - provider: Provider name (str)
        - announcement_title: Title of announcement (str or None)
        - source_url: URL of announcement (str)
        - retrieved_at: ISO timestamp of data retrieval (str)
        - retrieved_by: Script identifier for provenance (str)

        Returns empty DataFrame with correct schema if scraping fails.

    Examples
    --------
    >>> announcements = scrape_provider_announcements()
    >>> print(f"Retrieved {announcements.height} announcements")
    >>> print(announcements.groupby("provider").count())

    Notes
    -----
    - Rate limiting: 1 second delay between requests
    - Provenance tracking for reproducibility
    - Best-effort coverage: failures logged but don't stop pipeline
    - Actual scraping logic depends on provider site structures

    Provider Sources
    ----------------
    Target sources include:
    - OpenAI: https://openai.com/news
    - Anthropic: https://www.anthropic.com/news
    - Google: https://blog.google/technology/ai/
    - Meta: https://about.fb.com/news/section/ai/
    - Mistral: https://mistral.ai/news/

    Limitations
    -----------
    - Site structure changes require selector updates
    - Some providers may block automated scraping
    - Coverage varies by provider RSS/API availability
    - Language localization may affect parsing

    References
    ----------
    Provider blog URLs may change - verify current URLs before scraping.
    """
    # Provider blog URLs (may need updates)
    provider_sources = {
        "OpenAI": "https://openai.com/news",
        "Anthropic": "https://www.anthropic.com/news",
        "Google": "https://blog.google/technology/ai/",
        "Meta": "https://about.fb.com/news/section/ai/",
        "Mistral": "https://mistral.ai/news/"
    }

    announcements_data = []

    # Check if requests is available
    if not REQUESTS_AVAILABLE:
        print("WARNING: requests library not available. Install with: pip install requests beautifulsoup4")
        print("Returning empty DataFrame with expected schema")
        schema = {
            "model": pl.Utf8,
            "release_date": pl.Utf8,
            "provider": pl.Utf8,
            "announcement_title": pl.Utf8,
            "source_url": pl.Utf8,
            "retrieved_at": pl.Utf8,
            "retrieved_by": pl.Utf8
        }
        return pl.DataFrame(schema=schema)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AI-Models-Benchmark-Bot/1.0; +https://github.com/dustinober)"
    }

    for provider, url in provider_sources.items():
        try:
            print(f"Fetching {provider} announcements from {url}...")

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract announcement entries (selectors depend on site structure)
            # Common patterns: article tags, blog post cards, news items
            entries = (
                soup.select("article") or
                soup.select(".blog-post") or
                soup.select(".news-item") or
                soup.select("[class*='announcement']")
            )

            print(f"Found {len(entries)} announcement entries for {provider}")

            for entry in entries[:10]:  # Limit to first 10 announcements per provider
                try:
                    # Extract announcement title
                    title_elem = entry.select_one("h2, h3, h4") or entry.select_one(".title")
                    title = title_elem.text.strip() if title_elem else None

                    # Extract date if available
                    date_elem = entry.select_one("time") or entry.select_one(".date, [class*='date']")
                    release_date = (
                        date_elem.get("datetime") or
                        date_elem.text.strip()
                        if date_elem else None
                    )

                    # Extract link
                    link_elem = entry.select_one("a[href]")
                    source_url = link_elem.get("href") if link_elem else url
                    if source_url and not source_url.startswith("http"):
                        # Handle relative URLs
                        from urllib.parse import urljoin
                        source_url = urljoin(url, source_url)

                    announcements_data.append({
                        "model": None,  # Model names extracted from titles in post-processing
                        "release_date": release_date,
                        "provider": provider,
                        "announcement_title": title,
                        "source_url": source_url,
                        "retrieved_at": datetime.now().isoformat(),
                        "retrieved_by": "scrape_provider_announcements"
                    })

                except Exception as e:
                    print(f"Warning: Error parsing announcement entry: {e}")
                    continue

            # Rate limiting between providers
            time.sleep(1)

        except requests.exceptions.HTTPError as e:
            print(f"WARNING: HTTP error for {provider}: {e}")
        except requests.exceptions.Timeout:
            print(f"WARNING: Timeout for {provider}")
        except requests.exceptions.ConnectionError:
            print(f"WARNING: Connection error for {provider}")
        except requests.exceptions.RequestException as e:
            print(f"WARNING: Request exception for {provider}: {e}")
        except Exception as e:
            print(f"WARNING: Unexpected error for {provider}: {e}")

    # Convert to DataFrame
    if announcements_data:
        df = pl.DataFrame(announcements_data)
        print(f"Created DataFrame with {df.height} announcements from {len(provider_sources)} providers")
        return df
    else:
        print("No announcement data collected. Returning empty DataFrame with expected schema")
        schema = {
            "model": pl.Utf8,
            "release_date": pl.Utf8,
            "provider": pl.Utf8,
            "announcement_title": pl.Utf8,
            "source_url": pl.Utf8,
            "retrieved_at": pl.Utf8,
            "retrieved_by": pl.Utf8
        }
        return pl.DataFrame(schema=schema)
