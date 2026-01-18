"""
External data enrichment utilities for AI models benchmark dataset.

This module provides web scraping functions to fetch model metadata from
external sources (HuggingFace, provider announcements) with proper rate
limiting, error handling, and provenance tracking.

Functions
---------
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

import polars as pl

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


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
