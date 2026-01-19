"""
Provider clustering utilities for AI models benchmark dataset.

This module provides functions for clustering AI model providers by performance
characteristics using KMeans clustering algorithm. Identifies market segments
and competitive positioning across regions.

Functions
---------
aggregate_by_provider(df: pl.DataFrame, features: list[str]) -> pl.DataFrame
    Aggregate model-level data to provider-level with mean features and counts.

find_optimal_clusters(X_scaled: np.ndarray, max_k: int = 10) -> dict
    Find optimal number of clusters using silhouette score and elbow method.

cluster_providers(df_provider: pl.DataFrame, features: list[str], n_clusters: int = None) -> pl.DataFrame
    Cluster providers using KMeans with feature scaling.

validate_clustering(X_scaled: np.ndarray, labels: np.ndarray) -> dict
    Compute clustering validation metrics (silhouette scores, centroids).

assign_region(creator: str) -> str
    Map provider names to regions (US, China, Europe, Other).

compare_regions(df: pl.DataFrame, metric: str) -> dict
    Compare providers by region on specified metric.
"""

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def aggregate_by_provider(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """
    Aggregate model-level data to provider-level with mean features and counts.

    Groups by Creator column and computes mean values for specified features
    (e.g., intelligence_index, price_usd, speed). Also adds count of models
    per provider. Filters to models with valid intelligence scores.

    Parameters
    ----------
    df : pl.DataFrame
        Model-level DataFrame with Creator column and feature columns.
    features : list[str]
        List of column names to aggregate by mean (e.g., ['intelligence_index', 'price_usd']).

    Returns
    -------
    pl.DataFrame
        Provider-level DataFrame with columns:
        - Creator: Provider name
        - avg_{feature}: Mean value for each feature
        - model_count: Number of models per provider

    Examples
    --------
    >>> df = pl.read_parquet("data/processed/ai_models_deduped.parquet")
    >>> df_valid = df.filter(pl.col('intelligence_index').is_not_null())
    >>> provider_df = aggregate_by_provider(df_valid, ['intelligence_index', 'price_usd'])
    >>> print(provider_df.head())

    Notes
    -----
    - Filters to models with non-null intelligence_index before aggregation
    - Computes mean of each feature by provider
    - Counts total models per provider (including those without intelligence scores)
    """
    # Filter to models with valid intelligence scores
    df_valid = df.filter(pl.col("intelligence_index").is_not_null())

    # Build aggregation expressions
    agg_exprs = []
    for feature in features:
        # Map feature names to actual column names
        if feature == "intelligence_index":
            col_name = "intelligence_index"
            alias_name = "avg_intelligence"
        elif feature == "price_usd":
            col_name = "price_usd"
            alias_name = "avg_price"
        elif feature == "Speed(median token/s)":
            col_name = "Speed(median token/s)"
            alias_name = "avg_speed"
        elif feature == "speed":
            col_name = "Speed(median token/s)"
            alias_name = "avg_speed"
        else:
            col_name = feature
            alias_name = f"avg_{feature}"

        agg_exprs.append(
            pl.col(col_name)
            .cast(pl.Float64)
            .mean()
            .alias(alias_name)
        )

    # Aggregate by provider
    provider_df = (
        df_valid
        .group_by("Creator")
        .agg([
            *agg_exprs,
            pl.len().alias("model_count")
        ])
        .sort("Creator")
    )

    return provider_df


def find_optimal_clusters(X_scaled: np.ndarray, max_k: int = 10) -> dict:
    """
    Find optimal number of clusters using silhouette score and elbow method.

    Tests KMeans clustering with k from 2 to max_k. For each k, computes:
    - Silhouette score (higher = better cluster separation, max 1.0)
    - Inertia (within-cluster sum of squares, lower = better)

    Optimal k is selected as the k with maximum silhouette score.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    max_k : int, default=10
        Maximum number of clusters to test.

    Returns
    -------
    dict
        Dictionary containing:
        - optimal_k: int - Number of clusters with highest silhouette score
        - silhouette_scores: list - Silhouette scores for k=2 to max_k
        - inertias: list - Inertia values for k=2 to max_k
        - all_labels: dict - Cluster labels for each k {k: labels}

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = provider_df.select(['avg_intelligence', 'avg_price']).to_numpy()
    >>> X_scaled = StandardScaler().fit_transform(X)
    >>> results = find_optimal_clusters(X_scaled, max_k=10)
    >>> print(f"Optimal K: {results['optimal_k']}")
    >>> print(f"Silhouette scores: {results['silhouette_scores']}")

    Notes
    -----
    - Uses random_state=42 for reproducibility
    - Uses n_init=10 for KMeans initialization (default sklearn behavior)
    - Silhouette score ranges from -1 to 1 (higher = better)
    - Inertia decreases monotonically with k - use elbow method for visual inspection
    """
    silhouette_scores = []
    inertias = []
    all_labels = {}

    # Test k from 2 to max_k
    k_range = range(2, max_k + 1)

    for k in k_range:
        # Fit KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Compute metrics
        sil_score = silhouette_score(X_scaled, labels)
        inertia = kmeans.inertia_

        silhouette_scores.append(sil_score)
        inertias.append(inertia)
        all_labels[k] = labels

    # Find optimal k (maximum silhouette score)
    optimal_k = int(np.argmax(silhouette_scores) + 2)  # +2 because k starts at 2

    return {
        "optimal_k": optimal_k,
        "silhouette_scores": silhouette_scores,
        "inertias": inertias,
        "all_labels": all_labels
    }


def cluster_providers(
    df_provider: pl.DataFrame,
    features: list[str],
    n_clusters: int = None
) -> pl.DataFrame:
    """
    Cluster providers using KMeans with feature scaling.

    Extracts specified features, scales them using StandardScaler, and applies
    KMeans clustering. If n_clusters is None, automatically determines optimal
    k using silhouette score analysis.

    Parameters
    ----------
    df_provider : pl.DataFrame
        Provider-level DataFrame with feature columns.
    features : list[str]
        List of feature column names to use for clustering.
    n_clusters : int, optional
        Number of clusters. If None, finds optimal k automatically.

    Returns
    -------
    pl.DataFrame
        Original DataFrame with added 'cluster' column (int type).

    Examples
    --------
    >>> features = ["avg_intelligence", "avg_price", "avg_speed"]
    >>> clustered_df = cluster_providers(provider_df, features)
    >>> print(clustered_df["cluster"].unique())

    Notes
    -----
    - Features are scaled before clustering (StandardScaler)
    - If n_clusters is None, uses silhouette score to find optimal k
    - Uses random_state=42 for reproducibility
    """
    # Extract features
    feature_cols = []
    for feature in features:
        if feature == "avg_speed":
            col_name = "avg_speed"
        elif feature.startswith("avg_"):
            col_name = feature
        else:
            col_name = f"avg_{feature}"

        if col_name in df_provider.columns:
            feature_cols.append(col_name)

    # Handle null values - drop rows with null features
    df_clean = df_provider.drop_nulls(subset=feature_cols)

    # Extract feature matrix
    X = df_clean.select(feature_cols).to_numpy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal clusters if not specified
    if n_clusters is None:
        results = find_optimal_clusters(X_scaled, max_k=10)
        n_clusters = results["optimal_k"]

    # Fit final KMeans with optimal k
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Add cluster column to DataFrame
    df_with_clusters = df_clean.with_columns(
        pl.Series("cluster", labels)
    )

    return df_with_clusters


def validate_clustering(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """
    Compute clustering validation metrics.

    Calculates silhouette score (overall and per sample) and cluster centroids
    in the scaled feature space.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    labels : np.ndarray
        Cluster labels for each sample.

    Returns
    -------
    dict
        Dictionary containing:
        - silhouette_score: float - Overall silhouette score
        - silhouette_samples: np.ndarray - Per-sample silhouette scores
        - centroids: np.ndarray - Cluster centroids in scaled space
        - n_clusters: int - Number of clusters

    Examples
    --------
    >>> validation = validate_clustering(X_scaled, labels)
    >>> print(f"Silhouette score: {validation['silhouette_score']:.3f}")
    >>> print(f"Centroids shape: {validation['centroids'].shape}")

    Notes
    -----
    - Silhouette score ranges from -1 to 1 (higher = better)
    - Scores > 0.5 indicate good cluster structure
    - Scores < 0.2 indicate poor cluster structure
    """
    # Overall silhouette score
    sil_score = silhouette_score(X_scaled, labels)

    # Per-sample silhouette scores
    from sklearn.metrics import silhouette_samples
    sil_samples = silhouette_samples(X_scaled, labels)

    # Compute cluster centroids
    n_clusters = len(np.unique(labels))
    centroids = np.array([
        X_scaled[labels == i].mean(axis=0)
        for i in range(n_clusters)
    ])

    return {
        "silhouette_score": sil_score,
        "silhouette_samples": sil_samples,
        "centroids": centroids,
        "n_clusters": n_clusters
    }


def assign_region(creator: str) -> str:
    """
    Map provider names to regions (US, China, Europe, Other).

    Uses case-insensitive matching to map known AI providers to their
    geographic regions based on company headquarters.

    Parameters
    ----------
    creator : str
        Provider/creator name from dataset.

    Returns
    -------
    str
        Region code: "US", "China", "Europe", or "Other".

    Examples
    --------
    >>> test_creators = ['OpenAI', 'DeepSeek', 'Mistral', 'Unknown Provider']
    >>> for creator in test_creators:
    ...     region = assign_region(creator)
    ...     print(f'{creator} -> {region}')
    OpenAI -> US
    DeepSeek -> China
    Mistral -> Europe
    Unknown Provider -> Other

    Notes
    -----
    - US: OpenAI, Anthropic, Google, Meta, Microsoft, Amazon, etc.
    - China: DeepSeek, Alibaba, Tencent, Baidu, etc.
    - Europe: Mistral, Aleph Alpha, etc.
    - Other: All other providers
    """
    # Normalize for case-insensitive matching
    creator_lower = creator.lower()

    # US providers
    us_providers = [
        'openai', 'anthropic', 'google', 'meta', 'microsoft',
        'amazon', 'facebook', 'ibm', 'nvidia', 'cohere',
        'inflection', 'a121', 'character.ai', 'reka'
    ]

    # Chinese providers
    china_providers = [
        'deepseek', 'alibaba', 'tencent', 'baidu', 'qwen',
        '01.ai', 'zhipu', 'minimax', 'moonshot'
    ]

    # European providers
    europe_providers = [
        'mistral', 'aleph alpha', 'bigscience', 'huggingface'
    ]

    # Check each region
    for provider in us_providers:
        if provider in creator_lower:
            return "US"

    for provider in china_providers:
        if provider in creator_lower:
            return "China"

    for provider in europe_providers:
        if provider in creator_lower:
            return "Europe"

    return "Other"


def compare_regions(df: pl.DataFrame, metric: str) -> dict:
    """
    Compare providers by region on specified metric.

    Groups by region column and computes mean, median, and standard deviation
    for the specified metric.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with 'region' column and metric column.
    metric : str
        Column name of metric to compare (e.g., 'intelligence_index', 'price_usd').

    Returns
    -------
    dict
        Dictionary with region names as keys and statistics as values.

    Examples
    --------
    >>> regional_stats = compare_regions(df, 'intelligence_index')
    >>> for region, stats in regional_stats.items():
    ...     print(f"{region}: {stats['mean']:.2f}")

    Notes
    -----
    - Computes mean, median, std for each region
    - Handles null values in metric column
    - Returns empty dict if metric not found
    """
    if metric not in df.columns or "region" not in df.columns:
        return {}

    try:
        regional_stats = (
            df
            .group_by("region")
            .agg([
                pl.col(metric).cast(pl.Float64).mean().alias("mean"),
                pl.col(metric).cast(pl.Float64).median().alias("median"),
                pl.col(metric).cast(pl.Float64).std().alias("std"),
                pl.len().alias("count")
            ])
            .sort("region")
        )

        # Convert to dict for easier access
        return {
            row["region"]: {
                "mean": row["mean"],
                "median": row["median"],
                "std": row["std"],
                "count": row["count"]
            }
            for row in regional_stats.iter_rows(named=True)
        }
    except Exception as e:
        print(f"Warning: Could not compute regional stats for {metric}: {e}")
        return {}
