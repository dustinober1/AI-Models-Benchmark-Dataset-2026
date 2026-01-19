# Provider Clustering Analysis

**Generated:** 2026-01-18 19:16:46

---

## 1. Cluster Validation

**Optimal number of clusters:** 2
**Silhouette score:** 0.390

**Interpretation:**
- Silhouette score > 0.25 indicates **moderate cluster structure**
- Some overlap between clusters, but meaningful segments exist

## 2. Silhouette Score Analysis

Silhouette scores were computed for k=2 to k=10 clusters:

| K | Silhouette Score |
|---|------------------|
| 2 | 0.390  ← **Optimal K** |
| 3 | 0.356  |
| 4 | 0.355  |
| 5 | 0.324  |
| 6 | 0.336  |
| 7 | 0.349  |
| 8 | 0.348  |
| 9 | 0.338  |
| 10 | 0.322  |

## 3. Cluster Profiles

### Characteristics by Cluster

#### Cluster 0
- **Mean Intelligence:** 17.9
- **Mean Price:** $0.35
- **Mean Speed:** 34.3 tokens/s
- **Providers (24):** AI21 Labs, Alibaba, Allen Institute for AI, Baidu, ByteDance Seed, DeepSeek, IBM, InclusionAI, Kimi, Korea Telecom, LG AI Research, Liquid AI, MBZUAI Institute of Foundation Models, Meta, Microsoft Azure, Motif Technologies, NVIDIA, Naver, Nous Research, Perplexity, Prime Intellect, Reka AI, TII UAE, Upstage

#### Cluster 1
- **Mean Intelligence:** 29.0
- **Mean Price:** $1.53
- **Mean Speed:** 117.4 tokens/s
- **Providers (12):** Amazon, Anthropic, Cohere, Google, KwaiKAT, MiniMax, Mistral, OpenAI, ServiceNow, Xiaomi, Z AI, xAI

## 4. Market Segments

Based on cluster profiles, the following market segments were identified:

**Cluster 0: Budget-Friendly Segment**
- Affordable providers ($0.35) with mid-tier intelligence (17.9)

**Cluster 1: Premium Performance Segment**
- High-intelligence (29.0), premium-priced ($1.53), high-speed (117.4 tokens/s) providers

## 5. Regional Comparison (STAT-04)

Provider performance compared across regions (US, China, Europe, Other):

### Intelligence

| Region | Mean | Median | Std Dev | Min | Max | Count |
|--------|------|--------|---------|-----|-----|-------|
| China | 22.2 | 20.5 | 8.7 | 6.0 | 41.0 | 40 |
| Europe | 18.8 | 19.0 | 4.3 | 12.0 | 27.0 | 12 |
| Other | 21.1 | 20.0 | 9.8 | 6.0 | 42.0 | 55 |
| US | 22.6 | 19.0 | 12.9 | 6.0 | 51.0 | 74 |

### Price

| Region | Mean | Median | Std Dev | Min | Max | Count |
|--------|------|--------|---------|-----|-----|-------|
| China | $0.93 | $0.59 | $0.88 | $0.00 | $3.00 | 40 |
| Europe | $0.55 | $0.17 | $0.77 | $0.00 | $2.75 | 12 |
| Other | $0.44 | $0.15 | $0.96 | $0.00 | $6.00 | 55 |
| US | $1.53 | $0.32 | $2.26 | $0.00 | $10.00 | 74 |

### Speed

| Region | Mean | Median | Std Dev | Min | Max | Count |
|--------|------|--------|---------|-----|-----|-------|
| China | 66.4 | 48.0 | 57.7 | 0.0 | 200.0 | 40 |
| Europe | 142.3 | 128.0 | 82.3 | 36.0 | 298.0 | 12 |
| Other | 59.9 | 54.0 | 60.3 | 0.0 | 209.0 | 55 |
| US | 118.4 | 90.0 | 113.9 | 0.0 | 550.0 | 74 |

## 6. Strategic Insights

### Market Structure
- Provider clustering reveals distinct market segments based on intelligence, price, and speed
- 2 market segments identified with good separation (silhouette: 0.390)

### Competitive Positioning

**Cluster 0:** AI21 Labs, Alibaba, Allen Institute for AI, Baidu, ByteDance Seed (+19 more)

**Cluster 1:** Amazon, Anthropic, Cohere, Google, KwaiKAT (+7 more)

### Regional Differences
- STAT-04 requirement: Regional comparison shows performance differences across US, China, and European providers
- See Regional Comparison section above for detailed statistics by region

### Implications
- Market segments can inform pricing strategy and competitive positioning
- Regional differences may reflect regulatory environments, market dynamics, or strategic priorities
- Cluster assignments available in `data/processed/provider_clusters.parquet` for further analysis