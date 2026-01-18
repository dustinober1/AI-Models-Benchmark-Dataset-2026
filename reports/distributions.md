# Distribution Analysis Report

**Generated:** 2026-01-18 18:11:22 UTC

**Total Records:** 188

---

## Summary Statistics

| Column | Count | Mean | Std | Median | Min | Max | Skewness | Kurtosis |
|--------|-------|------|-----|--------|-----|-----|----------|----------|
| context_window | 188 | 359898.94 | 797142.08 | 200000.00 | 4000.00 | 10000000.00 | 9.63 | 114.20 |
| intelligence_index | 182 | 21.73 | 10.73 | 20.00 | 6.00 | 51.00 | 0.67 | 2.63 |
| price_usd | 188 | 0.97 | 1.64 | 0.30 | 0.00 | 10.00 | 2.82 | 12.58 |
| Speed(median token/s) | 188 | 89.77 | 92.27 | 72.00 | 0.00 | 550.00 | 1.73 | 7.38 |
| Latency (First Answer Chunk /s) | 188 | 3.31 | 12.68 | 0.52 | 0.00 | 113.01 | 7.11 | 58.73 |

## Distribution Interpretation

### context_window

- **Skewness:** 9.63 - Right-skewed (tail extends toward higher values)
- **Kurtosis:** 114.20 - Heavy-tailed (more outliers than normal distribution)
- **Normality test:** p-value = 0.0000 - **Not normally distributed** (reject null hypothesis)

### intelligence_index

- **Skewness:** 0.67 - Right-skewed (tail extends toward higher values)
- **Kurtosis:** 2.63 - Normal-like tail behavior
- **Normality test:** p-value = 0.0012 - **Not normally distributed** (reject null hypothesis)

### price_usd

- **Skewness:** 2.82 - Right-skewed (tail extends toward higher values)
- **Kurtosis:** 12.58 - Heavy-tailed (more outliers than normal distribution)
- **Normality test:** p-value = 0.0000 - **Not normally distributed** (reject null hypothesis)

### Speed(median token/s)

- **Skewness:** 1.73 - Right-skewed (tail extends toward higher values)
- **Kurtosis:** 7.38 - Heavy-tailed (more outliers than normal distribution)
- **Normality test:** p-value = 0.0000 - **Not normally distributed** (reject null hypothesis)

### Latency (First Answer Chunk /s)

- **Skewness:** 7.11 - Right-skewed (tail extends toward higher values)
- **Kurtosis:** 58.73 - Heavy-tailed (more outliers than normal distribution)
- **Normality test:** p-value = 0.0000 - **Not normally distributed** (reject null hypothesis)

## Outlier Analysis

- **Method:** Isolation Forest (contamination=0.05)
- **Total outliers detected:** 10 (5.32%)
- **Total inliers:** 178 (94.68%)

### Outlier Details

Models flagged as outliers:

| Model | Creator | Price | Intelligence | Speed | Latency | Outlier Score |
|-------|---------|-------|--------------|-------|---------|---------------|
| GPT-5.2 (xhigh) | OpenAI | $4.81 | 51 | 100.0 | 44.3 | -0.634 |
| Claude Opus 4.5 | Anthropic | $10.00 | 49 | 79.0 | 1.7 | -0.606 |
| Gemini 3 Pro Preview (high) | Google | $4.50 | 48 | 128.0 | 32.2 | -0.599 |
| GPT-5.1 (high) | OpenAI | $3.44 | 47 | 127.0 | 26.5 | -0.553 |
| Claude Opus 4.5 | Anthropic | $10.00 | 43 | 72.0 | 2.0 | -0.589 |
| GPT-5 mini (high) | OpenAI | $0.69 | 41 | 72.0 | 113.0 | -0.656 |
| Gemini 2.5 Pro | Google | $3.44 | 34 | 160.0 | 33.0 | -0.585 |
| GPT-5 nano (high) | OpenAI | $0.14 | 27 | 119.0 | 111.2 | -0.651 |
| Gemini 2.5 Flash-Lite (Sep) | Google | $0.17 | 22 | 550.0 | 4.2 | -0.597 |
| Llama 4 Scout | Meta | $0.28 | 14 | 117.0 | 0.4 | -0.644 |

## Visualizations

Distribution plots have been generated for all numerical columns:

- **context_window:** `reports/figures/context_window_distribution.png`
- **intelligence_index:** `reports/figures/intelligence_index_distribution.png`
- **price_usd:** `reports/figures/price_usd_distribution.png`
- **Speed(median token/s):** `reports/figures/Speedmedian_token_s_distribution.png`
- **Latency (First Answer Chunk /s):** `reports/figures/Latency_First_Answer_Chunk__s_distribution.png`

Each plot includes:
- Histogram with KDE curve (distribution shape)
- Box plot (quartiles and outliers)
- Q-Q plot (normality assessment)
