# Data Structure Documentation

**Generated:** 2026-01-18 22:57 UTC
**Source:** `data/raw/ai_models_performance.csv`
**Checkpoint:** `data/interim/01_loaded.parquet`
**Pipeline Stage:** 01 - Load (pre-cleaning)

---

## Dataset Overview

The AI Models Benchmark Dataset 2026 contains performance metrics for 188 Large Language Models (LLMs) from 37 different providers. This dataset captures key characteristics including context window size, intelligence scores, pricing, speed, and latency metrics.

| Metric | Value |
|--------|-------|
| **Total Rows** | 188 models |
| **Total Columns** | 7 metrics |
| **Unique Creators** | 37 organizations |
| **Data Format** | CSV (raw), Parquet (checkpoint) |
| **Loading Status** | Loaded with lenient schema (all Utf8) |

---

## Column Documentation

| Column Name | Data Type | Description | Valid Range | Sample Values |
|-------------|-----------|-------------|-------------|---------------|
| **Model** | String | Model name/identifier | Any text | "GPT-5.2 (xhigh)", "Claude Opus 4.5", "Gemini 3 Pro Preview (high)" |
| **Context Window** | String | Maximum context size in tokens | "16k" to "10m" | "400k", "1m", "200k", "128k", "10m" |
| **Creator** | String | Organization/lab that created the model | 37 unique values | "OpenAI", "Anthropic", "Google", "Alibaba", "DeepSeek" |
| **Intelligence Index** | String | Performance/IQ score (0-100 scale) | 0-100 (or "--" for missing) | "51", "49", "48", "47", "--" |
| **Price (Blended USD/1M Tokens)** | String | Price per 1M tokens in USD | "\$0.00 " to "\$100.00+" | "\$4.81 ", "\$10.00 ", "\$3.44 ", "\$1.13 " |
| **Speed(median token/s)** | String | Median tokens per second generated | 0+ | "100", "79", "128", "224", "264" |
| **Latency (First Answer Chunk /s)** | String | Time to first token in seconds | 0+ | "44.29", "1.7", "32.19", "26.5", "11.62" |

---

## Sample Data (First 5 Rows)

| Row | Model | Context Window | Creator | Intelligence Index | Price | Speed | Latency |
|-----|-------|----------------|---------|--------------------|-------|-------|---------|
| 1 | GPT-5.2 (xhigh) | 400k | OpenAI | 51 | \$4.81 | 100 | 44.29 |
| 2 | Claude Opus 4.5 | 200k | Anthropic | 49 | \$10.00 | 79 | 1.7 |
| 3 | Gemini 3 Pro Preview (high) | 1m | Google | 48 | \$4.50 | 128 | 32.19 |
| 4 | GPT-5.1 (high) | 400k | OpenAI | 47 | \$3.44 | 127 | 26.5 |
| 5 | Gemini 3 Flash | 1m | Google | 46 | \$1.13 | 224 | 11.62 |

---

## Data Quality Notes

### Messy Data Requiring Cleaning

The raw CSV contains several formatting issues that will be addressed in the cleaning stage (02_clean.py):

1. **Context Window Column**
   - **Issue:** Contains suffixes like "k" (thousand) and "m" (million)
   - **Examples:** "400k", "1m", "200k", "10m"
   - **Action:** Convert to numeric Int64 by parsing suffixes (1k = 1000, 1m = 1,000,000)
   - **Expected Range:** 0 to 2,000,000 tokens

2. **Price Column**
   - **Issue:** Contains dollar sign prefix and trailing space
   - **Examples:** "\$4.81 ", "\$10.00 ", "\$3.44 "
   - **Action:** Strip "\$" and whitespace, convert to Float64
   - **Expected Range:** \$0.00 to \$100.00+ per 1M tokens

3. **Intelligence Index Column**
   - **Issue:** Contains placeholder value "--" for missing data
   - **Examples:** "51", "49", "--"
   - **Action:** Replace "--" with null, convert valid values to Int64
   - **Expected Range:** 0 to 100

4. **Speed and Latency Columns**
   - **Issue:** Currently stored as strings
   - **Action:** Convert to Float64 after validation
   - **Expected Range:** 0+ (non-negative)

### Potential Data Issues

- **Missing Intelligence Index values:** Some rows have "--" instead of numeric scores
- **Context Window range:** Values range from "16k" to "10m" (16,000 to 10,000,000 tokens)
- **Quoted multi-line values:** CSV contains embedded newlines in quoted fields (e.g., "41\nE" in o3 model row)

---

## Unique Creators (37 Organizations)

The dataset includes models from major AI labs and companies worldwide:

| Category | Organizations |
|----------|---------------|
| **US Big Tech** | OpenAI, Google, Anthropic, Microsoft Azure, Amazon, Meta, NVIDIA, IBM |
| **Chinese Labs** | Alibaba, Baidu, ByteDance, Kimi, Z AI, MiniMax, DeepSeek |
| **European** | Mistral (France), Aleph Alpha (Germany), AI21 Labs (Israel) |
| **Middle East** | TII UAE, MBZUAI Institute of Foundation Models |
| **Korean** | Naver, LG AI Research, Korea Technology, Upstage |
| **Other** | xAI, Cohere, Reka AI, Liquid AI, Perplexity, ServiceNow |

---

## Schema Validation Results

**Status:** Not yet validated with Pandera schema

The dataset was loaded with a **lenient schema** (all Utf8) to accommodate messy formatting. Full schema validation with Pandera will occur **after** the cleaning stage (02_clean.py) when:

- Context Window is converted to Int64 (0 to 2,000,000)
- Price is converted to Float64 (>= 0)
- Intelligence Index is converted to Int64 (0 to 100)
- Speed is converted to Float64 (>= 0)
- Latency is converted to Float64 (>= 0)

### Expected Validation Checks

After cleaning, the Pandera schema (`src/validate.py::AIModelsSchema`) will enforce:

- **Type validation:** All columns match expected Polars data types
- **Range validation:** Intelligence Index 0-100, Context Window 0-2M, prices >= 0
- **Custom checks:** Context window realism check (<= 2,000,000 tokens)
- **Quarantine:** Invalid records will be separated to `data/quarantine/01_invalid_records.csv`

---

## Next Steps

The data pipeline continues with:

1. **02_clean.py** - Clean messy values and convert to proper types
   - Parse context window suffixes (k, m)
   - Strip dollar signs and whitespace from prices
   - Handle missing Intelligence Index values
   - Convert numeric columns to proper types

2. **Schema Validation** - Validate with Pandera `AIModelsSchema`
   - Enforce type constraints
   - Enforce range constraints
   - Quarantine invalid records

3. **03_analyze_distributions.py** - Statistical analysis
   - Distribution analysis for each numeric column
   - Skewness, kurtosis, normality tests

4. **04_detect_outliers.py** - Outlier detection
   - Isolation Forest algorithm
   - Multivariate outlier detection
   - Quarantine outliers with reason codes

5. **05_quality_report.py** - Quality assessment
   - Completeness metrics
   - Accuracy metrics
   - Consistency metrics
   - Visualizations and reports

---

## Technical Details

**Loading Method:** Polars LazyFrame with `scan_csv()`
**Schema Strategy:** Lenient (all Utf8) to handle messy data
**Checkpoint Format:** Parquet (compressed, columnar)
**File Size:** ~0.01 MB (188 rows × 7 columns)
**Memory Usage:** < 1 MB (lazy evaluation)

**Pipeline Pattern:**
```
Raw CSV → Lenient Load → Clean → Validate → Quarantine → Checkpoint
```

---

## Metadata

- **Generation Date:** 2026-01-18 22:57:00 UTC
- **Generator:** `scripts/01_load.py` via `src.load.document_structure()`
- **Dependencies:** Polars 1.0+, Pandera 0.21+
- **Checkpoint Path:** `data/interim/01_loaded.parquet`
- **Quarantine Path:** `data/quarantine/01_invalid_records.csv` (created if validation fails)
