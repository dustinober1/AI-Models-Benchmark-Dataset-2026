# Duplicate Resolution Report

**Generated:** 2026-01-18 19:02:00
**Input:** ai_models_enriched.parquet
**Output:** ai_models_deduped.parquet

---

## Summary

**Original Models:** 188
**Duplicate Model Names:** 34
**Resolved Models:** 187
**Unique Model IDs:** 187
**True Duplicates Removed:** 1

---

## Resolution Strategy

**Primary:** Context Window Disambiguation

Unique model_id created using pattern: `ModelName_ContextWindow`

Example:
- `GPT-4_128000` for GPT-4 with 128k context window
- `Claude_2_200000` for Claude 2 with 200k context window

**Secondary:** Intelligence Index Disambiguation

For models with same name AND context window, add Intelligence Index:

Example:
- `NVIDIA_Nemotron_3_Nano_1000000_25` for IQ=25
- `NVIDIA_Nemotron_3_Nano_1000000_14` for IQ=14

**Tertiary:** True Duplicate Removal

Models identical in all columns (name, context, IQ, price, speed, etc.) are true duplicates.
These are removed, keeping only the first occurrence.

---

## Before/After Statistics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total Rows | 188 | 187 | -1 |
| Unique Model IDs | N/A | 187 | +187 |
| Duplicate Names | 34 | 0 | -34 |

---

## Duplicates Found

**Total Duplicate Model Names:** 34

| Model | Count | Context Windows |
|-------|-------|-----------------|
| GLM-4.6V | 2 | 128k |
| DeepSeek V3.2 | 2 | 128k |
| Qwen3 VL 8B | 2 | 256k |
| Exaone 4.0 1.2B | 2 | 64k |
| Gemini 2.5 Flash-Lite (Sep) | 2 | 1m |
| DeepSeek V3.1 Terminus | 2 | 128k |
| Qwen3 Omni 30B A3B | 2 | 66k |
| NVIDIA Nemotron 3 Nano | 2 | 1m |
| Qwen3 VL 32B | 2 | 256k |
| NVIDIA Nemotron Nano 12B v2 VL | 2 | 128k |
| Hermes 4 70B | 2 | 128k |
| Qwen3 Next 80B A3B | 2 | 262k |
| EXAONE 4.0 32B | 2 | 131k |
| Gemini 3 Flash | 2 | 1m |
| K-EXAONE | 2 | 256k |
| Claude Opus 4.5 | 2 | 200k |
| Llama 3.3 Nemotron Super 49B | 2 | 128k |
| Claude 4.5 Haiku | 2 | 200k |
| NVIDIA Nemotron Nano 9B V2 | 2 | 131k |
| Qwen3 1.7B | 2 | 32k |
| Qwen3 VL 235B A22B | 2 | 262k |
| Grok 4 Fast | 2 | 2m |
| Solar Pro 2 | 2 | 66k |
| Qwen3 4B 2507 | 2 | 262k |
| Qwen3 VL 4B | 2 | 256k |
| Qwen3 0.6B | 2 | 32k |
| Hermes 4 405B | 2 | 128k |
| Qwen3 30B A3B 2507 | 2 | 262k |
| GLM-4.7 | 2 | 200k |
| Qwen3 VL 30B A3B | 2 | 256k |
| MiMo-V2-Flash | 2 | 256k |
| Grok 4.1 Fast | 2 | 2m |
| Llama Nemotron Super 49B v1.5 | 2 | 128k |
| Claude 4.5 Sonnet | 2 | 1m |

---

## Validation Results

**Status:** ✓ PASS
**Original Duplicates:** 34
**Resolved Count:** 187
**Unique Model IDs:** 187
**Remaining Duplicates:** 0

**Message:**
> Validation passed: 187 unique model_ids for 187 models (0 remaining duplicates)

---

## Resolution Strategy Distribution

| Strategy | Count | Percentage |
|----------|-------|------------|
| context_window_deduped | 187 | 100.0% |

---

## Examples of Resolved Duplicates

### Primary Disambiguation (Context Window)

Models with same name but different context windows:

| Original Model | Context Window | New model_id |
|----------------|----------------|--------------|
| Grok 4.1 Fast | 2,000,000 | Grok_4_1_Fast_2000000_38 |
| Grok 4.1 Fast | 2,000,000 | Grok_4_1_Fast_2000000_23 |
| Qwen3 Next 80B A3B | 262,000 | Qwen3_Next_80B_A3B_262000_20 |
| Qwen3 Next 80B A3B | 262,000 | Qwen3_Next_80B_A3B_262000_27 |
| Hermes 4 70B | 128,000 | Hermes_4_70B_128000_14 |
| Hermes 4 70B | 128,000 | Hermes_4_70B_128000_20 |
| Qwen3 VL 235B A22B | 262,000 | Qwen3_VL_235B_A22B_262000_21 |
| Qwen3 VL 235B A22B | 262,000 | Qwen3_VL_235B_A22B_262000_27 |
| Llama 3.3 Nemotron Super 49B | 128,000 | Llama_3_3_Nemotron_Super_49B_128000_18 |
| Llama 3.3 Nemotron Super 49B | 128,000 | Llama_3_3_Nemotron_Super_49B_128000_14 |

### Secondary Disambiguation (Intelligence Index)

Models with same name and context window but different Intelligence Index:

| Original Model | Context Window | Intelligence Index | New model_id |
|----------------|----------------|-------------------|--------------|

---

## Data Quality Notes

**True Duplicate Removed:**
- 1 model had identical rows in all columns (Exaone 4.0 1.2B)
- These were data entry errors and removed to prevent aggregation issues

**Intelligence Index Null Handling:**
- 6 models have null Intelligence Index (filled with -1 for disambiguation)
- These models are identified by model_id ending in '_-1'
- Intelligence-specific analyses should filter to n=181 models with valid IQ scores

---

## Next Steps

✓ **Dataset ready for Phase 2 statistical analysis**

**Output files:**
- `data/processed/ai_models_deduped.parquet` - Deduplicated dataset (187 models, 18 columns)
- Unique `model_id` column for accurate group-by operations
- Original `Model` column preserved for reference

**Recommended for correlation analysis:**
- Use `model_id` for all group-by operations
- Use Spearman correlation (non-parametric) for skewed distributions
- Consider log-transformation for Context Window (extreme skewness: 9.63)
- Filter to n=181 for intelligence-specific analyses (exclude null IQ scores)

---

## Metadata

**Generation Timestamp:** 2026-01-18 19:02:00
**Pipeline Version:** Phase 2 - Statistical Analysis & Domain Insights
**Plan:** 02-01 (Duplicate Resolution)

**Dependencies:**
- polars >= 1.0.0
- Input: `data/processed/ai_models_enriched.parquet`
- Output: `data/processed/ai_models_deduped.parquet`

*End of Report*