# Pareto Frontier Analysis Report

**Generated:** 2026-01-18 19:11:11
**Input:** ai_models_deduped.parquet
**Models Analyzed:** 181/187 (with valid Intelligence Index)

---

## Executive Summary

Pareto frontier analysis identifies models that offer optimal tradeoffs between competing objectives. A model is **Pareto-efficient** if no other model dominates it (is better in all objectives). These models represent the "efficient frontier" - the best choices for different use cases.

This analysis examines three objective spaces:

1. **Intelligence vs Price** - Value proposition: Best intelligence per dollar
2. **Speed vs Intelligence** - Performance leadership: Fast models with high intelligence
3. **Multi-Objective** - Overall excellence: Balancing intelligence, speed, price, and latency

---

## Key Findings

### 1. Price-Performance Frontier (Intelligence vs Price)

**Pareto-Efficient Models:** 8/181 (4.4%)

These models offer the best intelligence per dollar. No other model provides higher intelligence at a lower price.

**Top Value Leaders:**

| Model | Intelligence Index | Price ($/1M tokens) | Creator |
|-------|-------------------|---------------------|---------|
| GPT-5.2 (xhigh) | 51 | $4.81 | OpenAI |
| Gemini 3 Pro Preview (high) | 48 | $4.50 | Google |
| GPT-5.1 (high) | 47 | $3.44 | OpenAI |
| Gemini 3 Flash | 46 | $1.13 | Google |
| GLM-4.7 | 42 | $0.94 | Z AI |
| DeepSeek V3.2 | 41 | $0.32 | DeepSeek |
| MiMo-V2-Flash | 39 | $0.15 | Xiaomi |
| KAT-Coder-Pro V1 | 36 | $0.00 | KwaiKAT |

**Value Proposition:**
- Budget-friendly options with competitive intelligence
- Premium models with intelligence justifying higher cost
- Best "bang for buck" models at each price point

---

### 2. Speed-Intelligence Frontier

**Pareto-Efficient Models:** 6/181 (3.3%)

These models dominate in both speed and intelligence. No other model is faster AND smarter.

**Performance Leaders:**

| Model | Intelligence Index | Speed (tokens/s) | Creator |
|-------|-------------------|-----------------|---------|
| GPT-5.2 (xhigh) | 51 | 100.0 | OpenAI |
| Gemini 3 Pro Preview (high) | 48 | 128.0 | Google |
| Gemini 3 Flash | 46 | 224.0 | Google |
| o3 | 41 | 264.0 | OpenAI |
| gpt-oss-120B (high) | 33 | 366.0 | OpenAI |
| Gemini 2.5 Flash-Lite (Sep) | 22 | 550.0 | Google |

**Performance Insights:**
- High-intelligence models with competitive speed
- Real-time capable models with good intelligence
- Throughput leaders for high-volume applications

---

### 3. Multi-Objective Frontier (Overall Excellence)

**Pareto-Efficient Models:** 41/181 (22.7%)

These models balance all four objectives: intelligence, speed, price, and latency. No other model is better in all dimensions.

**Overall Optimal Models:**

| Model | Intelligence | Speed | Price | Latency | Creator |
|-------|-------------|-------|-------|---------|---------|
| GPT-5.2 (xhigh) | 51 | 100.0 | $4.81 | 44.3 | OpenAI |
| Claude Opus 4.5 | 49 | 79.0 | $10.00 | 1.7 | Anthropic |
| Gemini 3 Pro Preview (high) | 48 | 128.0 | $4.50 | 32.2 | Google |
| GPT-5.1 (high) | 47 | 127.0 | $3.44 | 26.5 | OpenAI |
| Gemini 3 Flash | 46 | 224.0 | $1.13 | 11.6 | Google |
| GPT-5.2 (medium) | 45 | 0.0 | $4.81 | 0.0 | OpenAI |
| Claude 4.5 Sonnet | 42 | 80.0 | $6.00 | 1.6 | Anthropic |
| GLM-4.7 | 42 | 78.0 | $0.94 | 0.6 | Z AI |
| o3 | 41 | 264.0 | $3.50 | 12.4 | OpenAI |
| GPT-5 mini (high) | 41 | 72.0 | $0.69 | 113.0 | OpenAI |

**Market Leaders:**
- These models represent the state-of-the-art across multiple dimensions
- No single model dominates all objectives - tradeoffs exist
- Different leaders for different use cases (budget, speed, intelligence)

---

## Market Insights

### Provider Dominance

Which providers dominate the Pareto frontiers:

**Price-Performance Frontier:**

- OpenAI: 2 models (25.0 of frontier)
- Google: 2 models (25.0 of frontier)
- Xiaomi: 1 models (12.5 of frontier)
- KwaiKAT: 1 models (12.5 of frontier)
- DeepSeek: 1 models (12.5 of frontier)

**Speed-Intelligence Frontier:**

- OpenAI: 3 models (50.0 of frontier)
- Google: 3 models (50.0 of frontier)

**Multi-Objective Frontier:**

- OpenAI: 8 models (19.5 of frontier)
- Google: 6 models (14.6 of frontier)
- Mistral: 5 models (12.2 of frontier)
- NVIDIA: 4 models (9.8 of frontier)
- Anthropic: 3 models (7.3 of frontier)

---

## Model Selection Recommendations

### For Different Use Cases

**1. Budget-Conscious Applications**
- Choose from: Intelligence vs Price Pareto frontier
- Prioritize: Low price per 1M tokens
- Best for: High-volume applications, cost-sensitive projects

**2. Performance-Critical Applications**
- Choose from: Speed vs Intelligence Pareto frontier
- Prioritize: High token throughput
- Best for: Real-time applications, high-volume processing

**3. Balanced Requirements**
- Choose from: Multi-objective Pareto frontier
- Prioritize: Overall excellence across all dimensions
- Best for: General-purpose applications, uncertain requirements

**4. Intelligence-Critical Applications**
- Choose: Highest intelligence model within budget
- Tradeoff: Accept higher price or lower speed
- Best for: Complex reasoning, code generation, analysis

---

## Frontier Interpretation

### What Does Pareto-Efficient Mean?

A model is **Pareto-efficient** if:
- No other model is better in **all** objectives being considered
- Any improvement in one objective would require sacrifice in another
- These models form the "efficient frontier" of the solution space

### What About Non-Efficient Models?

Models NOT on the Pareto frontier are **dominated** - there exists at least one other model that is better in all objectives. These models are generally not recommended unless:
- You have specific constraints not captured in the analysis
- You require features not measured (e.g., specific capabilities, ecosystem)
- The model has other advantages (e.g., ease of use, documentation)

### Frontier Density

- **Price-Performance:** 8 efficient models (4.4% of analyzed models)
- **Speed-Intelligence:** 6 efficient models (3.3% of analyzed models)
- **Multi-Objective:** 41 efficient models (22.7% of analyzed models)

A smaller frontier indicates clearer leaders. A larger frontier indicates more tradeoff options.

---

## Limitations and Considerations

**Analysis Scope:**
- Only models with valid Intelligence Index scores ({total_models_with_iq} of {total_models} models)
- Objectives limited to: Intelligence, Speed, Price, Latency
- Does not consider: Capabilities, ecosystem, documentation, ease of use

**Data Quality:**
- Scores represent benchmarks at time of data collection
- Performance may vary by task and implementation
- Prices may change over time

**Recommendation:**
Use Pareto analysis as a starting point, but consider qualitative factors (features, support, ecosystem) for final model selection.

---

## Technical Details

### Pareto Dominance Algorithm

For two models A and B with objectives [o1, o2, ..., on]:

**A dominates B if:**
- A is better or equal to B in **all** objectives
- A is strictly better than B in **at least one** objective

**Implementation:**
- Maximization objectives: Higher is better (e.g., intelligence)
- Minimization objectives: Lower is better (e.g., price, latency)
- Negate minimization objectives to convert to maximization
- Check dominance using vectorized numpy operations

### Output Files

- `data/processed/pareto_frontier.parquet` - Dataset with Pareto flags
- `reports/figures/pareto_frontier_intelligence_price.png` - Price-performance visualization
- `reports/figures/pareto_frontier_speed_intelligence.png` - Speed-intelligence visualization

---

## Metadata

**Generation Timestamp:** 2026-01-18 19:11:11
**Pipeline Version:** Phase 2 - Statistical Analysis & Domain Insights
**Plan:** 02-03 (Pareto Frontier Analysis)

**Dependencies:**
- polars >= 1.0.0
- numpy >= 1.24.0
- matplotlib >= 3.10.0
- Input: `data/processed/ai_models_deduped.parquet`
- Output: `data/processed/pareto_frontier.parquet`

*End of Report*