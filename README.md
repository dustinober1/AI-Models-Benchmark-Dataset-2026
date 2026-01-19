# AI Models Benchmark Analysis 2026

**Comprehensive exploratory data analysis of 187 AI models across 37 providers**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.14+-brightgreen.svg)](https://www.python.org/)

## Overview

This project performs a rigorous statistical analysis of the 2026 AI Models Benchmark Dataset to uncover insights about model performance, pricing strategies, and market dynamics. We use non-parametric methods (Spearman correlation, Mann-Whitney U, Kruskal-Wallis) due to right-skewed distributions and apply bootstrap resampling for uncertainty quantification.

**Key Findings:**
- Only 8 models (4.4%) are Pareto-efficient for price-performance
- Market has split into Budget (24 providers) vs Premium (12 providers) segments
- All 10 pairwise correlations statistically significant after FDR correction
- Intelligence correlates moderately with price (ρ=0.590) but weakly with speed (ρ=0.261)

## Installation

**Requirements:**
- Python 3.14+
- Poetry 2.3.0+

**Setup:**
```bash
# Clone repository
git clone https://github.com/yourusername/AI-Models-Benchmark-Dataset-2026.git
cd AI-Models-Benchmark-Dataset-2026

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Project Structure

```
AI-Models-Benchmark-Dataset-2026/
├── data/                          # Dataset storage
│   ├── raw/                       # Original CSV
│   ├── interim/                   # Intermediate checkpoints
│   └── processed/                 # Final datasets
├── scripts/                       # Analysis scripts (01-15)
│   ├── 01_load.py                # Load and validate
│   ├── 02_clean.py               # Data cleaning
│   ├── 03_analyze_distributions.py
│   ├── 04_detect_outliers.py
│   ├── 05_quality_report.py
│   ├── 06_enrich_external.py
│   ├── 07_duplicate_resolution.py
│   ├── 08_correlation_analysis.py
│   ├── 09_pareto_frontier.py
│   ├── 10_statistical_tests.py
│   ├── 11_provider_clustering.py
│   ├── 12_trend_predictions.py
│   ├── 13_distribution_viz.py
│   ├── 14_provider_frontier_viz.py
│   └── 15_linked_brushing_viz.py
├── src/                           # Reusable modules
│   ├── load.py, clean.py, analyze.py
│   ├── pareto.py, clustering.py, bootstrap.py
│   └── visualize.py
├── reports/                       # Generated outputs
│   ├── figures/                  # Pre-generated visualizations
│   └── *.md                      # Analysis reports
├── ai_models_benchmark_analysis.ipynb  # Kaggle notebook
└── README.md
```

## Reproducing the Analysis

**Option 1: Run full pipeline**
```bash
# Run all analysis scripts in order
cd scripts
for script in 0*.py; do python "$script"; done
```

**Option 2: Run specific phases**
```bash
# Phase 1: Data pipeline
python scripts/01_load.py
python scripts/02_clean.py
# ... etc

# Phase 2: Statistical analysis
python scripts/08_correlation_analysis.py
python scripts/09_pareto_frontier.py
# ... etc

# Phase 3: Visualizations
python scripts/13_distribution_viz.py
python scripts/14_provider_frontier_viz.py
python scripts/15_linked_brushing_viz.py
```

**Option 3: Interactive notebook**
```bash
jupyter notebook ai_models_benchmark_analysis.ipynb
```

## Data Sources

- **Primary dataset:** `ai_models_performance.csv` (188 models)
- **External enrichment:** Model release dates, provider announcements (0% coverage - web scraping failed)
- **Processed outputs:** `data/processed/ai_models_deduped.parquet` (187 models)

## Key Analysis Results

### Correlation Analysis
- All 10 pairwise correlations significant after FDR correction
- Intelligence-Price: ρ=0.590 (moderate)
- Intelligence-Speed: ρ=0.261 (weak)

### Pareto Frontier Analysis
- Price-performance frontier: 8 models (4.4%)
- Speed-intelligence frontier: 6 models (3.3%)
- Multi-objective frontier: 41 models (22.7%)

### Provider Clustering
- K=2 segments (Budget vs Premium)
- Budget: 24 providers, $0.35 mean price
- Premium: 12 providers, $1.53 mean price

### Regional Comparison
- Intelligence: Similar across regions
- Price: US highest ($1.53), Europe lowest ($0.55)
- Speed: Europe fastest (142 token/s), China slowest (66)

## Statistical Methods

- **Correlation:** Spearman rank correlation (non-parametric)
- **Multiple testing:** Benjamini-Hochberg FDR correction
- **Clustering:** KMeans with silhouette validation
- **Uncertainty:** Bootstrap BCa confidence intervals (95%)
- **Group comparisons:** Mann-Whitney U, Kruskal-Wallis tests

## Kaggle Notebook

The main notebook (`ai_models_benchmark_analysis.ipynb`) follows an **insight-first structure**:
1. Executive summary (key findings first)
2. Data quality assessment
3. Correlation analysis
4. Pareto frontier analysis
5. Provider clustering
6. Speed-intelligence tradeoff
7. 2027 trend predictions
8. Conclusions and recommendations

**Notebook features:**
- Pre-generated visualizations (fast loading)
- Imports from src/ modules (no duplicate code)
- 2:1 markdown-to-code ratio
- "So what?" explanations for practical relevance

## Requirements

See `pyproject.toml` for full dependency list:
- polars >= 1.0.0
- numpy >= 1.24.0
- scipy >= 1.14.0
- scikit-learn >= 1.6.0
- plotly >= 6.5.0
- jupyter >= 1.1.0

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit pull requests or open issues.

## Citation

If you use this analysis, please cite:
```
@software{ai_models_benchmark_2026,
  title={AI Models Benchmark Analysis 2026},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/AI-Models-Benchmark-Dataset-2026}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Analysis Date:** January 2026
**Dataset:** AI Models Benchmark Dataset 2026 (187 models, 37 providers)
**Method:** Non-parametric statistics, bootstrap resampling, Pareto optimization
