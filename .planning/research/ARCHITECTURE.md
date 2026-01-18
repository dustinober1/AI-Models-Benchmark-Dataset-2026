# Architecture Research

**Domain:** Exploratory Data Analysis (EDA) Projects
**Researched:** 2026-01-18
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Analysis Layer (scripts/)                    │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────┐│
│  │ 01_load_     │  │ 02_clean_    │  │ 03_analyze_  │  │ 04_viz_ ││
│  │ data.py      │  │ transform.py │  │ correlations.py│  .py    ││
│  │              │  │              │  │              │  │         ││
│  │ → Load raw   │  │ → Clean &   │  │ → Compute   │  │ → Gen   ││
│  │   dataset    │  │   transform │  │   metrics    │  │   plots  ││
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────┬────┘│
│         │                 │                 │                │     │
└─────────┼─────────────────┼─────────────────┼────────────────┼─────┘
          │                 │                 │                │
          ↓                 ↓                 ↓                ↓
┌─────────┴─────────────────┴─────────────────┴────────────────┴─────┐
│                        Data Storage Layer (data/)                   │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ raw/         │  │ interim/     │  │ processed/   │             │
│  │ (immutable)  │  │ (transformed)│  │ (analysis)  │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                  ↑
                                  │ imports
                                  │
┌─────────────────────────────────┴─────────────────────────────────────┐
│                    Narrative Layer (notebooks/)                      │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ final_analysis.ipynb                                           │ │
│  │                                                                │ │
│  │ - Imports functions from scripts/                              │ │
│  │ - Loads processed data from data/processed/                    │ │
│  │ - Synthesizes findings into narrative                          │ │
│  │ - Creates Kaggle-ready visualization                           │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Data scripts** (`scripts/01_load_*.py`) | Load raw dataset, validate schema, save to parquet | Polars LazyFrame with `.collect()` and `.write_parquet()` |
| **Transform scripts** (`scripts/02_clean_*.py`) | Clean data, handle missing values, feature engineering | Polars expression pipeline with `.with_columns()` |
| **Analysis scripts** (`scripts/03_analyze_*.py`) | Domain-specific analysis (price-performance, correlations, etc.) | Pure functions returning DataFrames with metrics |
| **Visualization scripts** (`scripts/04_viz_*.py`) | Generate Plotly figures, save to HTML/JSON | Plotly Express with `.write_html()` or `.to_json()` |
| **Narrative notebook** (`notebooks/final_analysis.ipynb`) | Synthesize all findings into Kaggle-ready story | Import functions, load processed data, narrative markdown cells |
| **Data storage** (`data/`) | Immutable raw → transformed → processed datasets | Parquet files for efficiency, CSV for raw data |

## Recommended Project Structure

```
ai-models-benchmark-2026/
├── data/                          # Data storage layer
│   ├── raw/                      # Original, immutable data
│   │   └── ai_models_benchmark_2026.csv
│   ├── interim/                  # Intermediate, transformed data
│   │   ├── models_cleaned.parquet
│   │   └── features_engineered.parquet
│   └── processed/                # Final analysis outputs
│       ├── price_performance_metrics.parquet
│       ├── speed_intelligence_tradeoffs.parquet
│       └── provider_comparisons.parquet
│
├── scripts/                       # Analysis layer - modular scripts
│   ├── __init__.py               # Make scripts importable as module
│   ├── 01_load_data.py           # Load and validate raw dataset
│   ├── 02_clean_transform.py     # Data cleaning and transformations
│   ├── 03_analyze_price_performance.py    # Price vs performance analysis
│   ├── 03_analyze_speed_intelligence.py   # Speed vs intelligence tradeoffs
│   ├── 03_analyze_providers.py            # Provider comparisons
│   ├── 03_analyze_predictions.py          # Statistical predictions
│   └── 04_generate_visualizations.py      # Create all Plotly figures
│
├── notebooks/                     # Narrative layer
│   └── final_analysis.ipynb       # Kaggle-ready narrative notebook
│
├── src/                          # Optional: reusable utility functions
│   ├── __init__.py
│   ├── config.py                 # Configuration, constants
│   ├── data_utils.py             # Helper functions for data operations
│   └── plot_utils.py             # Helper functions for plotting
│
├── reports/                      # Generated reports and figures
│   └── figures/                  # Saved visualizations
│
├── tests/                        # Unit tests for script functions
│   └── test_*.py
│
├── .gitignore
├── pyproject.toml                # Package configuration
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

### Structure Rationale

- **`data/raw/`:** Immutable original dataset - never modify after download
- **`data/interim/`:** Intermediate transformations - cache for expensive operations
- **`data/processed/`:** Final analysis outputs - what the notebook consumes
- **`scripts/`:** Modular analysis scripts - numbered to indicate execution order
  - Numbering (`01_`, `02_`, `03_`) makes dependencies explicit
  - Each script is independently runnable and testable
  - Scripts are importable as modules (need `__init__.py`)
- **`notebooks/`:** Final narrative notebook that synthesizes all findings
  - Imports functions from `scripts/` rather than duplicating code
  - Focuses on storytelling and visualization, not computation
- **`src/`:** Shared utilities used across multiple scripts
  - Configuration management, helper functions, custom plotting utilities
- **`tests/`:** Unit tests ensure script functions work correctly
  - Critical for reproducibility and catching data errors

## Architectural Patterns

### Pattern 1: Script-as-Module with Importable Functions

**What:** Each analysis script is structured as a Python module with importable functions, not just a standalone script.

**When to use:** Whenever you want to reuse analysis logic in notebooks or other scripts.

**Trade-offs:**
- **Pros:** Testable, reusable, DRY (Don't Repeat Yourself), notebook can import and call functions
- **Cons:** Slightly more upfront work than procedural scripts

**Example:**
```python
# scripts/03_analyze_price_performance.py
import polars as pl
from pathlib import Path

def calculate_price_performance_metrics(
    data_path: Path,
    output_path: Path
) -> pl.DataFrame:
    """
    Calculate price-performance metrics for AI models.

    Args:
        data_path: Path to cleaned data (data/interim/models_cleaned.parquet)
        output_path: Path to save results (data/processed/price_performance_metrics.parquet)

    Returns:
        DataFrame with price-performance metrics
    """
    df = pl.read_parquet(data_path)

    metrics = (
        df.group_by("provider")
        .agg([
            pl.col("price_per_1k_tokens").mean().alias("avg_price"),
            pl.col("benchmark_score").mean().alias("avg_score"),
            pl.col("benchmark_score").max().alias("max_score"),
            (pl.col("benchmark_score") / pl.col("price_per_1k_tokens")).alias("value_ratio")
        ])
        .sort("value_ratio", descending=True)
    )

    # Save for notebook to use
    metrics.write_parquet(output_path)

    return metrics

if __name__ == "__main__":
    # Can run standalone: python scripts/03_analyze_price_performance.py
    calculate_price_performance_metrics(
        data_path=Path("data/interim/models_cleaned.parquet"),
        output_path=Path("data/processed/price_performance_metrics.parquet")
    )
```

**Then in notebook:**
```python
# notebooks/final_analysis.ipynb
import sys
sys.path.append("..")
from scripts.03_analyze_price_performance import calculate_price_performance_metrics

# Either run the analysis or load pre-computed results
metrics_path = Path("../data/processed/price_performance_metrics.parquet")
if metrics_path.exists():
    price_performance = pl.read_parquet(metrics_path)
else:
    price_performance = calculate_price_performance_metrics(
        data_path=Path("../data/interim/models_cleaned.parquet"),
        output_path=metrics_path
    )
```

### Pattern 2: LazyFrame Pipeline with Checkpointing

**What:** Use Polars LazyFrame for transformation pipelines with periodic checkpointing to parquet files.

**When to use:** For any multi-step data transformation that benefits from query optimization and intermediate caching.

**Trade-offs:**
- **Pros:** Automatic query optimization, parallel execution, cache intermediate results
- **Cons:** Need to manage checkpoint files explicitly

**Example:**
```python
# scripts/02_clean_transform.py
import polars as pl
from pathlib import Path

def clean_and_transform_data(
    input_path: Path,
    output_path: Path
) -> pl.DataFrame:
    """Clean and transform raw data using LazyFrame pipeline."""
    # Use LazyFrame for optimization
    df_lazy = pl.scan_csv(input_path)

    # Build transformation pipeline (not executed yet)
    cleaned = (
        df_lazy
        .filter(pl.col("price_per_1k_tokens").is_not_null())
        .with_columns([
            pl.col("release_date").str.to_date(),
            pl.col("price_per_1k_tokens").log().alias("log_price"),
            (pl.col("benchmark_score") / pl.col("price_per_1k_tokens")).alias("value_ratio")
        ])
        .with_columns([
            pl.col("provider").cast(pl.Categorical)
        ])
    )

    # Execute and checkpoint (save to parquet)
    result = cleaned.collect()
    result.write_parquet(output_path)

    return result
```

### Pattern 3: Visualization Generation Script

**What:** Separate script that generates all Plotly figures and saves them as HTML/JSON files.

**When to use:** When you have complex visualizations that are expensive to compute or need to be reused.

**Trade-offs:**
- **Pros:** Notebook loads pre-generated figures (fast), figures are version-controlled, can be used outside notebook
- **Cons:** Need to coordinate between script outputs and notebook imports

**Example:**
```python
# scripts/04_generate_visualizations.py
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def create_price_performance_scatter(
    data_path: Path,
    output_path: Path
) -> go.Figure:
    """Create interactive scatter plot of price vs performance."""
    df = pl.read_parquet(data_path)

    fig = px.scatter(
        df.to_pandas(),  # Plotly works better with pandas
        x="price_per_1k_tokens",
        y="benchmark_score",
        color="provider",
        size="model_parameters",
        hover_data=["model_name"],
        title="AI Models: Price vs Performance (2026)",
        log_x=True
    )

    fig.update_layout(
        xaxis_title="Price per 1K Tokens (USD, log scale)",
        yaxis_title="Benchmark Score"
    )

    # Save as HTML for notebook to load
    fig.write_html(output_path)

    return fig

if __name__ == "__main__":
    create_price_performance_scatter(
        data_path=Path("data/processed/price_performance_metrics.parquet"),
        output_path=Path("reports/figures/price_performance_scatter.html")
    )
```

**Then in notebook:**
```python
# notebooks/final_analysis.ipynb
from IPython.display import IFrame

# Display pre-generated figure
IFrame(src="../reports/figures/price_performance_scatter.html", width=1000, height=600)
```

## Data Flow

### Analysis Execution Flow

```
1. Load Raw Data
   scripts/01_load_data.py
   → reads: data/raw/*.csv
   → writes: data/interim/models_raw.parquet
   ↓
2. Clean & Transform
   scripts/02_clean_transform.py
   → reads: data/interim/models_raw.parquet
   → writes: data/interim/models_cleaned.parquet
   ↓
3a. Price-Performance Analysis
   scripts/03_analyze_price_performance.py
   → reads: data/interim/models_cleaned.parquet
   → writes: data/processed/price_performance_metrics.parquet
   ↓
3b. Speed-Intelligence Analysis
   scripts/03_analyze_speed_intelligence.py
   → reads: data/interim/models_cleaned.parquet
   → writes: data/processed/speed_intelligence_tradeoffs.parquet
   ↓
3c. Provider Comparisons
   scripts/03_analyze_providers.py
   → reads: data/interim/models_cleaned.parquet
   → writes: data/processed/provider_comparisons.parquet
   ↓
4. Generate Visualizations
   scripts/04_generate_visualizations.py
   → reads: data/processed/*.parquet
   → writes: reports/figures/*.html
   ↓
5. Narrative Notebook
   notebooks/final_analysis.ipynb
   → imports: scripts/* functions
   → reads: data/processed/*.parquet
   → reads: reports/figures/*.html
   → writes: final Kaggle notebook
```

### Import Flow in Notebook

```
notebook/final_analysis.ipynb
    │
    ├─→ import sys; sys.path.append("..")
    │
    ├─→ from scripts.03_analyze_price_performance import calculate_price_performance_metrics
    │   → either: run function OR load pre-computed data/processed/price_performance_metrics.parquet
    │
    ├─→ from scripts.03_analyze_speed_intelligence import analyze_speed_tradeoffs
    │   → either: run function OR load pre-computed data/processed/speed_intelligence_tradeoffs.parquet
    │
    ├─→ from scripts.04_generate_visualizations import create_price_performance_scatter
    │   → either: run function OR load pre-computed reports/figures/*.html
    │
    └─→ Synthesize findings into narrative
        → Markdown cells tell the story
        → Code cells load data and display figures
        → Final output: Kaggle-ready notebook
```

### Key Data Flows

1. **Data preparation flow:** Raw → Interim → Processed (unidirectional, no backward dependencies)
2. **Analysis flow:** Cleaned data → Multiple parallel analyses → Separate output files
3. **Notebook consumption:** Notebook imports functions OR loads pre-computed results (can re-run or use cached)
4. **Visualization flow:** Analysis results → Plotly figures → HTML files → Notebook display

## Build Order and Dependencies

### Execution Order (Numbered Scripts)

Scripts should be executed in numerical order due to dependencies:

```
Phase 1: Data Preparation (Sequential)
├── 01_load_data.py              # No dependencies
└── 02_clean_transform.py        # Depends on: 01

Phase 2: Analysis (Parallel)
├── 03_analyze_price_performance.py    # Depends on: 02
├── 03_analyze_speed_intelligence.py   # Depends on: 02 (can run in parallel)
├── 03_analyze_providers.py            # Depends on: 02 (can run in parallel)
└── 03_analyze_predictions.py          # Depends on: 02 (can run in parallel)

Phase 3: Visualization (Sequential)
└── 04_generate_visualizations.py      # Depends on: all 03_* scripts

Phase 4: Narrative (Final)
└── notebooks/final_analysis.ipynb     # Depends on: all scripts OR their outputs
```

### Dependency Graph

```
01_load_data
    ↓
02_clean_transform
    ├─→ 03_analyze_price_performance ──┐
    ├─→ 03_analyze_speed_intelligence ─┤
    ├─→ 03_analyze_providers ──────────┼─→ 04_generate_visualizations
    └─→ 03_analyze_predictions ────────┘
                                          ↓
                                    final_analysis.ipynb
```

### Implications for Development

1. **Start with data pipeline:** Build `01_load_data.py` and `02_clean_transform.py` first
2. **Independent analysis development:** Each `03_*` script can be developed in parallel
3. **Notebook can be developed early:** Use sample/subset data while waiting for full pipeline
4. **Incremental validation:** Test each script independently before moving to next
5. **Cached execution:** Once scripts run, notebook can load cached results (faster iteration)

## Anti-Patterns

### Anti-Pattern 1: Monolithic Notebook with All Logic

**What people do:** Put all data loading, cleaning, analysis, and visualization code in a single Jupyter notebook with 50+ code cells.

**Why it's wrong:**
- Not reproducible (hard to run from command line)
- Not testable (can't unit test notebook cells)
- Not reusable (can't import analysis logic into other projects)
- Slow iteration (need to re-run entire notebook for small changes)
- Version control issues (notebook JSON merges poorly)

**Do this instead:**
- Extract logic into `scripts/` with importable functions
- Notebook only imports and calls functions, doesn't define logic
- Number scripts for execution order
- Use `data/` directory for checkpointing

### Anti-Pattern 2: Scripts That Can't Be Imported

**What people do:** Write scripts with only procedural code in `if __name__ == "__main__"` block, no functions.

**Why it's wrong:**
- Notebook can't reuse the logic
- Can't unit test the code
- Can't run specific analyses independently

**Do this instead:**
```python
# Bad: Only procedural code
df = pl.read_parquet("data.parquet")
result = df.group_by("x").agg(pl.col("y").mean())
result.write_parquet("output.parquet")

# Good: Function-based
def analyze_by_group(input_path: Path, output_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(input_path)
    result = df.group_by("x").agg(pl.col("y").mean())
    result.write_parquet(output_path)
    return result

if __name__ == "__main__":
    analyze_by_group(
        input_path=Path("data.parquet"),
        output_path=Path("output.parquet")
    )
```

### Anti-Pattern 3: No Intermediate Checkpointing

**What people do:** Load raw data and do all transformations in memory without saving intermediate results.

**Why it's wrong:**
- Expensive recomputations during development
- Can't debug intermediate steps
- Can't parallelize analyses

**Do this instead:**
- Save outputs of each script to `data/interim/` or `data/processed/`
- Use parquet format for efficiency
- Scripts should accept input/output paths as arguments

### Anti-Pattern 4: Modifying Raw Data

**What people do:** Load data, modify it, and save back to the same file.

**Why it's wrong:**
- Can't reproduce analysis from original data
- Can't debug data quality issues
- Violates immutability principle

**Do this instead:**
- `data/raw/` is read-only and immutable
- Always save transformed data to new files in `data/interim/` or `data/processed/`
- Document transformation steps in code

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1-10 analysis scripts | Flat `scripts/` directory is fine |
| 10-50 analysis scripts | Group by domain: `scripts/analysis/`, `scripts/visualization/` |
| 50+ analysis scripts | Consider package structure: `scripts/analysis/price/`, `scripts/analysis/speed/` |

### Scaling Priorities

1. **First bottleneck:** Script execution time
   - **Fix:** Use Polars LazyFrame for query optimization
   - **Fix:** Add intermediate checkpointing to avoid recomputation

2. **Second bottleneck:** Notebook loading time
   - **Fix:** Pre-generate all visualizations as HTML files
   - **Fix:** Load processed parquet files instead of recomputing

3. **Third bottleneck:** Managing many analysis scripts
   - **Fix:** Group related scripts into subdirectories
   - **Fix:** Create a `make` or `nox` file for running common workflows

## Integration Points

### Data Storage Integration

| Storage | Integration Pattern | Notes |
|---------|---------------------|-------|
| Local parquet files | `pl.read_parquet()` / `.write_parquet()` | Fast, efficient for Polars |
| CSV files | `pl.read_csv()` / `.write_csv()` | Use for raw data import only |
| Kaggle datasets | Download to `data/raw/` then process | Treat as immutable source |

### Notebook ↔ Scripts Integration

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Script → Notebook | Import functions OR load parquet outputs | Notebook chooses to re-run or use cached |
| Notebook → Script | Notebook should NOT modify script behavior | Scripts are standalone, notebook is consumer |

### External Tools Integration

| Tool | Integration Pattern | Notes |
|------|---------------------|-------|
| Git + GitHub | Version control for scripts and notebook | Add `data/` to `.gitignore` (use DVC if needed) |
| pytest | Unit tests for script functions | Test with sample data in `tests/` |
| pre-commit | Run linting/tests before commit | Ensure code quality |

## Polars-Specific Patterns

### LazyFrame for Pipeline Optimization

Polars uses a DAG (Directed Acyclic Graph) to optimize query execution:

```python
# Build pipeline (not executed yet)
df_lazy = pl.scan_csv("data.csv").filter(...).with_columns(...)

# Polars optimizes the entire pipeline before execution
df = df_lazy.collect()  # Execution happens here
```

**Benefits:**
- Automatic query optimization
- Parallel execution
- Better memory management

### Expression-Based Transformations

Polars uses expressions instead of iterative operations:

```python
# Polars way (vectorized, fast)
df.with_columns([
    (pl.col("score") / pl.col("price")).alias("value_ratio"),
    pl.col("date").str.to_date(),
    pl.col("category").cast(pl.Categorical)
])

# Not: iterative row-by-row operations
```

### Selector API for Column Selection

```python
import polars.selectors as cs

# Select all numeric columns
numeric_df = df.select(cs.numeric())

# Select all string columns
string_df = df.select(cs.string())
```

## Sources

- [Cookiecutter Data Science - Official Documentation](https://cookiecutter-data-science.drivendata.org/) (HIGH confidence - official project template)
- [Cookiecutter Data Science - GitHub Repository](https://github.com/drivendataorg/cookiecutter-data-science) (HIGH confidence - official source)
- [How to set up your data science projects - Kaan Öztürk (Feb 2025)](https://mkozturk.com/posts/en/2025/set-up-data-science-projects/) (MEDIUM confidence - recent best practices)
- [EDA with Polars: Step-by-Step Guide (Towards Data Science)](https://towardsdatascience.com/eda-with-polars-step-by-step-guide-for-pandas-users-part-1-b2ec500a1008/) (HIGH confidence - detailed Polars patterns)
- [Building a Repeatable Data Analysis Process with Jupyter](https://pbpython.com/notebook-process.html) (MEDIUM confidence - workflow patterns)
- [Jupyter + IDE: How to Make It Work](https://medium.com/data-science/jupyter-ide-how-to-make-it-work-6253f78eec67) (MEDIUM confidence - script/notebook integration)
- [Modularise your Notebook into Scripts](https://towardsdatascience.com/modularise-your-notebook-into-scripts-5d5ccaf3f4f3) (MEDIUM confidence - refactoring patterns)
- [Pandas vs Polars: Why the 2025 Evolution Changes Everything](https://dev.to/dataformathub/pandas-vs-polars-why-the-2025-evolution-changes-everything-5ad1) (LOW confidence - web search, need verification)
- [How I Automated an Entire Data Analysis Pipeline with Python](https://medium.com/@michaelpreston515/how-i-automated-an-entire-data-analysis-pipeline-with-python-and-why-ill-never-go-back-bc99a74ce17f) (LOW confidence - web search, need verification)

---
*Architecture research for: EDA Projects with Separate Scripts and Narrative Notebook*
*Researched: 2026-01-18*
