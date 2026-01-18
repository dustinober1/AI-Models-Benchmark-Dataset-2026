# Technology Stack

**Project:** Comprehensive EDA of 2026 AI Models Benchmark Dataset
**Researched:** 2026-01-18
**Focus:** Modern Python stack with Polars and Plotly for exploratory data analysis

## Recommended Stack

### Core Data Processing
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Polars** | 1.x (latest) | High-performance DataFrame library | Rust-backed, 3-10x faster than pandas, multi-threaded, lazy evaluation, Apache Arrow compatible. Production-ready in 2025 with mature ecosystem. |
| **NumPy** | 2.4.0+ | Numerical computing foundation | Required dependency for Polars and statistical libraries. NumPy 2.x (Dec 2025) improves free-threaded Python support and user dtypes. |
| **SciPy** | 1.16.1+ | Scientific computing & statistical tests | Provides scipy.stats for correlation analysis, hypothesis testing, and statistical functions. Latest version (July 2025) supports Python 3.14. |

### Visualization
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **Plotly** | 5.x (latest) | Interactive visualizations | Industry standard for interactive charts. Supports 40+ chart types, integrates with Jupyter, exports to HTML/PDF. Plotly Express provides high-level API for rapid EDA. |
| **Kaleido** | latest | Static image export | Required for exporting Plotly figures to static formats (PNG, SVG, PDF) for notebook exports and publications. |

### Statistical Analysis & Prediction
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **scikit-learn** | 1.8.0+ | Machine learning & regression | Standard library for regression analysis, feature engineering, and predictive modeling. Simple, efficient, well-documented API. |
| **statsmodels** | 0.14.6+ | Statistical modeling & inference | Complements scikit-learn with detailed statistical output (p-values, confidence intervals). Better for understanding relationships than prediction. Essential for: <br>• OLS regression with full statistics <br>• Time series analysis (ARIMA) <br>• Statistical tests <br>• Econometric modeling |

### Jupyter & Notebook Development
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **JupyterLab** | 4.x (latest) | Notebook IDE | Modern Jupyter environment with better UX than classic notebook. Native Plotly support with anywidget integration. |
| **anywidget** | 0.9.13+ | Interactive widgets in Jupyter | Required for Plotly FigureWidget interactivity in JupyterLab >= 7.0. |
| **papermill** | 2.6.0+ | Notebook parameterization & execution | Execute notebooks programmatically with different parameters. Critical for separating narrative notebook from analysis scripts. |
| **nbformat** | latest | Notebook file format manipulation | Low-level library for reading/writing .ipynb files programmatically. |

### Data I/O & Format Support
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **pyarrow** | latest | Apache Arrow support | Polars uses Apache Arrow memory model. Enables zero-copy data sharing and efficient I/O for Parquet/Feather formats. |

### Development & Testing
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **pytest** | latest | Testing framework | Standard Python testing library. Test analysis scripts and reusable functions extracted from notebooks. |
| **ruff** | latest | Fast Python linter | 10-100x faster than existing linters. Replaces flake8, black, isort. Modern Python tooling. |
| **pre-commit** | latest | Git hooks for code quality | Automate linting, formatting, and testing before commits. Essential for maintaining code quality in analysis projects. |

### Documentation & Reproducibility
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **quarto** | latest | Scientific publishing | Render Jupyter notebooks to professional reports (HTML, PDF, DOCX). Better than nbconvert for publication-quality output with citations, cross-refs, and themes. |
| **python-dotenv** | latest | Environment variable management | Load API keys, configuration from .env files. Keep secrets out of notebooks. |

## Installation

### Core Installation
```bash
# Create dedicated environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Core data stack (latest versions)
pip install polars numpy scipy

# Visualization
pip install plotly kaleido

# Statistical analysis
pip install scikit-learn statsmodels

# Jupyter ecosystem
pip install jupyterlab anywidget

# Notebook automation
pip install papermill nbformat

# Data I/O
pip install pyarrow

# Publishing
pip install quarto-cli  # Or: brew install quarto
```

### Development Dependencies
```bash
# Testing & code quality
pip install pytest ruff pre-commit

# Set up pre-commit hooks
pre-commit install
```

### Version Pinning (for reproducibility)
```bash
# Freeze exact versions
pip freeze > requirements.txt

# Or use uv for faster dependency resolution (recommended for 2025)
pip install uv
uv pip compile pyproject.toml -o requirements.lock
```

## Alternatives Considered

### DataFrame Libraries
| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| **Polars** | Pandas | Pandas is 3-30x slower. Copy-on-Write (CoW) in pandas 2.0+ helps but doesn't match Polars' Rust performance. Project requirement specifies Polars. |
| **Polars** | DuckDB | DuckDB is excellent for SQL-on-dataframes, but overkill for 188-row dataset. DuckDB shines at scale (>1M rows). Polars provides more Pythonic API for EDA. |
| **Polars** | Modin | Modin offers pandas-compatible API with parallelization, but less mature than Polars in 2025. Smaller community, fewer features. |

### Visualization Libraries
| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| **Plotly** | Matplotlib + Seaborn | Static charts only. Less engaging for Kaggle audience. Plotly's interactivity (zoom, hover, pan) is superior for EDA exploration. |
| **Plotly** | Bokeh | Bokeh is powerful but steeper learning curve. Plotly Express provides simpler API for common charts. Plotly has larger community and better Kaggle integration. |
| **Plotly** | Altair | Altair has elegant grammar-of-graphics API but creates Vega-Lite specs. Slower for large datasets. Plotly better for interactive dashboards. |

### Statistical Libraries
| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| **statsmodels + scikit-learn** | Pingouin | Pingouin provides simpler API for statistical tests but less comprehensive than statsmodels. Smaller ecosystem. statsmodels is standard in econometrics/statistics. |
| **statsmodels** | only scikit-learn | scikit-learn focuses on prediction, not inference. Lacks detailed statistical output (p-values, confidence intervals). statsmodels required for understanding relationships, not just predicting. |

### Notebook Tools
| Recommended | Alternative | Why Not |
|-------------|-------------|---------|
| **papermill** | nbconvert | nbconvert only converts formats, doesn't parameterize/execute. papermill is specifically designed for notebook execution with parameters. |
| **papermill** | Airflow + JupyterOperator | Overkill for single-analysis project. Airflow is production DAG scheduler. papermill is simple library for notebook execution. |
| **Quarto** | Jupyter Book | Jupyter Book is excellent for book-length projects but heavier setup. Quarto is lighter, focused on single-notebook publishing to multiple formats. |

## Polars-Specific Ecosystem Libraries

### Core Polars Features (Built-in)
- **Lazy evaluation** (`pl.scan_csv()`, `.lazy()`) - Query optimization before execution
- **Expression API** - Chainable, composable operations
- **Streaming API** - Process datasets larger than memory
- **Multi-threaded execution** - Automatic parallelization
- **Apache Arrow backend** - Zero-copy interoperability

### Complementary Libraries
| Library | Purpose | When to Use |
|---------|---------|-------------|
| **narwhals** | DataFrame-agnostic API | If you need code that works with both Polars and pandas. Not needed here since we're Polars-only. |
| **pointblank** | Data validation | For production data pipelines. Overkill for single-dataset EDA but good practice for data quality checks. |
| **lets-plot** | Grammar-of-graphics plotting | Alternative to Plotly with ggplot2-like syntax. Not recommended since Plotly is required. |

## Integration Patterns

### Polars + Plotly Integration
**MEDIUM Confidence** - Limited official integration, two approaches:

1. **Convert to pandas (simplest)**
   ```python
   import polars as pl
   import plotly.express as px

   df_polars = pl.read_csv("data.csv")
   df_pandas = df_polars.to_pandas()  # Zero-copy via Arrow
   fig = px.scatter(df_pandas, x="price", y="performance")
   ```

2. **Use Polars for data prep, convert only for plotting**
   ```python
   # Do all heavy lifting in Polars
   result = (
       pl.scan_csv("data.csv")
       .filter(pl.col("price") > 0)
       .group_by("provider")
       .agg(pl.col("performance").mean())
       .collect()
   )

   # Convert only final result for plotting
   fig = px.bar(result.to_pandas(), x="provider", y="performance")
   ```

**Recommendation:** Approach 2 minimizes conversion overhead. Polars handles data manipulation, convert final aggregated results for Plotly.

### Polars + Statistical Libraries

**With scikit-learn:**
```python
import polars as pl
from sklearn.linear_model import LinearRegression

# Prepare data in Polars
X = df.select(pl.numeric().fill_null(0)).to_numpy()
y = df["price"].to_numpy()

# Fit model
model = LinearRegression()
model.fit(X, y)
```

**With statsmodels:**
```python
import polars as pl
import statsmodels.api as sm

# Statsmodels works best with pandas
df_pd = df.to_pandas()
X = sm.add_constant(df_pd[["performance", "speed"]])
model = sm.OLS(df_pd["price"], X).fit()
print(model.summary())  # Full statistical output
```

**Pattern:** Use Polars for data manipulation, convert to NumPy/pandas for modeling. All three libraries (Polars, NumPy, pandas) use Apache Arrow for zero-copy conversion.

## Version-Specific Notes

### Python Version
- **Recommended:** Python 3.11 or 3.12
- **Why:** Best balance of ecosystem stability and performance. Python 3.14 support is emerging (SciPy 1.16.1+) but ecosystem still maturing.
- **Avoid:** Python 3.9 (EOL Oct 2025), Python 3.10 (security fixes only)

### NumPy 2.x Compatibility
- NumPy 2.4.0 (Dec 2025) is recommended
- Breaking changes from 1.x: check `numpy.distutils` removal
- Polars fully compatible with NumPy 2.x

### Polars API Stability
- Polars 1.x is stable, production-ready
- API still evolving but backwards-compatible for core operations
- Check release notes for deprecation warnings

## What NOT to Use

### Deprecated/Legacy
- **pandas** - Replaced by Polars per project requirements
- **matplotlib** - Replaced by Plotly for interactive visualizations
- **seaborn** - Statistical visualization replaced by Plotly
- **orca** - Deprecated Plotly image export (use Kaleido instead)
- **Jupyter Notebook (<7.0)** - Replaced by JupyterLab for modern UX

### Over-Engineering for This Project
- **Dash** - Full web app framework. Overkill for single-notebook analysis. Use Plotly inline in Jupyter.
- **Dask** - Distributed computing. Not needed for 188-row dataset.
- **Apache Spark** - Big data processing. Massive overkill.
- **Airflow/Prefect** - Workflow orchestration. Not needed for single analysis.
- **MLflow** - Experiment tracking. Overkill for EDA without model training.

### Alternative Ecosystems (Not Python)
- **R** - Excellent for statistics (tidyverse, ggplot2) but project requires Python
- **Julia** - High-performance but smaller ecosystem. Not required.

## Environment Configuration

### .python-version
```
3.12
```

### .env.example
```bash
# API Keys (if needed for external data)
# OPENAI_API_KEY=your_key_here

# Data paths
DATA_DIR=data
OUTPUT_DIR=output
```

### pyproject.toml (modern replacement for setup.py)
```toml
[project]
name = "ai-models-eda"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "polars>=1.0",
    "numpy>=2.4",
    "scipy>=1.16",
    "plotly>=5.0",
    "kaleido",
    "scikit-learn>=1.8",
    "statsmodels>=0.14",
    "jupyterlab>=4.0",
    "anywidget>=0.9",
    "papermill>=2.6",
    "pyarrow",
    "python-dotenv",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff",
    "pre-commit",
]

[tool.ruff]
line-length = 100
target-version = "py311"
```

### .gitignore
```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
.DS_Store
output/
*.html
*.pdf
```

## Kaggle-Specific Setup

### Kaggle Notebooks
- Kaggle provides pre-installed Polars, Plotly, scikit-learn
- Verify versions in notebook:
  ```python
  import polars as pl
  import plotly
  import sklearn
  print(f"Polars: {pl.__version__}")
  print(f"Plotly: {plotly.__version__}")
  print(f"scikit-learn: {sklearn.__version__}")
  ```

### Kaggle Dataset Integration
```python
# Kaggle datasets are in /kaggle/input/
import polars as pl
df = pl.read_csv("/kaggle/input/ai-models-benchmark-dataset-2026/data.csv")
```

### Publishing to Kaggle
```bash
# Install Kaggle API
pip install kaggle

# Create dataset
kaggle datasets create -p /path/to/data

# Create notebook (publish from Jupyter)
kaggle notebooks publish -m "EDA of AI Models" -c COMPETITION_NAME
```

## Sources

### Official Documentation (HIGH Confidence)
- Polars: https://pola.rs/ - Official homepage, features, and documentation
- Plotly: https://plotly.com/python/getting-started/ - Official getting started guide
- NumPy News: https://numpy.org/news/ - NumPy 2.4.0 release (Dec 20, 2025)
- SciPy News: https://scipy.org/news/ - SciPy 1.16.1 release (July 27, 2025)
- scikit-learn: https://scikit-learn.org/ - Official documentation
- statsmodels: https://www.statsmodels.org/ - Official documentation

### WebSearch Verification (MEDIUM Confidence)
- "Pandas vs Polars: Why the 2025 Evolution Changes Everything" (Dec 27, 2025)
- "Why Data Teams Are Moving from Pandas to Polars in 2025" (4 months ago)
- "Python for Data Engineering: Polars vs Pandas Performance" (Dec 27, 2025)
- "Polars, DuckDB, and Arrow Are Replacing Pandas" (Nov 27, 2025)
- "Top Python Libraries for Data Science and AI in 2025" (May 28, 2025)
- "Open Source Data Engineering Landscape 2025" (Feb 11, 2025)
- "Data Validation Libraries for Polars (2025 Edition)" (Jun 4, 2025)
- "Hypermodern Data Science Toolbox 2025"
- "Plotly for Data Visualization Guide with Features and Usage"
- "Top 10 Python Data Visualization Libraries in 2025" (Jan 27, 2025)
- "14 Essential Data Visualization Libraries for Python in 2025" (Mar 17, 2025)
- "Top 15 Python Libraries for Data Analytics [2025 updated]" (Jul 23, 2025)
- "Top 31 Python Libraries for Data Science in 2026"
- "The Rise of Notebook-First Data Engineering" (Medium)
- "Automate Jupyter Notebooks with Papermill" (Nov 15, 2024)
- "nbdev – Create delightful software with Jupyter Notebooks"
- "Statistical Analysis Using SciPy" (Medium)

### Community Resources (LOW Confidence - Verify before use)
- "Generating Plotly charts for water data with Python and Polars" (Dec 12, 2025)
- "How to use Polars with Plotly without converting to Pandas?" (StackOverflow)
- "plotly/polars-open-source-app: A Dash app showcase" (GitHub)

## Next Steps

After setting up stack:

1. **Create development environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set up JupyterLab**
   ```bash
   jupyter lab
   ```

3. **Verify installation**
   - Create test notebook importing all libraries
   - Test Polars + Plotly integration pattern
   - Test statistical library imports

4. **Set up code quality tools**
   ```bash
   pre-commit install
   ```

5. **Configure Quarto for publishing**
   ```bash
   quarto install tinytex  # For PDF output
   ```

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Core Stack (Polars, Plotly) | HIGH | Official documentation verified, WebSearch confirms 2025 best practices |
| Statistical Libraries (scikit-learn, statsmodels) | HIGH | Official documentation verified, standard Python statistical stack |
| NumPy/SciPy Versions | HIGH | Official release pages verify latest versions (Dec 2025, July 2025) |
| Jupyter Ecosystem | HIGH | Official Plotly Jupyter integration docs verified |
| Polars + Plotly Integration | MEDIUM | Pattern documented in community, no official integration library |
| Papermill for Automation | MEDIUM | WebSearch confirms usage pattern, official docs exist |
| Quarto for Publishing | MEDIUM | WebSearch confirms 2025 adoption, but Jupyter Book also viable |
| Development Tools (ruff, pre-commit) | MEDIUM | WebSearch confirms 2025 best practices |

**Overall Confidence: HIGH**

All core recommendations verified with official documentation or multiple credible sources from 2025. Statistical and visualization stacks are industry standards. Polars ecosystem is mature and production-ready in 2025.
