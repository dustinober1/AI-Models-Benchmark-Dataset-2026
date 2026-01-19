---
phase: 03-interactive-visualizations
plan: 01
subsystem: visualization
tags: [plotly, interactive-visualization, distribution-analysis, correlation-heatmap, box-plots, outlier-detection]

# Dependency graph
requires:
  - phase: 02-statistical-analysis
    provides: [deduplicated dataset, correlation matrix, Pareto frontier data, provider clusters]
provides:
  - Interactive Plotly visualizations for distributions, correlations, and outliers
  - Reusable visualization utilities in src/visualize.py
  - Standalone HTML files with hover, zoom, and pan capabilities
affects: [03-02, narrative-creation, kaggle-publication]

# Tech tracking
tech-stack:
  added: [plotly>=6.0.0, narwhals>=1.15.1]
  patterns: [plotly.graph_objects for figures, CDN-based standalone HTML, pyarrow-avoidance pattern]

key-files:
  created: [src/visualize.py, scripts/13_distribution_viz.py, reports/figures/interactive_*.html (13 files)]
  modified: [pyproject.toml, poetry.lock]

key-decisions:
  - "Use plotly.graph_objects directly instead of plotly.express to avoid pyarrow dependency"
  - "Save figures as standalone HTML files with Plotly CDN for easy sharing"
  - "Apply consistent theme (plotly_white) across all visualizations"

patterns-established:
  - "Polars-to-NumPy conversion pattern: cast to Float64 before to_numpy() for numerical operations"
  - "String-to-numeric casting: Speed and Latency columns stored as strings, require Float64 casting"
  - "Intelligence-specific filtering: Filter to n=181 models with valid intelligence_index for IQ-based plots"

# Metrics
duration: 2min
completed: 2026-01-19
---

# Phase 3 Plan 1: Interactive Distribution Visualizations Summary

**13 interactive Plotly visualizations with hover tooltips, zoom, and pan for exploring AI model distributions, correlations, and provider-based outliers**

## Performance

- **Duration:** 2 min
- **Started:** 2026-01-19T00:42:05Z
- **Completed:** 2026-01-19T00:44:46Z
- **Tasks:** 2
- **Files modified:** 17

## Accomplishments

- **Created Plotly visualization utilities** (src/visualize.py) with 3 reusable functions for histograms, box plots, and correlation heatmaps
- **Generated 13 interactive visualizations** (5 histograms + 5 box plots + 1 heatmap + 2 combined dashboards) as standalone HTML files
- **VIZ-01 satisfied:** Distribution histograms for intelligence, price, speed, latency, and context window with hover statistics
- **VIZ-02 satisfied:** Box plots segmented by Provider with jitter for individual model outlier detection
- **VIZ-04 satisfied:** Interactive correlation heatmap with RdBu divergent color scale and annotated coefficients
- **Avoided pyarrow dependency:** Used plotly.graph_objects directly to prevent ModuleNotFoundError with df.to_pandas()

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Plotly dependency and create visualization utilities** - `e613e6a` (feat)
2. **Task 2: Execute distribution visualization pipeline** - `57c270c` (feat)

**Plan metadata:** [to be added after SUMMARY/STATE commit]

## Files Created/Modified

### Created
- `src/visualize.py` - Plotly visualization utilities (3 functions: create_distribution_histogram, create_box_plot, create_correlation_heatmap)
- `scripts/13_distribution_viz.py` - Interactive visualization pipeline (13 visualizations, 687 lines)
- `reports/figures/interactive_intelligence_histogram.html` - Intelligence Index distribution (n=181)
- `reports/figures/interactive_price_histogram.html` - Price distribution (n=187)
- `reports/figures/interactive_speed_histogram.html` - Speed distribution (n=187)
- `reports/figures/interactive_latency_histogram.html` - Latency distribution (n=187)
- `reports/figures/interactive_context_window_histogram.html` - Context Window distribution (n=187)
- `reports/figures/interactive_intelligence_box_plot.html` - Intelligence by Provider (n=181)
- `reports/figures/interactive_price_box_plot.html` - Price by Provider (n=187)
- `reports/figures/interactive_speed_box_plot.html` - Speed by Provider (n=187)
- `reports/figures/interactive_latency_box_plot.html` - Latency by Provider (n=187)
- `reports/figures/interactive_context_window_box_plot.html` - Context Window by Provider (n=187)
- `reports/figures/interactive_correlation_heatmap.html` - 5x5 Spearman correlation matrix
- `reports/figures/interactive_distributions.html` - Combined histogram dashboard (5 subplots)
- `reports/figures/interactive_box_plots.html` - Combined box plot dashboard (5 subplots)

### Modified
- `pyproject.toml` - Added plotly>=6.0.0 dependency
- `poetry.lock` - Locked plotly 6.5.2 and narwhals 2.15.0

## Decisions Made

**1. Use plotly.graph_objects directly instead of plotly.express**
- **Rationale:** plotly.express with df.to_pandas() requires pyarrow, which is not installed. Using plotly.graph_objects directly avoids this dependency.
- **Impact:** All visualization functions use go.Figure() and add_trace() instead of px.histogram() and px.box()

**2. Save figures as standalone HTML with Plotly CDN**
- **Rationale:** Standalone HTML files are easy to share and open in any browser without Python/Plotly installation. CDN loading reduces file size.
- **Impact:** All figures use include_plotlyjs="cdn" and full_html=True

**3. Filter intelligence-specific plots to n=181 models**
- **Rationale:** 6 models have null intelligence_index scores. Intelligence-specific analyses should exclude these for accuracy.
- **Impact:** Intelligence histogram and box plot show n=181, other plots show n=187

**4. Cast Speed and Latency columns to Float64**
- **Rationale:** These columns are stored as String type in the dataset, requiring numeric casting for visualization.
- **Impact:** Added cast_string_to_numeric() helper function in pipeline script

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed pyarrow dependency issue in visualization functions**
- **Found during:** Task 1 (create_distribution_histogram function)
- **Issue:** df.to_pandas() in plotly.express requires pyarrow, which is not installed. ModuleNotFoundError: No module named 'pyarrow'
- **Fix:** Rewrote create_distribution_histogram() and create_box_plot() to use plotly.graph_objects directly. Extract data with df[column].drop_nulls().cast(pl.Float64).to_numpy() instead of df.to_pandas()
- **Files modified:** src/visualize.py (lines 69-254)
- **Verification:** Pipeline runs successfully, all 13 visualizations generated
- **Committed in:** 57c270c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Auto-fix was necessary for correct operation. No scope creep.

## Issues Encountered

**1. pyarrow ModuleNotFoundError with plotly.express**
- **Problem:** plotly.express functions (px.histogram, px.box) use df.to_pandas() internally, which requires pyarrow. Error: "ModuleNotFoundError: No module named 'pyarrow'"
- **Solution:** Switched from plotly.express to plotly.graph_objects, extracting data as NumPy arrays via df[column].cast(pl.Float64).to_numpy()
- **Outcome:** All visualizations generated successfully without adding pyarrow dependency

## User Setup Required

None - no external service configuration required.

## Verification Results

All verification criteria from plan satisfied:

1. **Interactive distributions with hover, zoom, pan:** All 13 HTML files include Plotly interactivity
2. **Correlation heatmap with hierarchical clustering:** 5x5 Spearman matrix with RdBu color scale
3. **Sample sizes in titles:** Intelligence plots show n=181, others show n=187
4. **Summary statistics in hover:** Mean, median, min, max displayed in histogram hover
5. **Box plots show Provider segmentation:** X-axis shows Creator, individual models visible via jitter
6. **Zoom/pan functional:** Plotly toolbar includes zoom, pan, reset, and lasso tools
7. **Correlation hover:** Shows variable pair and correlation coefficient (hovertemplate = "%{y} vs %{x}: %{z:.3f}")
8. **Consistent theme:** All figures use plotly_white template with 12pt font, 14pt titles
9. **Standalone HTML files:** All visualizations saved as full HTML with CDN loading

## Success Criteria Met

- [x] Plotly>=6.0.0 added to pyproject.toml and installed via poetry (version 6.5.2)
- [x] src/visualize.py created with 3 reusable visualization functions
- [x] 5 interactive distribution histograms generated (intelligence, price, speed, latency, context_window)
- [x] 5 interactive box plots generated for outlier detection, segmented by Provider
- [x] 1 interactive correlation heatmap generated with hierarchical clustering
- [x] All 11 individual figures saved as standalone HTML files with hover tooltips, zoom, and pan
- [x] Combined dashboards created: histograms (interactive_distributions.html) and box plots (interactive_box_plots.html)
- [x] VIZ-01 (distribution histograms), VIZ-02 (box plots with provider segmentation), and VIZ-04 (correlation heatmap) satisfied

## Next Phase Readiness

**Ready for Phase 3 Plan 2 (if applicable):**
- Visualization utilities available for reuse in future plans
- Pattern established for avoiding pyarrow dependency
- HTML files can be embedded in Jupyter notebooks or Kaggle kernels

**Ready for narrative creation (Phase 4):**
- Interactive visualizations available for narrative-driven storytelling
- Distribution insights (VIZ-01) can support sections on model characteristics
- Outlier detection (VIZ-02) can support provider analysis sections
- Correlation heatmap (VIZ-04) can support relationship discussions

**No blockers or concerns.**

---
*Phase: 03-interactive-visualizations*
*Completed: 2026-01-19*
