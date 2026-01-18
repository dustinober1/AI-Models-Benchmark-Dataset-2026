# Domain Pitfalls

**Domain:** Exploratory Data Analysis & Statistical Predictions (AI Models Benchmark Dataset)
**Researched:** 2025-01-18
**Confidence:** HIGH

## Executive Summary

EDA projects on AI benchmark datasets face unique challenges: small sample sizes (188 models), potential data quality issues in benchmark datasets, and the temptation to over-interpret correlations as causal relationships. This document catalogs critical pitfalls specific to statistical analysis, Polars usage, and narrative structure in data storytelling.

## Critical Pitfalls

Mistakes that cause rewrites, invalid conclusions, or misleading insights.

### Pitfall 1: Overfitting Predictions on Small Samples

**What goes wrong:** With only 188 models, any predictive model (regression, classification) will likely "learn noise" rather than true patterns. Small samples produce deceptively high accuracy (>95%) that doesn't generalize.

**Why it happens:**
- Models memorize training data instead of learning patterns
- Large standard errors make parameter estimates unreliable
- Cross-validation produces large error bars even with proper technique

**Consequences:**
- Inflated performance metrics that collapse on new data
- False confidence in predictions
- Wasted time building complex models on insufficient data

**Prevention:**
- Use **simple models only**: linear regression, decision trees (limited depth)
- Focus on **effect sizes and confidence intervals**, not point predictions
- Apply **strong regularization** (L1/L2 penalties)
- Use **bootstrap resampling** to estimate uncertainty
- Avoid complex models: neural nets, ensembles, deep learning

**Detection:**
- Training accuracy >> validation accuracy (e.g., 95% vs 70%)
- Confidence intervals so wide they span the entire plausible range
- Model performance changes dramatically with different random seeds

**Which phase should address:** Phase 2 (Statistical Analysis). Before building any prediction model, validate that sample size is sufficient. Rule of thumb: 10-20 observations per predictor minimum.

**Sources:**
- [Cross-validation failure: Small sample sizes lead to large error bars](https://www.sciencedirect.com/science/article/abs/pii/S1053811917305311) (798 citations)
- [A Survey on Small Sample Imbalance Problem](https://arxiv.org/html/2504.14800v1) (2025)
- [Unreliable clinical prediction models with small samples](https://www.jclinepi.com/article/S0895-4356(20)31209-9/fulltext) (138 citations)

---

### Pitfall 2: Data Leakage in Cross-Validation

**What goes wrong:** Performing feature selection, preprocessing, or hyperparameter tuning on the full dataset before cross-validation. This leaks test set information into training, producing optimistic but biased performance estimates.

**Why it happens:**
- Treating feature selection as a preprocessing step rather than part of model training
- Tuning hyperparameters using the same CV loop used for evaluation
- Scaling/normalizing data before train-test split

**Consequences:**
- Reported model accuracy is 5-20% higher than true performance
- Features appear predictive when they're not
- Hyperparameters tuned to exploit statistical quirks of the sample
- Production models fail dramatically

**Prevention:**
- **ALL** data preparation (feature selection, scaling, imputation) must happen **inside** each CV fold
- Use **nested cross-validation** when tuning hyperparameters:
  - Inner loop: hyperparameter search
  - Outer loop: performance estimation
- For final model: Apply the winning procedure to the full dataset
- Remember: CV tests a **procedure**, not a single model instance

**Detection:**
- Performance seems "too good to be true"
- Different random seeds produce wildly different feature importances
- Model fails in production despite strong CV scores

**Which phase should address:** Phase 2 (Statistical Analysis). Any time you use cross-validation for model evaluation, verify no preprocessing happens before the CV loop.

**Sources:**
- [How cross-validation can go wrong and how to fix it](https://towardsdatascience.com/how-cross-validation-can-go-wrong-and-how-to-fix-it-feature-selection-use-case-with-sample-code-abf928be9080/) (Towards Data Science)
- [Avoiding Data Leakage in Cross-Validation](https://medium.com/@silva.f.francis/avoiding-data-leakage-in-cross-validation-ba344d4d55c0) (Medium)
- [scikit-learn common pitfalls documentation](https://scikit-learn.org/stable/common_pitfalls.html)

---

### Pitfall 3: Correlation-Causation Fallacies in Narrative

**What goes wrong:** Presenting correlations (e.g., "higher-priced models score better on benchmarks") as if they imply causal relationships ("pricing causes better performance"). This misleads readers about the nature of AI model development.

**Why it happens:**
- Narrative pressure to tell a coherent "story"
- Confusing association with mechanism
- Ignoring confounding variables (e.g., model size, provider resources)
- Stakeholder demand for actionable insights

**Consequences:**
- Misguided business decisions (e.g., "raise prices to improve quality")
- Loss of credibility when correlations are spurious
- Oversimplified understanding of AI ecosystem dynamics
- Potential publication issues or retractions

**Prevention:**
- Always use precise language: "associated with," "correlated with," "tends to"
- Explicitly state: "Correlation ≠ causation" in relevant sections
- Identify and discuss **confounding variables**:
  - Model size (parameter count)
  - Training data quality/size
  - Provider R&D budget
  - Release date (temporal effects)
- Use **domain knowledge** to assess plausibility of causal claims
- Consider **DAGs (Directed Acyclic Graphs)** to map causal hypotheses

**Detection:**
- Using causal language ("causes," "leads to," "drives") for observational data
- Not discussing alternative explanations
- Presenting single correlational finding as definitive proof

**Which phase should address:** Phase 3 (Narrative Development). Every insight should be scrutinized for causal language before being included in the narrative.

**Sources:**
- [Correlation vs Causation: A Crucial Distinction in Data Analysis](https://www.kaggle.com/discussions/getting-started/560357) (Kaggle)
- [Four common pitfalls to avoid in Exploratory Data Analysis](https://towardsdatascience.com/four-common-pitfalls-to-avoid-in-exploratory-data-analysis-85d822dd5e34/) (Towards Data Science)

---

### Pitfall 4: P-Hacking and Multiple Hypothesis Testing

**What goes wrong:** Testing dozens of hypotheses (e.g., "price correlates with X," "speed correlates with Y," "provider A differs from B"...) without correcting for multiple comparisons. This inflates Type I error rates—finding "significant" results that are actually false positives.

**Why it happens:**
- Exploratory analysis naturally involves many tests
- Pressure to find "publishable" insights
- Lack of awareness about multiple testing correction
- Treating p < 0.05 as a threshold rather than a continuum

**Consequences:**
- "Significant" findings that don't replicate
- 5% of tests are false positives by definition (at α = 0.05)
- With 20 tests, ~1 false positive is expected
- Loss of credibility when findings fail to reproduce

**Prevention:**
- **Pre-register** hypotheses when possible (document what you'll test before testing)
- Apply **multiple testing corrections**:
  - Bonferroni: α_corrected = α / n_tests (conservative)
  - Benjamini-Hochberg FDR (less conservative, controls false discovery rate)
- Focus on **effect sizes and confidence intervals** over p-values
- Use **Bayesian methods** when appropriate (provide posterior probabilities)
- Be transparent: Report all tests conducted, not just significant ones

**Detection:**
- Many significance tests reported without correction
- "Marginally significant" (p = 0.053) treated as meaningful
- Only significant results discussed, null findings omitted

**Which phase should address:** Phase 2 (Statistical Analysis). Before conducting any hypothesis tests, plan how many tests you'll run and apply appropriate corrections.

**Sources:**
- [P-Hacking: How to (Not) Manipulate the P-Value](https://www.datacamp.com/tutorial/p-hacking) (DataCamp, 2025)
- [Big little lies: p-hacking compendium and simulation](https://royalsocietypublishing.org/rsos/article/10/2/220346/92017/Big-little-lies-a-compendium-and-simulation-of-p) (180 citations)
- [What is P-Hacking in Data Science](https://herovired.com/learning-hub/blogs/p-hacking-in-data-science-and-machine-learning-how-to-avoid-it) (2024)

---

### Pitfall 5: Benchmark Dataset Quality Issues

**What goes wrong:** AI benchmark datasets frequently contain errors, biases, and contamination. MMLU (a common LLM benchmark) has ~6.5% errors. Many models "see" benchmark questions during training (data contamination), inflating scores.

**Why it happens:**
- Human annotation is inherently error-prone
- Datasets are static, rarely updated after errors found
- Training data (Common Crawl) includes benchmark questions
- Lack of standardized quality control

**Consequences:**
- Analyzing noisy/error-laden data produces unreliable insights
- Overfitting to benchmark quirks rather than true patterns
- Comparisons between models are unfair if benchmarks are contaminated
- Narrative may emphasize "findings" that are actually dataset artifacts

**Prevention:**
- **Assume data quality issues exist**; document this limitation
- Perform **sanity checks**:
  - Spot-check outlier values for data entry errors
  - Check for impossible values (e.g., scores > 100%, negative prices)
  - Verify distributions make sense
- Look for **data contamination indicators**:
  - Suspiciously high benchmark scores (possible memorization)
  - Performance clustered around round numbers (95%, 90%)
- Use **qualitative analysis**: Manually examine a subset of data
- **Acknowledge limitations** in narrative; don't overstate conclusions

**Detection:**
- Values outside plausible ranges (e.g., benchmark score > 100%)
- Duplicate rows with different values
- Inconsistent formats (e.g., price as "$100/month" vs "100")
- High correlation between unrelated variables (possible batch effects)

**Which phase should address:** Phase 1 (Data Loading & Quality Assessment). Always begin EDA with systematic quality checks before any analysis.

**Sources:**
- [Quality issues in LLM Benchmark datasets](https://medium.com/@vbsowmya/quality-issues-in-llm-benchmark-datasets-324cc34e5511) (Medium, 2025)
- [Squashing 'Fantastic Bugs': Researchers Look to Fix Flaws in AI Benchmarks](https://hai.stanford.edu/news/squashing-fantastic-bugs-researchers-look-to-fix-flaws-in-ai-benchmarks) (Stanford HAI)
- [9 Hidden Biases & Limits of AI Benchmarks](https://www.chatbench.org/what-are-the-limitations-and-potential-biases-of-using-ai-benchmarks-to-compare-ai-framework-performance/) (ChatBench, 2025)

---

### Pitfall 6: Narrative Bias and Storytelling Traps

**What goes wrong:** The pressure to tell a compelling story leads to cherry-picking data, ignoring contradictory evidence, and oversimplifying complex findings. Narrative bias causes analysts to unconsciously shape conclusions to fit a predetermined storyline.

**Why it happens:**
- Humans are wired for coherent narratives, not nuanced probabilities
- Stakeholders demand clear "takeaways"
- "Positive findings" get more attention
- Cognitive dissonance: Inconvenient data is downplayed

**Consequences:**
- Misleading conclusions that don't reflect full picture
- Loss of trust when oversimplified narratives prove wrong
- Poor decisions based on incomplete understanding
- Kaggle notebook may appear impressive but be scientifically unsound

**Prevention:**
- **Pre-commit** to analysis plan before seeing results
- Report **null findings** alongside significant ones
- Use **multi-metric dashboards** rather than single numbers
- Include **uncertainty** in all visualizations (error bars, confidence intervals)
- Have **peer review** of narrative by someone who didn't do the analysis
- Explicitly discuss **limitations and alternative explanations**
- Ask: "What would I conclude if the opposite pattern appeared?"

**Detection:**
- Narrative only presents "clean" stories, ignores complexity
- Contradictory evidence omitted from discussion
- Confidence in claims exceeds what data supports
- Visualizations emphasize dramatic patterns but don't show noise

**Which phase should address:** Phase 3 (Narrative Development). Every claim should be cross-checked against the full distribution of evidence, not just supporting examples.

**Sources:**
- [How Not to Mislead with Your Data-Driven Story](https://towardsdatascience.com/how-not-to-mislead-with-your-data-driven-story/) (Towards Data Science, 2025)
- [Narrative Biases: When Storytelling HURTS User Experience](https://www.nngroup.com/articles/narrative-biases/) (Nielsen Norman Group)
- [The Illusion Of Insight: When Data Tells Stories Instead of the Truth](https://www.forbes.com/councils/forbesbusinesscouncil/2025/08/01/the-illusion-of-insight-when-data-tells-stories-instead-of-the-truth/) (Forbes, 2025)

---

## Moderate Pitfalls

Mistakes that cause delays, technical debt, or weakened analysis quality.

### Pitfall 7: Polars Lazy Evaluation Misunderstanding

**What goes wrong:** Converting LazyFrames to DataFrames too early, not understanding query plan optimization, or using Python operations incompatible with lazy evaluation. This negates Polars' performance benefits and can cause unexpected behavior.

**Why it happens:**
- Polars lazy evaluation is different from pandas eager execution
- Developers materialize intermediate results unnecessarily
- Not all Python operations work in lazy context
- Unclear when `.collect()` is needed

**Consequences:**
- Slower code (negating Polars' speed advantage)
- Memory issues from unnecessary materialization
- Confusion about when operations actually execute
- Debugging difficulty due to deferred execution

**Prevention:**
- Keep data **lazy as long as possible**; only `.collect()` when needed
- Understand **query plan optimization**: Predicate/projection pushdown only works in lazy mode
- Use Polars expression API, not Python loops/comprehensions
- Check **optimized query plan** with `.explain()` before `.collect()`
- Avoid unnecessary type casting
- Remember: LazyFrames don't contain data, they contain **instructions**

**Detection:**
- Frequent `.collect()` calls in code
- Converting between LazyFrame and DataFrame multiple times
- Using `for` loops over data instead of Polars operations
- Performance similar to pandas despite using Polars

**Which phase should address:** Phase 1 (Data Loading & Cleaning). Establish Polars best practices early, especially for large datasets (188 models is small, but code patterns matter for future work).

**Sources:**
- [How to Work With Polars LazyFrames](https://realpython.com/polars-lazyframe/) (Real Python)
- [Ultimate guide to the polars library in python](https://deepnote.com/blog/ultimate-guide-to-the-polars-library-in-python) (Deepnote)
- [Polars vs Pandas: Why 2025 Is the Year...](https://medium.com/@gowthamimm196/polars-vs-pandas-why-2025-is-the-year-python-data-scientists-must-learn-this-game-changing-library-bd367a8e79f4) (Medium, 2025)

---

### Pitfall 8: Misleading Visualizations

**What goes wrong:** Using wrong chart types, manipulating axis scales, or using poor color choices that hide patterns instead of revealing them. Bad visualizations distort data and mislead readers.

**Why it happens:**
- Unconscious desire to make patterns appear stronger
- Lack of training in data visualization best practices
- Using default chart types without considering data distribution
- Not accounting for colorblind accessibility

**Consequences:**
- Readers misinterpret data
- Patterns appear stronger/weaker than they are
- Loss of credibility when visualizations are questioned
- Key insights hidden by poor visualization choices

**Prevention:**
- **Match chart type to question**:
  - Distributions → histograms, KDEs, box plots
  - Relationships → scatter plots (with jitter for discrete data)
  - Comparisons → bar charts (horizontal for long labels)
  - Time series → line charts
- **Always start y-axis at zero** for bar charts (not always for line charts)
- **Avoid pie charts** (hard to compare angles)
- Use **colorblind-safe palettes** (viridis, plasma, ColorBrewer)
- Include **uncertainty** (error bars, confidence bands) where appropriate
- **Label axes clearly** with units

**Detection:**
- Visualizations require "explaining" to be understood
- Multiple chart types trying to show the same relationship
- Y-axis doesn't start at zero for bar charts
- More than 5-7 colors in a single plot

**Which phase should address:** Phase 3 (Narrative Development). Every visualization should be reviewed for clarity and accuracy before inclusion.

**Sources:**
- [10 Common EDA Mistakes: Using the Wrong Visualization](https://medium.com/@vamarnath/10-common-eda-mistakes-that-can-break-your-data-analysis-and-how-to-avoid-them-251fcbbddd5c) (Medium)
- [Bad Data Visualization: 5 Examples](https://online.hbs.edu/blog/post/bad-data-visualization) (Harvard Business School, 2021)
- [Four common pitfalls to avoid in EDA: Bad visualizations](https://towardsdatascience.com/four-common-pitfalls-to-avoid-in-exploratory-data-analysis-85d822dd5e34/) (Towards Data Science)

---

### Pitfall 9: Shallow Insights ("Stating the Obvious")

**What goes wrong:** Presenting trivial observations as insights (e.g., "larger models are more expensive"). Stakeholders respond with "so what?" because findings don't add value or aren't actionable.

**Why it happens:**
- Focusing on what's easy to measure rather than what's important
- Not understanding business context deeply enough
- Pressure to produce "quick wins"
- Lack of domain expertise

**Consequences:**
- Analysis dismissed as uninteresting
- Stakeholders disengage from findings
- Reputation damage ("analysis doesn't tell us anything new")
- Wasted effort on trivial discoveries

**Prevention:**
- **Ask "so what?" for every insight** before including it in narrative
- **Focus on counterintuitive findings** (e.g., "premium pricing doesn't correlate with performance")
- **Segment analysis**: Find patterns within subgroups (e.g., by provider, by model type)
- **Combine multiple variables**: Look for interactions (e.g., price-performance varies by model size)
- **Consult stakeholders early**: What questions do they actually care about?
- **Prioritize actionability**: "What should someone do differently based on this?"

**Detection:**
- Stakeholder feedback: "We already knew this"
- Insights are universally true across all subsets (no nuance)
- Analysis doesn't lead to recommendations or next steps

**Which phase should address:** Phase 3 (Narrative Development). Every "insight" should be stress-tested against the "so what?" question before inclusion.

**Sources:**
- [Four common pitfalls to avoid in EDA: Shallow insights](https://towardsdatascience.com/four-common-pitfalls-to-avoid-in-exploratory-data-analysis-85d822dd5e34/) (Towards Data Science)

---

### Pitfall 10: Unclear Business Problem

**What goes wrong:** Analysis proceeds without clarity on what business question it's answering. This leads to investigating everything superficially rather than deeply answering a specific question.

**Why it happens:**
- Pressure to start "doing data science" before defining objectives
- Stakeholders can't articulate what they want
- Analysts aren't business-savvy, managers aren't data experts
- Available data doesn't align with key questions

**Consequences:**
- Analysis becomes unfocused exploration
- Final deliverable doesn't address stakeholder needs
- Time wasted on irrelevant analyses
- Difficulty converting findings to recommendations

**Prevention:**
- **Define 1-3 core questions** analysis will answer before starting
- **Get stakeholder feedback early** on proposed questions
- **Align questions with available data** (if data can't answer question, acknowledge this)
- **Treat EDA as iterative**: Cycle back to stakeholders with preliminary findings
- **Write down questions** and refer back to them throughout analysis

**Example questions for this project:**
1. Which providers offer the best price-performance ratio?
2. How has model pricing evolved over time, and what predicts price changes?
3. Are there "undiscovered gems"—models with strong performance but low pricing?

**Detection:**
- Analysis feels like fishing expedition rather than investigation
- Final notebook lacks clear narrative arc
- Stakeholders ask "but what does this mean for us?"

**Which phase should address:** Phase 0 (Project Setup). Before writing any analysis code, explicitly define the questions you're answering.

**Sources:**
- [Four common pitfalls to avoid in EDA: Unclear business problems](https://towardsdatascience.com/four-common-pitfalls-to-avoid-in-exploratory-data-analysis-85d822dd5e34/) (Towards Data Science)

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

### Pitfall 11: Assuming Normality

**What goes wrong:** Assuming data follows normal distribution without checking. Many statistical tests (t-tests, ANOVA) assume normality, but real-world data (prices, benchmark scores) is often skewed or multimodal.

**Prevention:**
- Always plot **histograms/density plots** before applying parametric tests
- Use **Shapiro-Wilk test** or **Q-Q plots** to assess normality
- If non-normal: Use **non-parametric alternatives** (Mann-Whitney U, Kruskal-Wallis)
- Consider **transformations** (log, Box-Cox) if appropriate

**Detection:**
- Skewed distributions (long tails)
- Multiple modes (peaks) in distribution
- Outliers driving mean away from median

---

### Pitfall 12: Outlier Removal Without Investigation

**What goes wrong:** Automatically removing outliers (e.g., > 3 SD from mean) without understanding why they're outliers. Some outliers are genuine insights; others are data errors.

**Prevention:**
- **Investigate each outlier** individually
- Ask: Is this a data error or a real phenomenon?
- If data error: Correct or remove
- If real: Report separately, discuss implications
- Consider **robust statistics** (median, MAD) instead of removing outliers

**Detection:**
- Outliers appear in multiple variables for same observation (likely data error)
- Outliers change dramatically with small dataset changes (likely influential points)

---

### Pitfall 13: Not Reproducible Analysis

**What goes wrong:** Analysis can't be reproduced because random seeds aren't set, data isn't versioned, or environment isn't specified.

**Prevention:**
- Set **random seeds** for all stochastic operations (`np.random.seed()`)
- Use **environment management** (conda, venv, requirements.txt)
- **Version control** data and code
- Document **Polars version** (API changes between versions)
- Use **Jupyter notebooks** for narrative but separate `.py` scripts for analysis

**Detection:**
- Re-running notebook produces different results
- Code fails on different machine (missing dependencies)

---

## Phase-Specific Warnings

| Phase | Critical Pitfall | Mitigation |
|-------|-----------------|------------|
| **Phase 0: Setup** | Unclear business problem | Explicitly document 1-3 core questions before starting |
| **Phase 1: Data Quality** | Benchmark dataset errors | Systematic sanity checks, spot-validation, acknowledge limitations |
| **Phase 1: Data Loading** | Polars lazy evaluation misuse | Keep data lazy, use `.explain()`, minimize `.collect()` |
| **Phase 2: Statistics** | Overfitting on small sample | Use simple models, bootstrap resampling, report uncertainty |
| **Phase 2: Statistics** | Data leakage in CV | All preprocessing inside CV loop, use nested CV |
| **Phase 2: Statistics** | P-hacking | Pre-register hypotheses, apply multiple testing corrections |
| **Phase 3: Narrative** | Correlation-causation fallacy | Use precise language, discuss confounders, acknowledge limits |
| **Phase 3: Narrative** | Narrative bias | Report null findings, include uncertainty, peer review |
| **Phase 3: Visualizations** | Misleading charts | Match chart to question, start y-axis at zero, use colorblind palettes |
| **All Phases** | Shallow insights | Ask "so what?", focus on counterintuitive findings, segment analysis |

---

## Project-Specific Risk Factors

**This dataset (188 AI models) has specific vulnerabilities:**

1. **Sample size (188) → High risk of overfitting**
   - Any model with > 10 predictors is suspect
   - Predictive models should be simple (linear regression, shallow trees)
   - Focus on description, not prediction

2. **Benchmark scores → Potential contamination**
   - Models may have "seen" test questions during training
   - Suspicious: Many models clustered at 90%, 95%, 99%
   - Check for models scoring identically (possible copy-paste errors)

3. **Pricing data → Provider strategic behavior**
   - Pricing may not reflect cost (could be strategic, competitive)
   - Different pricing models (per-token, per-month, enterprise)
   - Temporal effects: Pricing may have changed (inflation, competition)

4. **Provider diversity → Imbalanced groups**
   - Some providers have 10+ models, others have 1-2
   - Group comparisons may be underpowered for smaller providers
   - Consider grouping providers by size (large vs small)

5. **Fast-moving field → Temporal confounding**
   - Newer models likely better (technology advance)
   - But also more expensive (inflation, premium pricing)
   - Release date confounds many relationships

---

## Quick Reference Checklist

Before submitting analysis:

### Statistical Validity
- [ ] All preprocessing inside CV loops (no data leakage)
- [ ] Multiple testing corrections applied (if > 3 tests)
- [ ] Confidence intervals reported (not just p-values)
- [ ] Effect sizes interpreted (not just significance)
- [ ] Model complexity appropriate for sample size

### Data Quality
- [ ] Sanity checks performed (outliers investigated)
- [ ] Missing data documented
- [ ] Duplicates checked for
- [ ] Benchmark scores verified (spot-check outliers)

### Narrative Integrity
- [ ] Causal language avoided for observational data
- [ ] Limitations section included
- [ ] Null findings reported (if any)
- [ ] Alternative explanations discussed
- [ ] "So what?" answered for each insight

### Visualization Quality
- [ ] Chart types match questions
- [ ] Axes labeled with units
- [ ] Uncertainty shown where appropriate
- [ ] Colorblind-safe palette used
- [ ] No distorted scales (y-axis at 0 for bar charts)

### Reproducibility
- [ ] Random seeds set
- [ ] Dependencies documented
- [ ] Data versioned
- [ ] Code commented

---

## Sources Summary

| Topic | Key Sources | Confidence |
|-------|-------------|------------|
| Small sample overfitting | ScienceDirect (798 citations), arXiv 2025, J Clin Epidemiol | HIGH |
| Data leakage in CV | Towards Data Science, scikit-learn docs, Medium | HIGH |
| Correlation-causation | Kaggle discussions, Towards Data Science | HIGH |
| P-hacking | DataCamp 2025, Royal Society (180 citations), HeroVired 2024 | HIGH |
| Benchmark data quality | Medium 2025, Stanford HAI, ChatBench 2025 | HIGH |
| Polars lazy evaluation | Real Python, Deepnote, Medium 2025 | HIGH |
| Misleading visualizations | Medium, HBS, Towards Data Science | HIGH |
| Narrative bias | Towards Data Science 2025, NNGroup, Forbes 2025 | MEDIUM |
| Shallow insights | Towards Data Science | MEDIUM |

**Overall Research Confidence: HIGH**

All critical findings are supported by multiple authoritative sources (peer-reviewed papers, official documentation, reputable data science publications). No critical claims rely solely on WebSearch results without verification.
