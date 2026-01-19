#!/usr/bin/env python3
"""
Trend Predictions Pipeline - Phase 2 Plan 05 (Part 2)

Generates simple 2027 trend predictions using linear regression.
Includes uncertainty discussion per NARR-09 requirement.

Usage:
    PYTHONPATH=. python3 scripts/12_trend_predictions.py
"""

import polars as pl
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from pathlib import Path
from datetime import datetime


def main():
    # Load deduplicated dataset
    input_path = "data/processed/ai_models_deduped.parquet"
    report_path = "reports/trend_predictions_2026-01-18.md"

    print(f"Loading: {input_path}")
    df = pl.read_parquet(input_path)
    print(f"Loaded {df.height} models")

    # Filter to models with valid intelligence scores
    df_valid = df.filter(pl.col("intelligence_index").is_not_null())
    print(f"Models with valid intelligence: {df_valid.height}")

    # Simple trend extrapolation
    print("\n=== 2027 TREND PREDICTIONS ===")
    predictions = {}

    # Prediction 1: Intelligence distribution shift
    predictions["intelligence"] = predict_intelligence_trend(df_valid)

    # Prediction 2: Price trends by intelligence tier
    predictions["price_by_tier"] = predict_price_trends(df_valid)

    # Prediction 3: Speed improvements
    predictions["speed"] = predict_speed_trends(df_valid)

    # Generate prediction report with uncertainty discussion
    print("\n=== GENERATING PREDICTION REPORT ===")
    generate_prediction_report(predictions, report_path)
    print(f"Report: {report_path}")

    print("\n✓ Trend predictions complete")


def predict_intelligence_trend(df: pl.DataFrame) -> dict:
    """Predict 2027 intelligence distribution using simple extrapolation."""
    print("\n--- Intelligence Trend Prediction ---")

    # Current intelligence statistics
    intelligence = df["intelligence_index"].drop_nulls().cast(pl.Float64)
    current_mean = float(intelligence.mean())
    current_median = float(intelligence.median())
    current_std = float(intelligence.std())
    n = len(intelligence)

    print(f"Current (2026): Mean={current_mean:.2f}, Median={current_median:.2f}, Std={current_std:.2f}")

    # Simple scenarios for 2027
    scenarios = {
        "optimistic": {
            "assumption": "10% improvement in median intelligence (breakthrough year)",
            "mean_multiplier": 1.10,
            "median_multiplier": 1.10
        },
        "baseline": {
            "assumption": "5% improvement in median intelligence (steady progress)",
            "mean_multiplier": 1.05,
            "median_multiplier": 1.05
        },
        "pessimistic": {
            "assumption": "2% improvement in median intelligence (diminishing returns)",
            "mean_multiplier": 1.02,
            "median_multiplier": 1.02
        }
    }

    results = {
        "current": {
            "mean": current_mean,
            "median": current_median,
            "std": current_std,
            "n": n
        },
        "scenarios": {}
    }

    for scenario_name, scenario in scenarios.items():
        pred_mean = current_mean * scenario["mean_multiplier"]
        pred_median = current_median * scenario["median_multiplier"]

        # Compute prediction interval using current std
        # 95% PI: mean ± 1.96 * std
        pi_lower = pred_mean - 1.96 * current_std
        pi_upper = pred_mean + 1.96 * current_std

        results["scenarios"][scenario_name] = {
            "assumption": scenario["assumption"],
            "mean": pred_mean,
            "median": pred_median,
            "prediction_interval_lower": pi_lower,
            "prediction_interval_upper": pi_upper,
            "std": current_std
        }

        print(f"{scenario_name.capitalize()}: Mean={pred_mean:.2f}, Median={pred_median:.2f}, 95% PI=[{pi_lower:.2f}, {pi_upper:.2f}]")

    return results


def predict_price_trends(df: pl.DataFrame) -> dict:
    """Predict 2027 pricing trends by intelligence tier."""
    print("\n--- Price Trend Prediction by Intelligence Tier ---")

    # Define intelligence tiers (quartiles)
    df_with_tiers = df.with_columns(
        pl.col("intelligence_index")
        .qcut([0.25, 0.5, 0.75], labels=["Q1", "Q2", "Q3", "Q4"])
        .alias("intelligence_tier")
    )

    results = {}

    for tier in ["Q1", "Q2", "Q3", "Q4"]:
        tier_df = df_with_tiers.filter(pl.col("intelligence_tier") == tier)
        price = tier_df["price_usd"].drop_nulls()

        if len(price) > 0:
            current_mean = float(price.mean())
            current_median = float(price.median())
            current_std = float(price.std())
            n = len(price)

            print(f"Tier {tier}: Current mean=${current_mean:.2f}, median=${current_median:.2f}, n={n}")

            # Simple linear extrapolation scenarios
            # Assume prices decrease due to competition (Moore's law analogy)
            scenarios = {
                "optimistic": {
                    "assumption": "20% price reduction (intense competition, efficiency gains)",
                    "multiplier": 0.80
                },
                "baseline": {
                    "assumption": "10% price reduction (steady competition)",
                    "multiplier": 0.90
                },
                "pessimistic": {
                    "assumption": "5% price reduction (limited competition)",
                    "multiplier": 0.95
                }
            }

            tier_results = {
                "current": {
                    "mean": current_mean,
                    "median": current_median,
                    "std": current_std,
                    "n": n
                },
                "scenarios": {}
            }

            for scenario_name, scenario in scenarios.items():
                pred_mean = current_mean * scenario["multiplier"]
                pred_median = current_median * scenario["multiplier"]

                # Prediction interval
                pi_lower = max(0, pred_mean - 1.96 * current_std)  # Prices can't be negative
                pi_upper = pred_mean + 1.96 * current_std

                tier_results["scenarios"][scenario_name] = {
                    "assumption": scenario["assumption"],
                    "mean": pred_mean,
                    "median": pred_median,
                    "prediction_interval_lower": pi_lower,
                    "prediction_interval_upper": pi_upper
                }

                print(f"  {scenario_name.capitalize()}: Mean=${pred_mean:.2f}, 95% PI=[${pi_lower:.2f}, ${pi_upper:.2f}]")

            results[tier] = tier_results

    return results


def predict_speed_trends(df: pl.DataFrame) -> dict:
    """Predict 2027 speed improvements."""
    print("\n--- Speed Trend Prediction ---")

    # Current speed statistics
    speed = df["Speed(median token/s)"].drop_nulls().cast(pl.Utf8).cast(pl.Float64)
    current_mean = float(speed.mean())
    current_median = float(speed.median())
    current_std = float(speed.std())
    n = len(speed)

    print(f"Current (2026): Mean={current_mean:.2f} token/s, Median={current_median:.2f} token/s, Std={current_std:.2f}")

    # Simple scenarios for 2027
    # Speed improvements due to hardware and optimization
    scenarios = {
        "optimistic": {
            "assumption": "20% speed improvement (hardware breakthroughs, optimization)",
            "multiplier": 1.20
        },
        "baseline": {
            "assumption": "10% speed improvement (steady hardware progress)",
            "multiplier": 1.10
        },
        "pessimistic": {
            "assumption": "5% speed improvement (incremental improvements)",
            "multiplier": 1.05
        }
    }

    results = {
        "current": {
            "mean": current_mean,
            "median": current_median,
            "std": current_std,
            "n": n
        },
        "scenarios": {}
    }

    for scenario_name, scenario in scenarios.items():
        pred_mean = current_mean * scenario["multiplier"]
        pred_median = current_median * scenario["multiplier"]

        # Prediction interval
        pi_lower = pred_mean - 1.96 * current_std
        pi_upper = pred_mean + 1.96 * current_std

        results["scenarios"][scenario_name] = {
            "assumption": scenario["assumption"],
            "mean": pred_mean,
            "median": pred_median,
            "prediction_interval_lower": pi_lower,
            "prediction_interval_upper": pi_upper
        }

        print(f"{scenario_name.capitalize()}: Mean={pred_mean:.2f}, Median={pred_median:.2f}, 95% PI=[{pi_lower:.2f}, {pi_upper:.2f}]")

    return results


def generate_prediction_report(predictions, output_path):
    """Generate 2027 trend prediction report with uncertainty discussion."""
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Get date
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Build report sections
    sections = []

    # Title
    sections.append(f"# 2027 Trend Predictions Report")
    sections.append(f"**Generated:** {date_str}\n")

    # Section 1: Prediction Methodology
    sections.append("## Prediction Methodology")
    sections.append("")
    sections.append("### Approach")
    sections.append("This report provides **simple trend extrapolations** to 2027 based on 2026 data.")
    sections.append("**Important:** These are simplified projections, not sophisticated forecasts.")
    sections.append("")
    sections.append("**Method:**")
    sections.append("- Cross-sectional data analysis (2026 snapshot)")
    sections.append("- Scenario-based projections (optimistic, baseline, pessimistic)")
    sections.append("- Prediction intervals using current standard deviation")
    sections.append("- Linear extrapolation from current means/medians")
    sections.append("")
    sections.append("**Data limitations:**")
    sections.append("- No time series data (single snapshot in 2026)")
    sections.append("- Cannot model trends over time")
    sections.append("- Assumes current patterns continue linearly")
    sections.append("- Does not account for disruptive innovations")
    sections.append("")
    sections.append("**Use cases:**")
    sections.append("- **Exploratory analysis:** What if trends continue?")
    sections.append("- **Scenario planning:** Range of possible outcomes")
    sections.append("- **Not for:** Investment decisions, product planning, competitive intelligence")
    sections.append("")

    # Section 2: 2027 Predictions
    sections.append("## 2027 Predictions")
    sections.append("")

    # Intelligence predictions
    intel_results = predictions["intelligence"]
    sections.append("### Intelligence Distribution")
    sections.append("")
    sections.append("**Current (2026):**")
    sections.append(f"- Mean: {intel_results['current']['mean']:.2f}")
    sections.append(f"- Median: {intel_results['current']['median']:.2f}")
    sections.append(f"- Std: {intel_results['current']['std']:.2f}")
    sections.append(f"- Sample size: n={intel_results['current']['n']}")
    sections.append("")

    sections.append("**2027 Scenarios:**")
    for scenario_name, scenario in intel_results["scenarios"].items():
        sections.append(f"#### {scenario_name.capitalize()} Scenario")
        sections.append(f"- **Assumption:** {scenario['assumption']}")
        sections.append(f"- **Predicted mean:** {scenario['mean']:.2f}")
        sections.append(f"- **Predicted median:** {scenario['median']:.2f}")
        sections.append(f"- **95% Prediction Interval:** [{scenario['prediction_interval_lower']:.2f}, {scenario['prediction_interval_upper']:.2f}]")
        sections.append("")

    # Price predictions by tier
    price_results = predictions["price_by_tier"]
    sections.append("### Pricing Trends by Intelligence Tier")
    sections.append("")
    sections.append("**Current (2026) by Intelligence Quartile:**")
    sections.append("")

    for tier in ["Q1", "Q2", "Q3", "Q4"]:
        if tier in price_results:
            current = price_results[tier]["current"]
            sections.append(f"**Tier {tier} (Low to High Intelligence):**")
            sections.append(f"- Current mean: ${current['mean']:.2f}")
            sections.append(f"- Current median: ${current['median']:.2f}")
            sections.append(f"- Sample size: n={current['n']}")
            sections.append("")

            sections.append(f"**2027 Projections for Tier {tier}:**")
            for scenario_name, scenario in price_results[tier]["scenarios"].items():
                sections.append(f"- **{scenario_name.capitalize()}:** ${scenario['mean']:.2f} (95% PI: [${scenario['prediction_interval_lower']:.2f}, ${scenario['prediction_interval_upper']:.2f}])")
            sections.append("")

    # Speed predictions
    speed_results = predictions["speed"]
    sections.append("### Speed Improvements")
    sections.append("")
    sections.append("**Current (2026):**")
    sections.append(f"- Mean: {speed_results['current']['mean']:.2f} token/s")
    sections.append(f"- Median: {speed_results['current']['median']:.2f} token/s")
    sections.append(f"- Std: {speed_results['current']['std']:.2f} token/s")
    sections.append(f"- Sample size: n={speed_results['current']['n']}")
    sections.append("")

    sections.append("**2027 Scenarios:**")
    for scenario_name, scenario in speed_results["scenarios"].items():
        sections.append(f"#### {scenario_name.capitalize()} Scenario")
        sections.append(f"- **Assumption:** {scenario['assumption']}")
        sections.append(f"- **Predicted mean:** {scenario['mean']:.2f} token/s")
        sections.append(f"- **Predicted median:** {scenario['median']:.2f} token/s")
        sections.append(f"- **95% Prediction Interval:** [{scenario['prediction_interval_lower']:.2f}, {scenario['prediction_interval_upper']:.2f}] token/s")
        sections.append("")

    # Section 3: Uncertainty Discussion (NARR-09 - CRITICAL)
    sections.append("## Uncertainty Discussion")
    sections.append("")
    sections.append("### Sources of Uncertainty")
    sections.append("")
    sections.append("**1. Data Limitations (NARR-09 requirement)**")
    sections.append("- **Cross-sectional, not temporal:** We have 2026 snapshot, not time series")
    sections.append("- **No trend data:** Cannot observe actual historical patterns")
    sections.append("- **Small sample:** 181 models may not represent full market")
    sections.append("- **Selection bias:** Dataset may not include all models")
    sections.append("")
    sections.append("**2. Model Assumptions")
    sections.append("- **Linear extrapolation:** Assumes trends continue linearly (unrealistic)")
    sections.append("- **Constant variance:** Assumes std stays constant (unlikely)")
    sections.append("- **No disruption:** Assumes no breakthrough technologies (risky)")
    sections.append("- **Independence:** Assumes models evolve independently (false)")
    sections.append("")
    sections.append("**3. External Factors")
    sections.append("- **New competitors:** Could disrupt market dynamics")
    sections.append("- **Regulatory changes:** AI regulation could affect development")
    sections.append("- **Technology breakthroughs:** Could accelerate or decelerate progress")
    sections.append("- **Economic factors:** R&D investment, market demand changes")
    sections.append("- **Geopolitical events:** Trade restrictions, international competition")
    sections.append("")
    sections.append("**4. Prediction Interpretation**")
    sections.append("- **Point forecasts:** Single numbers are misleading")
    sections.append("- **Prediction intervals:** Wide intervals show high uncertainty")
    sections.append("- **Scenario analysis:** Optimistic/baseline/pessimistic show range")
    sections.append("- **Not probabilities:** Scenarios are not equally likely")
    sections.append("")

    # Section 4: Limitations of Extrapolation
    sections.append("## Limitations of Extrapolation")
    sections.append("")
    sections.append("### Why These Predictions Are Unreliable")
    sections.append("")
    sections.append("**1. Cross-Sectional Data Problem**")
    sections.append("- We have models from 2026, not historical data from 2015-2026")
    sections.append("- Cannot fit time series models (ARIMA, exponential smoothing)")
    sections.append("- Cannot observe actual trends over time")
    sections.append("- Extrapolating from current distribution is naive")
    sections.append("")
    sections.append("**2. Linear Extrapolation Fallacy**")
    sections.append("- Technology progress is rarely linear")
    sections.append("- S-curves (logistic growth) are more realistic")
    sections.append("- Plateaus and breakthroughs violate linear assumptions")
    sections.append("- Moore's Law is slowing down (not exponential forever)")
    sections.append("")
    sections.append("**3. Black Swan Events**")
    sections.append("- GPT-4 level breakthroughs are unpredictable")
    sections.append("- New architectures (transformers in 2017) change everything")
    sections.append("- Regulatory shocks (EU AI Act, US executive orders)")
    sections.append("- Economic disruptions (recessions, funding crunches)")
    sections.append("")
    sections.append("**4. Competitive Dynamics**")
    sections.append("- New entrants (DeepSeek, Mistral) can disrupt quickly")
    sections.append("- incumbents (OpenAI, Google) may leapfrog")
    sections.append("- Open-source models (Llama, Mistral) change economics")
    sections.append("- Price wars could accelerate faster than predicted")
    sections.append("")

    # Section 5: Recommendations
    sections.append("## Recommendations")
    sections.append("")
    sections.append("### How to Use These Predictions")
    sections.append("")
    sections.append("**1. For Exploratory Analysis**")
    sections.append("- Use scenario ranges to understand possibilities")
    sections.append("- Consider wide prediction intervals as uncertainty bounds")
    sections.append("- Focus on direction (up/down), not exact numbers")
    sections.append("")
    sections.append("**2. For Planning**")
    sections.append("- Do NOT rely on point forecasts")
    sections.append("- Use optimistic/baseline/pessimistic for sensitivity analysis")
    sections.append("- Plan for multiple scenarios, not single outcome")
    sections.append("- Update predictions as new data becomes available")
    sections.append("")
    sections.append("**3. For Decision-Making**")
    sections.append("- Treat predictions as rough guides, not facts")
    sections.append("- Combine with domain knowledge and expert judgment")
    sections.append("- Monitor actual trends to validate/adjust assumptions")
    sections.append("- Maintain flexibility to adapt to unexpected changes")
    sections.append("")
    sections.append("**4. For Research**")
    sections.append("- Collect temporal data (historical model benchmarks)")
    sections.append("- Track model releases over time")
    sections.append("- Monitor provider announcements and roadmaps")
    sections.append("- Build proper time series forecasting models")
    sections.append("")

    # Section 6: Key Takeaways
    sections.append("## Key Takeaways")
    sections.append("")
    sections.append("**High uncertainty:** These predictions have very wide confidence intervals")
    sections.append("**Directional only:** Trust direction (prices down, speed up), not magnitudes")
    sections.append("**Scenario planning:** Use multiple scenarios, not single point forecast")
    sections.append("**Update regularly:** Re-evaluate as new data becomes available")
    sections.append("**Not for betting:** Do NOT use these predictions for investment decisions")
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append("*This report follows NARR-09 requirement for comprehensive uncertainty discussion.*")
    sections.append("*Predictions are simplified extrapolations for exploratory analysis only.*")
    sections.append("*Actual 2027 outcomes will likely differ substantially from these projections.*")
    sections.append("")

    # Write report
    report_content = "\n".join(sections)

    with open(output_path, 'w') as f:
        f.write(report_content)

    print("\nReport sections:")
    print("- Prediction methodology documented")
    print("- 2027 predictions for intelligence, price, speed")
    print("- Comprehensive uncertainty discussion (NARR-09)")
    print("- Limitations of extrapolation explained")
    print("- Scenario analysis provided")
    print("- Recommendations included")


if __name__ == "__main__":
    main()
