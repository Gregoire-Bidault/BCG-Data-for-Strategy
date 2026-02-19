"""
Deep-dive Correlation Analysis: 2011 Barley Yield Collapse vs Climate Features
================================================================================

Uses gold.parquet to compare 2010, 2011, 2012 and diagnose whether heat stress,
drought, or a specific combination during a sensitive phenological stage drove the
2011 yield drop.

Outputs are written to  notebooks/output/
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # headless backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

from constants.path import GOLD_PATH

# ── Output dir ────────────────────────────────────────────────────────────────
OUT_DIR = PROJECT_ROOT / "notebooks" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 0. Load & filter ─────────────────────────────────────────────────────────

def load_data():
    df = pd.read_parquet(GOLD_PATH / "gold.parquet")
    years = [2010, 2011, 2012]
    df = df[df["year"].isin(years)].copy()

    # Columns to exclude from weather features
    exclude = {"department", "year", "yield", "area", "production", "code_dep"}
    weather_cols = [c for c in df.columns if c not in exclude]

    return df, weather_cols


# ── 1. Feature Variance Ranking ──────────────────────────────────────────────

def feature_variance_ranking(df: pd.DataFrame, weather_cols: list[str]):
    """
    For each weather feature compute the national median per year, then
    %‑change 2010→2011 and 2011→2012.  Rank by absolute anomaly in 2011.
    """
    medians = df.groupby("year")[weather_cols].median()

    pct_10_11 = ((medians.loc[2011] - medians.loc[2010]) / medians.loc[2010].replace(0, np.nan) * 100)
    pct_11_12 = ((medians.loc[2012] - medians.loc[2011]) / medians.loc[2011].replace(0, np.nan) * 100)

    ranking = pd.DataFrame({
        "median_2010": medians.loc[2010],
        "median_2011": medians.loc[2011],
        "median_2012": medians.loc[2012],
        "pct_change_2010_2011": pct_10_11,
        "pct_change_2011_2012": pct_11_12,
    })
    ranking["abs_anomaly_2011"] = ranking["pct_change_2010_2011"].abs()
    ranking = ranking.sort_values("abs_anomaly_2011", ascending=False)

    ranking.to_csv(OUT_DIR / "feature_variance_ranking.csv")
    print("\n" + "=" * 72)
    print("TOP-20 MOST ABNORMAL FEATURES IN 2011 (vs 2010)")
    print("=" * 72)
    print(ranking.head(20).to_string())
    return ranking

# ── 2. Correlation Heatmap ────────────────────────────────────────────────────

def correlation_heatmap(df: pd.DataFrame, weather_cols: list[str]):
    """
    Pearson correlation of each weather feature with yield, pooled across
    the three years.  Plotted as a horizontal-bar-style single-column heatmap.
    """
    corrs = df[weather_cols + ["yield"]].corr()["yield"].drop("yield").sort_values()

    fig, ax = plt.subplots(figsize=(8, max(10, len(corrs) * 0.28)))
    colors = ["#d73027" if v < 0 else "#4575b4" for v in corrs.values]
    bars = ax.barh(range(len(corrs)), corrs.values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(corrs)))
    ax.set_yticklabels(corrs.index, fontsize=7)
    ax.set_xlabel("Pearson r  with yield", fontsize=10)
    ax.set_title("Weather-Feature Correlation with Yield (2010–2012)", fontsize=12, weight="bold")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    print("\n" + "=" * 72)
    print("TOP-10 FEATURES MOST CORRELATED WITH YIELD (2010–2012)")
    print("=" * 72)
    top10 = corrs.abs().sort_values(ascending=False).head(10)
    for feat in top10.index:
        print(f"  {feat:45s}  r = {corrs[feat]:+.3f}")
    return corrs

# ── 3. ZD49 Scatter Plots (Apr–May features) ─────────────────────────────────

def zd49_scatter(df: pd.DataFrame, ranking: pd.DataFrame):
    """
    Among the month-4 and month-5 features, pick the top-3 most abnormal
    in 2011 and scatter them against yield, colored by year.
    """
    zd49_feats = [f for f in ranking.index if f.startswith("m4_") or f.startswith("m5_")]
    top3 = zd49_feats[:3]

    if len(top3) == 0:
        print("No m4/m5 features found – skipping ZD49 scatter.")
        return

    year_colors = {2010: "#4575b4", 2011: "#d73027", 2012: "#91bfdb"}
    fig, axes = plt.subplots(1, len(top3), figsize=(6 * len(top3), 5))
    if len(top3) == 1:
        axes = [axes]

    for ax, feat in zip(axes, top3):
        for yr, color in year_colors.items():
            sub = df[df["year"] == yr]
            ax.scatter(sub[feat], sub["yield"], s=28, alpha=0.7, c=color, label=str(yr), edgecolors="white", linewidths=0.3)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("Yield (t/ha)", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_title(feat, fontsize=10, weight="bold")
        ax.grid(alpha=0.25)

    fig.suptitle("ZD49 Window Features vs Yield (2010 – 2012)", fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "zd49_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nZD49 scatter saved  →  top features: {top3}")

# ── 4. Spatial Consistency ────────────────────────────────────────────────────

def spatial_consistency(df: pd.DataFrame, ranking: pd.DataFrame):
    """
    Per department: compute the 2011 yield anomaly (% change vs 2010) and the
    anomaly of the single most abnormal climate feature.  Scatter + Pearson r.
    """
    # Pick the top climate-stress feature
    top_feat = ranking.index[0]

    pivot_yield = df.pivot_table(index="code_dep", columns="year", values="yield")
    pivot_feat  = df.pivot_table(index="code_dep", columns="year", values=top_feat)

    # % change 2010 → 2011
    yield_anom = ((pivot_yield[2011] - pivot_yield[2010]) / pivot_yield[2010] * 100).dropna()
    feat_anom  = ((pivot_feat[2011]  - pivot_feat[2010])  / pivot_feat[2010].replace(0, np.nan) * 100).dropna()

    common = yield_anom.index.intersection(feat_anom.index)
    yield_anom = yield_anom.loc[common]
    feat_anom  = feat_anom.loc[common]

    r, p = stats.pearsonr(feat_anom, yield_anom)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(feat_anom, yield_anom, s=38, alpha=0.7, c="#d73027", edgecolors="white", linewidths=0.4)

    # regression line
    m, b = np.polyfit(feat_anom, yield_anom, 1)
    xs = np.linspace(feat_anom.min(), feat_anom.max(), 100)
    ax.plot(xs, m * xs + b, "--", color="grey", linewidth=1)

    ax.set_xlabel(f"% change in  {top_feat}  (2010→2011)", fontsize=10)
    ax.set_ylabel("% change in  yield  (2010→2011)", fontsize=10)
    ax.set_title("Spatial Consistency:\nDepartments with extreme weather ↔ sharpest yield drops?",
                 fontsize=11, weight="bold")
    ax.annotate(f"Pearson r = {r:.3f}    p = {p:.2e}", xy=(0.05, 0.95),
                xycoords="axes fraction", fontsize=10, va="top",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spatial_consistency.png", dpi=180)
    plt.close(fig)

    print(f"\nSpatial-consistency scatter (top feature = {top_feat}):  r = {r:.3f},  p = {p:.2e}")
    return top_feat, r, p

# ── 5. Diagnostic Summary ────────────────────────────────────────────────────

def diagnostic_summary(df, ranking, corrs, top_feat, sp_r, sp_p):
    """
    Print + save a structured text report.
    """
    # national median yield per year
    med_yield = df.groupby("year")["yield"].median()

    # Separate drought / heat / combined features in the top-10 anomaly list
    top10 = ranking.head(10)
    heat_kws   = {"heat", "tmax", "gdd", "hot_dry", "consec_heat"}
    drought_kws = {"dry", "precip", "rain", "wet"}

    heat_feats   = [f for f in top10.index if any(k in f for k in heat_kws)]
    drought_feats = [f for f in top10.index if any(k in f for k in drought_kws)]

    # strongest correlated feature
    strongest_corr_feat = corrs.abs().idxmax()
    strongest_corr_val  = corrs[strongest_corr_feat]

    lines = []
    lines.append("=" * 72)
    lines.append("DIAGNOSTIC SUMMARY  –  2011 BARLEY YIELD COLLAPSE")
    lines.append("=" * 72)
    lines.append("")
    lines.append("1.  NATIONAL MEDIAN YIELD (t/ha)")
    for yr in [2010, 2011, 2012]:
        lines.append(f"       {yr}:  {med_yield[yr]:.2f}")
    drop = (med_yield[2011] - med_yield[2010]) / med_yield[2010] * 100
    lines.append(f"       Drop 2010→2011:  {drop:+.1f} %")
    lines.append("")
    lines.append("2.  TOP-10 MOST ABNORMAL FEATURES IN 2011")
    for i, feat in enumerate(top10.index, 1):
        pct = top10.loc[feat, "pct_change_2010_2011"]
        lines.append(f"       {i:2d}. {feat:45s}  Δ = {pct:+.1f} %")
    lines.append("")
    lines.append("3.  HEAT vs DROUGHT BREAKDOWN (among top-10)")
    lines.append(f"       Heat-related features :  {len(heat_feats)}  →  {heat_feats}")
    lines.append(f"       Drought-related feats :  {len(drought_feats)}  →  {drought_feats}")
    lines.append("")
    lines.append("4.  STRONGEST YIELD CORRELATE (2010-2012)")
    lines.append(f"       {strongest_corr_feat}   r = {strongest_corr_val:+.3f}")
    lines.append("")
    lines.append("5.  SPATIAL CONSISTENCY")
    lines.append(f"       Top feature: {top_feat}")
    lines.append(f"       Dept-level correlation with yield anomaly: r = {sp_r:.3f}, p = {sp_p:.2e}")
    sig = "YES – statistically significant" if sp_p < 0.05 else "NO – not significant"
    lines.append(f"       Significant at α=0.05?  {sig}")
    lines.append("")
    lines.append("6.  CONCLUSION")
    if len(heat_feats) >= len(drought_feats) and len(heat_feats) >= 3:
        driver = "primarily HEAT STRESS"
    elif len(drought_feats) > len(heat_feats) and len(drought_feats) >= 3:
        driver = "primarily DROUGHT"
    else:
        driver = "a COMBINATION of heat stress and drought"
    # Check which phenological window dominates
    spring_feats = [f for f in top10.index if "spring" in f or f.startswith("m4_") or f.startswith("m5_")]
    summer_feats = [f for f in top10.index if "summer" in f or f.startswith("m6_") or f.startswith("m7_") or f.startswith("m8_")]
    if len(spring_feats) > len(summer_feats):
        window = "the spring window (Mar–May, ZD30–ZD49)"
    elif len(summer_feats) > len(spring_feats):
        window = "the summer window (Jun–Aug, grain fill)"
    else:
        window = "both the spring and summer windows"

    lines.append(f"       The 2011 yield collapse was driven by {driver},")
    lines.append(f"       concentrated during {window}.")
    lines.append(f"       Departments that experienced the most extreme {top_feat}")
    if sp_r < -0.2 or sp_r > 0.2:
        lines.append(f"       also suffered the sharpest yield drops (r = {sp_r:.3f}),")
        lines.append(f"       confirming SPATIAL CONSISTENCY of the climate signal.")
    else:
        lines.append(f"       showed only weak spatial correlation with yield drops (r = {sp_r:.3f}).")
    lines.append("")
    lines.append("=" * 72)

    report = "\n".join(lines)
    print("\n" + report)
    (OUT_DIR / "diagnostic_summary.txt").write_text(report, encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading gold.parquet …")
    df, weather_cols = load_data()
    print(f"  {len(df)} rows · {len(weather_cols)} weather features · years {sorted(df['year'].unique())}")

    ranking = feature_variance_ranking(df, weather_cols)
    corrs   = correlation_heatmap(df, weather_cols)
    zd49_scatter(df, ranking)
    top_feat, sp_r, sp_p = spatial_consistency(df, ranking)
    diagnostic_summary(df, ranking, corrs, top_feat, sp_r, sp_p)

    print(f"\nAll outputs saved to  {OUT_DIR}/")


if __name__ == "__main__":
    main()
