"""
Build projection climate features for 2019–2050 from silver climate files.
Reuses climate_features_year() from silver_to_gold.py.

Usage:
    python src/data/build_projections.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.data.silver_to_gold import climate_features_year
from constants.path import SILVER_PATH, GOLD_PATH


def build_projections(scenario: str) -> pd.DataFrame:
    """Build climate features for years > 2018 for a given scenario."""
    parquet_name = f"climate_{scenario}.parquet"
    climate_df = pd.read_parquet(SILVER_PATH / parquet_name)
    climate_df["year"] = climate_df["time"].dt.year

    # Keep only future years (2019-2050)
    climate_df = climate_df[climate_df["year"] > 2018].copy()
    print(f"  {scenario}: {climate_df.shape[0]} daily rows, "
          f"years {climate_df['year'].min()}-{climate_df['year'].max()}")

    features = (
        climate_df
        .groupby(["code_dep", "year"], group_keys=False)
        .apply(climate_features_year)
        .reset_index()
    )

    return features


def main():
    for scenario in ["middle", "worst"]:
        print(f"Building projections for '{scenario}' scenario…")
        df = build_projections(scenario)
        out_path = GOLD_PATH / f"projections_{scenario}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path}  ({df.shape[0]} rows)")
    print("Done ✓")


if __name__ == "__main__":
    main()
