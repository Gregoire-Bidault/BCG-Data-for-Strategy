import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np

from constants.path import SILVER_PATH, GOLD_PATH

def max_consecutive_true(series):
    if series.isna().all() or len(series) == 0:
        return 0
    s = series.fillna(False).astype(int)
    groups = (s != s.shift()).cumsum()
    return int(s.groupby(groups).sum().max())

def climate_features_year(group):
    g = group.sort_values("time").copy()
    g["month"] = g["time"].dt.month

    feats = {}

    # Thresholds
    DRY_THRESH = 1.0          # mm/day
    HEAT_THRESH = 303.15      # 30°C in Kelvin
    FROST_THRESH = 273.15     # 0°C in Kelvin
    BASE_TEMP = 283.15        # 10°C in Kelvin for GDD

    # Required columns (names from your file)
    tmax = g["daily_maximum_near_surface_air_temperature"]
    tmean = g["near_surface_air_temperature"]
    precip = g["precipitation"]

    # -------- Annual features --------
    feats["tmean_mean"] = float(tmean.mean())
    feats["tmax_max"] = float(tmax.max())
    feats["tmax_p95"] = float(tmax.quantile(0.95))

    heat = tmax > HEAT_THRESH
    frost = tmean < FROST_THRESH
    dry = precip < DRY_THRESH
    wet = precip >= DRY_THRESH
    hot_dry = heat & dry

    feats["heat_days"] = int(heat.sum())
    feats["max_consecutive_heat_days"] = max_consecutive_true(heat)

    feats["frost_days"] = int(frost.sum())
    feats["max_consecutive_frost_days"] = max_consecutive_true(frost)

    feats["gdd"] = float((tmean - BASE_TEMP).clip(lower=0).sum())

    feats["total_precip"] = float(precip.sum())
    feats["rain_days"] = int((precip >= DRY_THRESH).sum())
    feats["max_1day_precip"] = float(precip.max())
    feats["p95_precip"] = float(precip.quantile(0.95))

    feats["max_consecutive_dry_days"] = max_consecutive_true(dry)
    feats["max_consecutive_wet_days"] = max_consecutive_true(wet)

    feats["max_3day_precip_sum"] = float(precip.rolling(3).sum().max())
    feats["max_5day_precip_sum"] = float(precip.rolling(5).sum().max())
    feats["max_7day_precip_sum"] = float(precip.rolling(7).sum().max())

    feats["precip_cv"] = float(precip.std() / precip.mean()) if precip.mean() != 0 else 0.0

    feats["hot_dry_days"] = int(hot_dry.sum())
    feats["max_consecutive_hot_dry_days"] = max_consecutive_true(hot_dry)

    # -------- Spring (Mar–May) --------
    spring = g[g["month"].isin([3, 4, 5])]
    if len(spring) > 0:
        tmax_s = spring["daily_maximum_near_surface_air_temperature"]
        tmean_s = spring["near_surface_air_temperature"]
        precip_s = spring["precipitation"]

        heat_s = tmax_s > HEAT_THRESH
        dry_s = precip_s < DRY_THRESH
        frost_s = tmean_s < FROST_THRESH

        feats["spring_heat_days"] = int(heat_s.sum())
        feats["spring_frost_days"] = int(frost_s.sum())
        feats["spring_max_consecutive_dry_days"] = max_consecutive_true(dry_s)
        feats["spring_gdd"] = float((tmean_s - BASE_TEMP).clip(lower=0).sum())
        feats["spring_total_precip"] = float(precip_s.sum())
    else:
        feats["spring_heat_days"] = 0
        feats["spring_frost_days"] = 0
        feats["spring_max_consecutive_dry_days"] = 0
        feats["spring_gdd"] = 0.0
        feats["spring_total_precip"] = 0.0

    # -------- Summer (Jun–Aug) --------
    summer = g[g["month"].isin([6, 7, 8])]
    if len(summer) > 0:
        tmax_u = summer["daily_maximum_near_surface_air_temperature"]
        tmean_u = summer["near_surface_air_temperature"]
        precip_u = summer["precipitation"]

        heat_u = tmax_u > HEAT_THRESH
        dry_u = precip_u < DRY_THRESH
        hot_dry_u = heat_u & dry_u

        feats["summer_heat_days"] = int(heat_u.sum())
        feats["summer_hot_dry_days"] = int(hot_dry_u.sum())
        feats["summer_max_consecutive_dry_days"] = max_consecutive_true(dry_u)
        feats["summer_gdd"] = float((tmean_u - BASE_TEMP).clip(lower=0).sum())
        feats["summer_total_precip"] = float(precip_u.sum())
        feats["summer_max_consecutive_heat_days"] = max_consecutive_true(heat_u)
    else:
        feats["summer_heat_days"] = 0
        feats["summer_hot_dry_days"] = 0
        feats["summer_max_consecutive_dry_days"] = 0
        feats["summer_gdd"] = 0.0
        feats["summer_total_precip"] = 0.0
        feats["summer_max_consecutive_heat_days"] = 0

    # -------- Winter (Dec–Feb) --------
    winter = g[g["month"].isin([12, 1, 2])]
    if len(winter) > 0:
        tmean_w = winter["near_surface_air_temperature"]
        frost_w = tmean_w < FROST_THRESH
        feats["winter_frost_days"] = int(frost_w.sum())
    else:
        feats["winter_frost_days"] = 0

    # -------- Monthly windows (Apr–Aug) --------
    for m in [4, 5, 6, 7, 8]:
        gm = g[g["month"] == m]
        if len(gm) > 0:
            tmax_m = gm["daily_maximum_near_surface_air_temperature"]
            tmean_m = gm["near_surface_air_temperature"]
            precip_m = gm["precipitation"]

            heat_m = tmax_m > HEAT_THRESH
            dry_m = precip_m < DRY_THRESH

            feats[f"m{m}_heat_days"] = int(heat_m.sum())
            feats[f"m{m}_max_consec_heat"] = max_consecutive_true(heat_m)
            feats[f"m{m}_dry_days"] = int(dry_m.sum())
            feats[f"m{m}_gdd"] = float((tmean_m - BASE_TEMP).clip(lower=0).sum())
            feats[f"m{m}_precip_sum"] = float(precip_m.sum())
        else:
            feats[f"m{m}_heat_days"] = 0
            feats[f"m{m}_max_consec_heat"] = 0
            feats[f"m{m}_dry_days"] = 0
            feats[f"m{m}_gdd"] = 0.0
            feats[f"m{m}_precip_sum"] = 0.0

    return pd.Series(feats)

def silver_to_gold() -> pd.DataFrame:
    climate_df = pd.read_parquet(SILVER_PATH / "climate_middle.parquet")
    barley_df = pd.read_parquet(SILVER_PATH / "barley.parquet")
    
    climate_df["year"] = climate_df["time"].dt.year

    climate_features = (
        climate_df
        .groupby(["code_dep", "year"], group_keys=False)
        .apply(climate_features_year)
        .reset_index()
    )

    df_merged = barley_df.merge(
        climate_features,
        on=["code_dep", "year"],
        how="left"
    )

    num_cols = df_merged.select_dtypes(include=np.number).columns
    for col in num_cols:
        if col != "yield":
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())

    return df_merged

if __name__ == "__main__":
    df = silver_to_gold()
    df.to_parquet(GOLD_PATH / "gold.parquet", index=False)