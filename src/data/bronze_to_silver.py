import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from constants.path import (
    CLIMATE_PATH,
    BARLEY_PATH,
    SILVER_PATH,
    GOLD_PATH,
)
from constants.constants import CLIMATE_COLUMNS, DEPARTMENT_COLUMNS

def bronze_to_silver_barley() -> pd.DataFrame:
    departments = pd.read_parquet(GOLD_PATH / "department.parquet")
    barley_df = pd.read_csv(BARLEY_PATH, sep=";")

    barley_df = barley_df.merge(departments, how="left", left_on="department", right_on="nom_dep")
    barley_df.drop(columns=["Unnamed: 0", "nom_dep"], inplace=True)

    barley_df["code_dep"] = barley_df["code_dep"].astype(str).str.strip()
    barley_df["department"] = barley_df["department"].astype(str).str.strip()

    barley_df["year"] = pd.to_numeric(barley_df["year"], errors="coerce")
    barley_df["yield"] = pd.to_numeric(barley_df["yield"], errors="coerce")
    barley_df["area"] = pd.to_numeric(barley_df["area"], errors="coerce")
    barley_df["production"] = pd.to_numeric(barley_df["production"], errors="coerce")

    barley_df["production"] = pd.to_numeric(barley_df["production"], errors="coerce")
    barley_df["area"] = pd.to_numeric(barley_df["area"], errors="coerce")
    barley_df["yield"] = pd.to_numeric(barley_df["yield"], errors="coerce")

    barley_df.dropna(subset=["production", "area", "yield"], thresh=2, inplace=True)

    barley_df["yield"] = barley_df["yield"].fillna(barley_df["production"] / barley_df["area"])
    barley_df["production"] = barley_df["production"].fillna(barley_df["area"] * barley_df["yield"])

    barley_df = barley_df.drop_duplicates()
    return barley_df

def bronze_to_silver_climate() -> pd.DataFrame:
    df = pd.read_parquet(CLIMATE_PATH)

    df_department = df[DEPARTMENT_COLUMNS].copy()
    df_department = df_department.drop_duplicates()

    mask_middle_scenario = (df["scenario"] == "historical") | (df["scenario"] == "ssp2_4_5")
    mask_worst_scenario = (df["scenario"] == "historical") | (df["scenario"] == "ssp5_8_5")

    df_middle_sc = df[mask_middle_scenario].copy()
    df_worst_sc = df[mask_worst_scenario].copy()

    df_middle_sc = df_middle_sc[CLIMATE_COLUMNS]
    df_worst_sc = df_worst_sc[CLIMATE_COLUMNS]

    df_middle_sc.loc[:, "time"] = pd.to_datetime(df_middle_sc["time"])
    df_worst_sc.loc[:, "time"] = pd.to_datetime(df_worst_sc["time"])

    df_middle_sc = (
        df_middle_sc.pivot_table(
            index=["code_dep", "time"],
            columns="metric",
            values="value",
            aggfunc="mean"
        )
        .reset_index()
    )
    df_worst_sc = (
        df_worst_sc.pivot_table(
            index=["code_dep", "time"],
            columns="metric",
            values="value",
            aggfunc="mean"
        )
        .reset_index()
    )

    df_middle_sc["precipitation"] = df_middle_sc["precipitation"] * 86400.0
    df_worst_sc["precipitation"] = df_worst_sc["precipitation"] * 86400.0

    return df_department, df_middle_sc, df_worst_sc

if __name__ == "__main__":
    barley_df = bronze_to_silver_barley()
    barley_df.to_parquet(SILVER_PATH / "barley.parquet", index=False)

    df_department, df_middle_sc, df_worst_sc = bronze_to_silver_climate()
    df_department.to_parquet(GOLD_PATH / "department.parquet", index=False)
    df_middle_sc.to_parquet(SILVER_PATH / "climate_middle.parquet", index=False)
    df_worst_sc.to_parquet(SILVER_PATH / "climate_worst.parquet", index=False)