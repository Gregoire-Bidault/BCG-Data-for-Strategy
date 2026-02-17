import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from constants.path import CLIMATE_PATH, BARLEY_PATH

climate_df = pd.read_parquet(CLIMATE_PATH)

def bronze_to_silver_barley() -> pd.DataFrame:
    barley_df = pd.read_csv(BARLEY_PATH, sep=";")

    barley_df.rename(columns={"Unnamed: 0":"code_dep"}, inplace=True)

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

