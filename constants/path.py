from pathlib import Path

# Project root = parent directory of the constants/ package
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"

CLIMATE_PATH = DATA_RAW_PATH / "climate" / "climate_data_from_1982.parquet"
BARLEY_PATH = DATA_RAW_PATH / "yields" / "barley_yield_from_1982.csv"

SILVER_PATH = DATA_PROCESSED_PATH / "silver"
GOLD_PATH = DATA_PROCESSED_PATH / "gold"