"""Card data loading and utilities."""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_card_data(lang: str = "EN") -> pd.DataFrame:
    path = DATA_DIR / f"{lang}_Card_Data.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def card_id_to_name(lang: str = "EN") -> dict[int, str]:
    df = load_card_data(lang)
    id_col = "Card ID"
    name_col = "Card Name"
    return dict(zip(df[id_col], df[name_col]))
