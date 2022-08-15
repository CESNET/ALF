import pandas as pd


def load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def save(path: str, df: pd.DataFrame) -> pd.DataFrame:
    return df.to_csv(index=False)
