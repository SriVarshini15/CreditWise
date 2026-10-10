from __future__ import annotations

import pandas as pd

from src.config import ID_COLUMN_CANDIDATES, RAW_DATA_PATH, TARGET_COLUMN_CANDIDATES


def detect_target_column(df: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    for col in df.columns:
        lc = str(col).lower()
        if "default" in lc and "payment" in lc and "next" in lc:
            return col
    raise ValueError(f"Could not identify a target/default column in the dataset. Columns: {list(df.columns)}")


def detect_id_column(df: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in ID_COLUMN_CANDIDATES:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    for col in df.columns:
        lc = str(col).lower()
        if "id" in lc or "customer" in lc:
            return col
    raise ValueError(f"Could not identify a customer ID column. Data columns: {list(df.columns)}")


def load_raw_dataset(path: str | None = None) -> pd.DataFrame:
    file_path = path or str(RAW_DATA_PATH)
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Dataset at {file_path} is empty.")
    if set(df.columns).issubset({f"X{i}" for i in range(1, 25)}) or "Y" in df.columns:
        rename_map = {
            **{f"X{i}": [
                "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
                "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
            ][i - 1] for i in range(1, 24)},
            "Y": "default payment next month",
        }
        df = df.rename(columns=rename_map)
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    target_col = detect_target_column(df)
    id_col = detect_id_column(df)
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "target_column": target_col,
        "id_column": id_col,
    }
