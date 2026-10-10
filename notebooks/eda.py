"""Quick EDA script for the CreditWise pipeline.

This script is intentionally lightweight and meant to accompany the formal model training workflow.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import inspect_dataset, load_raw_dataset
from src.data_validation import validate_dataset


def run_eda(path: str = "data/raw/credit_card_default.csv") -> dict:
    df = load_raw_dataset(path)
    info = inspect_dataset(df)
    target_col = info["target_column"]
    id_col = info["id_column"]
    validation = validate_dataset(df, target_col, id_col)
    print(f"Rows: {info['rows']} | Columns: {len(info['columns'])}")
    print(f"Target column: {target_col} | ID column: {id_col}")
    print(df[[target_col]].value_counts(normalize=True).head())
    print("Warnings:", validation["warnings"])
    return validation


if __name__ == "__main__":
    run_eda()
