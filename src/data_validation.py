from __future__ import annotations

import pandas as pd


def validate_dataset(df: pd.DataFrame, target_col: str, id_col: str) -> dict:
    report = {
        "shape": df.shape,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().to_dict(),
        "target_distribution": df[target_col].value_counts(dropna=False).to_dict(),
        "invalid_target_values": sorted(pd.unique(df[target_col].dropna().astype(str)))[:20],
        "numeric_summary": {
            col: {
                "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
            }
            for col in df.select_dtypes(include=['number']).columns
        },
        "warnings": [],
    }
    if report["duplicate_rows"] > 0:
        report["warnings"].append("Duplicate rows detected.")
    if df[id_col].nunique() != len(df):
        report["warnings"].append("Customer IDs are not unique; review duplicate or repeated account records.")
    if df[target_col].isna().any():
        report["warnings"].append("Missing target values detected.")
    if report["target_distribution"].get("1", 0) == 0 or report["target_distribution"].get("0", 0) == 0:
        report["warnings"].append("Target distribution is highly imbalanced or missing one class.")
    return report


def build_data_dictionary(df: pd.DataFrame, target_col: str, id_col: str, used_columns: list[str]) -> list[dict]:
    dictionary = []
    for col in df.columns:
        s = df[col]
        meta = {
            "column_name": col,
            "business_meaning": "Not documented in the source metadata; reviewed during EDA",
            "data_type": str(s.dtype),
            "observed_values": str(s.unique()[:10]) if not pd.api.types.is_numeric_dtype(s) else "Numeric range",
            "missing_value_count": int(s.isna().sum()),
            "used_for_modeling": col not in {id_col, target_col} and col in used_columns,
            "reason": "Included as a model feature" if col in used_columns else "Excluded from primary model",
        }
        dictionary.append(meta)
    return dictionary


def assert_id_and_target_excluded(X: pd.DataFrame, id_col: str, target_col: str) -> None:
    if id_col in X.columns:
        raise AssertionError(f"Customer ID column '{id_col}' is still in model features.")
    if target_col in X.columns:
        raise AssertionError(f"Target column '{target_col}' is still in model features.")
