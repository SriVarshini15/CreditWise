from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DEMOGRAPHIC_COLUMNS, PAY_COLUMNS


def prepare_feature_frame(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    excluded_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    excluded = {id_col, target_col}
    if excluded_columns:
        excluded |= set(excluded_columns)
    feature_columns = [col for col in df.columns if col not in excluded]
    X = df.loc[:, feature_columns].copy()
    y = df[target_col].astype(int)
    return X, y, feature_columns


def build_preprocessor(
    X: pd.DataFrame,
    categorical_columns: Sequence[str] | None = None,
    pay_columns: Sequence[str] | None = None,
) -> ColumnTransformer:
    categorical_columns = list(categorical_columns or [])
    pay_columns = list(pay_columns or PAY_COLUMNS)
    available_pay_columns = [col for col in pay_columns if col in X.columns]
    all_cat = [col for col in categorical_columns if col in X.columns]
    if available_pay_columns:
        all_cat = list(dict.fromkeys(available_pay_columns + all_cat))

    numeric_columns = [
        col for col in X.columns
        if col not in set(all_cat)
        and pd.api.types.is_numeric_dtype(X[col])
    ]
    transformers: list[tuple] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if available_pay_columns:
        transformers.append(
            (
                "pay_status",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                available_pay_columns,
            )
        )
    if all_cat:
        other_cats = [col for col in all_cat if col not in available_pay_columns]
        if other_cats:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                        ]
                    ),
                    other_cats,
                )
            )
    if not transformers:
        transformers.append(
            (
                "empty",
                Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]),
                X.columns.tolist(),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_primary_model_features(df: pd.DataFrame) -> list[str]:
    base = [col for col in df.columns if col not in {"ID", "id", "customer_id", "client_id"}]
    return [col for col in base if col not in DEMOGRAPHIC_COLUMNS]
