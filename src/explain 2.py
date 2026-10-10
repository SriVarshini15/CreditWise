from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_feature_importance(pipeline, feature_names: list[str]) -> list[dict]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_names = list(preprocessor.get_feature_names_out())
    if hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_).reshape(-1)
        importance = np.abs(coefficients)
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    else:
        return []
    rankings = np.argsort(importance)[::-1][:10]
    reasons = []
    for idx in rankings:
        raw_name = transformed_names[idx]
        contribution = float(importance[idx])
        reasons.append({
            "feature_name": raw_name,
            "importance": contribution,
            "label": _label_feature(raw_name),
        })
    return reasons


def explain_customer(pipeline, row: pd.Series, feature_names: list[str]) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(pd.DataFrame([row[feature_names]]))
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        scores = np.asarray(model.coef_).reshape(-1)
        feature_score = transformed[0] * scores
    elif hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_)
        feature_score = transformed[0] * scores
    else:
        return ["Model explains risk from recent repayment and bill behaviour."]
    transformed_names = list(preprocessor.get_feature_names_out())
    top_idx = np.argsort(np.abs(feature_score))[-3:][::-1]
    reasons = []
    for idx in top_idx:
        name = transformed_names[idx]
        reasons.append(_label_feature(name))
    return reasons[:3]


def _label_feature(feature_name: str) -> str:
    if "PAY_" in feature_name:
        return "Recent repayment delays detected"
    if "BILL_AMT" in feature_name:
        return "High outstanding balances relative to recent payments"
    if "PAY_AMT" in feature_name:
        return "Recent payment amounts are low relative to the account balance"
    if "LIMIT" in feature_name:
        return "Credit utilization and limit exposure are elevated"
    return "Behavioral credit risk signal is elevated"
