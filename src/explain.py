from __future__ import annotations

import numpy as np
import pandas as pd


PAY_STATUS_MEANINGS = {
    -2: "no consumption",
    -1: "paid duly",
    0: "revolving credit",
}


def summarize_feature_importance(pipeline, feature_names: list[str]) -> list[dict]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed_names = list(preprocessor.get_feature_names_out())
    if hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_).reshape(-1))
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    else:
        return []

    indices = np.argsort(importance)[::-1][:10]
    return [
        {
            "feature_name": transformed_names[index],
            "importance": float(importance[index]),
            "label": _label_feature(transformed_names[index]),
        }
        for index in indices
    ]


def local_risk_reasons(
    pipeline,
    row: pd.Series,
    reference_values: pd.Series,
    feature_names: list[str],
    top_k: int = 3,
    minimum_score_change: float = 0.005,
) -> list[str]:
    """Explain score increases by comparing each value with its training-set reference.

    Each reason reports a one-feature-at-a-time perturbation, not a causal effect.
    """
    features = [name for name in feature_names if name in row.index and name in reference_values.index]
    if not features:
        return []

    customer = row.loc[features].to_dict()
    current_score = float(pipeline.predict_proba(pd.DataFrame([customer]))[0, 1])
    counterfactuals = []
    changed_features = []

    for feature in features:
        if pd.isna(customer[feature]):
            continue
        reference_value = reference_values[feature]
        if pd.isna(reference_value) or customer[feature] == reference_value:
            continue

        counterfactual = customer.copy()
        counterfactual[feature] = reference_value
        counterfactuals.append(counterfactual)
        changed_features.append(feature)

    if not counterfactuals:
        return []

    comparison_scores = pipeline.predict_proba(pd.DataFrame(counterfactuals))[:, 1]
    effects: list[tuple[float, str]] = []
    for feature, comparison_score in zip(changed_features, comparison_scores):
        score_change = current_score - float(comparison_score)
        if score_change < minimum_score_change:
            continue

        reason = _format_local_reason(
            feature,
            customer[feature],
            reference_values[feature],
            score_change,
        )
        effects.append((score_change, reason))

    effects.sort(key=lambda item: (-item[0], item[1]))
    return [reason for _, reason in effects[:top_k]]


def _format_local_reason(feature: str, actual_value, reference_value, score_change: float) -> str:
    change_pp = score_change * 100
    if feature.startswith("PAY_") and feature.removeprefix("PAY_").isdigit():
        code = int(actual_value)
        if code > 0:
            detail = f"records a {code}-month repayment delay"
        else:
            detail = f"has status “{PAY_STATUS_MEANINGS.get(code, f'code {code}')}”"
        return (
            f"{feature} {detail}; replacing it with the training-set typical status "
            f"({reference_value:g}) lowers this account's model score by {change_pp:.1f} percentage points."
        )

    if feature.startswith("BILL_AMT"):
        label = f"Billed balance ({feature})"
    elif feature.startswith("PAY_AMT"):
        label = f"Payment amount ({feature})"
    elif feature == "LIMIT_BAL":
        label = "Credit limit (LIMIT_BAL)"
    else:
        label = feature.replace("_", " ").title()

    return (
        f"{label} is {actual_value:,.0f} (training-set median: {reference_value:,.0f}); "
        f"replacing it with that reference lowers this account's model score by {change_pp:.1f} percentage points."
    )


def _label_feature(feature_name: str) -> str:
    if "PAY_" in feature_name:
        return "Repayment status"
    if "BILL_AMT" in feature_name:
        return "Billed balance"
    if "PAY_AMT" in feature_name:
        return "Payment amount"
    if "LIMIT" in feature_name:
        return "Credit limit"
    return feature_name.replace("_", " ").strip().title()
