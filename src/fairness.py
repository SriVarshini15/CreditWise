from __future__ import annotations

import pandas as pd

from src.evaluate import compute_metrics


def fairness_audit(df: pd.DataFrame, score_col: str, target_col: str, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame(columns=["group", "accuracy", "precision", "recall", "f1", "roc_auc"])
    audits = []
    for group, group_df in df.groupby(group_col):
        y_true = group_df[target_col]
        y_scores = group_df[score_col]
        metrics = compute_metrics(y_true, y_scores)
        audits.append({
            "group": group,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        })
    return pd.DataFrame(audits)
