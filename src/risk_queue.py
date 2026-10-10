from __future__ import annotations

import pandas as pd


def assign_risk_band(probability: float) -> str:
    if probability >= 0.7:
        return "Critical"
    if probability >= 0.5:
        return "High"
    if probability >= 0.3:
        return "Moderate"
    return "Low"


def build_review_queue(
    df: pd.DataFrame,
    id_col: str,
    score_col: str,
    reason_col: str = "primary_risk_reason",
    top_n: int = 500,
) -> pd.DataFrame:
    queue = df[[id_col, score_col]].copy()
    queue = queue.rename(columns={id_col: "customer_id", score_col: "predicted_default_probability"})
    queue["risk_band"] = queue["predicted_default_probability"].apply(assign_risk_band)
    queue["risk_rank"] = queue["predicted_default_probability"].rank(method="first", ascending=False).astype(int)
    queue = queue.sort_values(["predicted_default_probability", "customer_id"], ascending=[False, True]).reset_index(drop=True)
    queue["risk_rank"] = range(1, len(queue) + 1)
    queue["primary_risk_reason"] = "Behavioral risk indicator"
    queue["secondary_risk_reason"] = "Repayment history trend"
    queue["review_status"] = "Pending"
    queue["recommended_action"] = "Unassigned"
    if len(queue) > top_n:
        queue = queue.iloc[:top_n].copy()
    queue["risk_rank"] = range(1, len(queue) + 1)
    return queue
