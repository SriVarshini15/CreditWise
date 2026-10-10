from __future__ import annotations

import pandas as pd


def determine_action(row: pd.Series) -> str:
    prob = float(row.get("predicted_default_probability", 0.0))
    recent_delay = float(row.get("recent_repayment_delay", 0.0))
    payment_ratio = float(row.get("payment_to_bill_ratio", 1.0))
    if prob >= 0.65 or recent_delay >= 2.0:
        return "Tier A — CALL"
    if prob >= 0.45 or payment_ratio < 0.35:
        return "Tier B — LIMIT REVIEW"
    return "Tier C — PAYMENT PLAN"


def build_action_recommendations(queue: pd.DataFrame, use_top_20: bool = True) -> pd.DataFrame:
    actions = queue.copy()
    actions["recent_repayment_delay"] = actions.get("recent_repayment_delay", 0.0)
    actions["payment_to_bill_ratio"] = actions.get("payment_to_bill_ratio", 1.0)
    actions["recommended_action"] = actions.apply(determine_action, axis=1)
    actions["action_explanation"] = actions.apply(
        lambda r: (
            "Early outreach is recommended because recent repayment behavior and predicted default risk are elevated."
            if r["recommended_action"].startswith("Tier A")
            else "A human review of the customer's credit limit and account circumstances is recommended."
            if r["recommended_action"].startswith("Tier B")
            else "A reviewed repayment arrangement may be appropriate if the customer reports hardship and policy permits."
        ),
        axis=1,
    )
    if use_top_20:
        return actions.head(20).copy()
    return actions
