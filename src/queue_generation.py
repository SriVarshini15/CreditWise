from __future__ import annotations

import pandas as pd

from src.explain import local_risk_reasons
from src.risk_queue import assign_risk_band, build_review_queue


def generate_review_queue(
    df: pd.DataFrame,
    X: pd.DataFrame,
    pipeline,
    reference_values: pd.Series,
    id_col: str,
    model_version: str,
    top_n: int = 500,
) -> pd.DataFrame:
    scored = df.copy()
    scored["predicted_default_probability"] = pipeline.predict_proba(X)[:, 1]
    scored["risk_band"] = scored["predicted_default_probability"].apply(assign_risk_band)
    queue = build_review_queue(
        scored,
        id_col=id_col,
        score_col="predicted_default_probability",
        top_n=top_n,
    )

    customers_by_id = X.set_index(df[id_col])
    reasons = []
    for customer_id in queue["customer_id"]:
        reasons.append(
            local_risk_reasons(
                pipeline,
                customers_by_id.loc[customer_id],
                reference_values,
                X.columns.tolist(),
                top_k=3,
            )
        )
    queue["primary_risk_reason"] = [
        items[0] if items else "The model score reflects combined signals; no single feature met the explanation threshold."
        for items in reasons
    ]
    queue["secondary_risk_reason"] = [items[1] if len(items) > 1 else "" for items in reasons]
    queue["tertiary_risk_reason"] = [items[2] if len(items) > 2 else "" for items in reasons]
    queue["review_status"] = "Pending"
    queue["recommended_action"] = "Unassigned"
    queue["model_version"] = model_version
    return queue
