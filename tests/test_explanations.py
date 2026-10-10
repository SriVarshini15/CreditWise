import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.explain import _format_local_reason, local_risk_reasons
from src.preprocessing import build_preprocessor


def test_local_reasons_reflect_customer_values_and_model_score():
    X_train = pd.DataFrame({"BILL_AMT1": np.arange(30, dtype=float)})
    y_train = (X_train["BILL_AMT1"] > 14).astype(int)
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", LogisticRegression(random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    reference_values = X_train.median()

    high_risk_reasons = local_risk_reasons(
        pipeline,
        X_train.iloc[-1],
        reference_values,
        ["BILL_AMT1"],
    )
    low_risk_reasons = local_risk_reasons(
        pipeline,
        X_train.iloc[0],
        reference_values,
        ["BILL_AMT1"],
    )

    assert len(high_risk_reasons) == 1
    assert "29" in high_risk_reasons[0]
    assert "lowers this account's model score" in high_risk_reasons[0]
    assert low_risk_reasons == []


def test_payment_amount_is_not_described_as_a_delay():
    reason = _format_local_reason("PAY_AMT1", 8500, 1000, 0.01)

    assert "payment amount" in reason.lower()
    assert "repayment delay" not in reason
