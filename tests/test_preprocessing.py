import pandas as pd

from src.preprocessing import build_preprocessor, prepare_feature_frame


def test_prepare_feature_frame_excludes_id_and_target():
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "LIMIT_BAL": [1000, 2000, 3000],
        "PAY_0": [0, 1, 2],
        "default payment next month": [0, 1, 0],
    })
    X, y, features = prepare_feature_frame(df, "default payment next month", "ID")
    assert "ID" not in X.columns
    assert "default payment next month" not in X.columns
    assert len(features) == 2
    assert y.shape[0] == 3


def test_build_preprocessor_handles_numeric_and_pay_columns():
    df = pd.DataFrame({
        "LIMIT_BAL": [500, 1000, 2000],
        "PAY_0": [0, 1, 2],
        "PAY_2": [1, 0, 1],
        "BILL_AMT1": [200, 400, 600],
    })
    pre = build_preprocessor(df, categorical_columns=[], pay_columns=["PAY_0", "PAY_2"])
    transformed = pre.fit_transform(df)
    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0
