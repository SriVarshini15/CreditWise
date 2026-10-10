import pandas as pd

from src.data_validation import assert_id_and_target_excluded
from src.preprocessing import prepare_feature_frame


def test_id_and_target_excluded_from_model_matrix():
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "LIMIT_BAL": [100, 200, 300],
        "PAY_0": [0, 1, 0],
        "default payment next month": [0, 1, 0],
    })
    X, y, features = prepare_feature_frame(df, "default payment next month", "ID")
    assert "ID" not in X.columns
    assert "default payment next month" not in X.columns
    assert "LIMIT_BAL" in X.columns
    assert_id_and_target_excluded(X, "ID", "default payment next month")
