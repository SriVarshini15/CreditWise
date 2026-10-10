import pandas as pd

from src.data_validation import validate_dataset


def test_validate_dataset_reports_shape_and_warnings():
    df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "LIMIT_BAL": [1000, 2000, 3000, 4000],
        "PAY_0": [0, 1, 2, 0],
        "default payment next month": [0, 1, 0, 1],
    })
    report = validate_dataset(df, "default payment next month", "ID")
    assert report["shape"] == (4, 4)
    assert "target_distribution" in report
    assert isinstance(report["warnings"], list)
