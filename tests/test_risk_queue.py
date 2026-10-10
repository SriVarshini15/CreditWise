import pandas as pd

from src.risk_queue import build_review_queue


def test_build_review_queue_orders_by_probability_and_caps_size():
    df = pd.DataFrame({
        "customer_id": ["A", "B", "C", "D"],
        "score": [0.9, 0.8, 0.7, 0.6],
    })
    queue = build_review_queue(df, id_col="customer_id", score_col="score", top_n=2)
    assert len(queue) == 2
    assert queue["customer_id"].tolist() == ["A", "B"]
    assert queue["risk_rank"].tolist() == [1, 2]
    assert queue["risk_band"].tolist()[0] == "Critical"
