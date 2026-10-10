from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "credit_card_default.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_credit_card_default.csv"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = ROOT_DIR / "reports"
MLFLOW_TRACKING_URI = "file:///" + str((ROOT_DIR / "mlruns").resolve())
TARGET_COLUMN_CANDIDATES = [
    "default payment next month",
    "default.payment.next.month",
    "default_next_month",
    "default",
    "default_payment_next_month",
]
ID_COLUMN_CANDIDATES = ["id", "customer_id", "client_id", "account_id", "credit_card_id"]
DEMOGRAPHIC_COLUMNS = ["SEX", "AGE", "EDUCATION", "MARRIAGE"]
PAY_COLUMNS = [f"PAY_{i}" for i in range(7)]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]
