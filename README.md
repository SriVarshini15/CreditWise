# CreditWise

CreditWise is a lightweight, explainable credit-card default risk platform built on the official UCI “Default of Credit Card Clients” dataset. It trains a leakage-free model, ranks customers for a TOP-500 review queue, and exposes the results through a Streamlit dashboard.

## Goals

- Predict next-month default risk for credit-card customers
- Rank accounts in a fixed-capacity review queue
- Explain the main behavioral drivers behind each prediction
- Recommend provisional action tiers for the top reviewed accounts
- Track model runs using MLflow locally

## Dataset and source

- Source: UCI Machine Learning Repository, dataset ID 350
- Official dataset: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients+dataset
- Data is downloaded programmatically through `ucimlrepo` and saved to `data/raw/credit_card_default.csv`
- The primary target is `default payment next month`
- The customer ID column is retained separately and excluded from model inputs
- Demographic variables (`SEX`, `AGE`, `EDUCATION`, `MARRIAGE`) are excluded from the primary deployed model to reduce fairness and governance risk

## Project structure

- `app.py`: Streamlit dashboard
- `src/download_data.py`: downloads and saves the raw dataset
- `src/data_loader.py`: schema and column inspection helpers
- `src/data_validation.py`: quality validation and assertions
- `src/preprocessing.py`: leakage-safe preprocessing pipeline
- `src/train.py`: training, evaluation, MLflow logging, queue generation
- `src/explain.py`: explanation helpers
- `src/risk_queue.py`: top-500 review queue logic
- `src/action_recommendations.py`: provisional action recommendations
- `src/fairness.py`: fairness audit scaffold
- `tests/`: focused automated tests

## Setup

```bash
cd /Users/srivarshini/Documents/Luminary
python3 -m pip install -r requirements.txt
```

## Download the official dataset

```bash
cd /Users/srivarshini/Documents/Luminary
PYTHONPATH=. python3 src/download_data.py
```

This saves the raw file to `data/raw/credit_card_default.csv`.

## Train the model and generate queue artifacts

```bash
cd /Users/srivarshini/Documents/Luminary
PYTHONPATH=. python3 src/train.py
```

Expected output includes:

- Model artifacts in `models/`
- Evaluable metrics in `reports/training_summary.json`
- Ranked export at `outputs/top_500_review_queue.csv`
- Local MLflow runs in `mlruns/`

Training fits both models and may take several minutes. If trained model artifacts and a training summary already exist and you only need to refresh the queue or its explanations, run:

```bash
PYTHONPATH=. python3 src/score_queue.py
```

## Run the dashboard

```bash
cd /Users/srivarshini/Documents/Luminary
PYTHONPATH=. streamlit run app.py
```

Open the local URL printed by Streamlit in the browser.

The dashboard includes an executive snapshot, interactive TOP-500 queue filters and downloads, a TOP-20 worklist, customer repayment/bill-payment history, and model/governance comparisons. Queue reason text is specific to each account and describes model-score comparisons, not causal effects.

## Deploy on Replit

This repository includes a Replit run and deployment configuration. Import
`https://github.com/SriVarshini15/CreditWise` into Replit, then select **Run** to
start the dashboard. On a fresh workspace, the launcher downloads the UCI data
and trains the candidate models if their artifacts are not present; the first
startup can take several minutes. It reuses available artifacts and only
regenerates the queue when needed.

To publish it, use Replit's **Publish** flow and choose a web deployment. The
app binds to Replit's `PORT` and listens on `0.0.0.0`. Model and queue files are
generated locally at startup because they are excluded from Git. Review Replit's
current deployment pricing and resource limits before publishing; the first
deployment startup includes model training.

## Tests

```bash
cd /Users/srivarshini/Documents/Luminary
PYTHONPATH=. pytest -q
```

## Ethical and governance notes

- The primary model excludes direct demographic inputs
- Each queue reason is generated from that account's actual model inputs. For each feature, the app compares the model score with a one-feature-at-a-time version where that value is replaced by its training-set reference (median for numeric features; most frequent status for repayment-status fields). Reasons are shown only when the replacement lowers the score by at least 0.5 percentage points.
- Repayment-status explanations distinguish documented positive delay codes from `-2` (no consumption), `-1` (paid duly), and `0` (revolving credit). A status code of `0` is not described as a payment delay.
- These local comparisons explain why the model assigned a higher score; they do not establish causation or confirm that a customer will default. If no individual feature meets the explanation threshold, the dashboard says the score reflects combined signals.
- Model outputs are decision-support signals and not final credit decisions
- Human review is required before any action is taken

## Limitations

- This project is designed as a local, explainable demonstration suitable for a hackathon or internal prototype
- Actual bank deployment would require formal governance, policy review, and compliant controls
- Model performance should be interpreted in the context of a fixed-capacity review workflow rather than raw predictive accuracy alone
