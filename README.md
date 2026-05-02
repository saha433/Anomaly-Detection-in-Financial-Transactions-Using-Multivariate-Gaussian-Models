# Anomaly Detection in Financial Transactions
### Multivariate Gaussian + Isolation Forest | bank_transactions_data_2

A production-ready unsupervised anomaly detection system that identifies suspicious financial transactions without needing labeled fraud data.

---

## What it does

- Learns what "normal" transactions look like using a **Multivariate Gaussian model**
- Flags transactions with unusually low probability as anomalies
- Compares against **Isolation Forest** as a benchmark model
- Serves results through a **Streamlit web app** with risk scores and downloadable reports

---

## Project structure

```
anomaly_detector/
├── data/
│   └── bank_transactions_data_2.csv   ← your dataset goes here
├── src/
│   ├── features.py    ← data loading, feature engineering, scaling
│   ├── model.py       ← MVG and Isolation Forest implementations
│   └── evaluate.py    ← metrics, plots, model comparison
├── models/            ← saved model files (auto-created)
├── outputs/           ← saved plots (auto-created)
├── train.py           ← full training pipeline (run this first)
├── app.py             ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put your CSV in the data folder
cp /path/to/bank_transactions_data_2.csv data/

# 3. Train the model
python train.py

# 4. Launch the app
streamlit run app.py
```

---

## Column auto-detection

The code automatically finds your column names. Supported aliases:

| Role   | Detected column names |
|--------|-----------------------|
| Amount | Amount, TransactionAmount, amt |
| Time   | Time, Timestamp, Date, TransactionDate |
| Label  | Class, isFraud, Fraud, Label |

If your columns use different names, edit `COLUMN_ALIASES` in `src/features.py`.

---

## Running without fraud labels

If your dataset has no fraud label column, use unsupervised mode:

```bash
python train.py --no-labels
```

The model flags the bottom 0.5% of transactions by probability (adjustable in the app sidebar).

---

## The math (in plain English)

**Multivariate Gaussian** estimates the mean vector μ and covariance matrix Σ from normal transactions. For any new transaction x, it computes p(x) — the probability of being "normal." Transactions with p(x) below threshold ε are flagged.

The covariance matrix is the key insight: it captures *relationships* between features. A large transaction at 3am from an unknown merchant is more suspicious than either signal alone.

**Why F1 and not accuracy?** With 0.17% fraud rate, predicting "all normal" gives 99.8% accuracy but catches zero fraud. F1 balances precision and recall, making it the right metric for imbalanced problems.

---

## Tech stack

`pandas` · `numpy` · `scipy` · `scikit-learn` · `matplotlib` · `seaborn` · `streamlit` · `joblib`

---

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py` → Deploy

You get a public URL in under 2 minutes. Put it on your resume.
