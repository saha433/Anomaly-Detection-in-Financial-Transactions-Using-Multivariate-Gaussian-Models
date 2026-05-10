# Anomaly Detection in Financial Transactions

### Multivariate Gaussian + Isolation Forest | financial_fraud_detection_dataset

A labeled anomaly detection system that identifies suspicious financial transactions with a mathematically implemented multivariate Gaussian model.

---

## What it does

- Learns what "normal" transactions look like using a **Multivariate Gaussian model**
- Flags transactions with unusually low probability as anomalies
- Evaluates predictions against the dataset's `is_fraud` label
- Compares against **Isolation Forest** as a benchmark model
- Serves results through a **Streamlit web app** with risk scores and downloadable reports

---

## Project structure

```
anomaly_detector/
├── data/
│   └── financial_fraud_detection_dataset.csv   ← optional local copy
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

# 2. Train on the labeled fraud dataset
python train.py --data /Users/saha/Downloads/financial_fraud_detection_dataset.csv --sample-rows 200000

# 3. Launch the app
streamlit run app.py
```

---

## Column auto-detection

The code automatically finds your column names. Supported aliases:

| Role   | Detected column names                                  |
| ------ | ------------------------------------------------------ |
| Amount | amount, Amount, TransactionAmount, transaction_amount  |
| Time   | timestamp, Timestamp, Time, Date, TransactionDate      |
| Label  | is_fraud, isFraud, Class, Fraud, Label                 |

If your columns use different names, edit `COLUMN_ALIASES` in `src/features.py`.

For `financial_fraud_detection_dataset.csv`, the feature pipeline uses:

- `amount_log`
- `amount_zscore`
- `hour_sin`
- `hour_cos`
- `time_since_last_transaction`
- `spending_deviation_score`
- `velocity_score`
- `geo_anomaly_score`

---

## Running without fraud labels

If your dataset has no fraud label column, use unsupervised mode:

```bash
python train.py --no-labels
```

The model flags the bottom 0.5% of transactions by probability (adjustable in the app sidebar).

---

## The math

The model trains only on normal transactions, where `is_fraud == False`.

For a transaction vector \(x \in \mathbb{R}^k\), the mean vector is:

```text
mu = (1 / m) * sum(x_i)
```

The covariance matrix is:

```text
Sigma = (1 / (m - 1)) * sum((x_i - mu)(x_i - mu)^T)
```

The multivariate Gaussian probability density is:

```text
p(x) = 1 / sqrt((2*pi)^k * |Sigma|)
       * exp(-0.5 * (x - mu)^T * Sigma^-1 * (x - mu))
```

This project implements that formula directly in `src/model.py`. It does not call `scipy.stats.multivariate_normal.pdf`; it computes the Mahalanobis distance, determinant term, and exponential density explicitly.

Transactions with lower `p(x)` are less likely under the learned normal distribution and are flagged as anomalies.

The covariance matrix is the key insight: it captures _relationships_ between features. A large transaction at 3am from an unknown merchant is more suspicious than either signal alone.

**Why F1 and not accuracy?** With 0.17% fraud rate, predicting "all normal" gives 99.8% accuracy but catches zero fraud. F1 balances precision and recall, making it the right metric for imbalanced problems.

---

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn` · `streamlit` · `joblib`

---

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py` → Deploy
