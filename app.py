import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os, sys, io

sys.path.insert(0, os.path.dirname(__file__))
from src.features import engineer_features, scale_features
from src.model    import MultivariateGaussianDetector, IsolationForestDetector

st.set_page_config(
    page_title="Anomaly Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("Financial Transaction Anomaly Detector")
st.caption("Multivariate Gaussian + Isolation Forest | bank_transactions_data_2")

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio("Detection model", ["Multivariate Gaussian", "Isolation Forest", "Both"])
    mode = st.radio("Mode", ["Use pre-trained model", "Train from uploaded file"])

    if mode == "Train from uploaded file":
        contamination = st.slider("Expected fraud rate (%)", 0.01, 5.0, 0.2, step=0.01) / 100
        has_labels = st.checkbox("My CSV has a fraud label column", value=True)
    else:
        contamination = 0.002
        has_labels = True

    threshold_pct = st.slider(
        "Sensitivity (if no labels) — flag bottom X%",
        min_value=0.1, max_value=5.0, value=0.5, step=0.1,
        help="Only used when no fraud labels are available"
    )

st.divider()

uploaded_file = st.file_uploader(
    "Upload your bank_transactions_data_2.csv (or any transaction CSV)",
    type=["csv", "xlsx"],
    help="The app auto-detects Amount, Time, and label columns."
)

if uploaded_file is None:
    st.info("Upload a CSV to get started. The model will auto-detect your column names.")
    st.stop()

@st.cache_data(show_spinner="Loading data...")
def load_df(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)

df = load_df(uploaded_file)

st.subheader("Dataset preview")
col1, col2, col3 = st.columns(3)
col1.metric("Total transactions", f"{len(df):,}")
col2.metric("Columns", df.shape[1])

from src.features import detect_column
label_col = detect_column(df, "label")
if label_col:
    fraud_count = int(df[label_col].sum())
    col3.metric("Known frauds", f"{fraud_count:,} ({fraud_count/len(df)*100:.3f}%)")
else:
    col3.metric("Label column", "Not found — unsupervised mode")

st.dataframe(df.head(10), use_container_width=True)

st.divider()

with st.spinner("Engineering features..."):
    X, y, feature_names = engineer_features(df)

st.success(f"Features ready: {feature_names}")

run_btn = st.button("Run anomaly detection", type="primary", use_container_width=True)

if not run_btn:
    st.stop()

with st.spinner("Training model..."):
    X_scaled = scale_features(X, fit=True)
    y_for_train = y if (has_labels and y is not None) else None
    X_normal = X_scaled[y == 0] if (y_for_train is not None) else X_scaled

    # --- Multivariate Gaussian ---
    mvg = MultivariateGaussianDetector()
    mvg.fit(X_normal)

    if y_for_train is not None:
        from sklearn.model_selection import train_test_split
        _, X_val, _, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        mvg.tune_epsilon(X_val, y_val)
    else:
        mvg.set_epsilon_percentile(X_scaled, percentile=threshold_pct)
        y_val = None

    p          = mvg.score(X_scaled)
    mvg_preds  = mvg.predict(X_scaled)
    mvg_risk   = mvg.risk_score(X_scaled)

    # --- Isolation Forest ---
    iso = IsolationForestDetector(contamination=contamination)
    iso.fit(X_normal)
    iso_preds = iso.predict(X_scaled)
    iso_risk  = iso.risk_score(X_scaled)

st.success("Detection complete!")
st.divider()
st.subheader("Results")

r1, r2, r3 = st.columns(3)
r1.metric("MVG flagged", f"{mvg_preds.sum():,}", f"{mvg_preds.mean()*100:.2f}% of total")
r2.metric("Isolation Forest flagged", f"{iso_preds.sum():,}", f"{iso_preds.mean()*100:.2f}% of total")
agreed = ((mvg_preds == 1) & (iso_preds == 1)).sum()
r3.metric("Both models agree", f"{agreed:,} anomalies")

st.subheader("Flagged transactions (MVG)")

result_df = df.copy()
result_df["mvg_anomaly"]   = mvg_preds
result_df["iso_anomaly"]   = iso_preds
result_df["mvg_risk_score"] = mvg_risk.round(1)
result_df["iso_risk_score"] = iso_risk.round(1)
result_df["both_agree"]    = ((mvg_preds == 1) & (iso_preds == 1)).astype(int)

flagged = result_df[result_df["mvg_anomaly"] == 1].sort_values("mvg_risk_score", ascending=False)

st.dataframe(
    flagged.head(200),
    use_container_width=True,
    column_config={
        "mvg_risk_score": st.column_config.ProgressColumn("MVG risk", min_value=0, max_value=100),
        "iso_risk_score": st.column_config.ProgressColumn("ISO risk", min_value=0, max_value=100),
    }
)

csv_bytes = flagged.to_csv(index=False).encode()
st.download_button("Download flagged transactions CSV", csv_bytes, "flagged_anomalies.csv", "text/csv")

st.divider()

st.subheader("Probability distribution chart")
st.caption("Shows how well the model separates anomalies from normal transactions")

fig, ax = plt.subplots(figsize=(10, 4))
log_p   = np.log(p + 1e-300)
log_eps = np.log(mvg.epsilon + 1e-300)

if y is not None:
    ax.hist(log_p[y == 0], bins=100, alpha=0.6, color="#378ADD", label="Normal")
    ax.hist(log_p[y == 1], bins=60,  alpha=0.7, color="#E24B4A", label="Fraud")
else:
    ax.hist(log_p, bins=100, alpha=0.6, color="#378ADD", label="All transactions")

ax.axvline(log_eps, color="#2C2C2A", linestyle="--", linewidth=1.5, label=f"Threshold ε")
ax.set_xlabel("log p(x)")
ax.set_ylabel("Count")
ax.legend()
ax.set_title("Transaction probability density")
plt.tight_layout()
st.pyplot(fig)

st.subheader("Risk score distribution")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.hist(mvg_risk, bins=100, color="#534AB7", alpha=0.7)
ax2.axvline(70, color="#E24B4A", linestyle="--", label="High-risk threshold (70)")
ax2.set_xlabel("Risk score (0–100)")
ax2.set_ylabel("Count")
ax2.legend()
ax2.set_title("MVG risk scores — all transactions")
plt.tight_layout()
st.pyplot(fig2)

if y is not None and y.sum() > 0:
    st.divider()
    st.subheader("Model performance metrics")

    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    import seaborn as sns

    mc1, mc2 = st.columns(2)

    for col_ui, name, preds in [(mc1, "Multivariate Gaussian", mvg_preds), (mc2, "Isolation Forest", iso_preds)]:
        with col_ui:
            st.markdown(f"**{name}**")
            p_s = precision_score(y, preds, zero_division=0)
            r_s = recall_score(y, preds, zero_division=0)
            f_s = f1_score(y, preds, zero_division=0)
            st.metric("Precision", f"{p_s:.4f}")
            st.metric("Recall",    f"{r_s:.4f}")
            st.metric("F1 Score",  f"{f_s:.4f}")

            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y, preds)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal","Fraud"], yticklabels=["Normal","Fraud"], ax=ax_cm)
            ax_cm.set_title(f"{name} — Confusion matrix")
            plt.tight_layout()
            st.pyplot(fig_cm)

st.divider()
st.caption("Built for Mathematical Foundations For Machine Learning Course Project | Dataset from Kaggle")
