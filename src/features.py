import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

COLUMN_ALIASES = {
    "amount": ["amount", "Amount", "TransactionAmount", "transaction_amount", "amt"],
    "time": ["timestamp", "Timestamp", "Time", "time", "Date", "date", "TransactionDate"],
    "label": ["is_fraud", "isFraud", "Class", "class", "Fraud", "fraud", "Label", "label"],
}

SCALER_PATH = "models/scaler.pkl"
FRAUD_DATA_NUMERIC_FEATURES = [
    "time_since_last_transaction",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
]


def detect_column(df: pd.DataFrame, role: str) -> str | None:
    """Find the actual column name in df for a given role (amount/time/label)."""
    for candidate in COLUMN_ALIASES[role]:
        if candidate in df.columns:
            return candidate
    normalized = {str(col).lower(): col for col in df.columns}
    for candidate in COLUMN_ALIASES[role]:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def load_data(filepath: str, nrows: int | None = None) -> pd.DataFrame:
    """Load a financial transaction CSV or Excel file."""
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, nrows=nrows)
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath, nrows=nrows)
    else:
        df = pd.read_csv(filepath, nrows=nrows)

    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """
    Build model features from raw dataframe.
    Returns: X (features array), y (labels or None), feature_names
    """
    df = df.copy()

    feature_cols = []

    amount_col = detect_column(df, "amount")
    if amount_col:
        df["amount_log"] = np.log1p(df[amount_col].clip(lower=0))
        df["amount_zscore"] = (df[amount_col] - df[amount_col].mean()) / (df[amount_col].std() + 1e-9)
        feature_cols += ["amount_log", "amount_zscore"]
        print(f"  Amount column detected: '{amount_col}'")
    else:
        print("  WARNING: No amount column found. Skipping amount features.")

    time_col = detect_column(df, "time")
    if time_col:
        col = df[time_col]
        if pd.api.types.is_numeric_dtype(col):
            hour = (col / 3600) % 24
        else:
            # parse as datetime
            col_dt = pd.to_datetime(col, errors="coerce")
            hour = col_dt.dt.hour + col_dt.dt.minute / 60
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feature_cols += ["hour_sin", "hour_cos"]
        print(f"  Time column detected: '{time_col}'")
    else:
        print("  WARNING: No time column found. Skipping time features.")

    fraud_numeric = [c for c in FRAUD_DATA_NUMERIC_FEATURES if c in df.columns]
    if fraud_numeric:
        feature_cols += fraud_numeric
        print(f"  Fraud-dataset numeric features detected: {fraud_numeric}")

    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    if v_cols:
        top_v = [c for c in ["V1","V2","V3","V4","V5","V10","V12","V14","V17"] if c in df.columns]
        feature_cols += top_v
        print(f"  PCA V-features detected: {top_v}")

    if not feature_cols:
        label_col = detect_column(df, "label")
        exclude = {amount_col, time_col, label_col} - {None}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in exclude]
        print(f"  Using all numeric columns as features: {feature_cols}")

    label_col = detect_column(df, "label")
    y = df[label_col].astype(int).values if label_col else None
    if label_col:
        print(f"  Label column detected: '{label_col}' | fraud rate: {y.mean()*100:.3f}%")
    else:
        print("  No label column found — running in unsupervised mode.")

    feature_cols = [c for c in feature_cols if c in df.columns]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    X = df[feature_cols].values.astype(float)

    print(f"  Feature matrix shape: {X.shape}")
    return X, y, feature_cols


def scale_features(X: np.ndarray, fit: bool = True) -> np.ndarray:
    """Scale features. fit=True trains a new scaler, fit=False loads saved one."""
    os.makedirs("models", exist_ok=True)
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
        print(f"  Scaler saved to {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
    return X_scaled
