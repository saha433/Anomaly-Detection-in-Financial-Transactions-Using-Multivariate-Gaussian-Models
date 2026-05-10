import argparse
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from src.features  import load_data, engineer_features, scale_features
from src.model     import MultivariateGaussianDetector, IsolationForestDetector
from src.evaluate  import print_report, plot_probability_distribution, plot_confusion_matrix, plot_precision_recall, compare_models

DEFAULT_DATA = "/Users/saha/Downloads/financial_fraud_detection_dataset.csv"
if not os.path.exists(DEFAULT_DATA):
    DEFAULT_DATA = "data/financial_fraud_detection_dataset.csv"


def main(data_path: str, has_labels: bool = True, sample_rows: int | None = None):
    print("\n" + "="*60)
    print("  Anomaly Detection — financial_fraud_detection_dataset")
    print("="*60 + "\n")

    print("[1/5] Loading data...")
    df = load_data(data_path, nrows=sample_rows)

    print("\n[2/5] Engineering features...")
    X, y, feature_names = engineer_features(df)
    X_scaled = scale_features(X, fit=True)
    print(f"  Features used: {feature_names}")

    print("\n[3/5] Splitting data...")
    if has_labels and y is not None:
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        X_normal = X_train[y_train == 0]
        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Normal (train): {len(X_normal):,}")
    else:
        X_train = X_val = X_scaled
        y_train = y_val = None
        X_normal = X_scaled
        has_labels = False
        print(f"  Unsupervised mode: {len(X_normal):,} transactions")

    print("\n[4/5] Fitting models...")
    os.makedirs("models", exist_ok=True)

    # Multivariate Gaussian
    mvg = MultivariateGaussianDetector()
    mvg.fit(X_normal)

    if has_labels:
        mvg.tune_epsilon(X_val, y_val)
    else:
        mvg.set_epsilon_percentile(X_val, percentile=0.5)

    mvg.save()

    # Isolation Forest
    iso = IsolationForestDetector(contamination=0.002 if not has_labels else max(float(y_train.mean()), 1e-5))
    iso.fit(X_normal)
    iso.save()

    print("\n[5/5] Evaluating...")

    mvg_preds  = mvg.predict(X_val)
    iso_preds  = iso.predict(X_val)
    p          = mvg.score(X_val)
    mvg_risk   = mvg.risk_score(X_val)
    iso_risk   = iso.risk_score(X_val)

    if has_labels:
        print_report(y_val, mvg_preds, "Multivariate Gaussian")
        print_report(y_val, iso_preds, "Isolation Forest")
        compare_models(y_val, {"MVG": mvg_preds, "Isolation Forest": iso_preds})
        plot_probability_distribution(p, y_val, mvg.epsilon)
        plot_confusion_matrix(y_val, mvg_preds, "MVG")
        plot_confusion_matrix(y_val, iso_preds, "Isolation Forest")
        plot_precision_recall(y_val, {"MVG": mvg_risk, "Isolation Forest": iso_risk})
    else:
        n_flagged = mvg_preds.sum()
        print(f"\n  MVG flagged {n_flagged:,} anomalies ({n_flagged/len(mvg_preds)*100:.2f}%)")
        plot_probability_distribution(p, None, mvg.epsilon)

    print("\nDone! Models saved in ./models/  |  Plots saved in ./outputs/")
    print("Next step: run  streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="Path to your CSV file")
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="Optional row limit for quick experiments on very large CSV files")
    parser.add_argument("--no-labels", action="store_true",
                        help="Run in unsupervised mode (no fraud labels needed)")
    args = parser.parse_args()
    main(args.data, has_labels=not args.no_labels, sample_rows=args.sample_rows)
