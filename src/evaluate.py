import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
import os

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_report(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"):
    """Print precision, recall, F1 for both classes."""
    print(f"\n{'='*50}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"], zero_division=0))
    if y_true.sum() > 0:
        auprc = average_precision_score(y_true, y_pred)
        print(f"  AU-PRC (area under precision-recall curve): {auprc:.4f}")


def plot_probability_distribution(p: np.ndarray, y: np.ndarray, epsilon: float, save: bool = True):
    """
    Histogram of log p(x) split by class.
    The signature chart — shows how well the model separates fraud from normal.
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    log_p = np.log(p + 1e-300)
    log_eps = np.log(epsilon + 1e-300)

    if y is not None:
        ax.hist(log_p[y == 0], bins=120, alpha=0.65, color="#378ADD", label="Normal transactions")
        ax.hist(log_p[y == 1], bins=80,  alpha=0.75, color="#E24B4A", label="Fraud transactions")
    else:
        ax.hist(log_p, bins=120, alpha=0.65, color="#378ADD", label="All transactions")

    ax.axvline(log_eps, color="#2C2C2A", linestyle="--", linewidth=1.5,
               label=f"Threshold ε = {epsilon:.2e}")
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1e6],
                     log_p.min(), log_eps, alpha=0.06, color="#E24B4A")

    ax.set_xlabel("log p(x)  [log probability density]", fontsize=12)
    ax.set_ylabel("Number of transactions", fontsize=12)
    ax.set_title("Multivariate Gaussian — probability distribution by class", fontsize=13)
    ax.legend(fontsize=11)
    ax.text(log_eps + 0.2, ax.get_ylim()[1] * 0.85, "← flagged as anomaly",
            color="#E24B4A", fontsize=10)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "probability_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "MVG", save: bool = True):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Fraud"],
                yticklabels=["Normal", "Fraud"], ax=ax)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=12)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.lower().replace(' ','_')}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


def plot_precision_recall(y_true: np.ndarray, scores: dict, save: bool = True):
    """
    scores = {"MVG": risk_scores_array, "Isolation Forest": risk_scores_array}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#185FA5", "#993C1D", "#3B6D11"]

    for (name, score), color in zip(scores.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax.plot(recall, precision, label=f"{name}  (AP={ap:.3f})", color=color, linewidth=2)

    # baseline (random classifier)
    baseline = y_true.mean()
    ax.axhline(baseline, linestyle="--", color="#888780", linewidth=1, label=f"Baseline (AP={baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.show()


def compare_models(y_true: np.ndarray, predictions: dict):
    """
    predictions = {"MVG": y_pred_array, "Isolation Forest": y_pred_array}
    Side-by-side summary table.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    print(f"\n{'='*60}")
    print(f"  Model Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'-'*60}")
    for name, preds in predictions.items():
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        print(f"  {name:<23} {p:>10.4f} {r:>10.4f} {f:>10.4f}")
    print()
