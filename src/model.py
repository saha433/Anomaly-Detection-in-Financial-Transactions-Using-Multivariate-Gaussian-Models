import numpy as np
import joblib
import os
from scipy.stats import multivariate_normal
from sklearn.ensemble import IsolationForest

MODEL_PATH = "models/mvg_params.pkl"
ISO_PATH   = "models/isolation_forest.pkl"


# ── Multivariate Gaussian ─────────────────────────────────────────────────────

class MultivariateGaussianDetector:
    """
    Unsupervised anomaly detector based on multivariate Gaussian distribution.
    Trains on normal transactions only. Flags low-probability transactions.
    """

    def __init__(self):
        self.mu = None
        self.sigma = None
        self.epsilon = None
        self.mvn = None

    def fit(self, X_normal: np.ndarray) -> "MultivariateGaussianDetector":
        """Estimate mu and sigma from normal (non-fraud) transactions."""
        self.mu    = np.mean(X_normal, axis=0)
        self.sigma = np.cov(X_normal, rowvar=False)

        # regularize covariance to avoid singular matrix issues
        self.sigma += np.eye(self.sigma.shape[0]) * 1e-6

        self.mvn = multivariate_normal(mean=self.mu, cov=self.sigma, allow_singular=True)
        print(f"MVG fitted on {len(X_normal):,} normal samples | features: {self.mu.shape[0]}")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return probability density p(x) for each transaction. Lower = more anomalous."""
        assert self.mvn is not None, "Call fit() first."
        return self.mvn.pdf(X)

    def tune_epsilon(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        Find the best probability threshold (epsilon) on a labeled validation set.
        Optimizes F1 score (better than accuracy for imbalanced data).
        """
        from sklearn.metrics import f1_score

        p = self.score(X_val)
        # search across bottom 5% of probability values
        thresholds = np.percentile(p, np.linspace(0.001, 5, 1000))

        best_f1, best_eps = 0.0, thresholds[0]
        for eps in thresholds:
            preds = (p < eps).astype(int)
            if preds.sum() == 0:
                continue
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_eps = eps

        self.epsilon = best_eps
        print(f"Best epsilon: {best_eps:.3e} | Best F1: {best_f1:.4f}")
        return best_eps

    def set_epsilon_percentile(self, X: np.ndarray, percentile: float = 0.5):
        """
        Set epsilon without labels: flag bottom `percentile`% as anomalies.
        Use this when you have NO fraud labels at all.
        """
        p = self.score(X)
        self.epsilon = np.percentile(p, percentile)
        n_flagged = (p < self.epsilon).sum()
        print(f"Epsilon set at {percentile}th percentile: {self.epsilon:.3e} | flagging {n_flagged} transactions")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions: 1 = anomaly, 0 = normal."""
        assert self.epsilon is not None, "Call tune_epsilon() or set_epsilon_percentile() first."
        return (self.score(X) < self.epsilon).astype(int)

    def risk_score(self, X: np.ndarray) -> np.ndarray:
        """Return a 0-100 risk score (100 = most anomalous) for UI display."""
        p = self.score(X)
        log_p = np.log(p + 1e-300)
        lo, hi = np.percentile(log_p, [1, 99])
        score = 100 * (1 - (log_p - lo) / (hi - lo + 1e-9))
        return np.clip(score, 0, 100)

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"mu": self.mu, "sigma": self.sigma, "epsilon": self.epsilon}, path)
        print(f"MVG model saved to {path}")

    def load(self, path: str = MODEL_PATH) -> "MultivariateGaussianDetector":
        data = joblib.load(path)
        self.mu      = data["mu"]
        self.sigma   = data["sigma"]
        self.epsilon = data["epsilon"]
        self.mvn     = multivariate_normal(mean=self.mu, cov=self.sigma, allow_singular=True)
        print(f"MVG model loaded from {path}")
        return self


# ── Isolation Forest (comparison model) ──────────────────────────────────────

class IsolationForestDetector:
    """Wrapper around sklearn IsolationForest for easy comparison."""

    def __init__(self, contamination: float = 0.002):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, X_normal: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X_normal)
        print(f"Isolation Forest fitted | contamination: {self.contamination}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # sklearn returns -1 for anomaly, 1 for normal
        return (self.model.predict(X) == -1).astype(int)

    def risk_score(self, X: np.ndarray) -> np.ndarray:
        raw = -self.model.score_samples(X)  # higher = more anomalous
        lo, hi = raw.min(), raw.max()
        return np.clip(100 * (raw - lo) / (hi - lo + 1e-9), 0, 100)

    def save(self, path: str = ISO_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str = ISO_PATH) -> "IsolationForestDetector":
        self.model = joblib.load(path)
        return self
