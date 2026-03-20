"""Sklearn-based classifiers (LR, RandomForest, XGBoost, LinearSVM)."""

import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.training.metrics import MetricsComputer

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter grids
# ──────────────────────────────────────────────────────────────────────────────

def get_param_grid(clf_type: str, clf_config: dict) -> List[Dict[str, Any]]:
    """
    Build a flat list of hyperparameter dicts from config section.

    Args:
        clf_type: classifier key ('logistic_regression', 'random_forest', etc.)
        clf_config: dict from YAML for this classifier

    Returns:
        List of param dicts, one per combination to try
    """
    import itertools

    if clf_type == "logistic_regression":
        Cs = clf_config.get("C", [0.01, 0.1, 1.0, 10.0, 100.0])
        return [{"C": c, "max_iter": 1000, "solver": "lbfgs",
                 "multi_class": "multinomial"} for c in Cs]

    elif clf_type == "random_forest":
        n_list    = clf_config.get("n_estimators", [100, 300])
        depth_list = clf_config.get("max_depth", [None, 10, 20])
        return [{"n_estimators": n, "max_depth": d, "n_jobs": -1, "random_state": 42}
                for n, d in itertools.product(n_list, depth_list)]

    elif clf_type == "xgboost":
        if not HAS_XGB:
            return []
        n_list  = clf_config.get("n_estimators", [100, 300])
        d_list  = clf_config.get("max_depth",    [3, 6])
        lr_list = clf_config.get("learning_rate", [0.1, 0.05])
        return [{"n_estimators": n, "max_depth": d, "learning_rate": lr,
                 "use_label_encoder": False, "eval_metric": "mlogloss",
                 "random_state": 42, "n_jobs": -1}
                for n, d, lr in itertools.product(n_list, d_list, lr_list)]

    elif clf_type == "svm":
        Cs = clf_config.get("C", [0.1, 1.0, 10.0])
        return [{"C": c, "max_iter": 2000, "random_state": 42} for c in Cs]

    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


# ──────────────────────────────────────────────────────────────────────────────
# Build classifier
# ──────────────────────────────────────────────────────────────────────────────

def build_classifier(clf_type: str, params: dict):
    """Instantiate an sklearn classifier from type + params."""
    if clf_type == "logistic_regression":
        return LogisticRegression(**params)
    elif clf_type == "random_forest":
        return RandomForestClassifier(**params)
    elif clf_type == "xgboost":
        return xgb.XGBClassifier(**params)
    elif clf_type == "svm":
        # LinearSVC is fast but needs CalibratedClassifierCV for probabilities
        base = LinearSVC(**params)
        return CalibratedClassifierCV(base, cv=3)
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


# ──────────────────────────────────────────────────────────────────────────────
# Train + evaluate one config
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    clf_type: str,
    params: Dict[str, Any],
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    num_classes: int,
    task: str,
) -> Dict[str, Any]:
    """
    Fit one classifier and evaluate on val and test.

    Returns dict with val_metrics, test_metrics, params.
    """
    clf = build_classifier(clf_type, params)
    clf.fit(X_train, y_train)

    val_probs  = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)

    val_metrics  = MetricsComputer.compute_probabilities_metrics(
        val_probs,  y_val,  num_classes, task)
    test_metrics = MetricsComputer.compute_probabilities_metrics(
        test_probs, y_test, num_classes, task)

    return {
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "params":       params,
        "classifier":   clf,    # in-memory, caller may save with joblib
    }
