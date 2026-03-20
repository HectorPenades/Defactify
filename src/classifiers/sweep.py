"""Classifier sweep: train all configured classifiers and select the best."""

import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from src.classifiers.sklearn_classifiers import (
    get_param_grid, train_and_evaluate, HAS_XGB
)
from src.classifiers.mlp import train_mlp
from src.utils.results_tracker import record_result


def run_sweep(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    num_classes: int,
    task: str,
    config: Dict[str, Any],
    exp_dir: Path,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Train all classifiers defined in config['classifiers'] and pick the best.

    Results are saved to:
        <exp_dir>/sweep_results.json  — all configs tried
        <exp_dir>/final_metrics.json  — best overall result
        <exp_dir>/models/             — best sklearn model per type (.joblib)

    Returns:
        dict with 'best_classifier', 'best_val_f1_macro', 'all_results'
    """
    clf_config = config.get("classifiers", {})
    models_dir = exp_dir / "models"
    models_dir.mkdir(exist_ok=True)

    all_results: Dict[str, Any] = {}
    best_val_f1 = -1.0
    best_clf_name = None
    best_test_metrics = None

    # ── Binary class weights (1:5 imbalance) ────────────────────────────────
    class_weights_np = None
    class_weights_torch = None
    if task == "binary":
        import torch
        class_weights_torch = torch.tensor([5.0, 1.0], dtype=torch.float32)
        # For sklearn: sample_weight is computed at fit time

    # ── Sklearn classifiers ──────────────────────────────────────────────────
    SKLEARN_TYPES = ["logistic_regression", "random_forest", "svm"]
    if HAS_XGB:
        SKLEARN_TYPES.append("xgboost")

    for clf_type in SKLEARN_TYPES:
        cfg = clf_config.get(clf_type, {})
        if not cfg.get("enabled", True):
            continue

        print(f"\n  [{clf_type}]")
        param_grid = get_param_grid(clf_type, cfg)
        if not param_grid:
            print(f"    Skipped (not available or empty grid)")
            continue

        best_for_type = None
        best_val_f1_for_type = -1.0
        results_for_type = []

        # Sample weights for binary imbalance
        sample_weights = None
        if task == "binary":
            sample_weights = np.where(y_train == 0, 5.0, 1.0)

        for i, params in enumerate(param_grid):
            print(f"    Config {i+1}/{len(param_grid)}: {params}", end=" ... ", flush=True)
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from src.classifiers.sklearn_classifiers import build_classifier
                clf_obj = build_classifier(clf_type, params)
                if sample_weights is not None:
                    clf_obj.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    clf_obj.fit(X_train, y_train)

                val_probs  = clf_obj.predict_proba(X_val)
                test_probs = clf_obj.predict_proba(X_test)

                from src.training.metrics import MetricsComputer
                val_m  = MetricsComputer.compute_probabilities_metrics(
                    val_probs,  y_val,  num_classes, task)
                test_m = MetricsComputer.compute_probabilities_metrics(
                    test_probs, y_test, num_classes, task)

                f1 = val_m["f1_macro"]
                print(f"val_f1={f1:.4f}")
                results_for_type.append({
                    "params": {k: v for k, v in params.items()
                               if k not in ("n_jobs", "random_state",
                                            "use_label_encoder", "eval_metric")},
                    "val_f1_macro":  f1,
                    "val_bal_acc":   val_m["balanced_accuracy"],
                    "test_f1_macro": test_m["f1_macro"],
                })

                if f1 > best_val_f1_for_type:
                    best_val_f1_for_type = f1
                    best_for_type = {
                        "params": params,
                        "val_metrics":  val_m,
                        "test_metrics": test_m,
                        "classifier_obj": clf_obj,
                    }

            except Exception as e:
                print(f"ERROR: {e}")

        if best_for_type is not None:
            all_results[clf_type] = {
                "best_params":    {k: v for k, v in best_for_type["params"].items()
                                   if k not in ("n_jobs", "random_state",
                                                "use_label_encoder", "eval_metric")},
                "best_val_f1_macro":  best_val_f1_for_type,
                "best_val_metrics":   _serialize(best_for_type["val_metrics"]),
                "best_test_metrics":  _serialize(best_for_type["test_metrics"]),
                "configs_tested":     len(results_for_type),
                "all_configs":        results_for_type,
            }
            # Save best model
            model_path = models_dir / f"{clf_type}_best.joblib"
            joblib.dump(best_for_type["classifier_obj"], model_path)

            if best_val_f1_for_type > best_val_f1:
                best_val_f1 = best_val_f1_for_type
                best_clf_name = clf_type
                best_test_metrics = best_for_type["test_metrics"]

    # ── MLP ─────────────────────────────────────────────────────────────────
    mlp_cfg = clf_config.get("mlp", {})
    if mlp_cfg.get("enabled", True):
        print(f"\n  [mlp]")
        hidden_list = mlp_cfg.get("hidden_dims", [[512], [512, 256], [1024, 512]])
        dropout     = float(mlp_cfg.get("dropout",    0.3))
        lr          = float(mlp_cfg.get("lr",         0.001))
        num_epochs  = int(mlp_cfg.get("num_epochs",  30))
        batch_size  = int(mlp_cfg.get("batch_size",  512))

        best_mlp = None
        best_mlp_val_f1 = -1.0
        mlp_results = []

        for hidden_dims in hidden_list:
            print(f"    hidden_dims={hidden_dims}", end=" ... ", flush=True)
            result = train_mlp(
                X_train, y_train, X_val, y_val, X_test, y_test,
                num_classes=num_classes, task=task,
                hidden_dims=hidden_dims, dropout=dropout,
                lr=lr, num_epochs=num_epochs, batch_size=batch_size,
                device=device,
                class_weights=class_weights_torch,
            )
            f1 = result["val_metrics"]["f1_macro"]
            print(f"val_f1={f1:.4f}")
            mlp_results.append({
                "hidden_dims": hidden_dims,
                "val_f1_macro":  f1,
                "test_f1_macro": result["test_metrics"]["f1_macro"],
            })
            if f1 > best_mlp_val_f1:
                best_mlp_val_f1 = f1
                best_mlp = result

        if best_mlp is not None:
            all_results["mlp"] = {
                "best_params":       best_mlp["params"],
                "best_val_f1_macro": best_mlp_val_f1,
                "best_val_metrics":  _serialize(best_mlp["val_metrics"]),
                "best_test_metrics": _serialize(best_mlp["test_metrics"]),
                "configs_tested":    len(mlp_results),
                "all_configs":       mlp_results,
            }
            if best_mlp_val_f1 > best_val_f1:
                best_val_f1 = best_mlp_val_f1
                best_clf_name = "mlp"
                best_test_metrics = best_mlp["test_metrics"]

    # ── Save sweep results ───────────────────────────────────────────────────
    sweep_path = exp_dir / "sweep_results.json"
    with open(sweep_path, "w") as f:
        json.dump({
            "experiment":    config.get("experiment_name", ""),
            "task":          task,
            "num_classes":   num_classes,
            "completed_at":  datetime.now().isoformat(),
            "best_classifier": best_clf_name,
            "best_val_f1_macro": best_val_f1,
            "results":       all_results,
        }, f, indent=2, default=str)

    # ── Save final_metrics.json (best result, consistent with other experiments)
    if best_test_metrics is not None:
        final = {
            "test":          _serialize(best_test_metrics),
            "best_classifier": best_clf_name,
            "best_val_f1_macro": best_val_f1,
            "all_classifiers_val_f1": {
                k: v["best_val_f1_macro"] for k, v in all_results.items()
            },
        }
        with open(exp_dir / "final_metrics.json", "w") as f:
            json.dump(final, f, indent=2, default=str)

    print(f"\n  Best: {best_clf_name} (val F1-macro={best_val_f1:.4f})")

    return {
        "best_classifier":  best_clf_name,
        "best_val_f1_macro": best_val_f1,
        "best_test_metrics": best_test_metrics,
        "all_results":      all_results,
    }


def _serialize(d: Dict) -> Dict:
    """Remove non-serializable values (confusion_matrix lists are kept)."""
    out = {}
    for k, v in d.items():
        if k == "per_class" and isinstance(v, dict):
            out[k] = {cls: {m: float(val) if isinstance(val, (int, float, np.floating))
                             else int(val)
                             for m, val in metrics.items()}
                      for cls, metrics in v.items()}
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, list):
            out[k] = v  # confusion_matrix
        else:
            out[k] = v
    return out
