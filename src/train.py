"""
train.py
========
The training script. This is the "ML loop" the playbook keeps mentioning, written
out end to end. Run it from the PROJECT ROOT:

    # simplest — trains both models with good defaults, saves to models/
    python -m src.train

    # with hyperparameter tuning (slower, usually a bit more accurate)
    python -m src.train --tune

    # also log the run to Weights & Biases (needs `wandb login` first)
    python -m src.train --wandb

What it does, in order:
    1. Load + clean train.csv                (src.features.load_and_clean)
    2. Drop rows we genuinely can't use      (no target, no GPS)
    3. Split off a held-out test set         (train_test_split)
    4. Cross-validate on the training part    (honest estimate, no leakage)
    5. (optional) Randomised hyperparam search
    6. Fit the final pipeline on all training data
    7. Evaluate ONCE on the held-out test set (the number you report)
    8. Save both pipelines + metadata + metrics to models/

Why a held-out test set AND cross-validation? CV tells you how stable the model is
across different slices of the *training* data (great for choosing settings). The
held-out set is data the model has never touched in any form — it's your unbiased
final grade. Touch it once, at the end.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)

from src.features import (
    CATEGORICAL_FEATURES,
    COORD_COLS,
    LATE_THRESHOLD_MIN,
    NUMERIC_FEATURES,
    TARGET,
    load_and_clean,
    make_late_target,
)
from src.pipeline import build_classifier_pipeline, build_regressor_pipeline

# Where trained artifacts land. The API reads from exactly here.
MODELS_DIR = "models"
DATA_PATH = os.path.join("data", "train.csv")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data(path: str):
    """Load, clean, drop unusable rows, and return X, y_reg, y_clf."""
    df = load_and_clean(path)
    n_start = len(df)

    # We can only learn from rows that have a target...
    df = df.dropna(subset=[TARGET])
    # ...and a real distance (both restaurant and delivery GPS present).
    df = df.dropna(subset=COORD_COLS)
    print(f"Rows: {n_start} loaded -> {len(df)} usable "
          f"({n_start - len(df)} dropped for missing target/GPS)")

    y_reg = df[TARGET].astype(float)          # minutes
    y_clf = make_late_target(y_reg)           # 1 = late, 0 = on time
    X = df.drop(columns=[TARGET])             # the pipeline engineers features from X
    return X, y_reg, y_clf


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------
def train_regressor(X_train, y_train, X_test, y_test, tune: bool):
    print("\n=== Regressor: predicting delivery time (minutes) ===")
    pipe = build_regressor_pipeline()

    # Cross-validation on the training set. scoring is negative MAE because sklearn
    # maximises scores; we negate back to a friendly "average minutes off".
    cv_mae = -cross_val_score(
        pipe, X_train, y_train, cv=5,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    print(f"5-fold CV MAE: {cv_mae.mean():.2f} +/- {cv_mae.std():.2f} min")

    if tune:
        pipe = _tune(
            pipe, X_train, y_train,
            scoring="neg_mean_absolute_error",
            label="regressor",
        )

    pipe.fit(X_train, y_train)

    # Final, honest evaluation on data the model has never seen.
    preds = pipe.predict(X_test)
    metrics = {
        "cv_mae_mean": float(cv_mae.mean()),
        "cv_mae_std": float(cv_mae.std()),
        "test_mae": float(mean_absolute_error(y_test, preds)),
        "test_rmse": float(root_mean_squared_error(y_test, preds)),
        "test_r2": float(r2_score(y_test, preds)),
    }
    print(f"Held-out  MAE : {metrics['test_mae']:.2f} min")
    print(f"Held-out  RMSE: {metrics['test_rmse']:.2f} min")
    print(f"Held-out  R^2 : {metrics['test_r2']:.3f}")
    return pipe, metrics


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def train_classifier(X_train, y_train, X_test, y_test, tune: bool):
    print("\n=== Classifier: will the order be late? ===")
    print(f"(late = takes more than {LATE_THRESHOLD_MIN} min; "
          f"{y_train.mean():.1%} of training orders are late)")
    pipe = build_classifier_pipeline()

    # ROC-AUC = how well the model ranks late vs on-time orders, regardless of the
    # 0.5 threshold. 0.5 is coin-flip, 1.0 is perfect.
    cv_auc = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1,
    )
    print(f"5-fold CV ROC-AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

    if tune:
        pipe = _tune(pipe, X_train, y_train, scoring="roc_auc", label="classifier")

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]   # P(late)
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds)),
        "test_recall": float(recall_score(y_test, preds)),
        "test_f1": float(f1_score(y_test, preds)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
    }
    print(f"Held-out  Accuracy : {metrics['test_accuracy']:.3f}")
    print(f"Held-out  Precision: {metrics['test_precision']:.3f}")
    print(f"Held-out  Recall   : {metrics['test_recall']:.3f}")
    print(f"Held-out  ROC-AUC  : {metrics['test_roc_auc']:.3f}")
    return pipe, metrics


# ---------------------------------------------------------------------------
# Hyperparameter tuning (shared by both models)
# ---------------------------------------------------------------------------
def _tune(pipe, X, y, scoring: str, label: str):
    """Randomised search over a few XGBoost knobs.

    RandomizedSearchCV samples ``n_iter`` random combinations instead of trying
    every one (grid search). For a handful of continuous knobs, random sampling
    finds a near-best config far faster. Note the ``model__`` prefix: it targets
    the step named "model" inside the Pipeline.
    """
    print(f"Tuning {label} (RandomizedSearchCV)...")
    param_distributions = {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [4, 5, 6, 8, 10],
        "model__learning_rate": [0.02, 0.05, 0.1, 0.15],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=15,
        scoring=scoring,
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X, y)
    print(f"Best {label} params: {search.best_params_}")
    print(f"Best {label} CV score: {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Metadata for the API / Flutter dropdowns
# ---------------------------------------------------------------------------
def build_metadata(X: pd.DataFrame) -> dict:
    """Save the valid dropdown options + numeric ranges so the app can build a form
    without hard-coding anything. The API serves this at /options."""
    options = {
        col: sorted(X[col].dropna().unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    numeric_ranges = {}
    for col in ["Delivery_person_Age", "Delivery_person_Ratings",
                "Vehicle_condition", "multiple_deliveries"]:
        s = pd.to_numeric(X[col], errors="coerce")
        numeric_ranges[col] = {"min": float(s.min()), "max": float(s.max()),
                               "median": float(s.median())}
    return {
        "categorical_options": options,
        "numeric_ranges": numeric_ranges,
        "late_threshold_min": LATE_THRESHOLD_MIN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train delivery time models.")
    parser.add_argument("--data", default=DATA_PATH, help="path to train.csv")
    parser.add_argument("--tune", action="store_true",
                        help="run RandomizedSearchCV hyperparameter tuning")
    parser.add_argument("--wandb", action="store_true",
                        help="log metrics to Weights & Biases (needs `wandb login`)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="fraction held out for final evaluation")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- data ---
    X, y_reg, y_clf = prepare_data(args.data)

    # One split, reused for both models so they're evaluated on identical rows.
    X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
        X, y_reg, y_clf, test_size=args.test_size, random_state=RANDOM_STATE,
    )
    print(f"Train rows: {len(X_train)} | Held-out rows: {len(X_test)}")

    # --- optional experiment tracking ---
    run = _init_wandb(args) if args.wandb else None

    # --- train both models ---
    reg_pipe, reg_metrics = train_regressor(
        X_train, yreg_train, X_test, yreg_test, args.tune)
    clf_pipe, clf_metrics = train_classifier(
        X_train, yclf_train, X_test, yclf_test, args.tune)

    # --- persist everything the API needs ---
    joblib.dump(reg_pipe, os.path.join(MODELS_DIR, "regressor.joblib"))
    joblib.dump(clf_pipe, os.path.join(MODELS_DIR, "classifier.joblib"))

    metadata = build_metadata(X)
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    all_metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "tuned": args.tune,
        "regressor": reg_metrics,
        "classifier": clf_metrics,
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    if run is not None:
        import wandb
        wandb.log({**_flatten("reg", reg_metrics), **_flatten("clf", clf_metrics)})
        run.finish()

    print("\nSaved to models/: regressor.joblib, classifier.joblib, "
          "metadata.json, metrics.json")
    print("Done. Next: `uvicorn api.main:app --reload` to serve predictions.")


def _init_wandb(args):
    import wandb
    return wandb.init(
        project="delivery-time-predictor",
        config={"tune": args.tune, "test_size": args.test_size},
        # 'online' if logged in; harmless if you just want the local dashboard.
        mode=os.environ.get("WANDB_MODE", "online"),
    )


def _flatten(prefix, d):
    return {f"{prefix}/{k}": v for k, v in d.items()}


if __name__ == "__main__":
    main()
