"""
pipeline.py
===========
Builds the scikit-learn ``Pipeline`` objects — the heart of the project.

A Pipeline chains steps so that ``.fit()`` and ``.predict()`` run the WHOLE
sequence (feature engineering -> encoding -> model) as one object. Two payoffs:

  * No leakage. During cross-validation each fold refits every step on that fold's
    training data only. If you one-hot-encoded or scaled *before* splitting, the
    validation fold would leak information into training. The Pipeline makes the
    correct thing the easy thing.
  * One artifact. ``joblib.dump(pipeline, ...)`` saves preprocessing AND the model.
    The API loads one file and calls ``.predict()`` on raw-ish input. No chance of
    the serving code and training code disagreeing.

We build two pipelines that share the SAME preprocessing:
  * a regressor  -> predicts minutes (XGBRegressor)
  * a classifier -> predicts P(late)  (XGBClassifier)

Coming from Flutter: a Pipeline is like a `StatelessWidget` tree — you compose
small widgets (transformers) into one thing you can build (`fit`) and render
(`predict`) as a unit.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

from src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    DeliveryFeatureEngineer,
)


def build_preprocessor() -> ColumnTransformer:
    """Turn engineered columns into a pure-numeric matrix XGBoost can eat.

    * Numeric columns: passthrough. XGBoost handles missing values (NaN) natively
      by learning which branch to send them down, so we don't impute or scale —
      trees don't care about feature scale.
    * Categorical columns: fill missing with the literal "Unknown" (a real category,
      not a guess), then one-hot encode. ``handle_unknown="ignore"`` means a category
      the model never saw at train time (e.g. a new city) becomes all-zeros instead
      of crashing the API.
    """
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# Sensible XGBoost defaults for tabular data of this size (~35k rows).
# Every one of these is a knob train.py can tune; these are just good starting points.
_XGB_DEFAULTS = dict(
    n_estimators=400,        # number of trees
    learning_rate=0.05,      # how much each tree corrects the previous ones
    max_depth=6,             # tree depth -> how many feature interactions per tree
    subsample=0.9,           # row sampling per tree -> fights overfitting
    colsample_bytree=0.9,    # column sampling per tree -> fights overfitting
    min_child_weight=3,      # minimum data in a leaf -> smooths predictions
    tree_method="hist",      # fast histogram algorithm
    n_jobs=-1,               # use all CPU cores
    random_state=42,         # reproducibility
)


def build_regressor_pipeline(**overrides) -> Pipeline:
    """Full pipeline that predicts delivery time in minutes."""
    params = {**_XGB_DEFAULTS, **overrides}
    return Pipeline(steps=[
        ("features", DeliveryFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", XGBRegressor(objective="reg:squarederror", **params)),
    ])


def build_classifier_pipeline(**overrides) -> Pipeline:
    """Full pipeline that predicts the probability an order is late."""
    params = {**_XGB_DEFAULTS, **overrides}
    return Pipeline(steps=[
        ("features", DeliveryFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **params,
        )),
    ])
