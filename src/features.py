"""
features.py
===========
Everything that turns the raw, messy delivery CSV into clean, model-ready numbers.

There are TWO very different jobs in this file, and keeping them separate is one of
the most important ideas in applied ML:

1. ``load_and_clean(path)``  -> "parsing".
   This knows about the *specific* quirks of THIS Kaggle CSV: values like
   ``"(min) 24"``, ``"conditions Sunny"``, the literal string ``"NaN"``, trailing
   spaces, sign-flipped GPS coordinates, etc. It runs ONCE when we read a file.
   It is NOT part of the model. Your live API will never see this junk because the
   Flutter app sends clean, typed JSON.

2. ``DeliveryFeatureEngineer``  -> a scikit-learn *transformer*.
   This DERIVES new features (distance in km, hour of day, weekend flag, kitchen
   prep lag) from already-clean columns. It lives INSIDE the pipeline so that the
   EXACT same maths runs during training and during a live prediction. That is how
   you avoid "train/serve skew" — the classic bug where your model scores great in
   the notebook and badly in production because the two code paths computed
   features slightly differently.

Coming from Flutter: think of ``load_and_clean`` as a one-off JSON-from-a-flaky-API
sanitiser, and ``DeliveryFeatureEngineer`` as a reusable widget you drop into every
screen so the rendering is always identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Constants — the single source of truth for "what does the model look at?"
# ---------------------------------------------------------------------------

# Business rule: we call a delivery "late" if it takes longer than this.
# The dataset has no promised-time column, so we invent a sensible SLA. Because
# ~30% of orders exceed 30 minutes, this gives us a nicely balanced classification
# target (not 99%/1%, which would be a nightmare to learn from).
LATE_THRESHOLD_MIN = 30

# The raw target column as it appears in train.csv (test.csv does NOT have it).
TARGET_RAW = "Time_taken(min)"
TARGET = "time_taken_min"  # cleaned name we use everywhere after load_and_clean

# The four GPS columns we turn into a single distance.
COORD_COLS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]

# After feature engineering, THESE are the columns the model actually trains on.
# Numeric features are passed straight to XGBoost (it handles missing values itself).
NUMERIC_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "distance_km",      # engineered
    "prep_lag_min",     # engineered
    "order_hour",       # engineered
    "order_dayofweek",  # engineered
    "is_weekend",       # engineered
]

# Categorical features get one-hot encoded in the pipeline.
CATEGORICAL_FEATURES = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

# The union above is the model's input contract. The API builds a row with these
# same raw columns and hands it to the pipeline.
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns the feature engineer needs as *input* to derive things (before selection).
_ENGINEER_INPUT_COLS = (
    COORD_COLS
    + ["Order_Date", "Time_Orderd", "Time_Order_picked"]
    + ["Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition", "multiple_deliveries"]
    + CATEGORICAL_FEATURES
)


# ---------------------------------------------------------------------------
# 1. Dataset-specific cleaning
# ---------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    """Read the Kaggle delivery CSV and return a tidy DataFrame.

    Steps, each fixing a real problem you can see with ``head data/train.csv``:
      * strip whitespace from every text cell ("High " -> "High")
      * turn the literal strings "NaN"/"nan" into real missing values
      * drop the "conditions " prefix on weather ("conditions Fog" -> "Fog")
      * parse the target "(min) 24" -> 24 (only if the column exists)
      * coerce numeric-looking text columns to real numbers
      * fix GPS: some coordinates are sign-flipped, and ~8% are near-zero junk
    """
    df = pd.read_csv(path)

    # --- strip whitespace on all text columns ("High " -> "High") ---
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # --- weather: "conditions Sunny" -> "Sunny" (do this BEFORE marking missing,
    #     otherwise "conditions NaN" becomes the bare string "NaN" and sneaks past
    #     the replace below, showing up as a fake category in the dropdowns) ---
    if "Weatherconditions" in df.columns:
        df["Weatherconditions"] = (
            df["Weatherconditions"].str.replace("conditions", "", regex=False).str.strip()
        )

    # --- normalise every missing-value marker to a real NaN ---
    df = df.replace({"NaN": np.nan, "nan": np.nan, "": np.nan})

    # --- target: "(min) 24" -> 24 (int). test.csv has no target, so guard it. ---
    if TARGET_RAW in df.columns:
        df[TARGET] = (
            df[TARGET_RAW]
            .astype(str)
            .str.replace("(min)", "", regex=False)
            .str.strip()
        )
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.drop(columns=[TARGET_RAW])

    # --- numeric columns stored as text -> real numbers ---
    for col in ["Delivery_person_Age", "Delivery_person_Ratings",
                "Vehicle_condition", "multiple_deliveries"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- GPS clean-up ---
    # Some latitudes/longitudes are negative (sign-flipped but otherwise valid) —
    # this is India data, so everything should be positive; abs() recovers them.
    # After that, values very close to 0 are genuinely missing/corrupt -> NaN.
    for col in COORD_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").abs()
            df.loc[df[col] < 1.0, col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Helpers used by the transformer (pure functions -> easy to test)
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points (vectorised, NumPy).

    "As the crow flies" distance. Not road distance, but a strong proxy: the
    farther the delivery point, the longer it takes. This single number replaces
    four raw coordinates the model can't use directly.
    """
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius_km * 2 * np.arcsin(np.sqrt(a))


def _parse_time_to_minutes(series: pd.Series) -> pd.Series:
    """'19:45:00' -> 1185.0 (minutes since midnight). Unparseable -> NaN."""
    t = pd.to_datetime(series, format="%H:%M:%S", errors="coerce")
    # Some rows use "H:MM" or other shapes; fall back to a lenient parse for those.
    missing = t.isna()
    if missing.any():
        t2 = pd.to_datetime(series[missing], errors="coerce")
        t.loc[missing] = t2
    return t.dt.hour * 60 + t.dt.minute


# ---------------------------------------------------------------------------
# 2. The feature-engineering transformer (goes INSIDE the pipeline)
# ---------------------------------------------------------------------------
class DeliveryFeatureEngineer(BaseEstimator, TransformerMixin):
    """Derive model features from clean columns.

    Why a class and not just a function? Because scikit-learn Pipelines speak the
    ``fit``/``transform`` protocol. By making feature engineering a transformer:
      * cross-validation refits it fold-by-fold (no leakage),
      * ``joblib.dump(pipeline)`` saves it together with the model,
      * the live API gets identical features for free — it just calls the pipeline.

    This transformer is *stateless*: ``fit`` learns nothing, it only validates and
    returns ``self``. All the work happens in ``transform``.
    """

    # This transformer doesn't learn parameters, but sklearn still calls fit().
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()  # never mutate the caller's DataFrame

        # --- distance: 4 coordinates -> 1 km number ---
        X["distance_km"] = haversine_km(
            X["Restaurant_latitude"], X["Restaurant_longitude"],
            X["Delivery_location_latitude"], X["Delivery_location_longitude"],
        )

        # --- time-of-day features from the order timestamp ---
        ordered = _parse_time_to_minutes(X["Time_Orderd"])
        picked = _parse_time_to_minutes(X["Time_Order_picked"])
        # If the order time is missing, fall back to the pickup time so we still
        # get an hour-of-day signal instead of throwing the whole row away.
        order_minutes = ordered.fillna(picked)
        X["order_hour"] = (order_minutes // 60).astype("float")

        # kitchen prep lag = pickup time - order time, in minutes.
        # Handle orders placed before midnight and picked up after (wrap +24h).
        lag = picked - ordered
        lag = lag.where(lag >= 0, lag + 24 * 60)
        X["prep_lag_min"] = lag

        # --- calendar features from the order date ---
        date = pd.to_datetime(X["Order_Date"], format="%d-%m-%Y", errors="coerce")
        X["order_dayofweek"] = date.dt.dayofweek.astype("float")   # Mon=0 .. Sun=6
        X["is_weekend"] = (date.dt.dayofweek >= 5).astype("float")

        # Return ONLY the columns the model consumes, in a stable order. The next
        # pipeline step (ColumnTransformer) selects these by name.
        return X[MODEL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        # Lets sklearn report readable feature names if asked.
        return np.array(MODEL_FEATURES)


def make_late_target(time_taken_min: pd.Series) -> pd.Series:
    """Regression target -> classification target: 1 if late, else 0."""
    return (time_taken_min > LATE_THRESHOLD_MIN).astype(int)
