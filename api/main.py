"""
api/main.py
===========
The FastAPI service. This is what turns a `.joblib` file on disk into something
your Flutter app can call over HTTP.

Run it from the PROJECT ROOT (so `import src...` resolves):

    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs — FastAPI auto-generates an interactive API
explorer from the type hints below. Try POSTing to /predict right there.

The mental model (familiar from Flutter's http/dio):
    Flutter form  --JSON-->  POST /predict  -->  pipeline.predict()  -->  JSON back

Endpoints:
    GET  /          -> service info + whether models loaded
    GET  /health    -> liveness probe (for deployment platforms)
    GET  /options   -> dropdown choices for the Flutter form (from metadata.json)
    POST /predict   -> the actual prediction
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.features import LATE_THRESHOLD_MIN

MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# Load models ONCE at startup, not per request. Loading joblib on every call
# would make the API painfully slow. The `lifespan` handler is FastAPI's modern
# way to run setup code before the server accepts traffic (and teardown after).
# ---------------------------------------------------------------------------
_regressor = None
_classifier = None
_metadata = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _regressor, _classifier, _metadata
    reg_path = os.path.join(MODELS_DIR, "regressor.joblib")
    clf_path = os.path.join(MODELS_DIR, "classifier.joblib")
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(reg_path):
        _regressor = joblib.load(reg_path)
    if os.path.exists(clf_path):
        _classifier = joblib.load(clf_path)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _metadata = json.load(f)
    yield  # <- server runs while we're "paused" here; code after yield is teardown


app = FastAPI(
    title="Delivery Time Predictor",
    description="Predicts food-delivery time and the probability an order is late.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow a browser/Flutter-web client on another origin to call us. For a real
# deployment you'd restrict allow_origins to your app's domain; "*" is fine locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request schema. Pydantic validates + documents the input automatically.
# Field names match the raw CSV columns EXACTLY, so we can build a one-row
# DataFrame and hand it straight to the pipeline (which does the rest).
#
# Everything has a sensible default so the Flutter form only has to send the few
# fields a user actually cares about (location + time). Defaults reflect a typical
# order and keep the demo working with minimal input.
# ---------------------------------------------------------------------------
class OrderRequest(BaseModel):
    # --- location (the two things the app really asks for) ---
    Restaurant_latitude: float = Field(..., examples=[12.9716])
    Restaurant_longitude: float = Field(..., examples=[77.5946])
    Delivery_location_latitude: float = Field(..., examples=[12.9352])
    Delivery_location_longitude: float = Field(..., examples=[77.6245])

    # --- when ---
    Order_Date: str = Field("14-07-2026", description="dd-mm-yyyy")
    Time_Orderd: str = Field("19:30:00", description="HH:MM:SS, 24-hour")
    Time_Order_picked: str = Field("19:40:00", description="HH:MM:SS, 24-hour")

    # --- context (would come from weather/traffic APIs or dropdowns) ---
    Weatherconditions: str = "Sunny"
    Road_traffic_density: str = "Medium"
    City: str = "Metropolitian"
    Festival: str = "No"
    Type_of_order: str = "Snack"
    Type_of_vehicle: str = "motorcycle"

    # --- driver / order stats (defaults = typical values) ---
    Delivery_person_Age: float = 30.0
    Delivery_person_Ratings: float = 4.6
    Vehicle_condition: int = 1
    multiple_deliveries: float = 1.0


class PredictionResponse(BaseModel):
    predicted_minutes: float
    late_probability: float
    will_be_late: bool
    late_threshold_min: int


@app.get("/")
def root():
    return {
        "service": "Delivery Time Predictor",
        "models_loaded": _regressor is not None and _classifier is not None,
        "docs": "/docs",
        "predict": "POST /predict",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    """Dropdown choices + numeric ranges for building the Flutter form."""
    if not _metadata:
        raise HTTPException(503, "Metadata not loaded. Train the model first.")
    return _metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderRequest):
    if _regressor is None or _classifier is None:
        raise HTTPException(
            503, "Models not loaded. Run `python -m src.train` first.")

    # One JSON object -> one-row DataFrame with the exact raw column names the
    # pipeline expects. The pipeline then engineers distance/time features,
    # one-hot encodes, and predicts — identical maths to training.
    row = pd.DataFrame([order.model_dump()])

    minutes = float(_regressor.predict(row)[0])
    late_prob = float(_classifier.predict_proba(row)[0][1])

    return PredictionResponse(
        predicted_minutes=round(minutes, 1),
        late_probability=round(late_prob, 3),
        will_be_late=late_prob >= 0.5,
        late_threshold_min=LATE_THRESHOLD_MIN,
    )
