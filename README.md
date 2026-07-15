# 🛵 Delivery Time Predictor

Predict how long a food order will take — and the probability it'll be **late** —
from a restaurant location, delivery location, time of day, weather and traffic.

An **XGBoost** model trained on a real (messy) Kaggle food-delivery dataset, wrapped
in a **FastAPI** service, called from a **Flutter** app.

This is **Phase 2** of the *Flutter → AI Engineer* playbook, and it's built to be
**learned from**, not just run. Every file is heavily commented, and this README
explains not only *what* was built but *why* and *how to think* when you build the
next one yourself.

> **Results on held-out data (8,391 orders the model never saw during training):**
> | Model | Metric | Score |
> |---|---|---|
> | Regressor (minutes) | MAE | **3.08 min** |
> | Regressor | R² | **0.826** |
> | Classifier (will be late?) | ROC-AUC | **0.984** |
> | Classifier | Accuracy | **0.934** |
>
> MAE of ~3 minutes means the typical prediction is within 3 minutes of the truth.

---

## Table of contents
1. [The 10-second demo](#the-10-second-demo)
2. [How to run it](#how-to-run-it)
3. [The mental model: the classical ML workflow](#the-mental-model-the-classical-ml-workflow)
4. [Project structure](#project-structure)
5. [Walkthrough: what each file does and why](#walkthrough-what-each-file-does-and-why)
6. [The ideas that matter (read this part twice)](#the-ideas-that-matter-read-this-part-twice)
7. [Flutter → Python: a translation guide](#flutter--python-a-translation-guide)
8. [Deploying the API](#deploying-the-api)
9. [How to approach a project like this from scratch](#how-to-approach-a-project-like-this-from-scratch)
10. [What I'd do next](#what-id-do-next)

---

## The 10-second demo

Start the API, then POST an order:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "Restaurant_latitude": 12.97, "Restaurant_longitude": 77.59,
  "Delivery_location_latitude": 13.10, "Delivery_location_longitude": 77.75,
  "Road_traffic_density": "Jam", "Weatherconditions": "Stormy", "Festival": "Yes"
}'
# -> {"predicted_minutes":42.9,"late_probability":0.972,"will_be_late":true,"late_threshold_min":30}
```

A near delivery on a sunny day with low traffic returns ~16 min and `will_be_late: false`.
The model learned real intuition: **jam + storm + festival = slow**.

---

## How to run it

**Prerequisites:** Python 3.11 and (on macOS, for XGBoost) the OpenMP runtime:

```bash
brew install libomp        # macOS only — XGBoost needs this native library
```

Everything else is already in the project's virtual environment (`.venv/`). If you
ever need to recreate it, this project uses [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync                    # installs the exact deps from uv.lock into .venv/
```

### 1. (optional) Explore the data

```bash
.venv/bin/jupyter lab notebooks/eda.ipynb
```

### 2. Train the models

```bash
.venv/bin/python -m src.train              # ~7 seconds, writes to models/
.venv/bin/python -m src.train --tune       # + hyperparameter search (slower)
.venv/bin/python -m src.train --wandb      # + log the run to Weights & Biases
```

This writes four files into `models/`: `regressor.joblib`, `classifier.joblib`,
`metadata.json` (dropdown options for the app), and `metrics.json` (the scores above).

### 3. Serve the API

```bash
.venv/bin/uvicorn api.main:app --reload
# open http://127.0.0.1:8000/docs  <- interactive, auto-generated API explorer
```

### 4. Run the Flutter app

```bash
cd flutter_app
flutter pub get
flutter run
```

> ⚠️ On an **Android emulator**, change `apiBaseUrl` in `flutter_app/lib/main.dart`
> to `http://10.0.2.2:8000` (the emulator's alias for your computer's localhost).
> On a **real phone**, use your computer's LAN IP, e.g. `http://192.168.1.20:8000`.

---

## The mental model: the classical ML workflow

This is the loop you'll repeat for the rest of your ML career. Burn it in:

```
   ┌─────────┐   ┌─────────┐   ┌──────────────┐   ┌────────┐   ┌──────────┐   ┌────────┐
   │  DATA   │──▶│  CLEAN  │──▶│  FEATURE-ENG  │──▶│ TRAIN  │──▶│ EVALUATE │──▶│  SERVE │
   └─────────┘   └─────────┘   └──────────────┘   └────────┘   └──────────┘   └────────┘
    train.csv     fix junk,     distance, hour,     XGBoost      held-out       FastAPI
                  types, NaN     weekend, prep       + CV         MAE / AUC       + Flutter
```

The single most important rule threaded through all of it: **the transformations you
apply during training must be applied *identically* at prediction time.** Get this
wrong and your model looks great in the notebook and fails in production. Almost every
design decision in this repo exists to guarantee it. (More on this below.)

---

## Project structure

```
DeliveryTimePredictor/
├── data/
│   ├── train.csv              # 45k labelled orders (has Time_taken)
│   └── test.csv               # 11k unlabelled (Kaggle-style; we don't use it for scoring)
├── notebooks/
│   └── eda.ipynb              # exploratory data analysis — understand before modelling
├── src/
│   ├── features.py            # cleaning + the feature-engineering transformer
│   ├── pipeline.py            # scikit-learn Pipeline builders (preprocessing + XGBoost)
│   └── train.py               # the training script: load→split→CV→tune→evaluate→save
├── api/
│   └── main.py                # FastAPI service exposing /predict
├── models/                    # OUTPUT of train.py (committed so the API runs on clone)
│   ├── regressor.joblib       # predicts minutes
│   ├── classifier.joblib      # predicts P(late)
│   ├── metadata.json          # dropdown options + ranges for the app
│   └── metrics.json           # the scores
├── flutter_app/
│   ├── lib/main.dart          # the mobile front-end that calls the API
│   └── pubspec.yaml
├── pyproject.toml / uv.lock   # dependencies
└── README.md                  # you are here
```

**Why this split?** The playbook's rule *"notebook to script"* in action. You *explore*
in a notebook (fast, throwaway, visual), then move the code that survives into `.py`
modules that are importable, testable, and reusable by both training and serving.
The notebook is lab equipment; the `.py` files are the product.

---

## Walkthrough: what each file does and why

### `src/features.py` — turning mess into numbers
Two clearly separated jobs live here, and separating them is a genuine skill:

1. **`load_and_clean(path)`** — *dataset-specific parsing.* This knows the ugly
   details of THIS csv: `"(min) 24"` targets, `"conditions Sunny"`, trailing spaces,
   the literal string `"NaN"`, and GPS coordinates that are sign-flipped or near-zero
   junk. It runs **once** when you read a file and is **not part of the model**. Your
   live API never sees this junk because the app sends clean, typed JSON.

2. **`DeliveryFeatureEngineer`** — a scikit-learn *transformer* that DERIVES features
   (`distance_km`, `order_hour`, `is_weekend`, `prep_lag_min`) from clean columns.
   This lives **inside the pipeline**, so training and serving compute features with
   the exact same code. This is the anti-skew guarantee.

   Notable feature: **haversine distance** collapses four useless raw GPS columns into
   one strong signal (how far, as the crow flies). Trees can't use latitude/longitude
   directly, but "km between the two points" is exactly what drives delivery time.

### `src/pipeline.py` — the Pipeline
Builds two scikit-learn `Pipeline`s that share the same preprocessing:

```
DeliveryFeatureEngineer → ColumnTransformer(passthrough numerics, one-hot categoricals) → XGBoost
```

- **Numeric columns are passed through untouched** — XGBoost handles missing values
  natively and doesn't care about feature scaling (trees split on thresholds, so
  "is age > 30" works whether age is in years or centuries).
- **Categoricals are one-hot encoded** with `handle_unknown="ignore"`, so a city the
  model never saw becomes all-zeros instead of a 500 error.
- The whole thing is **one object**. `pipeline.predict(raw_row)` does *everything*.

### `src/train.py` — the ML loop
Load → clean → drop unusable rows → **hold out a test set** → **cross-validate** →
optionally **tune** → fit → **evaluate once** → save. Run it and read its console
output; it narrates each step. It also writes `metadata.json` so the app's dropdowns
are generated from the data, never hard-coded.

### `api/main.py` — the server
Loads the two `.joblib` pipelines **once at startup** (via FastAPI's `lifespan`
handler), then exposes:
- `POST /predict` — the prediction (Pydantic validates the JSON body automatically)
- `GET /options` — dropdown choices for the Flutter form
- `GET /health` — a liveness check for deployment platforms
- `GET /docs` — a free, interactive API explorer generated from the type hints

### `flutter_app/lib/main.dart` — your edge
A single-screen app: pick traffic/weather/festival/distance, tap a button, POST to
`/predict`, render the result. This is where a Flutter developer pulls ahead — very
few ML beginners can put a model behind a real mobile UI.

---

## The ideas that matter (read this part twice)

These six ideas are the transferable lessons. The delivery dataset is just the vehicle.

### 1. Train/serve skew is the #1 production ML bug — the Pipeline prevents it
If you clean and encode data in a notebook, train a raw model, then re-implement the
cleaning by hand in your API, the two code paths **will** drift apart. Predictions
silently get worse. The fix: put every transformation *inside* a `Pipeline`, save the
whole pipeline with `joblib`, and have the API call `.predict()`. One code path. This
is why feature engineering here is a **transformer**, not a loose function.

### 2. Data leakage: cross-validate the *whole* pipeline, never the data first
"Leakage" = information from the validation/test set sneaking into training, giving a
falsely high score. The classic mistake: fit a scaler or encoder on the *entire*
dataset before splitting — now the training fold has "seen" the validation fold's
statistics. Because we pass the **pipeline** (not pre-transformed data) to
`cross_val_score`, scikit-learn refits preprocessing on each fold's training portion
only. Correct by construction.

### 3. Cross-validation vs. a held-out test set — you need both, for different jobs
- **Cross-validation** (5-fold here): splits the *training* data 5 ways, trains on 4,
  validates on 1, rotates. Gives a *stable* score with an error bar — great for
  comparing settings and catching a lucky/unlucky split. Our CV MAE was
  `3.12 ± 0.02` min: low variance, so the model is stable.
- **Held-out test set** (20%): data touched *once*, at the very end, for the number
  you actually report. Our held-out MAE was `3.08` min — consistent with CV, which is
  exactly what you want to see.

> Rule of thumb: use CV to *make decisions*, use the held-out set to *report a grade*.
> If you tune against the test set, you've leaked and the grade is fiction.

### 4. Regression vs. classification — same features, different questions
We train **two** models on the same inputs:
- a **regressor** answering *"how many minutes?"* (a number → MAE, RMSE, R²)
- a **classifier** answering *"will it be late?"* (a yes/no → accuracy, precision,
  recall, ROC-AUC)

They're independent, so they can mildly disagree near the boundary (e.g. the regressor
says 37 min while the classifier says 44% late). That's expected, not a bug — pick the
one that fits the UI. "84% chance you'll wait longer than 30 min" is often more useful
to a user than a single point estimate.

Why these classifier metrics? **Accuracy** alone lies when classes are imbalanced
(here ~30% late, so always guessing "on time" scores 70% while being useless).
**Precision** = of the orders we flagged late, how many were? **Recall** = of the
orders that were actually late, how many did we catch? **ROC-AUC** = how well the
model *ranks* late vs on-time regardless of the 0.5 cutoff (0.5 = coin flip, 1.0 =
perfect).

### 5. Why gradient boosting (XGBoost) wins on tabular data
Neural nets rule images and text, but for **rows-and-columns** data, gradient-boosted
trees are still the champion in 2026. They handle mixed numeric/categorical features,
missing values, and non-linear interactions (traffic × distance × weather) with almost
no tuning, and they train in seconds. "Boosting" = build many shallow trees where each
new tree corrects the previous ensemble's errors. Reach for XGBoost/LightGBM first on
any tabular problem; only escalate to deep learning if they plateau.

### 6. Real data is dirty, and cleaning is most of the job
This dataset had: a target formatted as `"(min) 24"`, a `"conditions "` prefix on
every weather value, the string `"NaN"` instead of real nulls, trailing spaces
everywhere, and ~8% of rows with corrupt GPS. **Roughly 60% of the work was cleaning,
~30% feature engineering, ~10% modelling.** That ratio is normal. A subtle bug lived
in the *order* of cleaning steps — stripping the `"conditions "` prefix had to happen
*before* normalising missing values, or `"conditions NaN"` survived as a fake weather
category. Small ordering bugs like that are the everyday texture of ML work.

---

## Flutter → Python: a translation guide

| You know (Dart/Flutter) | Meet (Python/ML) |
|---|---|
| `pubspec.yaml` + `flutter pub get` | `pyproject.toml` + `uv sync` (deps via `uv`) |
| `List`, `Map`, `for` loops | Python lists/dicts, **but** use NumPy/Pandas vectorised ops instead of loops |
| A `StatelessWidget` tree you compose then build | A scikit-learn `Pipeline` you compose then `.fit()` |
| `http`/`dio` calling a REST API | Same idea — your Flutter app calls FastAPI |
| `json_serializable` models | **Pydantic** models (validate + document JSON automatically) |
| Hot reload | Jupyter cells + `uvicorn --reload` |
| `flutter test` | `pytest` (a natural next addition to this repo) |
| Null safety (`String?`) | Type hints (`float`, `str`) — advisory, not enforced at runtime |
| A widget's `build()` runs every frame | A transformer's `transform()` runs every fit/predict — keep it pure & fast |

The biggest mental shift: in app dev you mostly write **imperative, deterministic**
code. In ML you spend your time **understanding data distributions** and **measuring**
whether a change helped. The code is often short; the judgement is the job.

---

## Deploying the API

The playbook suggests **Modal** or **Railway** (free tiers). The service is a standard
FastAPI app, so any of these work. Minimal Modal example:

```python
# deploy.py  — run: modal deploy deploy.py
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi", "uvicorn", "scikit-learn", "xgboost", "pandas", "joblib")
    .add_local_dir("models", "/root/models")
    .add_local_dir("src", "/root/src")
    .add_local_dir("api", "/root/api")
)
app = modal.App("delivery-time-predictor", image=image)

@app.function()
@modal.asgi_app()
def fastapi_app():
    from api.main import app as web_app
    return web_app
```

Then point the Flutter app's `apiBaseUrl` at the URL Modal prints. That's the
"model running on your own deployed server, called from your phone" moment the
playbook is aiming at.

**Deployment checklist:** commit `models/` (already done), pin your dependencies
(`uv.lock`), restrict CORS `allow_origins` to your app's domain, and add a `/health`
check (already there) so the platform knows the service is alive.

---

## How to approach a project like this from scratch

The reusable recipe, so you can do the next one without a playbook:

1. **Write the README first** (a stub). State what you're predicting and what
   "success" looks like. It forces clarity before code.
2. **Look at the raw data by hand.** `head` the file, open it, read 20 rows. You can't
   clean what you haven't seen.
3. **Explore in a notebook.** Plot the target's distribution. Group by each category
   and eyeball which ones move the target. Form hypotheses.
4. **Decide the target(s).** Regression? Classification? Both? Define fuzzy business
   terms concretely (here: "late" ≡ `> 30 min`).
5. **Build cleaning + feature engineering as reusable functions/transformers** — never
   inline in the notebook only. This is what you'll ship.
6. **Start with a baseline model** and good defaults. Get the *plumbing* working
   end-to-end before optimising anything.
7. **Cross-validate to compare, hold out a test set to report.** Never tune on the
   test set.
8. **Only then tune** hyperparameters — and check the gain is worth the complexity.
9. **Serialize the whole pipeline** and wrap it in an API. Test with `curl`.
10. **Put a UI on it** and ship. A used demo beats a perfect notebook.

If you internalise one thing: **get the end-to-end skeleton working first
(data → dumb model → API → app), then improve each piece.** Beginners try to perfect
the model before they have a pipeline. Build the pipe, then improve the water.

---

## What I'd do next

Ideas to extend this and deepen the learning (in rough order of value):

- **Add a `tests/` folder** with `pytest`: assert `load_and_clean` handles the junk,
  assert the pipeline round-trips a single row. Your Flutter `flutter test` instinct,
  applied here.
- **Run `--tune`** and compare `metrics.json` before/after. Is the accuracy gain worth
  the longer training? (Often it's marginal — a good lesson.)
- **Log to Weights & Biases** (`--wandb`) and compare a few runs visually.
- **Feature honesty:** `prep_lag_min` uses the pickup time, which isn't actually known
  the instant a customer orders. The model barely relies on it (it's not in the top
  features), but a rigorous version would drop it or default it. Good exercise in
  spotting *deployment-time* leakage.
- **Quantile regression** for a *range* ("25–32 min") instead of a point estimate —
  much better UX for a delivery app.
- **Deploy to Modal** and write the blog post the playbook asks for:
  *"I trained an XGBoost model and shipped it to my phone."*

---

*Built as Phase 2 of the Flutter → AI Engineer playbook. Dataset: Kaggle food-delivery
time. Stack: pandas · scikit-learn · XGBoost · FastAPI · Flutter · (optional) W&B.*
