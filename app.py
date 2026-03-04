from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "processed", "housing_feature_selected.csv")

FEATURE_COLUMNS: List[str] = [
    "latitude",
    "housing_median_age",
    "median_income",
    "ocean_proximity_INLAND",
    "ocean_proximity_NEAR BAY",
    "ocean_proximity_NEAR OCEAN",
    "rooms_per_household",
    "bedrooms_per_room",
]

FEATURE_LABELS = {
    "latitude":                   "Latitude",
    "housing_median_age":         "Housing Median Age",
    "median_income":              "Median Income",
    "ocean_proximity_INLAND":     "Ocean Proximity: Inland",
    "ocean_proximity_NEAR BAY":   "Ocean Proximity: Near Bay",
    "ocean_proximity_NEAR OCEAN": "Ocean Proximity: Near Ocean",
    "rooms_per_household":        "Rooms per Household",
    "bedrooms_per_room":          "Bedrooms per Room",
}


@dataclass
class HouseFeatures:
    latitude: float
    housing_median_age: float
    median_income: float
    ocean_proximity: str
    rooms_per_household: float
    bedrooms_per_room: float

    def to_model_vector(self) -> np.ndarray:
        oceans = {"INLAND": 0.0, "NEAR BAY": 0.0, "NEAR OCEAN": 0.0}
        key = self.ocean_proximity.upper()
        if key in oceans:
            oceans[key] = 1.0
        return np.array([
            self.latitude,
            self.housing_median_age,
            self.median_income,
            oceans["INLAND"],
            oceans["NEAR BAY"],
            oceans["NEAR OCEAN"],
            self.rooms_per_household,
            self.bedrooms_per_room,
        ], dtype=float)


def train() -> Tuple[GradientBoostingRegressor, StandardScaler, dict]:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run the feature-selection notebook first."
        )

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Tuned hyperparameters from 04_gradient_boosting.ipynb
    model = GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = float(r2_score(y_test, y_pred))

    importances = [
        {"feature": FEATURE_LABELS[col], "importance": round(float(imp), 4)}
        for col, imp in zip(FEATURE_COLUMNS, model.feature_importances_)
    ]
    importances.sort(key=lambda x: x["importance"], reverse=True)

    meta = {
        "model_type": "GradientBoostingRegressor",
        "hyperparameters": {
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_samples_split": 5,
            "n_estimators": 200,
        },
        "test_rmse":  round(rmse, 2),
        "test_r2":    round(r2, 4),
        "cv_rmse":    64789,
        "cv_rmse_std": 1434,
        "train_split": "75% train / 25% test",
        "cv_strategy": "5-fold cross-validation",
        "features":  FEATURE_COLUMNS,
        "feature_importances": importances,
        "dataset_size": len(df),
        "dataset_source": "California Housing (1990 U.S. Census)",
    }
    print(f"  Model ready — Test RMSE: ${rmse:,.0f}  |  R²: {r2:.4f}")
    return model, scaler, meta


# ── App ─────────────────────────────────────────────────────────
app   = Flask(__name__, static_folder="frontend", static_url_path="/")
model: GradientBoostingRegressor | None = None
scaler: StandardScaler | None = None
meta_cache: dict = {}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/metadata")
def metadata():
    return jsonify(meta_cache)


@app.route("/api/examples")
def examples():
    """Preset scenarios useful for a live demo."""
    return jsonify([
        {
            "label": "Bay Area — High Income",
            "values": {
                "latitude": 37,
                "housing_median_age": 30,
                "median_income": 8.5,
                "ocean_proximity": "NEAR BAY",
                "rooms_per_household": 6.0,
                "bedrooms_per_room": 0.15,
            },
        },
        {
            "label": "Coastal — Mid Income",
            "values": {
                "latitude": 34,
                "housing_median_age": 20,
                "median_income": 5.0,
                "ocean_proximity": "NEAR OCEAN",
                "rooms_per_household": 5.5,
                "bedrooms_per_room": 0.22,
            },
        },
        {
            "label": "Inland Valley — Low Income",
            "values": {
                "latitude": 36,
                "housing_median_age": 15,
                "median_income": 2.5,
                "ocean_proximity": "INLAND",
                "rooms_per_household": 4.5,
                "bedrooms_per_room": 0.30,
            },
        },
        {
            "label": "Central Valley — Average",
            "values": {
                "latitude": 36,
                "housing_median_age": 25,
                "median_income": 3.5,
                "ocean_proximity": "INLAND",
                "rooms_per_household": 5.0,
                "bedrooms_per_room": 0.25,
            },
        },
    ])


# Input bounds from training data (avoids extrapolation)
INPUT_BOUNDS = {
    "latitude": (32.0, 41.0),
    "housing_median_age": (1.0, 52.0),
    "median_income": (1.5, 9.0),
    "rooms_per_household": (1.0, 8.0),
    "bedrooms_per_room": (0.05, 0.35),
}


def _clamp(val: float, key: str) -> float:
    lo, hi = INPUT_BOUNDS[key]
    return max(lo, min(hi, val))


@app.route("/api/predict", methods=["POST"])
def predict():
    global model, scaler
    if model is None or scaler is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True) or {}
    try:
        lat = _clamp(float(data["latitude"]), "latitude")
        feats = HouseFeatures(
            latitude=round(lat),
            housing_median_age=_clamp(float(data["housing_median_age"]), "housing_median_age"),
            median_income=_clamp(float(data["median_income"]), "median_income"),
            ocean_proximity=str(data.get("ocean_proximity", "INLAND")),
            rooms_per_household=_clamp(float(data["rooms_per_household"]), "rooms_per_household"),
            bedrooms_per_room=_clamp(float(data["bedrooms_per_room"]), "bedrooms_per_room"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": "Invalid input", "details": str(exc)}), 400

    vec    = feats.to_model_vector().reshape(1, -1)
    vec_s  = scaler.transform(vec)
    pred   = float(model.predict(vec_s)[0])

    return jsonify({
        "input": asdict(feats),
        "prediction": {
            "value":     round(pred, 2),
            "formatted": f"${pred:,.0f}",
        },
    })


def main():
    global model, scaler, meta_cache
    print("Training Gradient Boosting model...")
    model, scaler, meta_cache = train()
    port = int(os.environ.get("PORT", 5001))
    print(f"  Open → http://localhost:{port}/")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
