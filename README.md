# California House Price Predictor

**ECS 171 Group Project (Winter 2026)** — source code and demo app for the final report submission.

Predicts California median house value from location and housing features using a **Gradient Boosting** model (1990 U.S. Census data). Flask backend + HTML/JS front-end; sliders and presets for the live demo.

**Team 3:** Tianrun Xu, Rakel Munshi, Nidhi Deshmukh, Zain Muhammad, Shreyans Porwal.

---

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5001** — adjust inputs or pick a preset, click **Predict Price**.

---

## Layout

- **`app.py`** — Trains Gradient Boosting (200 trees, depth 5), serves `/`, `/api/metadata`, `/api/examples`, `/api/predict`.
- **`frontend/`** — Single page: form, result, feature-importance bars (from model).
- **`notebooks/`** — EDA, feature selection, model notebooks.
- **`data/`** — Raw and processed CSVs; EDA figures.
- **`report/`** — LaTeX source for the final project report (PDF submitted separately).

---

## Model

8 features (latitude, housing_median_age, median_income, ocean_proximity one-hot, rooms_per_household, bedrooms_per_room). StandardScaler + GradientBoostingRegressor. Test RMSE ≈ $67k, R² ≈ 0.66. Top importance: median_income (~62%), then ocean (e.g. INLAND).
