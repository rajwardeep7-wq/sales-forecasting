# ============================================================
# PROJECT 3: Sales Forecasting & Predictive Analytics
# Stack: Python, Pandas, NumPy, Scikit-learn, Matplotlib
#
# WHAT IT DOES
#   1. Loads daily sales data (synthetic retail sales out of the box)
#   2. EDA - trend line, monthly totals, weekly seasonality
#   3. Feature engineering - calendar features + lag & rolling means
#   4. Time-based split (last 90 days = test; no shuffling!)
#   5. Trains Linear Regression, Random Forest, Gradient Boosting
#   6. Evaluation - MAE, RMSE, R²
#
# TO USE A REAL DATASET (e.g. Kaggle "Superstore sales" / "Walmart sales"):
#   df = pd.read_csv("sales.csv", parse_dates=["date"])
#   df = df.groupby("date", as_index=False)["sales"].sum()  # daily total
#   Just replace the "LOAD DATA" section. Expects columns: date, sales
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(IMG_DIR, exist_ok=True)
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

# ---------------- 1. LOAD DATA ----------------
# --- Option A: synthetic daily sales (trend + weekly + yearly seasonality) ---
def make_sales_data(start="2022-01-01", days=1095, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    sales = (500                                    # baseline
             + 0.35 * t                             # upward trend
             + 180 * np.sin(2 * np.pi * (t % 7) / 7)        # weekly pattern (weekday vs weekend)
             + 120 * np.sin(2 * np.pi * t / 365.25)         # yearly seasonality
             + rng.normal(0, 60, days))             # noise
    sales += np.where((dates.month == 12) & (dates.day >= 15), 250, 0)  # holiday spike
    return pd.DataFrame({"date": dates, "sales": np.round(sales, 2)})

df = make_sales_data()
# --- Option B: real data ---
# df = pd.read_csv("sales.csv", parse_dates=["date"])
# df = df.groupby("date", as_index=False)["sales"].sum()

df = df.sort_values("date").reset_index(drop=True)
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nStats:\n", df["sales"].describe().round(2))

# ---------------- 2. EDA ----------------
fig, axes = plt.subplots(2, 1, figsize=(13, 8))
axes[0].plot(df["date"], df["sales"], lw=0.8, color="steelblue")
axes[0].set_title("Daily Sales Over Time"); axes[0].set_ylabel("Sales")
df.set_index("date")["sales"].resample("ME").sum().plot(kind="bar", ax=axes[1], color="teal")
axes[1].set_title("Monthly Sales"); axes[1].set_ylabel("Total sales"); axes[1].tick_params(axis="x", labelsize=6)
plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, "sales_trend.png")); plt.close()

# ---------------- 3. FEATURE ENGINEERING ----------------
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek          # 0 = Monday
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
# Lag features: sales from N days ago (most predictive for time series)
for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = df["sales"].shift(lag)
# Rolling statistics: recent average sales
df["rolling_mean_7"] = df["sales"].shift(1).rolling(7).mean()
df["rolling_mean_30"] = df["sales"].shift(1).rolling(30).mean()
df = df.dropna().reset_index(drop=True)  # drop rows with NaN lags

FEATURES = ["year", "month", "day", "dayofweek", "is_weekend",
            "lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_30"]
X, y = df[FEATURES], df["sales"]

# ---------------- 4. TIME-BASED SPLIT (no shuffling for time series!) ----------------
test_days = 90
X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]
y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]
dates_test = df["date"].iloc[-test_days:]
print(f"\nTrain: {len(X_train)} days | Test: {len(X_test)} days")

# ---------------- 5. TRAIN & EVALUATE MODELS ----------------
models = {
    "Linear Regression":   LinearRegression(),
    "Random Forest":       RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                     random_state=RANDOM_STATE),
}

plt.figure(figsize=(13, 5))
plt.plot(dates_test, y_test.values, label="Actual", color="black", lw=2)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name:20s} MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")

    plt.plot(dates_test, y_pred, label=name, alpha=0.85, lw=1.5)

plt.title("Sales Forecast vs Actual (last 90 days)"); plt.xlabel("Date"); plt.ylabel("Sales")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, "sales_forecast_vs_actual.png")); plt.close()

# ---------------- 6. FEATURE IMPORTANCE (best tree model) ----------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
importances.plot(kind="barh", color="teal", figsize=(8, 5))
plt.title("Feature Importance - Random Forest"); plt.tight_layout(); plt.savefig(os.path.join(IMG_DIR, "sales_feature_importance.png")); plt.close()

print("\nSaved plots: sales_trend.png, sales_forecast_vs_actual.png, sales_feature_importance.png")
